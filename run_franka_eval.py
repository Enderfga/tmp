"""
run_franka_eval.py

Evaluates a trained OpenVLA-OFT policy on real Franka robot hardware.
Similar structure to run_libero_eval.py but adapted for real robot deployment.

Usage:
    # Basic usage
    python run_franka_eval.py \
        --pretrained_checkpoint ZechenBai/OpenVLA-OFT \
        --external_camera 327122079691 \
        --wrist_camera 218622273043 \
        --nuc_ip 192.168.1.112 \
        --instruction "pick up the grey mug"
    
    # With custom unnorm key
    python run_franka_eval.py \
        --pretrained_checkpoint ZechenBai/OpenVLA-OFT \
        --custom_unnorm_key robot_rlds_kevin \
        --external_camera 327122079691 \
        --wrist_camera 218622273043 \
        --nuc_ip 192.168.1.112 \
        --instruction "pick up the grey mug"
    
    # Multiple trials
    python run_franka_eval.py \
        --pretrained_checkpoint ZechenBai/OpenVLA-OFT \
        --num_trials 10 \
        --external_camera 327122079691 \
        --wrist_camera 218622273043 \
        --nuc_ip 192.168.1.112 \
        --instruction "pick up the grey mug"
"""

import contextlib
import dataclasses
import datetime
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union

import cv2
import draccus
import numpy as np
import torch
from moviepy import ImageSequenceClip
from scipy.spatial.transform import Rotation as R
from PIL import Image

# Add parent directory to path (for robot_utils and prismatic)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import OpenVLA utilities — openvla_utils is local (in deploy/), robot_utils is in parent repo
from openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Import local modules
import camera_utils
from franka_interface import FrankaInterface, MockRobot
from action_unnormalization import (
    CustomActionUnnormalizer, 
    create_unnormalizer,
    create_kevin_unnormalizer,  # For backward compatibility
    print_action_stats,
    AVAILABLE_BOUNDS,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FrankaEvalConfig:
    """Configuration for Franka robot evaluation."""
    # fmt: off
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    
    use_l1_regression: bool = False                  # If True, uses continuous action head with L1 regression
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (1 = external only, 2 = external + wrist)
    use_proprio: bool = False                        # Whether to include proprio state in input
    
    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 24                     # Number of actions to execute open-loop before requerying policy
    
    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    custom_unnorm_key: str = ""                      # Customized unnorm key name (used if unnorm_key not in model)
    
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    #################################################################################################################
    # Camera parameters
    #################################################################################################################
    external_camera: Optional[str] = None            # External RealSense camera serial number
    wrist_camera: Optional[str] = None               # Wrist RealSense camera serial number
    camera_width: int = 640                          # Camera resolution width
    camera_height: int = 480                         # Camera resolution height
    camera_fps: int = 30                             # Camera FPS
    use_mock_cameras: bool = False                   # Use mock cameras for testing
    
    #################################################################################################################
    # Robot parameters
    #################################################################################################################
    nuc_ip: str = "192.168.1.143"                    # IP address of NUC running Polymetis
    nuc_port: int = 4242                             # Port of ZeroRPC server
    control_mode: str = "eef"                        # Control mode: 'eef' (Cartesian) or 'joint'
    control_frequency: int = 10                      # Control frequency in Hz
    use_mock_robot: bool = False                     # Use mock robot for testing
    
    # Action scaling (only used if use_custom_unnormalization=False)
    position_scale: float = 0.05                     # Position delta scaling (meters)
    rotation_scale: float = 0.05                     # Rotation delta scaling (radians)
    
    # Custom unnormalization (replaces position_scale and rotation_scale if True)
    use_custom_unnormalization: bool = False         # Use custom action bounds from data collection
    action_bounds_name: str = "robot_rlds_kevin"     # Name of bounds to use (see action_unnormalization.py)
    
    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    instruction: str = "pick up the object"          # Task instruction
    num_trials: int = 1                              # Number of trials to run
    max_timesteps: int = 600                         # Maximum timesteps per trial
    
    #################################################################################################################
    # Visualization and logging
    #################################################################################################################
    show_cameras: bool = True                        # Display camera feeds
    save_video: bool = True                          # Save rollout videos
    video_fps: int = 10                              # Video FPS
    
    run_id_note: Optional[str] = None                # Extra note to add to run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    
    seed: int = 7                                    # Random seed
    
    # fmt: on


def validate_config(cfg: FrankaEvalConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    
    # if "image_aug" in str(cfg.pretrained_checkpoint):
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    
    # Check camera serials if not using mock cameras
    if not cfg.use_mock_cameras:
        assert cfg.external_camera is not None, "external_camera serial must be provided (or use --use_mock_cameras)"
        # wrist_camera is optional if num_images_in_input == 1
        if cfg.num_images_in_input > 1:
            assert cfg.wrist_camera is not None, "wrist_camera serial must be provided when num_images_in_input > 1"
    
    # Validate control mode
    assert cfg.control_mode in ["eef", "joint"], f"Invalid control_mode: {cfg.control_mode}"
    if cfg.control_mode == "joint":
        logger.warning("Joint control mode is not fully implemented for delta EE actions!")


def initialize_model(cfg: FrankaEvalConfig):
    """Initialize model and associated components."""
    logger.info("Loading model...")
    
    # Load model
    model = get_model(cfg)
    
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        logger.info("Loading proprioception projector...")
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=PROPRIO_DIM,  # Use constant from prismatic.vla.constants
        )
    
    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        logger.info("Loading action head...")
        action_head = get_action_head(cfg, model.llm_dim)
    
    # Get OpenVLA processor
    processor = None
    if cfg.model_family == "openvla":
        logger.info("Loading processor...")
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)
    
    logger.info("✓ Model loaded successfully")
    logger.info(f"  Device: {next(model.parameters()).device}")
    logger.info(f"  Dtype: {next(model.parameters()).dtype}")
    
    return model, action_head, proprio_projector, processor


def check_unnorm_key(cfg: FrankaEvalConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # If unnorm_key already set, check if it exists
    if cfg.unnorm_key:
        unnorm_key = cfg.unnorm_key
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        
        if unnorm_key in model.norm_stats:
            cfg.unnorm_key = unnorm_key
            logger.info(f"Using unnorm_key: {cfg.unnorm_key}")
            return
    
    # Try custom unnorm key
    if cfg.custom_unnorm_key:
        logger.warning(f"Provided unnorm_key not found in model. Trying custom_unnorm_key: {cfg.custom_unnorm_key}")
        unnorm_key = cfg.custom_unnorm_key
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        
        if unnorm_key in model.norm_stats:
            cfg.unnorm_key = unnorm_key
            logger.info(f"Using custom unnorm_key: {cfg.unnorm_key}")
            return
    
    # List available keys
    logger.error(f"Could not find valid unnorm_key!")
    logger.error(f"Available keys in model: {list(model.norm_stats.keys())}")
    raise RuntimeError(f"Invalid unnorm_key. Available keys: {list(model.norm_stats.keys())}")


def setup_logging(cfg: FrankaEvalConfig):
    """Set up logging to file."""
    # Create run ID
    safe_instruction = "".join(c if c.isalnum() or c == " " else "_" for c in cfg.instruction[:30])
    safe_instruction = safe_instruction.replace(" ", "_")
    run_id = f"EVAL-FRANKA-{safe_instruction}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")
    
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def setup_cameras(cfg: FrankaEvalConfig, log_file=None):
    """Initialize RealSense cameras."""
    log_message("Setting up cameras...", log_file)
    
    if cfg.use_mock_cameras:
        log_message("  Using MOCK cameras (no real hardware)", log_file)
        external_cam = camera_utils.MockCamera(
            width=cfg.camera_width,
            height=cfg.camera_height,
        )
        wrist_cam = camera_utils.MockCamera(
            width=cfg.camera_width,
            height=cfg.camera_height,
        ) if cfg.num_images_in_input > 1 else None
    else:
        # List available cameras
        devices = camera_utils.list_realsense_devices()
        log_message(f"  Found {len(devices)} RealSense device(s)", log_file)
        
        if not devices:
            raise RuntimeError("No RealSense cameras found!")
        
        for i, dev in enumerate(devices):
            log_message(f"    [{i}] {dev['name']} (Serial: {dev['serial_number']})", log_file)
        
        # Initialize cameras
        log_message(f"  Initializing external camera (serial: {cfg.external_camera})...", log_file)
        external_cam = camera_utils.RealSenseCamera(
            serial_number=cfg.external_camera,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
        )
        
        wrist_cam = None
        if cfg.num_images_in_input > 1:
            log_message(f"  Initializing wrist camera (serial: {cfg.wrist_camera})...", log_file)
            wrist_cam = camera_utils.RealSenseCamera(
                serial_number=cfg.wrist_camera,
                width=cfg.camera_width,
                height=cfg.camera_height,
                fps=cfg.camera_fps,
            )
    
    log_message("✓ Cameras ready", log_file)
    return external_cam, wrist_cam


def setup_robot(cfg: FrankaEvalConfig, log_file=None):
    """Connect to Franka robot."""
    if cfg.use_mock_robot:
        log_message("Using MOCK robot (no real hardware)...", log_file)
        robot = MockRobot(
            ip=cfg.nuc_ip,
            port=cfg.nuc_port
        )
    else:
        log_message(f"Connecting to Franka NUC at {cfg.nuc_ip}:{cfg.nuc_port}...", log_file)
        
        try:
            robot = FrankaInterface(
                ip=cfg.nuc_ip,
                port=cfg.nuc_port
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to robot: {e}")
    
    # Test connection
    joint_pos = robot.get_joint_positions()
    gripper_pos = robot.get_gripper_position()
    
    log_message("✓ Connected to robot", log_file)
    log_message(f"  Joint positions: {joint_pos}", log_file)
    log_message(f"  Gripper position: {gripper_pos[0]:.3f}", log_file)
    
    # Initialize robot controller
    if cfg.control_mode == "joint":
        log_message("  Starting joint impedance controller...", log_file)
        robot.start_joint_impedance(Kq=None, Kqd=None)
    elif cfg.control_mode == "eef":
        log_message("  Starting Cartesian impedance controller...", log_file)
        # [x y z roll pitch yaw]
        Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
        Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
        # Kx = np.array([600.0, 600.0, 600.0, 120.0, 120.0, 120.0])
        # Kxd = np.array([30.0, 30.0, 30.0, 15.0, 15.0, 15.0])
        # Kx = np.array([650.0, 650.0, 650.0, 115.0, 115.0, 115.0])
        # Kxd = np.array([30.0, 30.0, 30.0, 14.0, 14.0, 14.0])
        robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
    
    log_message(f"✓ Robot controller started (mode: {cfg.control_mode})", log_file)
    
    return robot


def apply_delta_pose(current_pose: np.ndarray, delta: np.ndarray,
                     position_scale: float = 0.05, rotation_scale: float = 0.1, current_gripper = 0.75) -> np.ndarray:
    """
    Apply delta pose to current pose.
    
    Args:
        current_pose: Current EE pose [x, y, z, qx, qy, qz, qw] (quaternion format)
        delta: Delta pose [dx, dy, dz, droll, dpitch, dyaw] (normalized -1 to 1)
               Rotation is in RPY (roll-pitch-yaw) format
        position_scale: Scaling factor for position delta (meters)
        rotation_scale: Scaling factor for rotation delta (radians)
        
    Returns:
        Target pose [x, y, z, qx, qy, qz, qw] (position + quaternion)
    """
    # Current pose must be in [x, y, z, qx, qy, qz, qw] format (7D)
    if len(current_pose) != 7:
        raise ValueError(f"Expected 7D pose [x,y,z,qx,qy,qz,qw], got shape {current_pose.shape}")
    
    position = current_pose[:3]
    quat = current_pose[3:7]
    
    # Apply position delta
    if current_gripper < 0.07:
        position_scale = 0.9
        rotation_scale = 0.9
        position_scale = min(position_scale, 1.0)
        rotation_scale = min(rotation_scale, 1.0)

    new_position = position + delta[:3] * position_scale
    
    # Apply rotation delta (RPY format)
    current_rot_euler = R.from_quat(quat).as_euler('xyz', degrees=False)
    
    # Create rotation from delta RPY (roll, pitch, yaw)
    delta_rpy = delta[3:6] * rotation_scale
    
    # Convert RPY to rotation matrix/quaternion
    # scipy uses 'xyz' extrinsic rotations for 'XYZ' convention (roll-pitch-yaw)
    # delta_rot = R.from_euler('xyz', delta_rpy, degrees=False)
    
    # Apply delta rotation to current rotation
    # new_rot = current_rot * delta_rot

    new_rot_euler = current_rot_euler + delta_rpy
    new_rot = R.from_euler('xyz', new_rot_euler, degrees=False)
    
    new_quat = new_rot.as_quat()

    # new_quat = quat
    
    # Return as [x, y, z, qx, qy, qz, qw]
    return np.concatenate([new_position, new_quat])


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until after policy inference completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)
    
    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True
    
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

def crop_right_square(image: Image.Image) -> Image.Image:
    """
    Crop a rightmost square region from a PIL image.
    The square's side length equals the shorter dimension of the image.

    Args:
        image: A PIL Image.Image object.

    Returns:
        A cropped PIL Image.Image of size (short_side, short_side).
    """
    width, height = image.size
    short_side = min(width, height)

    # Align the crop to the right edge, and vertically center if needed
    left = width - short_side
    top = (height - short_side) // 2
    right = left + short_side
    bottom = top + short_side

    cropped = image.crop((left, top, right, bottom))

    return cropped

def prepare_observation(external_img, wrist_img, robot, resize_size):
    """Prepare observation for policy input."""
    # Convert BGR to RGB
    log_message("Preparing observation...", None)
    external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)
    external_img_rgb_pil = Image.fromarray(external_img_rgb)
    external_img_rgb_pil = crop_right_square(external_img_rgb_pil)
    external_img_rgb = np.array(external_img_rgb_pil)
    
    # Resize external image
    from openvla_utils import resize_image_for_policy
    external_img_resized = resize_image_for_policy(external_img_rgb, resize_size)
    
    # Handle wrist image if present
    wrist_img_resized = None
    if wrist_img is not None:
        wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
        wrist_img_resized = resize_image_for_policy(wrist_img_rgb, resize_size)
    
    # Get robot state
    joint_pos = robot.get_joint_positions()
    gripper_pos = robot.get_gripper_position()
    
    # Prepare observation dict
    observation = {
        "full_image": external_img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate([joint_pos, gripper_pos]),
    }
    
    return observation, external_img_rgb


def process_action(action, cfg: FrankaEvalConfig):
    """Process action before sending to robot."""
    # Note: Model outputs gripper in normalized [-1, 1] range
    # After unnormalization, it will be in [0, 1] range where:
    #   0 = close, 1 = open
    # No inversion needed - the unnormalization handles the conversion
    
    return action


def run_episode(
    cfg: FrankaEvalConfig,
    robot,
    external_cam,
    wrist_cam,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    unnormalizer=None,
    log_file=None,
):
    """Run a single episode on the robot."""
    log_message(f"\n{'='*70}", log_file)
    log_message(f"TASK: {cfg.instruction}", log_file)
    log_message(f"{'='*70}", log_file)
    
    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})!",
            log_file
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    
    # Tracking variables
    video_frames = []
    inference_times = []
    action_log = []  # Store [target_ee_pose, current_gripper_state] for each step
    dt = 1.0 / cfg.control_frequency
    
    # Gripper state tracking (require 2 consecutive commands before changing)
    prev_gripper_cmd = None
    current_gripper_state = None  # None = unknown, True = closed, False = open
    
    # Log unnormalization mode
    if cfg.use_custom_unnormalization and unnormalizer is not None:
        log_message("Using CUSTOM unnormalization with data collection bounds", log_file)
    else:
        log_message(f"Using SCALING unnormalization (pos={cfg.position_scale}, rot={cfg.rotation_scale})", log_file)
    
    log_message("Running episode... (Press Ctrl+C to stop early)", log_file)
    
    target_ee_pose = None
    try:
        for t_step in range(cfg.max_timesteps):
            step_start_time = time.time()
            
            # Capture camera images
            ret_ext, external_img, _ = external_cam.read()
            ret_wrist, wrist_img, _ = (True, None, None) if wrist_cam is None else wrist_cam.read()
            
            if not ret_ext or (wrist_cam is not None and not ret_wrist):
                log_message(f"  [Step {t_step}] Failed to capture images!", log_file)
                break
            
            # Prepare observation
            observation, external_img_rgb = prepare_observation(
                external_img, wrist_img, robot, resize_size
            )
            
            # Save frame for video
            if cfg.save_video:
                video_frames.append(external_img_rgb.copy())
            
            # Display cameras
            if cfg.show_cameras and t_step % 5 == 0:
                if wrist_img is not None:
                    display_img = np.hstack([external_img, wrist_img])
                    cv2.imshow('Cameras: External (left) | Wrist (right)', display_img)
                else:
                    cv2.imshow('Camera: External', external_img)
                cv2.waitKey(1)
            
            # Query policy if action queue is empty
            if len(action_queue) == 0:
                with prevent_keyboard_interrupt():
                    inference_start = time.time()
                    
                    # Get action chunk from policy
                    print("Generating action...")
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        cfg.instruction,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    
                    inference_time = (time.time() - inference_start) * 1000
                    inference_times.append(inference_time)
                
                action_queue.extend(actions)
                
                # Debug: Print action info on first query
                if t_step == 0:
                    log_message(f"\n  📊 ACTION FORMAT DEBUG:", log_file)
                    log_message(f"     Action chunk shape: {np.array(actions).shape}", log_file)
                    log_message(f"     First action: {actions[0]}", log_file)
                    log_message(f"     Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]", log_file)
                    log_message(f"     Expected: [dx, dy, dz, droll, dpitch, dyaw, gripper]", log_file)
                    log_message(f"     Note: Rotation is RPY (roll-pitch-yaw) format\n", log_file)
                
                log_message(
                    f"  [Step {t_step:3d}] New action chunk | Inference: {inference_time:.1f}ms",
                    log_file
                )
            
            # Get action from queue
            action = action_queue.popleft()
            
            # Process action (inverts gripper if OpenVLA)
            action = process_action(action, cfg)
            print("Raw action:", action)
            # Unnormalize action if using custom bounds
            if cfg.use_custom_unnormalization and unnormalizer is not None:
                # Action is in [-1, 1] range, unnormalize to physical units
                action_unnormalized = unnormalizer.unnormalize(action)
                print("Unnormalized action:", action_unnormalized)
                
                # Debug: Print unnormalization on first step
                if t_step == 0:
                    log_message(f"\n  🔧 UNNORMALIZATION DEBUG:", log_file)
                    log_message(f"     Normalized action:   {action}", log_file)
                    log_message(f"     Unnormalized action: {action_unnormalized}", log_file)
                    log_message(f"     Position delta (m):  [{action_unnormalized[0]:.4f}, {action_unnormalized[1]:.4f}, {action_unnormalized[2]:.4f}]", log_file)
                    log_message(f"     Rotation delta (rad): [{action_unnormalized[3]:.4f}, {action_unnormalized[4]:.4f}, {action_unnormalized[5]:.4f}]", log_file)
                    log_message(f"     Gripper:             {action_unnormalized[6]:.2f}\n", log_file)
                
                # Use unnormalized action (already in physical units)
                ee_delta = action_unnormalized[:6]
                gripper_cmd = action_unnormalized[6]
            else:
                # Parse action: [6D end-effector delta (xyz + RPY), 1 gripper]
                # Action is in [-1, 1] range, will be scaled by position_scale and rotation_scale
                ee_delta = action[:6]
                gripper_cmd = action[6]
            
            # Debug: Print action details periodically
            if t_step % 10 == 0:
                log_message(
                    f"  [Step {t_step:3d}] Action: pos_delta={ee_delta[:3]}, "
                    f"rot_delta={ee_delta[3:6]}, gripper={gripper_cmd:.2f}",
                    log_file
                )
            
            # Execute action
            if cfg.control_mode == "eef":
                # Get current EE pose
                if target_ee_pose is None:
                    current_ee_pose = robot.get_ee_pose()  # [x, y, z, qx, qy, qz, qw]
                else:
                    current_ee_pose = target_ee_pose
                
                if cfg.use_custom_unnormalization and unnormalizer is not None:
                    # ee_delta is already in physical units, apply directly
                    target_ee_pose = apply_delta_pose(
                        current_ee_pose,
                        ee_delta,
                        position_scale=1.0,  # No additional scaling needed
                        rotation_scale=1.0,   # No additional scaling needed
                        # rotation_scale=0.0   # debugging
                        current_gripper = robot.get_joint_positions_w_gripper()[-1]
                    )
                else:
                    # ee_delta is in [-1, 1] range, apply scaling
                    target_ee_pose = apply_delta_pose(
                        current_ee_pose,
                        ee_delta,
                        position_scale=cfg.position_scale,
                        rotation_scale=cfg.rotation_scale,
                        current_gripper = robot.get_joint_positions_w_gripper()[-1]
                    )
                
                # Read current joint positions BEFORE sending new target
                current_joint_positions = robot.get_joint_positions_w_gripper()
                action_log.append(list(current_joint_positions))
                
                # Send target pose to robot
                print(f"Target EE Pose: {target_ee_pose}")
                robot.update_desired_ee_pose(target_ee_pose)
            else:
                if t_step == 0:
                    log_message("  ⚠️  WARNING: Joint control mode not fully implemented!", log_file)
            
            # Process gripper command with 2-step confirmation
            # After unnormalization: gripper is in [0, 1] range where 0=close, 1=open
            # Franka: control_gripper(True) = close, control_gripper(False) = open
            # Require 2 consecutive steps with same command before changing state
            
            desired_gripper_close = bool(gripper_cmd < 0.7)  # 0=close (True), 1=open (False)
            print('desired_gripper_close:', desired_gripper_close)
            
            # Directly apply gripper command without dual timestep confirmation
            # current_gripper_state = desired_gripper_close
            # robot.control_gripper(current_gripper_state)

            # Check if this is the second consecutive step with the same command
            if prev_gripper_cmd is not None:
                prev_desired_close = bool(prev_gripper_cmd < 0.7)
                
                # If both current and previous command agree, update gripper state
                if desired_gripper_close == prev_desired_close:
                    if current_gripper_state != desired_gripper_close:
                        current_gripper_state = desired_gripper_close
                        robot.control_gripper(current_gripper_state)
                        if t_step % 10 == 0 or t_step < 5:
                            log_message(
                                f"  [Step {t_step:3d}] Gripper state changed to: {'CLOSE' if current_gripper_state else 'OPEN'}",
                                log_file
                            )
                # If commands disagree, keep current state (do nothing)
            else:
                # First step: initialize gripper state based on first command
                current_gripper_state = desired_gripper_close
                robot.control_gripper(current_gripper_state)
            
            
            # Update previous gripper command for next iteration
            prev_gripper_cmd = gripper_cmd
            
            # Regulate control frequency
            elapsed = time.time() - step_start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        log_message("\n\n  Episode interrupted by user (Ctrl+C)", log_file)
    
    finally:
        # Print statistics
        if inference_times:
            log_message(f"\nInference statistics:", log_file)
            log_message(f"  Mean: {np.mean(inference_times):.1f}ms", log_file)
            log_message(f"  Std:  {np.std(inference_times):.1f}ms", log_file)
            log_message(f"  Min:  {np.min(inference_times):.1f}ms", log_file)
            log_message(f"  Max:  {np.max(inference_times):.1f}ms", log_file)
        
        # Save video
        if cfg.save_video and video_frames:
            save_video(video_frames, cfg, log_file)
        
        # Save action log
        if action_log:
            save_action_log(action_log, cfg, log_file)
        
        # Close visualization window
        if cfg.show_cameras:
            cv2.destroyAllWindows()
    
    log_message(f"\nEpisode complete ({len(video_frames)} steps)", log_file)
    return len(video_frames)


def save_video(frames: list, cfg: FrankaEvalConfig, log_file=None):
    """Save rollout video."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_instruction = "".join(c if c.isalnum() or c == " " else "_" for c in cfg.instruction[:30])
    safe_instruction = safe_instruction.replace(" ", "_")
    filename = f"VLAST_examples/rollout_{timestamp}_{safe_instruction}.mp4"
    
    log_message(f"\nSaving video to {filename}...", log_file)
    try:
        clip = ImageSequenceClip(frames, fps=cfg.video_fps)
        try:
            clip.write_videofile(filename, codec="libx264", logger=None)
        except TypeError:
            clip.write_videofile(filename, codec="libx264")
        log_message(f"  ✓ Video saved: {filename}", log_file)
    except Exception as e:
        log_message(f"  ✗ Failed to save video: {e}", log_file)


def save_action_log(action_log: list, cfg: FrankaEvalConfig, log_file=None):
    """Save action log as JSON.
    
    Args:
        action_log: List of action vectors, each containing [joint1, joint2, ..., joint7, gripper, gripper]
        cfg: Configuration object
        log_file: Optional log file handle
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_instruction = "".join(c if c.isalnum() or c == " " else "_" for c in cfg.instruction[:30])
    safe_instruction = safe_instruction.replace(" ", "_")
    filename = f"VLAST_examples/actions_{timestamp}_{safe_instruction}.json"
    
    log_message(f"Saving action log to {filename}...", log_file)
    try:
        # Create the data structure
        action_data = {
            "actions": action_log,
            "metadata": {
                "instruction": cfg.instruction,
                "timestamp": timestamp,
                "num_actions": len(action_log),
                "action_format": "[joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper, gripper]",
                "action_type": "joint positions (9D)",
                "joint_unit": "radians",
                "gripper_unit": "meters (gripper width)",
                "note": "Actions are read from robot.get_joint_positions_w_gripper() before sending new target"
            }
        }
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(action_data, f, indent=2)
        
        log_message(f"  ✓ Action log saved: {filename} ({len(action_log)} actions)", log_file)
    except Exception as e:
        log_message(f"  ✗ Failed to save action log: {e}", log_file)


def cleanup(robot, external_cam, wrist_cam, log_file=None):
    """Clean up resources."""
    log_message("\nCleaning up...", log_file)
    
    if robot is not None:
        try:
            robot.terminate_current_policy()
            robot.close()
            log_message("  ✓ Robot disconnected", log_file)
        except Exception as e:
            log_message(f"  ✗ Error disconnecting robot: {e}", log_file)
    
    if external_cam is not None:
        external_cam.release()
    
    if wrist_cam is not None:
        wrist_cam.release()
    
    cv2.destroyAllWindows()
    log_message("  ✓ Cleanup complete", log_file)


@draccus.wrap()
def main(cfg: FrankaEvalConfig):
    """Main evaluation loop."""
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Validate configuration
    validate_config(cfg)
    
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    log_message("="*70, log_file)
    log_message("OpenVLA-OFT Franka Robot Evaluation", log_file)
    log_message("="*70, log_file)
    log_message(f"Run ID: {run_id}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Instruction: {cfg.instruction}", log_file)
    log_message(f"Num trials: {cfg.num_trials}", log_file)
    log_message("="*70, log_file)
    
    robot = None
    external_cam = None
    wrist_cam = None
    
    try:
        # Initialize model
        model, action_head, proprio_projector, processor = initialize_model(cfg)
        
        # Get image resize size
        resize_size = get_image_resize_size(cfg)
        log_message(f"Image resize size: {resize_size}", log_file)
        
        # Initialize custom unnormalizer if requested
        unnormalizer = None
        if cfg.use_custom_unnormalization:
            log_message("\n" + "="*70, log_file)
            log_message("Initializing custom action unnormalization...", log_file)
            log_message(f"Selected bounds: '{cfg.action_bounds_name}'", log_file)
            try:
                unnormalizer = create_unnormalizer(cfg.action_bounds_name)
                print_action_stats(unnormalizer)
            except ValueError as e:
                log_message(f"\nERROR: {e}", log_file)
                log_message("\nAvailable bounds configurations:", log_file)
                for name in AVAILABLE_BOUNDS.keys():
                    log_message(f"  - {name}", log_file)
                raise
            log_message("="*70, log_file)
        
        # Setup cameras
        external_cam, wrist_cam = setup_cameras(cfg, log_file)
        
        # Setup robot
        robot = setup_robot(cfg, log_file)
        
        log_message("\n" + "="*70, log_file)
        log_message("Setup complete! Starting evaluation...", log_file)
        log_message("="*70 + "\n", log_file)
        
        # Run trials
        total_steps = 0
        for trial_idx in range(cfg.num_trials):
            log_message(f"\n{'='*70}", log_file)
            log_message(f"TRIAL {trial_idx + 1}/{cfg.num_trials}", log_file)
            log_message(f"{'='*70}", log_file)
            
            # Run episode
            steps = run_episode(
                cfg,
                robot,
                external_cam,
                wrist_cam,
                model,
                resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                unnormalizer=unnormalizer,
                log_file=log_file,
            )
            
            total_steps += steps
            
            # If running multiple trials, wait for user confirmation to continue
            if trial_idx < cfg.num_trials - 1:
                log_message("\n" + "-"*70, log_file)
                input("Press Enter to start next trial (or Ctrl+C to stop)...")
                robot = setup_robot(cfg, log_file)
        
        # Final summary
        log_message(f"\n{'='*70}", log_file)
        log_message("EVALUATION COMPLETE", log_file)
        log_message(f"{'='*70}", log_file)
        log_message(f"Total trials: {cfg.num_trials}", log_file)
        log_message(f"Total steps: {total_steps}", log_file)
        log_message(f"Average steps per trial: {total_steps / cfg.num_trials:.1f}", log_file)
        log_message(f"{'='*70}", log_file)
        
    except Exception as e:
        log_message(f"\n❌ Error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
    
    finally:
        # Cleanup
        cleanup(robot, external_cam, wrist_cam, log_file)
        
        if log_file:
            log_file.close()
            log_message(f"\nLog saved to: {local_log_filepath}", None)


if __name__ == "__main__":
    main()
