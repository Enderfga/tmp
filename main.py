"""Main script for running OpenVLA-OFT policy on Franka robot.

This script runs on your workstation with GPU and:
1. Loads the OpenVLA-OFT policy locally (on GPU)
2. Captures images from RealSense cameras
3. Connects to the Franka NUC via ZeroRPC
4. Runs policy inference and executes actions

Usage:
    # With real hardware:
    python main.py --instruction "pick up the red block"
    
    # With mock cameras and robot (testing):
    python main.py --use-mock-cameras --use-mock-robot --instruction "test task"
    
    # Custom config:
    python main.py --nuc-ip 192.168.1.143 --external-camera <serial> --instruction "..."
"""

import contextlib
import dataclasses
import datetime
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from moviepy import ImageSequenceClip
from scipy.spatial.transform import Rotation as R

# Add parent directory to path to import experiments modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import OpenVLA utilities
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Import local modules
import camera_utils
import config as cfg
from franka_interface import FrankaInterface, MockRobot


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.
    
    Args:
        euler: Euler angles [rx, ry, rz] in radians
        
    Returns:
        Quaternion [qx, qy, qz, qw]
    """
    rot = R.from_euler('xyz', euler)
    quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
    return quat


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles.
    
    Args:
        quat: Quaternion [qx, qy, qz, qw]
        
    Returns:
        Euler angles [rx, ry, rz] in radians
    """
    rot = R.from_quat(quat)
    euler = rot.as_euler('xyz')
    return euler


def apply_delta_pose(current_pose: np.ndarray, delta: np.ndarray, 
                     position_scale: float = 0.05, rotation_scale: float = 0.1) -> np.ndarray:
    """
    Apply delta pose to current pose.
    
    Args:
        current_pose: Current EE pose [x, y, z, qx, qy, qz, qw] (quaternion format)
        delta: Delta pose [dx, dy, dz, dax, day, daz] (normalized -1 to 1)
               Rotation is in axis-angle format (as used by LIBERO/robosuite)
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
    new_position = position + delta[:3] * position_scale
    
    # Apply rotation delta (axis-angle format)
    # Convert current quaternion to rotation matrix
    current_rot = R.from_quat(quat)
    
    # Create rotation from delta axis-angle
    # Axis-angle is already in the correct format: [ax, ay, az] where magnitude = angle
    delta_axis_angle = delta[3:6] * rotation_scale
    angle = np.linalg.norm(delta_axis_angle)
    
    if angle > 1e-6:  # Only apply rotation if angle is significant
        axis = delta_axis_angle / angle
        delta_rot = R.from_rotvec(delta_axis_angle)  # rotvec = axis * angle
        
        # Compose rotations
        new_rot = current_rot * delta_rot
    else:
        new_rot = current_rot
    
    new_quat = new_rot.as_quat()
    
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


class OpenVLAFrankaRunner:
    """Manages OpenVLA policy inference and Franka robot control."""
    
    def __init__(self, config: cfg.Config):
        self.config = config
        self.robot: Optional[FrankaInterface] = None
        self.external_cam = None
        self.wrist_cam = None
        
        # OpenVLA components
        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        
    def setup(self):
        """Initialize all components."""
        print("=" * 70)
        print("Setting up OpenVLA-OFT Policy Runner for Franka")
        print("=" * 70)
        
        # 1. Setup cameras
        self._setup_cameras()
        
        # 2. Connect to robot
        self._connect_robot()
        
        # 3. Load OpenVLA policy model (locally on this GPU workstation)
        self._load_policy()
        
        print("\n" + "=" * 70)
        print("Setup complete! Ready to run.")
        print("=" * 70 + "\n")
    
    def _setup_cameras(self):
        """Initialize RealSense cameras."""
        print("\n[1/3] Setting up cameras...")
        
        if self.config.camera.use_mock_cameras:
            print("  Using MOCK cameras (no real hardware)")
            self.external_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
        else:
            # List available cameras
            devices = camera_utils.list_realsense_devices()
            print(f"  Found {len(devices)} RealSense device(s)")
            
            if not devices:
                raise RuntimeError(
                    "No RealSense cameras found! "
                    "Run 'python camera_utils.py' to list devices, "
                    "or use --use-mock-cameras for testing."
                )
            
            for i, dev in enumerate(devices):
                print(f"    [{i}] {dev['name']} (Serial: {dev['serial_number']})")
            
            # Initialize cameras
            print(f"  Initializing external camera (serial: {self.config.camera.external_camera_serial})...")
            self.external_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.external_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            
            print(f"  Initializing wrist camera (serial: {self.config.camera.wrist_camera_serial})...")
            self.wrist_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.wrist_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
        
        print("  ✓ Cameras ready")
    
    def _connect_robot(self):
        """Connect to Franka robot via ZeroRPC."""
        if self.config.robot.use_mock_robot:
            print(f"\n[2/3] Using MOCK robot (no real hardware)...")
            self.robot = MockRobot(
                ip=self.config.robot.nuc_ip,
                port=self.config.robot.nuc_port
            )
        else:
            print(f"\n[2/3] Connecting to Franka NUC at {self.config.robot.nuc_ip}:{self.config.robot.nuc_port}...")
            
            try:
                self.robot = FrankaInterface(
                    ip=self.config.robot.nuc_ip,
                    port=self.config.robot.nuc_port
                )
            except Exception as e:
                raise RuntimeError(f"Failed to connect to robot: {e}")
        
        # Test connection by getting robot state
        joint_pos = self.robot.get_joint_positions()
        gripper_pos = self.robot.get_gripper_position()
        
        print(f"  ✓ Connected to robot")
        print(f"    Joint positions: {joint_pos}")
        print(f"    Gripper position: {gripper_pos[0]:.3f}")
        
        # Initialize robot controller
        if self.config.robot.control_mode == "joint":
            print("  Starting joint impedance controller...")
            self.robot.start_joint_impedance(Kq=None, Kqd=None)
        elif self.config.robot.control_mode == "eef":
            print("  Starting Cartesian impedance controller...")
            Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
            self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
        
        print(f"  ✓ Robot controller started")
        print(f"  Control mode: {self.config.robot.control_mode}")
        if self.config.robot.control_mode == "eef":
            print(f"  ℹ️  Using Cartesian (end-effector) impedance control")
        else:
            print(f"  ⚠️  WARNING: Using joint control mode - may not work with EE delta actions!")
    
    def _load_policy(self):
        """Load OpenVLA-OFT policy model on local GPU."""
        print(f"\n[3/3] Loading OpenVLA-OFT policy model...")
        print(f"  Checkpoint: {self.config.policy.pretrained_checkpoint}")
        print(f"  Unnorm key: {self.config.policy.unnorm_key}")
        
        # Load VLA model
        print("  Loading VLA model...")
        self.vla = get_vla(self.config.policy)
        
        # Load processor
        print("  Loading image processor...")
        self.processor = get_processor(self.config.policy)
        
        # Load action head for continuous actions
        if self.config.policy.use_l1_regression or self.config.policy.use_diffusion:
            print("  Loading action head...")
            self.action_head = get_action_head(self.config.policy, self.vla.llm_dim)
        
        # Load proprio projector if using proprioception
        if self.config.policy.use_proprio:
            print("  Loading proprioception projector...")
            self.proprio_projector = get_proprio_projector(
                self.config.policy, 
                self.vla.llm_dim, 
                PROPRIO_DIM
            )
        
        print("  ✓ Policy loaded and ready")
        print(f"    Model device: {next(self.vla.parameters()).device}")
        print(f"    Model dtype: {next(self.vla.parameters()).dtype}")
    
    def run_rollout(self, instruction: str):
        """
        Execute one rollout with the given instruction.
        
        Args:
            instruction: Natural language task instruction
        """
        print(f"\n{'=' * 70}")
        print(f"TASK: {instruction}")
        if self.config.camera.use_mock_cameras or self.config.robot.use_mock_robot:
            print(f"MODE: TESTING (Mock: cameras={self.config.camera.use_mock_cameras}, "
                  f"robot={self.config.robot.use_mock_robot})")
        print(f"{'=' * 70}")
        
        # Tracking variables
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        
        # For video recording
        video_frames = []
        
        # Control timing
        dt = 1.0 / self.config.robot.control_frequency
        
        print("Running rollout... (Press Ctrl+C to stop early)")
        if self.config.robot.use_mock_robot:
            print("  [Note: Robot commands are simulated, not executed on hardware]")
        
        # Statistics tracking
        inference_times = []
        
        try:
            for t_step in range(self.config.max_timesteps):
                step_start_time = time.time()
                
                # 1. Get robot state
                joint_pos = self.robot.get_joint_positions()  # (7,)
                gripper_pos = self.robot.get_gripper_position()  # (1,) array
                
                # 2. Capture camera images
                ret_ext, external_img, _ = self.external_cam.read()
                ret_wrist, wrist_img, _ = self.wrist_cam.read()
                
                if not ret_ext or not ret_wrist:
                    print(f"  [Step {t_step}] Failed to capture images!")
                    break
                
                # Convert BGR to RGB
                external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)
                wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                
                # Save frame for video
                if self.config.save_video:
                    video_frames.append(external_img_rgb.copy())
                
                # Display cameras
                if self.config.show_cameras and t_step % 5 == 0:
                    display_img = np.hstack([external_img, wrist_img])
                    cv2.imshow('Cameras: External (left) | Wrist (right)', display_img)
                    cv2.waitKey(1)
                
                # 3. Query policy if needed
                if (actions_from_chunk_completed == 0 or 
                    actions_from_chunk_completed >= self.config.policy.open_loop_horizon):
                    
                    actions_from_chunk_completed = 0
                    
                    # Prepare observation dict for OpenVLA
                    observation = {
                        "full_image": external_img_rgb,
                        "wrist_image": wrist_img_rgb,
                        "state": np.concatenate([joint_pos, gripper_pos]),
                    }
                    
                    # Run OpenVLA policy inference (locally on this GPU)
                    with prevent_keyboard_interrupt():
                        inference_start = time.time()
                        
                        # Get action chunk from OpenVLA
                        pred_action_chunk = get_vla_action(
                            self.config.policy,
                            self.vla,
                            self.processor,
                            observation,
                            instruction,
                            action_head=self.action_head,
                            proprio_projector=self.proprio_projector,
                            use_film=self.config.policy.use_film,
                        )
                        
                        inference_time = (time.time() - inference_start) * 1000
                        inference_times.append(inference_time)
                    
                    # Debug: Print action chunk info on first query
                    if t_step == 0:
                        print(f"\n  📊 ACTION FORMAT DEBUG:")
                        print(f"     Action chunk shape: {np.array(pred_action_chunk).shape}")
                        print(f"     First action: {pred_action_chunk[0]}")
                        print(f"     Action range: [{np.min(pred_action_chunk):.3f}, {np.max(pred_action_chunk):.3f}]")
                        print(f"     Action dim per step: {len(pred_action_chunk[0])}")
                        print(f"     Expected format: [dx, dy, dz, dax, day, daz, gripper]")
                        print(f"     Where: dx/dy/dz = position delta (normalized)")
                        print(f"            dax/day/daz = axis-angle rotation delta (normalized)")
                        print(f"            gripper = 0 (close) to 1 (open)")
                        print(f"     Note: Rotation is axis-angle (used by LIBERO/robosuite)")
                        print(f"           NOT euler angles!\n")
                    
                    print(f"  [Step {t_step:3d}] New action chunk | "
                          f"Inference: {inference_time:.1f}ms | "
                          f"Chunk size: {len(pred_action_chunk)}")
                
                # 4. Select and execute action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                
                # Parse action: OpenVLA outputs delta end-effector pose + gripper
                # action is a numpy array of shape (7,) -> [6D end-effector delta (xyz + axis-angle), 1 gripper]
                # Note: LIBERO uses axis-angle rotation representation, not euler angles!
                # Gripper: 0 = close, 1 = open (normalized to [0, 1])
                ee_delta = action[:6]  # [dx, dy, dz, dax, day, daz] - axis-angle rotation!
                gripper_cmd = action[6]
                
                # Debug: Print action details every 10 steps
                if t_step % 10 == 0:
                    print(f"  [Step {t_step:3d}] Action: pos_delta={ee_delta[:3]}, "
                          f"rot_delta={ee_delta[3:6]}, gripper={gripper_cmd:.2f}")
                
                if self.config.robot.control_mode == "eef":
                    # For Cartesian impedance control
                    # Get current end-effector pose in quaternion format
                    current_ee_pose = self.robot.get_ee_pose()  # [x, y, z, qx, qy, qz, qw]
                    
                    # Apply delta and get target pose in [x, y, z, qx, qy, qz, qw] format
                    # Adjust scaling factors based on your workspace size
                    position_scale = 0.05  # 5cm max delta per action
                    rotation_scale = 0.05   # ~5.7 degrees max delta per action
                    
                    target_ee_pose = apply_delta_pose(
                        current_ee_pose, 
                        ee_delta, 
                        position_scale=position_scale,
                        rotation_scale=rotation_scale
                    )
                    
                    # Send target pose to robot (expects 7D: [x, y, z, qx, qy, qz, qw])
                    self.robot.update_desired_ee_pose(target_ee_pose)
                    
                else:
                    # For joint impedance control, we need inverse kinematics
                    # This is more complex - for now, print a warning
                    if t_step == 0:
                        print("  ⚠️  WARNING: Joint control mode with EEF actions is not yet implemented!")
                        print("      Actions will be ignored. Please use --control-mode eef in config")
                
                # Process gripper command
                # OpenVLA: 0 = close, 1 = open
                # Franka: control_gripper(True) = close, control_gripper(False) = open
                gripper_open = bool(gripper_cmd > 0.5)  # Threshold at 0.5, convert to native Python bool
                self.robot.control_gripper(not gripper_open)  # Invert: True=close, False=open
                
                # 5. Regulate control frequency
                elapsed = time.time() - step_start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                
        except KeyboardInterrupt:
            print("\n\n  Rollout interrupted by user (Ctrl+C)")
        
        finally:
            # Print statistics
            if inference_times:
                print(f"\nInference statistics:")
                print(f"  Mean: {np.mean(inference_times):.1f}ms")
                print(f"  Std:  {np.std(inference_times):.1f}ms")
                print(f"  Min:  {np.min(inference_times):.1f}ms")
                print(f"  Max:  {np.max(inference_times):.1f}ms")
            
            # Save video
            if self.config.save_video and video_frames:
                self._save_video(video_frames, instruction)
            
            # Close visualization window
            if self.config.show_cameras:
                cv2.destroyAllWindows()
        
        print(f"\nRollout complete ({len(video_frames)} steps)")
    
    def _save_video(self, frames: list, instruction: str):
        """Save rollout video."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instruction = "".join(c if c.isalnum() or c == " " else "_" for c in instruction[:30])
        safe_instruction = safe_instruction.replace(" ", "_")
        filename = f"videos/rollout_{timestamp}_{safe_instruction}.mp4"
        
        print(f"\nSaving video to {filename}...")
        try:
            clip = ImageSequenceClip(frames, fps=self.config.video_fps)
            # Try with logger parameter only (newer moviepy versions don't support verbose)
            try:
                clip.write_videofile(filename, codec="libx264", logger=None)
            except TypeError:
                # Fallback for older versions
                clip.write_videofile(filename, codec="libx264")
            print(f"  ✓ Video saved: {filename}")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.robot is not None:
            try:
                self.robot.terminate_current_policy()
                self.robot.close()
                print("  ✓ Robot disconnected")
            except Exception as e:
                print(f"  ✗ Error disconnecting robot: {e}")
        
        if self.external_cam is not None:
            self.external_cam.release()
        
        if self.wrist_cam is not None:
            self.wrist_cam.release()
        
        cv2.destroyAllWindows()
        print("  ✓ Cleanup complete")


def main(
    # Task instruction
    instruction: str = "pick up the red block",
    
    # Camera settings
    external_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    use_mock_cameras: bool = False,
    
    # Robot settings
    nuc_ip: str = "192.168.1.143",
    nuc_port: int = 4242,
    use_mock_robot: bool = False,
    
    # Policy settings
    # checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    checkpoint: str = "ZechenBai/OpenVLA-OFT",
    unnorm_key: str = "libero_spatial_no_noops",
    use_8bit: bool = False,
    use_4bit: bool = False,
    
    # Rollout settings
    max_timesteps: int = 600,
    open_loop_horizon: int = 8,
    show_cameras: bool = True,
    save_video: bool = True,
):
    """
    Run OpenVLA-OFT policy on Franka robot.
    
    Args:
        instruction: Natural language task instruction
        external_camera: Serial number of external RealSense camera
        wrist_camera: Serial number of wrist RealSense camera
        use_mock_cameras: Use mock cameras for testing (no real hardware)
        nuc_ip: IP address of NUC running Polymetis
        nuc_port: Port of ZeroRPC server on NUC
        use_mock_robot: Use mock robot for testing (no real hardware)
        checkpoint: Model checkpoint path (HuggingFace Hub or local)
        unnorm_key: Action unnormalization key (must match training data)
        use_8bit: Load model with 8-bit quantization
        use_4bit: Load model with 4-bit quantization
        max_timesteps: Maximum steps per rollout
        open_loop_horizon: Number of actions to execute before re-querying
        show_cameras: Display camera feeds during rollout
        save_video: Save rollout video
    """
    
    # Build configuration
    config = cfg.Config(
        camera=cfg.CameraConfig(
            external_camera_serial=external_camera,
            wrist_camera_serial=wrist_camera,
            use_mock_cameras=use_mock_cameras,
        ),
        robot=cfg.RobotConfig(
            nuc_ip=nuc_ip,
            nuc_port=nuc_port,
            use_mock_robot=use_mock_robot,
        ),
        policy=cfg.PolicyConfig(
            pretrained_checkpoint=checkpoint,
            unnorm_key=unnorm_key,
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
            open_loop_horizon=open_loop_horizon,
        ),
        max_timesteps=max_timesteps,
        show_cameras=show_cameras,
        save_video=save_video,
    )
    
    # Create runner
    runner = OpenVLAFrankaRunner(config)
    
    try:
        # Setup
        runner.setup()
        
        # Run rollout
        runner.run_rollout(instruction)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        runner.cleanup()


if __name__ == "__main__":
    tyro.cli(main)
