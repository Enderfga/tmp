"""Configuration for Franka robot with OpenVLA-OFT policy."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class CameraConfig:
    """Configuration for cameras."""
    # Camera serial numbers - find these by running: python camera_utils.py
    external_camera_serial: Optional[str] = None  # e.g., "123456789"
    wrist_camera_serial: Optional[str] = None     # e.g., "987654321"
    
    # Camera settings
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Use mock cameras for testing without hardware
    use_mock_cameras: bool = False


@dataclasses.dataclass
class RobotConfig:
    """Configuration for robot connection."""
    # NUC connection (your Polymetis ZeroRPC server)
    nuc_ip: str = "192.168.1.143"
    nuc_port: int = 4242
    
    # Control mode: 'eef' for end-effector control (required for OpenVLA delta actions)
    #               'joint' for joint control (not yet implemented for OpenVLA)
    control_mode: str = "eef"
    
    # Control frequency (Hz) - OpenVLA typically uses 15-30 Hz
    control_frequency: int = 10
    
    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s (for safety checks)
    max_gripper_width: float = 0.085  # meters (Franka max)
    
    # Use mock robot for testing without hardware
    use_mock_robot: bool = False


@dataclasses.dataclass
class PolicyConfig:
    """Configuration for OpenVLA policy inference."""
    # Model checkpoint - use one of the pretrained checkpoints from HuggingFace Hub:
    # - "moojink/openvla-7b-oft-finetuned-libero-spatial"
    # - "moojink/openvla-7b-oft-finetuned-libero-object"
    # - "moojink/openvla-7b-oft-finetuned-libero-goal"
    # - "moojink/openvla-7b-oft-finetuned-libero-10"
    # - "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    # Or use a local checkpoint path
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    
    # Model architecture settings
    use_l1_regression: bool = False       # Use continuous action head with L1 regression
    use_diffusion: bool = False          # Use diffusion-based action prediction
    use_film: bool = False               # Use FiLM to infuse language into vision
    num_images_in_input: int = 1         # Number of images (1 external + 1 wrist)
    use_proprio: bool = False             # Include proprioception state
    center_crop: bool = True             # Apply center crop to images
    
    # Quantization (to reduce memory usage)
    load_in_8bit: bool = False           # Load model with 8-bit quantization
    load_in_4bit: bool = False           # Load model with 4-bit quantization
    
    # Action normalization key - MUST match your training data!
    # Common keys: "libero_spatial_no_noops", "libero_object_no_noops", 
    #              "libero_goal_no_noops", "libero_10_no_noops"
    # unnorm_key: str = "libero_spatial_no_noops"
    unnorm_key: str = "robot_rlds_kevin"
    
    # Action execution
    action_horizon: int = 25              # Number of actions predicted by model
    open_loop_horizon: int = 8            # Execute N actions before re-querying
    
    # Diffusion settings (only used if use_diffusion=True)
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    
    # LoRA rank (must match training configuration)
    lora_rank: int = 32
    
    # Default prompt if none provided
    default_prompt: Optional[str] = None


@dataclasses.dataclass
class Config:
    """Main configuration."""
    camera: CameraConfig = dataclasses.field(default_factory=CameraConfig)
    robot: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)
    
    # Rollout settings
    max_timesteps: int = 600
    
    # Visualization
    show_cameras: bool = True
    save_video: bool = True
    video_fps: int = 10


# Example configurations

def get_default_config() -> Config:
    """Get default configuration - UPDATE WITH YOUR CAMERA SERIALS."""
    return Config(
        camera=CameraConfig(
            external_camera_serial=None,  # Set your camera serial here
            wrist_camera_serial=None,      # Set your camera serial here
            use_mock_cameras=False,        # Set to True for testing without cameras
        ),
        robot=RobotConfig(
            nuc_ip="192.168.1.143",
            nuc_port=4242,
            use_mock_robot=False,
        ),
        policy=PolicyConfig(
            pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
            unnorm_key="libero_spatial_no_noops",
        ),
    )


def get_test_config() -> Config:
    """Get test configuration with mock hardware."""
    return Config(
        camera=CameraConfig(
            use_mock_cameras=True,
        ),
        robot=RobotConfig(
            nuc_ip="192.168.1.143",
            nuc_port=4242,
            use_mock_robot=True,
        ),
        policy=PolicyConfig(
            pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
            unnorm_key="libero_spatial_no_noops",
        ),
    )


def get_libero_object_config() -> Config:
    """Configuration for LIBERO Object tasks."""
    cfg = get_default_config()
    cfg.policy.pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-object"
    cfg.policy.unnorm_key = "libero_object_no_noops"
    return cfg


def get_libero_goal_config() -> Config:
    """Configuration for LIBERO Goal tasks."""
    cfg = get_default_config()
    cfg.policy.pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-goal"
    cfg.policy.unnorm_key = "libero_goal_no_noops"
    return cfg


def get_libero_10_config() -> Config:
    """Configuration for LIBERO-10 tasks."""
    cfg = get_default_config()
    cfg.policy.pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-10"
    cfg.policy.unnorm_key = "libero_10_no_noops"
    return cfg
