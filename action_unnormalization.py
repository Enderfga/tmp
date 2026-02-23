"""
action_unnormalization.py

Custom action unnormalization utilities for Franka robot deployment.
Handles conversion from normalized actions (model output) to physical robot commands.

Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper]
- Position deltas in meters
- Orientation deltas in RPY (roll-pitch-yaw) radians
- Gripper: 0.0 = close, 1.0 = open
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ======================================================================================
# ACTION BOUNDS CONFIGURATION - EDIT HERE FOR DIFFERENT MODELS
# ======================================================================================

# Kevin's Dataset (robot_rlds_kevin) - Current/Default
KEVIN_BOUNDS_1_2_0 = {
    "name": "robot_rlds_kevin",
    # "position_min": np.array([-0.02053109, -0.02344218, -0.01927894], dtype=np.float32),
    # "position_max": np.array([0.02424264, 0.02225738, 0.01838928], dtype=np.float32),
    "position_min": np.array([-0.02053109 * 3, -0.02344218 * 3, -0.01927894], dtype=np.float32),
    "position_max": np.array([0.02424264 * 3, 0.02225738 * 3, 0.01838928], dtype=np.float32),
    # "orientation_min": np.array([-0.03959246, -0.02551901, -0.05755476], dtype=np.float32),
    # "orientation_max": np.array([0.02692719, 0.02277117, 0.07725246], dtype=np.float32),
    "orientation_min": np.array([-10.0, -10.0, -10.0], dtype=np.float32),
    "orientation_max": np.array([10.0, 10.0, 10.0], dtype=np.float32),
    # "orientation_min": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    # "orientation_max": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

KEVIN_BOUNDS_1_4_0 = {
    "name": "robot_rlds_kevin",
    "position_min": np.array([-0.01466781, -0.01822115, -0.01563790], dtype=np.float32),
    "position_max": np.array([0.01742128, 0.01146988, 0.02446541], dtype=np.float32),
    "orientation_min": np.array([-0.02732539, -0.03591533, -0.08691083], dtype=np.float32) * 10,
    "orientation_max": np.array([0.04289465, 0.02818049, 0.11987307], dtype=np.float32) * 10,
    # "orientation_min": np.array([-10.0, -10.0, -10.0], dtype=np.float32),
    # "orientation_max": np.array([10.0, 10.0, 10.0], dtype=np.float32),
    # "orientation_min": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    # "orientation_max": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}


PICK_CUP_GREEN_PLATE = {
    "name": "pick_cup_green_plate",
    "position_min": np.array([-0.01055485, -0.01778186, -0.01517773], dtype=np.float32),
    "position_max": np.array([0.02017602, 0.03290551, 0.02640338], dtype=np.float32),
    "orientation_min": np.array([-0.01637130, -0.01816135, -0.03264310], dtype=np.float32),
    "orientation_max": np.array([0.01911882, 0.01777281, 0.06029939], dtype=np.float32),
    # "orientation_min": np.array([-10.0, -10.0, -10.0], dtype=np.float32),
    # "orientation_max": np.array([10.0, 10.0, 10.0], dtype=np.float32),
    # "orientation_min": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    # "orientation_max": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

PICK_TEST_CUBE_IN_CUP = {
    "name": "pick_test_cube_in_cup",
    "position_min": np.array([-0.008407831192016602, -0.0110735222697258, -0.015861153602600098], dtype=np.float32),
    "position_max": np.array([0.01985916495323181, 0.02518712729215622, 0.03769785165786743], dtype=np.float32),
    "orientation_min": np.array([-0.0027124977204948664, -0.01014685072004795, -0.010627374053001404], dtype=np.float32),
    "orientation_max": np.array([0.004058344289660454, 0.020218973979353905, 0.01053987629711628], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

# Example: LIBERO Spatial bounds (for reference)
LIBERO_SPATIAL_BOUNDS = {
    "name": "libero_spatial",
    "position_min": np.array([-0.021, -0.023, -0.019], dtype=np.float32),
    "position_max": np.array([0.024, 0.022, 0.018], dtype=np.float32),
    "orientation_min": np.array([-0.103, -0.070, -0.058], dtype=np.float32),
    "orientation_max": np.array([0.104, 0.065, 0.077], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

GUIAN_SPATIAL_BOUNDS = {
    "name": "orange_cube_train_temporal_aligned",
    "position_min": np.array([0.4033, -0.2258, -0.0468], dtype=np.float32),
    "position_max": np.array([0.6562, 0.2024, 0.3731], dtype=np.float32),
    "orientation_min": np.array([0.9014, -0.070, -0.058], dtype=np.float32),
    "orientation_max": np.array([0.104, 0.065, 0.077], dtype=np.float32),
    "gripper_min": 0.0483,
    "gripper_max": 0.1062,
}
# pick_n_place_ee — 25-episode Franka pick-and-place dataset
# Bounds extracted from dataset_statistics.json (action: [dx, dy, dz, droll, dpitch, dyaw, gripper])
PICK_N_PLACE_EE = {
    "name": "pick_n_place_ee",
    "position_min": np.array([-0.005163, -0.011432, -0.009680], dtype=np.float32),
    "position_max": np.array([ 0.004788,  0.012703,  0.014425], dtype=np.float32),
    "orientation_min": np.array([-0.015043, -0.031847, -0.007038], dtype=np.float32),
    "orientation_max": np.array([ 0.019807,  0.016383,  0.005300], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

# Registry of all available bounds
AVAILABLE_BOUNDS = {
    "robot_rlds_kevin": KEVIN_BOUNDS_1_2_0,
    "robot_rlds_kevin_v1_4_0": KEVIN_BOUNDS_1_4_0,
    "libero_spatial": LIBERO_SPATIAL_BOUNDS,
    "pick_cup_green_plate": PICK_CUP_GREEN_PLATE,
    "pick_test_cube_in_cup": PICK_TEST_CUBE_IN_CUP,
    "orange_cube_train_temporal_aligned": GUIAN_SPATIAL_BOUNDS,
    "pick_n_place_ee": PICK_N_PLACE_EE,
}

# Default bounds to use
DEFAULT_BOUNDS = "robot_rlds_kevin"

# ======================================================================================


class CustomActionUnnormalizer:
    """
    Unnormalizes actions using custom min/max bounds from data collection.
    
    During training, actions were normalized to [-1, 1] range using:
        normalized = 2 * (action - min) / (max - min) - 1
    
    During inference, we need to reverse this:
        action = (normalized + 1) * (max - min) / 2 + min
    """
    
    def __init__(
        self,
        position_min: np.ndarray,
        position_max: np.ndarray,
        orientation_min: np.ndarray,
        orientation_max: np.ndarray,
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
    ):
        """
        Initialize unnormalizer with custom bounds.
        
        Args:
            position_min: Minimum position deltas [dx_min, dy_min, dz_min]
            position_max: Maximum position deltas [dx_max, dy_max, dz_max]
            orientation_min: Minimum orientation deltas [droll_min, dpitch_min, dyaw_min] (RPY)
            orientation_max: Maximum orientation deltas [droll_max, dpitch_max, dyaw_max] (RPY)
            gripper_min: Minimum gripper value (default: 0.0 = close)
            gripper_max: Maximum gripper value (default: 1.0 = open)
        """
        self.position_min = np.array(position_min, dtype=np.float32)
        self.position_max = np.array(position_max, dtype=np.float32)
        self.orientation_min = np.array(orientation_min, dtype=np.float32)
        self.orientation_max = np.array(orientation_max, dtype=np.float32)
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        
        # Compute ranges for efficiency
        self.position_range = self.position_max - self.position_min
        self.orientation_range = self.orientation_max - self.orientation_min
        self.gripper_range = self.gripper_max - self.gripper_min
        
        # Log initialization
        print(f"  CustomActionUnnormalizer initialized:")
        print(f"    Position bounds: min={self.position_min}, max={self.position_max}")
        print(f"    Orientation bounds: min={self.orientation_min}, max={self.orientation_max}")
        print(f"    Gripper bounds: min={self.gripper_min}, max={self.gripper_max}")
    
    def unnormalize(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action from [-1, 1] to physical units.
        
        Args:
            normalized_action: Normalized action in [-1, 1] range
                              Shape: (7,) = [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        Returns:
            Unnormalized action in physical units
            Shape: (7,) = [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if len(normalized_action) != 7:
            raise ValueError(f"Expected 7D action, got shape {normalized_action.shape}")
        
        # Unnormalize: action = (normalized + 1) * (max - min) / 2 + min
        unnormalized = np.zeros(7, dtype=np.float32)
        
        # Position (indices 0-2)
        unnormalized[0:3] = (normalized_action[0:3] + 1) * self.position_range / 2 + self.position_min
        
        # Orientation (indices 3-5) - RPY (roll-pitch-yaw) representation
        unnormalized[3:6] = (normalized_action[3:6] + 1) * self.orientation_range / 2 + self.orientation_min
        
        # Gripper (index 6)
        unnormalized[6] = (normalized_action[6] + 1) * self.gripper_range / 2 + self.gripper_min
        
        return unnormalized
    
    def unnormalize_batch(self, normalized_actions: np.ndarray) -> np.ndarray:
        """
        Unnormalize a batch of actions.
        
        Args:
            normalized_actions: Normalized actions in [-1, 1] range
                               Shape: (batch_size, 7)
        
        Returns:
            Unnormalized actions in physical units
            Shape: (batch_size, 7)
        """
        if normalized_actions.ndim != 2 or normalized_actions.shape[1] != 7:
            raise ValueError(f"Expected shape (N, 7), got {normalized_actions.shape}")
        
        batch_size = normalized_actions.shape[0]
        unnormalized = np.zeros((batch_size, 7), dtype=np.float32)
        
        for i in range(batch_size):
            unnormalized[i] = self.unnormalize(normalized_actions[i])
        
        return unnormalized
    
    def get_stats_dict(self) -> Dict[str, np.ndarray]:
        """
        Get normalization statistics in the format expected by OpenVLA.
        
        Returns:
            Dictionary with 'q01' (min) and 'q99' (max) keys
        """
        # Combine all bounds into single arrays
        action_min = np.concatenate([
            self.position_min,
            self.orientation_min,
            [self.gripper_min]
        ])
        
        action_max = np.concatenate([
            self.position_max,
            self.orientation_max,
            [self.gripper_max]
        ])
        
        return {
            "q01": action_min.tolist(),
            "q99": action_max.tolist(),
        }


def create_unnormalizer(bounds_name: Optional[str] = None) -> CustomActionUnnormalizer:
    """
    Create unnormalizer with specified bounds.
    
    Args:
        bounds_name: Name of bounds configuration to use (e.g., "robot_rlds_kevin", "libero_spatial").
                     If None, uses DEFAULT_BOUNDS.
    
    Returns:
        CustomActionUnnormalizer configured with the specified bounds
    
    Raises:
        ValueError: If bounds_name is not found in AVAILABLE_BOUNDS
    
    Example:
        >>> unnormalizer = create_unnormalizer("robot_rlds_kevin")
        >>> unnormalizer = create_unnormalizer()  # Uses default
    """
    if bounds_name is None:
        bounds_name = DEFAULT_BOUNDS
    
    if bounds_name not in AVAILABLE_BOUNDS:
        available = ", ".join(AVAILABLE_BOUNDS.keys())
        raise ValueError(
            f"Unknown bounds '{bounds_name}'. Available bounds: {available}\n"
            f"Add your custom bounds to the AVAILABLE_BOUNDS dictionary in action_unnormalization.py"
        )
    
    bounds = AVAILABLE_BOUNDS[bounds_name]
    print(f"  Loading action bounds: '{bounds['name']}'")
    
    return CustomActionUnnormalizer(
        position_min=bounds["position_min"],
        position_max=bounds["position_max"],
        orientation_min=bounds["orientation_min"],
        orientation_max=bounds["orientation_max"],
        gripper_min=bounds["gripper_min"],
        gripper_max=bounds["gripper_max"],
    )


def create_kevin_unnormalizer() -> CustomActionUnnormalizer:
    """
    Create unnormalizer with Kevin's data collection bounds.
    
    DEPRECATED: Use create_unnormalizer("robot_rlds_kevin") instead.
    This function is kept for backward compatibility.
    
    Returns:
        CustomActionUnnormalizer configured with Kevin's bounds
    """
    return create_unnormalizer("robot_rlds_kevin")


def print_action_stats(unnormalizer: CustomActionUnnormalizer):
    """Print statistics about the action bounds."""
    print("\nAction Unnormalization Statistics:")
    print("=" * 70)
    
    # Position stats
    pos_range = unnormalizer.position_max - unnormalizer.position_min
    print(f"Position Delta Ranges (meters):")
    print(f"  X: [{unnormalizer.position_min[0]:.4f}, {unnormalizer.position_max[0]:.4f}] (range: {pos_range[0]:.4f}m)")
    print(f"  Y: [{unnormalizer.position_min[1]:.4f}, {unnormalizer.position_max[1]:.4f}] (range: {pos_range[1]:.4f}m)")
    print(f"  Z: [{unnormalizer.position_min[2]:.4f}, {unnormalizer.position_max[2]:.4f}] (range: {pos_range[2]:.4f}m)")
    
    # Orientation stats
    ori_range = unnormalizer.orientation_max - unnormalizer.orientation_min
    print(f"\nOrientation Delta Ranges (RPY, radians):")
    print(f"  droll: [{unnormalizer.orientation_min[0]:.4f}, {unnormalizer.orientation_max[0]:.4f}] (range: {ori_range[0]:.4f} rad = {np.degrees(ori_range[0]):.1f}°)")
    print(f"  dpitch: [{unnormalizer.orientation_min[1]:.4f}, {unnormalizer.orientation_max[1]:.4f}] (range: {ori_range[1]:.4f} rad = {np.degrees(ori_range[1]):.1f}°)")
    print(f"  dyaw: [{unnormalizer.orientation_min[2]:.4f}, {unnormalizer.orientation_max[2]:.4f}] (range: {ori_range[2]:.4f} rad = {np.degrees(ori_range[2]):.1f}°)")
    
    # Gripper stats
    print(f"\nGripper Range:")
    print(f"  [{unnormalizer.gripper_min:.1f}, {unnormalizer.gripper_max:.1f}] (0=close, 1=open)")
    
    print("=" * 70 + "\n")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACTION UNNORMALIZATION UTILITY")
    print("="*70)
    
    # Show available bounds
    print("\nAvailable bounds configurations:")
    for name in AVAILABLE_BOUNDS.keys():
        marker = " (default)" if name == DEFAULT_BOUNDS else ""
        print(f"  - {name}{marker}")
    
    # Create unnormalizer with default bounds
    print(f"\nCreating unnormalizer with default bounds ('{DEFAULT_BOUNDS}')...")
    unnormalizer = create_unnormalizer()
    
    # Print stats
    print_action_stats(unnormalizer)
    
    # Test unnormalization
    print("Testing unnormalization:")
    print("-" * 70)
    
    # Test with normalized actions at extremes
    test_actions = [
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),  # All minimum
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),         # All maximum
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),         # All zero
    ]
    
    for i, normalized in enumerate(test_actions):
        unnormalized = unnormalizer.unnormalize(normalized)
        print(f"\nTest {i+1}:")
        print(f"  Normalized:   {normalized}")
        print(f"  Unnormalized: {unnormalized}")
        print(f"    Position:    [{unnormalized[0]:.4f}, {unnormalized[1]:.4f}, {unnormalized[2]:.4f}] m")
        print(f"    Orientation: [{unnormalized[3]:.4f}, {unnormalized[4]:.4f}, {unnormalized[5]:.4f}] rad")
        print(f"    Gripper:     {unnormalized[6]:.2f}")
