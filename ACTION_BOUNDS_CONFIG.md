# Action Bounds Configuration Guide

This guide explains how to configure and manage action bounds for different models.

## Action Format

The model outputs 7D actions: **[dx, dy, dz, droll, dpitch, dyaw, gripper]**

- **Position deltas (dx, dy, dz)**: Translation in meters
- **Orientation deltas (droll, dpitch, dyaw)**: Rotation in RPY (roll-pitch-yaw) radians
- **Gripper**: 0.0 = close, 1.0 = open

During training, actions are normalized to [-1, 1] range. During deployment, they must be unnormalized back to physical units.

## Quick Start

### Viewing Available Bounds

```bash
python action_unnormalization.py
```

This will show all configured action bounds.

### Using Different Bounds

**Option 1: Edit the shell script**
```bash
# In run_eval_franka.sh, change:
--action_bounds_name robot_rlds_kevin

# To your desired bounds:
--action_bounds_name my_custom_bounds
```

**Option 2: Direct command line**
```bash
python run_franka_eval.py \
  --action_bounds_name robot_rlds_kevin \
  --use_custom_unnormalization True \
  ... other args ...
```

## Adding New Action Bounds

Edit `action_unnormalization.py` at the top of the file:

```python
# ======================================================================================
# ACTION BOUNDS CONFIGURATION - EDIT HERE FOR DIFFERENT MODELS
# ======================================================================================

# Your new bounds
MY_MODEL_BOUNDS = {
    "name": "my_model_v1",
    "position_min": np.array([...], dtype=np.float32),
    "position_max": np.array([...], dtype=np.float32),
    "orientation_min": np.array([...], dtype=np.float32),
    "orientation_max": np.array([...], dtype=np.float32),
    "gripper_min": 0.0,
    "gripper_max": 1.0,
}

# Add to registry
AVAILABLE_BOUNDS = {
    "robot_rlds_kevin": KEVIN_BOUNDS,
    "libero_spatial": LIBERO_SPATIAL_BOUNDS,
    "my_model_v1": MY_MODEL_BOUNDS,  # <-- Add your bounds here
}

# Optionally change default
DEFAULT_BOUNDS = "my_model_v1"  # <-- Change default if desired
```

## Current Bounds

### robot_rlds_kevin (Default)
```python
position_min: [-0.02053109, -0.02344218, -0.01927894]
position_max: [0.02424264, 0.02225738, 0.01838928]
orientation_min: [-0.03959246, -0.02551901, -0.05755476]
orientation_max: [0.02692719, 0.02277117, 0.07725246]
```

**Ranges:**
- Position: ~4.5cm per action
- Rotation: 3.8°, 2.8°, 7.7° per action

### libero_spatial (Reference)
```python
position_min: [-0.021, -0.023, -0.019]
position_max: [0.024, 0.022, 0.018]
orientation_min: [-0.103, -0.070, -0.058]
orientation_max: [0.104, 0.065, 0.077]
```

**Ranges:**
- Position: ~4.5cm per action
- Rotation: 11.9°, 7.7°, 7.7° per action

## Finding Your Bounds

If you need to determine bounds for a new dataset:

```python
import numpy as np
from your_dataset import load_actions

# Load your action data
actions = load_actions()  # Shape: (N, 7) for [dx,dy,dz,dax,day,daz,gripper]

# Compute bounds
position_min = np.min(actions[:, :3], axis=0)
position_max = np.max(actions[:, :3], axis=0)
orientation_min = np.min(actions[:, 3:6], axis=0)
orientation_max = np.max(actions[:, 3:6], axis=0)

print(f"position_min: {position_min}")
print(f"position_max: {position_max}")
print(f"orientation_min: {orientation_min}")
print(f"orientation_max: {orientation_max}")
```

## Command Line Reference

### With Custom Unnormalization
```bash
python run_franka_eval.py \
  --use_custom_unnormalization True \
  --action_bounds_name robot_rlds_kevin \
  ... other args ...
```

### With Simple Scaling (Old Method)
```bash
python run_franka_eval.py \
  --use_custom_unnormalization False \
  --position_scale 0.05 \
  --rotation_scale 0.05 \
  ... other args ...
```

## Testing Your Bounds

After adding new bounds:

```bash
# Test the configuration
python action_unnormalization.py

# Run with mock robot first
python run_franka_eval.py \
  --use_mock_robot \
  --use_custom_unnormalization True \
  --action_bounds_name your_new_bounds \
  --instruction "test task"
```

## Troubleshooting

**Error: "Unknown bounds 'xyz'"**
- Check that your bounds name is added to `AVAILABLE_BOUNDS` dictionary
- Run `python action_unnormalization.py` to see available bounds

**Robot moves too little/much**
- Verify your bounds match your training data normalization
- Check position/orientation ranges are in correct units (meters/radians)

**Wrong format**
- Ensure all arrays use `dtype=np.float32`
- Verify shape: position (3,), orientation (3,), gripper (scalar)
