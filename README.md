# OpenVLA-OFT Franka Deployment

Deploy **OpenVLA-OFT** (Optimized Fine-Tuning for Vision-Language-Action models) on your Franka Panda robot.

## System Architecture

```
┌─────────────────────────────────────┐
│   Workstation (GPU)                 │
│   - OpenVLA-OFT policy inference    │
│   - Camera capture (RealSense)      │
│   - Robot control (ZeroRPC client)  │
│   - main.py runs here               │
└─────────────────┬───────────────────┘
                  │
                  │ Network (TCP/IP)
                  │
┌─────────────────▼───────────────────┐
│   NUC (192.168.1.143)               │
│   - Polymetis                       │
│   - ZeroRPC server (port 4242)      │
│   - Real-time kernel                │
└─────────────────┬───────────────────┘
                  │
                  │ FCI
                  │
┌─────────────────▼───────────────────┐
│   Franka Panda Robot                │
└─────────────────────────────────────┘

RealSense cameras → USB 3.0 → Workstation
```

## Installation

### 1. Set Up OpenVLA-OFT Environment

Follow the main repository's [SETUP.md](../SETUP.md):

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch (adjust for your CUDA version)
pip3 install torch torchvision torchaudio

# Clone and install openvla-oft
cd /path/to/openvla-oft
pip install -e .

# Install Flash Attention 2 (optional but recommended)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 2. Install Franka-Specific Dependencies

```bash
cd franka_openvla

# Install additional dependencies
pip install pyrealsense2>=2.55.0 zerorpc>=0.6.3 opencv-python moviepy tyro
```

### 3. Verify Installation

```bash
# List available RealSense cameras
python camera_utils.py

# Test policy loading (mock mode)
python test_policy.py

# Test robot connection (requires NUC)
python test_robot.py --nuc-ip 192.168.1.143
```

## Hardware Setup

### Camera Setup

**Requirements:**
- 2 Intel RealSense cameras (D435/D455 recommended)
- USB 3.0 connection to workstation
- Good lighting conditions

**Camera Placement:**

1. **External Camera** (Primary/Third-person view)
   - Mount 1-2m away from workspace
   - Elevation angle: 30-45°
   - Should capture entire manipulation area
   - Clear view of objects and robot

2. **Wrist Camera** (First-person view)
   - Mount on or near Franka gripper
   - May need long USB 3.0 cable or active USB extension
   - Should see objects being manipulated
   - Close-up view of manipulation region

### Find Camera Serial Numbers

```bash
python camera_utils.py
```

Output example:
```
Device 0:
  Serial:   123456789
  Name:     Intel RealSense D435
  Firmware: 5.13.0.50

Device 1:
  Serial:   987654321
  Name:     Intel RealSense D455
  Firmware: 5.13.0.50
```

**Important:** Note down the serial numbers - you'll need them for configuration!

### Robot Setup

Ensure your Franka robot's NUC is:
1. Running Polymetis with ZeroRPC server
2. Accessible on the network (default: 192.168.1.143:4242)
3. Robot is in a safe starting configuration

## Quick Start

### 1. Test with Mock Hardware (No Robot Required)

```bash
python main.py \
  --use-mock-cameras \
  --use-mock-robot \
  --instruction "pick up the red block"
```

This will:
- ✓ Load the OpenVLA-OFT model
- ✓ Simulate cameras with random images
- ✓ Simulate robot control
- ✓ Verify the entire pipeline works

### 2. Run with Real Hardware

```bash
python main.py \
  --external-camera "123456789" \
  --wrist-camera "987654321" \
  --nuc-ip "192.168.1.143" \
  --instruction "pick up the red block"
```

### 3. Use Different Checkpoints

```bash
# LIBERO Spatial tasks
python main.py \
  --checkpoint "moojink/openvla-7b-oft-finetuned-libero-spatial" \
  --unnorm-key "libero_spatial_no_noops" \
  --instruction "pick up the red block"

# LIBERO Object tasks
python main.py \
  --checkpoint "moojink/openvla-7b-oft-finetuned-libero-object" \
  --unnorm-key "libero_object_no_noops" \
  --instruction "open the drawer"

# LIBERO Goal tasks
python main.py \
  --checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
  --unnorm-key "libero_goal_no_noops" \
  --instruction "put the bowl on the plate"
```

## Available Checkpoints

OpenVLA-OFT provides several pretrained checkpoints on HuggingFace Hub:

| Checkpoint | Unnorm Key | Description |
|------------|------------|-------------|
| `moojink/openvla-7b-oft-finetuned-libero-spatial` | `libero_spatial_no_noops` | Spatial reasoning tasks |
| `moojink/openvla-7b-oft-finetuned-libero-object` | `libero_object_no_noops` | Object manipulation tasks |
| `moojink/openvla-7b-oft-finetuned-libero-goal` | `libero_goal_no_noops` | Goal-based tasks |
| `moojink/openvla-7b-oft-finetuned-libero-10` | `libero_10_no_noops` | Combined 10-task suite |
| `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | Multiple suites | Multi-task checkpoint |

**CRITICAL:** The `unnorm_key` must match the checkpoint's training data, or actions will be incorrect!

## Configuration

### Command-Line Arguments

```bash
python main.py --help
```

Key arguments:

**Task:**
- `--instruction`: Natural language task instruction (required)

**Cameras:**
- `--external-camera`: Serial number of external camera
- `--wrist-camera`: Serial number of wrist camera
- `--use-mock-cameras`: Use mock cameras for testing

**Robot:**
- `--nuc-ip`: IP address of NUC (default: 192.168.1.143)
- `--nuc-port`: ZeroRPC port (default: 4242)
- `--use-mock-robot`: Use mock robot for testing

**Policy:**
- `--checkpoint`: Model checkpoint path (HF Hub or local)
- `--unnorm-key`: Action unnormalization key (MUST match training data!)
- `--use-8bit`: Load model with 8-bit quantization (saves memory)
- `--use-4bit`: Load model with 4-bit quantization (saves more memory)
- `--open-loop-horizon`: Number of actions to execute before re-querying (default: 8)

**Rollout:**
- `--max-timesteps`: Maximum steps per rollout (default: 600)
- `--show-cameras`: Display camera feeds (default: True)
- `--save-video`: Save rollout video (default: True)

### Config File (config.py)

For more complex configurations, edit `config.py`:

```python
from config import get_default_config

cfg = get_default_config()

# Update camera serials
cfg.camera.external_camera_serial = "123456789"
cfg.camera.wrist_camera_serial = "987654321"

# Update robot settings
cfg.robot.nuc_ip = "192.168.1.143"
cfg.robot.control_frequency = 15  # Hz

# Update policy settings
cfg.policy.pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial"
cfg.policy.unnorm_key = "libero_spatial_no_noops"
cfg.policy.open_loop_horizon = 8
```

## Understanding OpenVLA-OFT Actions

### Action Format

OpenVLA-OFT outputs **delta end-effector poses** (not absolute positions):

```python
action = [dx, dy, dz, drx, dry, drz, gripper]
#         └──────────┬──────────┘ └────┬────┘ └──┬──┘
#          position delta (m)   rotation   gripper
#                               delta (rad)  (0=close, 1=open)
```

**Key Differences from Other Policies:**

| Policy | Action Type | Action Dim | Control Freq |
|--------|-------------|------------|--------------|
| OpenVLA-OFT | End-effector deltas | 7 | 15 Hz |
| DROID (pi05) | Joint velocities | 8 | 15 Hz |
| Octo | End-effector deltas | 7 | 10 Hz |

### Action Execution

1. **Action Chunking**: Model predicts 25 actions at once (1.67s at 15 Hz)
2. **Open-Loop Execution**: Execute 8 actions before re-querying (0.53s)
3. **EEF Control**: Uses Cartesian impedance controller for smooth motion
4. **Gripper Control**: 0 = close, 1 = open (thresholded at 0.5)

### Action Normalization

**CRITICAL:** Actions are normalized during training and must be unnormalized during inference!

- Normalization statistics are stored in `dataset_statistics.json`
- The `unnorm_key` specifies which dataset's statistics to use
- **Mismatch between checkpoint and unnorm_key will cause incorrect actions!**

Example `dataset_statistics.json`:
```json
{
  "libero_spatial_no_noops": {
    "action": {
      "mean": [...],
      "std": [...],
      "max": [...],
      "min": [...]
    },
    "proprio": {
      "mean": [...],
      "std": [...]
    }
  }
}
```

## Troubleshooting

### Camera Issues

**Problem:** "No RealSense cameras found!"

Solutions:
1. Check USB 3.0 connection (blue port)
2. List devices: `rs-enumerate-devices`
3. Try different USB port
4. Update firmware: `realsense-viewer`
5. Check permissions: `sudo usermod -a -G video $USER`

**Problem:** Camera initialization fails intermittently

Solutions:
1. Unplug and replug USB cable
2. Use powered USB 3.0 hub
3. Reduce USB bandwidth: Lower resolution/FPS
4. Check cable quality (use high-quality USB 3.0 cables)

### Robot Connection Issues

**Problem:** "Failed to connect to robot"

Solutions:
1. Verify NUC is accessible: `ping 192.168.1.143`
2. Check ZeroRPC server is running on NUC
3. Verify port 4242 is open: `telnet 192.168.1.143 4242`
4. Check firewall settings

### Model Loading Issues

**Problem:** CUDA out of memory

Solutions:
1. Use 8-bit quantization: `--use-8bit`
2. Use 4-bit quantization: `--use-4bit` (more aggressive)
3. Reduce batch size (not applicable for single inference)
4. Close other GPU programs

**Problem:** "Action un-norm key not found"

Solutions:
1. Check that `unnorm_key` matches checkpoint training data
2. Use correct checkpoint-unnorm_key pair (see table above)
3. For custom checkpoints, ensure `dataset_statistics.json` exists

### Action Issues

**Problem:** Robot moves erratically or unsafe

Solutions:
1. Verify correct `unnorm_key` for checkpoint
2. Check control frequency matches training (usually 15 Hz)
3. Increase `open_loop_horizon` for smoother motion
4. Check camera calibration/placement
5. Verify task instruction matches training distribution

**Problem:** Robot doesn't move at all

Solutions:
1. Check joint impedance controller is started
2. Verify robot is not in error state
3. Check action values are reasonable (print actions)
4. Ensure gripper is not stuck

## Performance Tips

### Inference Speed

Typical inference times on RTX 5090:
- Full precision (bf16): ~100-150ms per action chunk
- 8-bit quantization: ~80-120ms per action chunk
- 4-bit quantization: ~60-100ms per action chunk

To maximize throughput:
1. Use Flash Attention 2 (installed by default)
2. Use quantization if memory allows
3. Warm up model before critical rollouts
4. Keep control frequency at 15 Hz (matches training)

### Memory Usage

| Configuration | VRAM Usage |
|---------------|------------|
| Full precision (bf16) | ~18 GB |
| 8-bit quantization | ~12 GB |
| 4-bit quantization | ~8 GB |

## Video Recording

Rollouts are automatically saved as MP4 videos:

```
rollout_20250126_143022_pick_up_the_red_block.mp4
```

Location: Current working directory

To disable: `--save-video False`

## Safety Considerations

1. **Always supervise** the robot during execution
2. **Emergency stop** button should be easily accessible
3. **Workspace** should be clear of obstacles
4. **Joint velocity limits** are enforced in software
5. **Test with mock hardware** first before real deployment
6. **Start with simple tasks** to verify behavior

## Differences from Pi0.5-DROID

If you're migrating from Pi0.5-DROID:

| Aspect | Pi0.5-DROID | OpenVLA-OFT |
|--------|-------------|-------------|
| Framework | OpenPI | Transformers |
| Action Type | Velocities | Positions |
| Action Horizon | 15 | 25 |
| Image Processing | Direct resize | Resize + optional center crop |
| Proprio | Concatenated | Projected via MLP |
| Checkpoints | Single file | VLA + action_head + proprio_projector |
| Unnormalization | N/A | Required via unnorm_key |

## Advanced Usage

### Custom Checkpoints

If you've fine-tuned your own checkpoint:

```bash
python main.py \
  --checkpoint "/path/to/your/checkpoint" \
  --unnorm-key "your_dataset_name" \
  --instruction "your custom task"
```

Ensure your checkpoint directory contains:
- `config.json`
- Model weights (`.safetensors` or `.bin`)
- `action_head--*_checkpoint.pt`
- `proprio_projector--*_checkpoint.pt`
- `dataset_statistics.json`

### Server-Client Mode

For remote inference or multi-robot setups, use server-client mode:

**On workstation (server):**
```bash
cd /path/to/openvla-oft
python vla-scripts/deploy.py \
  --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-spatial" \
  --unnorm_key "libero_spatial_no_noops" \
  --port 8777
```

**On client:**
```python
from experiments.robot.openvla_utils import get_action_from_server

action = get_action_from_server(
    observation,
    server_endpoint="http://192.168.1.100:8777/act"
)
```

## Citation

If you use this code, please cite the OpenVLA-OFT paper:

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```

## Support

For issues:
1. Check this README and troubleshooting section
2. Open a GitHub issue on the main repository
3. Email: moojink@cs.stanford.edu

## License

Same license as the main OpenVLA-OFT repository.
