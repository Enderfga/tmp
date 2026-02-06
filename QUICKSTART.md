# Quick Start Guide: OpenVLA-OFT on Franka

Get up and running in 5 minutes!

## Prerequisites

- ✅ OpenVLA-OFT environment installed (see [../SETUP.md](../SETUP.md))
- ✅ 2x Intel RealSense cameras (D435/D455)
- ✅ Franka robot with Polymetis + ZeroRPC server on NUC
- ✅ GPU with ~18GB VRAM (or use quantization)

## 30-Second Test (No Hardware)

Test the entire pipeline without any hardware:

```bash
cd franka_openvla
python main.py \
  --use-mock-cameras \
  --use-mock-robot \
  --instruction "pick up the red block"
```

This will:
- ✅ Load OpenVLA-OFT model (~18GB)
- ✅ Simulate cameras
- ✅ Simulate robot
- ✅ Run inference and "execute" actions
- ✅ Save video

**Expected output:** Video file `rollout_YYYYMMDD_HHMMSS_pick_up_the_red_block.mp4`

## 5-Minute Real Deployment

### Step 1: Install Dependencies (1 min)

```bash
cd franka_openvla
./setup.sh
```

### Step 2: Find Camera Serials (1 min)

```bash
python camera_utils.py
```

Note down your camera serial numbers:
- External camera: `_______________`
- Wrist camera: `_______________`

### Step 3: Test Cameras (1 min)

```bash
python test_cameras.py \
  --external-camera YOUR_EXTERNAL_SERIAL \
  --wrist-camera YOUR_WRIST_SERIAL
```

Press 'q' to quit once you see both camera feeds.

### Step 4: Test Robot (1 min)

```bash
python test_robot.py --nuc-ip 192.168.1.143
```

Skip the motion test (just press 'N' when asked).

### Step 5: Run Your First Task! (1 min)

```bash
python main.py \
  --external-camera YOUR_EXTERNAL_SERIAL \
  --wrist-camera YOUR_WRIST_SERIAL \
  --nuc-ip 192.168.1.143 \
  --instruction "pick up the red block"
```

**🎉 You're now running OpenVLA-OFT on your Franka robot!**

## Common Tasks

### Pick and Place

```bash
python main.py \
  --external-camera XXX \
  --wrist-camera YYY \
  --instruction "pick up the red block and place it in the bowl"
```

### Opening Drawers

```bash
python main.py \
  --external-camera XXX \
  --wrist-camera YYY \
  --checkpoint "moojink/openvla-7b-oft-finetuned-libero-object" \
  --unnorm-key "libero_object_no_noops" \
  --instruction "open the drawer"
```

### Custom Checkpoint

```bash
python main.py \
  --external-camera XXX \
  --wrist-camera YYY \
  --checkpoint "/path/to/your/checkpoint" \
  --unnorm-key "your_dataset_name" \
  --instruction "your custom task"
```

## Troubleshooting

### Issue: "No RealSense cameras found"

**Solution:**
```bash
# Check cameras are detected
rs-enumerate-devices

# Update permissions
sudo usermod -a -G video $USER
# Then log out and log back in
```

### Issue: "Failed to connect to robot"

**Solution:**
```bash
# Verify NUC is reachable
ping 192.168.1.143

# Check if ZeroRPC server is running
telnet 192.168.1.143 4242
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use 8-bit quantization
python main.py ... --use-8bit

# Or 4-bit quantization (even less memory)
python main.py ... --use-4bit
```

### Issue: Robot moves erratically

**Solution:**
- ✅ Verify `--unnorm-key` matches your `--checkpoint`
- ✅ Check camera placement and lighting
- ✅ Ensure task instruction is similar to training data

## Next Steps

1. **Read the full README:** [README.md](README.md)
2. **Understand configurations:** [config.py](config.py)
3. **Fine-tune your own model:** [../LIBERO.md](../LIBERO.md)
4. **Optimize performance:** See "Performance Tips" in README

## Configuration File Method

Instead of passing arguments every time, edit `config.py`:

```python
# Edit config.py
def get_default_config():
    return Config(
        camera=CameraConfig(
            external_camera_serial="YOUR_EXTERNAL_SERIAL",
            wrist_camera_serial="YOUR_WRIST_SERIAL",
        ),
        robot=RobotConfig(
            nuc_ip="192.168.1.143",
        ),
        policy=PolicyConfig(
            pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
            unnorm_key="libero_spatial_no_noops",
        ),
    )
```

Then simply run:
```bash
python main.py --instruction "pick up the red block"
```

## Available Checkpoints

| Task Type | Checkpoint | Unnorm Key |
|-----------|------------|------------|
| Spatial reasoning | `moojink/openvla-7b-oft-finetuned-libero-spatial` | `libero_spatial_no_noops` |
| Object manipulation | `moojink/openvla-7b-oft-finetuned-libero-object` | `libero_object_no_noops` |
| Goal-based tasks | `moojink/openvla-7b-oft-finetuned-libero-goal` | `libero_goal_no_noops` |
| Multi-task (10 tasks) | `moojink/openvla-7b-oft-finetuned-libero-10` | `libero_10_no_noops` |

## Help

```bash
# Get all available options
python main.py --help

# Test individual components
python test_policy.py --help
python test_robot.py --help
python test_cameras.py --help
```

## Support

- 📖 Full documentation: [README.md](README.md)
- 🐛 Issues: [GitHub Issues](https://github.com/moojink/openvla-oft/issues)
- 📧 Email: moojink@cs.stanford.edu
