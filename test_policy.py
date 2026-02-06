"""Test script for validating OpenVLA-OFT policy loading.

This script loads the OpenVLA-OFT model and verifies all components
without requiring robot hardware or cameras.

Usage:
    python test_policy.py
    python test_policy.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-object
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import tyro

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
import config as cfg


def test_policy(
    checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    unnorm_key: str = "libero_spatial_no_noops",
    use_8bit: bool = False,
    use_4bit: bool = False,
):
    """
    Test OpenVLA-OFT policy loading and inference.
    
    Args:
        checkpoint: Model checkpoint path
        unnorm_key: Action unnormalization key
        use_8bit: Use 8-bit quantization
        use_4bit: Use 4-bit quantization
    """
    print("=" * 70)
    print("OpenVLA-OFT Policy Test")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Unnorm key: {unnorm_key}")
    print(f"Quantization: {'8-bit' if use_8bit else '4-bit' if use_4bit else 'None (bf16)'}")
    
    # Create config
    policy_cfg = cfg.PolicyConfig(
        pretrained_checkpoint=checkpoint,
        unnorm_key=unnorm_key,
        load_in_8bit=use_8bit,
        load_in_4bit=use_4bit,
    )
    
    print("\n" + "=" * 70)
    print("Step 1: Loading VLA Model")
    print("=" * 70)
    
    try:
        vla = get_vla(policy_cfg)
        print("\n✓ VLA model loaded successfully!")
        print(f"  Device: {next(vla.parameters()).device}")
        print(f"  Dtype: {next(vla.parameters()).dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in vla.parameters())
        trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n✗ Failed to load VLA model: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Step 2: Loading Image Processor")
    print("=" * 70)
    
    try:
        processor = get_processor(policy_cfg)
        print("\n✓ Processor loaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to load processor: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Step 3: Loading Action Head")
    print("=" * 70)
    
    try:
        action_head = get_action_head(policy_cfg, vla.llm_dim)
        print("\n✓ Action head loaded successfully!")
        print(f"  Device: {next(action_head.parameters()).device}")
        print(f"  Dtype: {next(action_head.parameters()).dtype}")
    except Exception as e:
        print(f"\n✗ Failed to load action head: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Step 4: Loading Proprio Projector")
    print("=" * 70)
    
    try:
        proprio_projector = get_proprio_projector(policy_cfg, vla.llm_dim, PROPRIO_DIM)
        print("\n✓ Proprio projector loaded successfully!")
        print(f"  Device: {next(proprio_projector.parameters()).device}")
        print(f"  Dtype: {next(proprio_projector.parameters()).dtype}")
    except Exception as e:
        print(f"\n✗ Failed to load proprio projector: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Step 5: Test Inference with Random Input")
    print("=" * 70)
    
    try:
        # Create dummy observation
        observation = {
            "full_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "state": np.random.randn(8).astype(np.float32),  # 7 joints + 1 gripper
        }
        
        task = "pick up the red block"
        
        print(f"\nTask: {task}")
        print(f"Image shapes: {observation['full_image'].shape}, {observation['wrist_image'].shape}")
        print(f"State shape: {observation['state'].shape}")
        
        # Run inference
        import time
        start_time = time.time()
        
        actions = get_vla_action(
            policy_cfg,
            vla,
            processor,
            observation,
            task,
            action_head=action_head,
            proprio_projector=proprio_projector,
            use_film=policy_cfg.use_film,
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        print(f"\n✓ Inference successful!")
        print(f"  Inference time: {inference_time:.1f}ms")
        print(f"  Action chunk size: {len(actions)}")
        print(f"  Action shape: {actions[0].shape}")
        print(f"  First action: {actions[0]}")
        print(f"  Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print("\nThe policy is ready to use with your Franka robot.")
    print("Next steps:")
    print("  1. Connect your RealSense cameras")
    print("  2. Update camera serial numbers in config.py")
    print("  3. Ensure robot NUC is accessible")
    print("  4. Run: python main.py --instruction 'your task'")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(test_policy)
    sys.exit(0 if success else 1)
