#!/bin/bash
# Script to run OpenVLA-OFT evaluation on real Franka robot
# Similar to run_eval_real_world.sh but for real robot deployment

# Example usage:
#   bash run_eval_franka.sh

# Configuration
CUDA_DEVICE=0  # Which GPU to use

CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/guian_ckpt/baseline"

UNNORM_KEY="pick_n_place_ee"  # dataset_statistics key used during training

# Camera serials (UPDATE THESE WITH YOUR CAMERA SERIALS)
# EXTERNAL_CAMERA="317222075319"
EXTERNAL_CAMERA="336222073740"
WRIST_CAMERA="218622273043"

# Robot connection (UPDATE WITH YOUR NUC IP)
NUC_IP="192.168.1.112"
NUC_PORT=4242

# Task instruction
INSTRUCTION="Pick up the orange cube and place it in the target location."

# Evaluation settings
NUM_TRIALS=61
MAX_TIMESTEPS=400

# Run evaluation
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run_franka_eval.py \
  --pretrained_checkpoint $CHECKPOINT \
  --custom_unnorm_key $UNNORM_KEY \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio False \
  --num_images_in_input 2 \
  --external_camera $EXTERNAL_CAMERA \
  --wrist_camera $WRIST_CAMERA \
  --nuc_ip $NUC_IP \
  --nuc_port $NUC_PORT \
  --instruction "$INSTRUCTION" \
  --lora_rank 32 \
  --num_trials $NUM_TRIALS \
  --max_timesteps $MAX_TIMESTEPS \
  --control_mode eef \
  --use_custom_unnormalization True \
  --action_bounds_name pick_n_place_ee \
  --save_video True \
  --show_cameras True \
  --position_scale 0.6 \
  --rotation_scale 0.6 \
  # --action_bounds_name pick_test_cube_in_cup \
  # --center_crop True \
  # --use_mock_robot True \
  # --action_bounds_name robot_rlds_kevin \

# To use different action bounds, change --action_bounds_name to one of:
#   - robot_rlds_kevin (current/default)
#   - libero_spatial
#   - your_custom_name (after adding to action_unnormalization.py)
#
# Or run: python action_unnormalization.py
# to see all available bounds
