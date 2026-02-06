#!/bin/bash
# Script to run OpenVLA-OFT evaluation on real Franka robot
# Similar to run_eval_real_world.sh but for real robot deployment

# Example usage:
#   bash run_eval_franka.sh

# Configuration
CUDA_DEVICE=0  # Which GPU to use
# CHECKPOINT="ZechenBai/OpenVLA-OFT"  # Your trained checkpoint
# CHECKPOINT="ZechenBai/OpenVLA-OFT-OneCam-L1Reg"  # Your trained checkpoint
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.0--parallel_dec--nonorm_8_acts_chunk--l1_reg--3rd_person_img--no_aug--20000_chkpt"  # Your trained checkpoint
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.0--parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_aug--20000_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b8+lr-0.0001+lora-r32+dropout-0.3--noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--800_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b8+lr-0.0001+lora-r32+dropout-0.3--noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--400_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b16+lr-0.0005+lora-r32+dropout-0.0--nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--10000_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b16+lr-0.0005+lora-r32+dropout-0.4--nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--6000_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/VLAST_ckpt/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.1--image_aug--VLAST_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--6000_chkpt"

# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/tube_ckpt/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.1--image_aug--tube_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--2000_chkpt"
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/tube_ckpt/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.2--tube_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--800_chkpt"

# RL Posttrain ckpts
CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/guian_ckpt/orange_cube_exp4_temporal_aligned/"
UNNORM_KEY="orange_cube_train_temporal_aligned"  # Your custom unnorm key
# CHECKPOINT="/home/showlab/openvla-oft/openvla-oft-xk/guian_ckpt/baseline/orange_cube_exp0_baseline"
# UNNORM_KEY="orange_cube_robot_20"  # Your custom unnorm key
# Camera serials (UPDATE THESE WITH YOUR CAMERA SERIALS)
EXTERNAL_CAMERA="317222075319"
#EXTERNAL_CAMERA="336222073740"
WRIST_CAMERA="218622273043"

# Robot connection (UPDATE WITH YOUR NUC IP)
NUC_IP="192.168.1.112"
NUC_PORT=4242

# Task instruction
# INSTRUCTION="place the cube on the purple gear"
INSTRUCTION="Pick up the orange cube and place it in the target location"
# INSTRUCTION="Pick up the test cube with the orange cap and put it in the grey cup."

# Evaluation settings
NUM_TRIALS=61
MAX_TIMESTEPS=450

# Run evaluation
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run_guian_eval.py \
  --pretrained_checkpoint $CHECKPOINT \
  --custom_unnorm_key $UNNORM_KEY \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio True \
  --center_crop True \
  --num_images_in_input 1 \
  --external_camera $EXTERNAL_CAMERA \
  --wrist_camera $WRIST_CAMERA \
  --nuc_ip $NUC_IP \
  --nuc_port $NUC_PORT \
  --instruction "$INSTRUCTION" \
  --lora_rank 32 \
  --num_trials $NUM_TRIALS \
  --max_timesteps $MAX_TIMESTEPS \
  --num_open_loop_steps 8 \
  --control_mode eef \
  --use_custom_unnormalization False \
  --save_video True \
  --show_cameras True \
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
