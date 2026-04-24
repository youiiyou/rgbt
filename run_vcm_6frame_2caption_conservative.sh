#!/bin/bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 \
python train.py \
  --name vcm_6frame_2caption_lr5e6_warmup1_ep10 \
  --dataset_name VCM \
  --root_dir /data/ydl/datasets \
  --loss_names 'sdm+id' \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_epoch 10 \
  --num_workers 2 \
  --num_frames 6 \
  --train_caption_mode double \
  --sampler random \
  --lr 5e-6 \
  --warmup_epochs 1
