#!/bin/bash

DATASET_NAME="VCM"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --name vcm_6frame_2caption_baseline \
  --dataset_name VCM \
  --root_dir /data/ydl/datasets \
  --loss_names 'sdm+id' \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_epoch 30 \
  --num_workers 2 \
  --num_frames 6 \
  --sampler random
