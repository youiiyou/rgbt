#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
  --name vcm_6frame_1caption_sdm_mlm_id_irgray_lr5e6_warmup1_ep30 \
  --dataset_name VCM \
  --root_dir /data/ydl/datasets \
  --loss_names 'sdm+mlm+id' \
  --MLM \
  --batch_size 8 \
  --test_batch_size 8 \
  --num_epoch 30 \
  --num_workers 2 \
  --num_frames 6 \
  --train_caption_mode single \
  --ir_grayscale \
  --sampler random \
  --lr 5e-6 \
  --warmup_epochs 1
