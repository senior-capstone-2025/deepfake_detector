#!/bin/sh

# Example script to run the deepfake detector training script with specified parameters

# Directory locations for combined dataset
model_name="TrainingModelExample"
real_dir="videos/original"
fake_dir="videos/deepfake"
output_dir="trained_models"
cache_dir="combined_cache"

python3 main.py \
    --model_name $model_name \
    --real_dir $real_dir \
    --fake_dir $fake_dir \
    --output_dir $output_dir \
    --cache_dir $cache_dir \
    --max_videos_per_dir 5 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --num_frames 64 \
    --use_cosine_scheduler \
    --early_stopping_patience 2

