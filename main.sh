#!/bin/sh

# This script runs the deepfake detector training script with the specified parameters.

# Directory locations for combined dataset
model_name="CombinedDatasetv2"
real_dir="/mnt/d/deepfake_detector/combined_dataset/real"
fake_dir="/mnt/d/deepfake_detector/combined_dataset/fake"
output_dir="/mnt/d/deepfake_detector/trained_models"
cache_dir="/mnt/d/deepfake_detector/combined_cache"

# Directory locations for FaceForensics dataset
# model_name = "FF32f32b"
# real_dir = "/mnt/d/deepfake_detector/faceforensics/original_sequences"
# fake_dir = "/mnt/d/deepfake_detector/faceforensics/manipulated_sequences"
# output_dir = "/mnt/d/deepfake_detector/trained_models"
# cache_dir = "/mnt/d/deepfake_detector/cache_FF_32f"

# Directory locations for Celeb-DF dataset
# model_name = "CDF64f16b"
# real_dir = "/mnt/d/deepfake_detector/celeb-df/original_sequences"
# fake_dir = "/mnt/d/deepfake_detector/celeb-df/manipulated_sequences"
# output_dir = "/mnt/d/deepfake_detector/trained_models"
# cache_dir = "/mnt/d/deepfake_detector/cache_CDF_64f"

python3 main.py \
    --model_name $model_name \
    --real_dir $real_dir \
    --fake_dir $fake_dir \
    --output_dir $output_dir \
    --cache_dir $cache_dir \
    --max_videos_per_dir 1300 \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.00005 \
    --weight_decay 1e-4 \
    --num_frames 64 \
    --use_cosine_scheduler \
    --early_stopping_patience 25

