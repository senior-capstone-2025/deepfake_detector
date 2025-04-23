#!/bin/sh

# Change arguments to suit your environment
# My C drive is almost full, so using D drive via WSL
python3 main.py \
    --real_dir /mnt/d/deepfake_detector/deepfake_dataset/original_sequences \
    --fake_dir /mnt/d/deepfake_detector/deepfake_dataset/manipulated_sequences \
    --output_dir /mnt/d/deepfake_detector/trained_models \
    --cache_dir /mnt/d/deepfake_detector/preprocessed_cache \
    --max_videos_per_dir 500 \
    --include_evaluation \
