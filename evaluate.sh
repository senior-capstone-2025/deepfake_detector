#!/bin/sh

python3 evaluate_model.py \
    --model_dir /mnt/d/deepfake_detector/trained_models/0422_14h00m52s_model \
    --real_dir /mnt/d/deepfake_detector/deepfake_dataset/original_sequences \
    --fake_dir /mnt/d/deepfake_detector/deepfake_dataset/manipulated_sequences \
    --cache_dir /mnt/d/deepfake_detector/preprocessed_cache \
    --batch_size 32 \
    --max_videos_per_dir 100
