#!/bin/sh

# Change arguments to suit your environment
# My C drive is almost full, so using D drive via WSL
python3 main.py \
    --model_name TestingTrainingCDF \
    --real_dir /mnt/d/deepfake_detector/celeb-df/original_sequences \
    --fake_dir /mnt/d/deepfake_detector/celeb-df/Celeb-synthesis \
    --output_dir /mnt/d/deepfake_detector/trained_models \
    --cache_dir /mnt/d/deepfake_detector/df_preprocessed_cache \
    --max_videos_per_dir 890 \
    --include_evaluation \
