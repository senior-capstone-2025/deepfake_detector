#!/bin/sh

# Change arguments to suit your environment
# My C drive is almost full, so using D drive via WSL
python3 main.py \
    --model_name CombinedDataset_64frames_1500perdir \
    --real_dir /mnt/d/deepfake_detector/combined_dataset/real \
    --fake_dir /mnt/d/deepfake_detector/combined_dataset/fake \
    --output_dir /mnt/d/deepfake_detector/trained_models \
    --cache_dir /mnt/d/deepfake_detector/combined_cache \
    --max_videos_per_dir 1600 \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --weight_decay 1e-5 \
    --num_frames 64 \
    --use_mixup \
    --use_cosine_scheduler \
    --early_stopping_patience 5 

#--real_dir /mnt/d/deepfake_detector/deepfake_dataset/original_sequences \
#--fake_dir /mnt/d/deepfake_detector/deepfake_dataset/manipulated_sequences \

# --real_dir /mnt/d/deepfake_detector/celeb-df/original_sequences \
# --fake_dir /mnt/d/deepfake_detector/celeb-df/manipulated_sequences \
# --cache_dir /mnt/d/deepfake_detector/cache_CDF_64f \