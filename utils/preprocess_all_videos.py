## file: preprocess.py
#
# Preprocesses all videos on the first pass.
# 
##

import os
import torch
import logging
import time
from glob import glob
import random
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

def preprocess_video_directory(
    dir_path, 
    preprocessor, 
    cache_dir, 
    label, 
    num_frames=64, 
    force_reprocess=False,
    max_videos_per_dir=None,
    metadata_file=None,
    metadata_label=None
):
    """
    Preprocess all videos in a directory and save to cache
    
    Args:
        dir_path: Directory containing videos
        preprocessor: DeepfakePreprocessor instance
        cache_dir: Directory to save cached tensors
        label: Label for these videos (0=real, 1=fake)
        num_frames: Number of frames to extract per video
        force_reprocess: Whether to force reprocessing even if cache exists
        max_videos_per_dir: Maximum number of videos to process from this directory
        metadata_file: Path to metadata file (JSON) for specific video selection
        metadata_label: Label to filter videos in metadata file
        
    Returns:
        List of tuples (cache_path, label, is_valid)
    """
    
    os.makedirs(cache_dir, exist_ok=True)
    
    video_info = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Read metadata file to find real/fake videos
    if metadata_file:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        videos = []
        for filename, info in metadata.items():
            if info['label'] == metadata_label:
                video_path = os.path.join(dir_path, filename)
                if os.path.exists(video_path):
                    videos.append((video_path))
                else:
                    logger.warning(f"Video {video_path} not found in directory {dir_path}")
    # # If no metadata file, find all videos in the directory (i.e. dataset/real or dataset/fake)
    else:
        videos = [f for f in os.listdir(dir_path) 
                if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    logger.info(f"Found {len(videos)} videos in {dir_path}")

    # If max_videos_per_dir is specified, randomly select that many videos
    if max_videos_per_dir and len(videos) > max_videos_per_dir:
        random.seed(42)
        videos = random.sample(videos, max_videos_per_dir)
        logger.info(f"Randomly selected {max_videos_per_dir} videos for processing")
    
    # Process each video
    for filename in tqdm(videos, desc=f"Processing videos in {os.path.basename(dir_path)}"):
        video_path = os.path.join(dir_path, filename)
        cache_key = os.path.basename(video_path).replace('.', '_')
        cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
        
        # Skip if already cached
        if os.path.exists(cache_path) and not force_reprocess:
            try:
                # Quick check for valid cache file
                cached_data = torch.load(cache_path)
                if isinstance(cached_data, tuple) and len(cached_data) == 3:
                    video_info.append((cache_path, label, True))
                    continue
            except Exception as e:
                logger.warning(f"Invalid cache for {video_path}: {e}")
        
        # Process video
        try:
            content_features, style_codes = preprocessor.process_video(
                video_path, num_frames=num_frames
            )
            
            if content_features is not None and style_codes is not None:
                # Save to cache
                torch.save((content_features, style_codes, True), cache_path)
                video_info.append((cache_path, label, True))
            else:
                # Mark as invalid
                torch.save((None, None, False), cache_path)
                video_info.append((cache_path, label, False))
                logger.warning(f"Failed to process video: {video_path}")
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            # Save as invalid
            torch.save((None, None, False), cache_path)
            video_info.append((cache_path, label, False))
    
    return video_info

def preprocess_all_videos(real_dir, fake_dir, preprocessor, cache_dir, num_frames=32, force_reprocess=False, max_videos_per_dir=None, metadata_file=None):
    """
    Preprocess all real and fake videos and return metadata for dataset creation
    
    Args:
        real_dir: Directory with real videos
        fake_dir: Directory with fake videos
        preprocessor: DeepfakePreprocessor instance  
        cache_dir: Base directory for cached files
        num_frames: Number of frames to extract per video
        force_reprocess: Whether to reprocess videos even if cached,
        max_videos_per_dir: Maximum number of videos to process from each directory
        metadata_file: Path to metadata file (JSON) for specific video selection
        
    Returns:
        List of tuples (cache_path, label, is_valid)
    """

    start_time = time.time()
    
    # Create cache directories
    real_cache_dir = os.path.join(cache_dir, 'real')
    fake_cache_dir = os.path.join(cache_dir, 'fake')
    os.makedirs(real_cache_dir, exist_ok=True)
    os.makedirs(fake_cache_dir, exist_ok=True)
    
    # Process real videos
    logger.info("Processing real videos...")
    real_videos = preprocess_video_directory(
        real_dir, 
        preprocessor, 
        real_cache_dir, 
        label=0,
        num_frames=num_frames,
        force_reprocess=force_reprocess,
        max_videos_per_dir=max_videos_per_dir,
        metadata_file=metadata_file,
        metadata_label='REAL'
    )
    
    # Process fake videos
    logger.info("Processing fake videos...")
    fake_videos = preprocess_video_directory(
        fake_dir, 
        preprocessor, 
        fake_cache_dir, 
        label=1,
        num_frames=num_frames,
        force_reprocess=force_reprocess,
        max_videos_per_dir=max_videos_per_dir,
        metadata_file=metadata_file,
        metadata_label='FAKE'
    )
    
    # Combine and return all video info
    all_videos = real_videos + fake_videos
    logger.info(f"Total processed videos: {len(all_videos)}")
    
    # Count valid videos
    valid_count = sum(1 for _, _, is_valid in all_videos if is_valid)
    logger.info(f"Valid videos: {valid_count}, Invalid: {len(all_videos) - valid_count}")
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("Time per video: {:.2f} seconds".format(total_time / len(all_videos) if all_videos else 0))

    return all_videos