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

from tqdm import tqdm
from preprocessor import DeepfakePreprocessor

logger = logging.getLogger(__name__)

import os
import torch
from glob import glob

def load_preprocessed_cache(cache_dir):
    """
    Load *all* of your cached .pt files (real vs fake) and return their metadata.
    
    Args:
        cache_dir: Base directory where you have subfolders 'real' and 'fake'
        
    Returns:
        List of tuples (cache_path, label, is_valid)
          - cache_path: full path to the .pt file
          - label: 0 for real, 1 for fake
          - is_valid: the third element of the saved tuple (or False on load‐error)
    """
    video_info = []
    for subfolder, label in (("real", 0), ("fake", 1)):
        folder = os.path.join(cache_dir, subfolder)
        # find all torch caches
        for cache_path in glob(os.path.join(folder, "*.pt")):
            try:
                data = torch.load(cache_path)
                # expect (content_features, style_codes, is_valid)
                is_valid = bool(data[2]) if isinstance(data, (list, tuple)) and len(data) >= 3 else False
            except Exception:
                is_valid = False
            video_info.append((cache_path, label, is_valid))
    return video_info

def preprocess_video_directory(
    dir_path, 
    preprocessor, 
    cache_dir, 
    label, 
    num_frames=64, 
    force_reprocess=False,
    max_videos_per_dir=None
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
        
    Returns:
        List of tuples (cache_path, label, is_valid)
    """
    
    os.makedirs(cache_dir, exist_ok=True)
    
    video_info = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Find all videos
    videos = [f for f in os.listdir(dir_path) 
              if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    logger.info(f"Found {len(videos)} videos in {dir_path}")

    # If max_videos_per_dir is specified, randomly select that many videos
    if max_videos_per_dir and len(videos) > max_videos_per_dir:
        random.see(42)
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



def preprocess_all_videos(real_dir, fake_dir, preprocessor, cache_dir, num_frames=32, force_reprocess=False, max_videos_per_dir=None):
    """
    Preprocess all real and fake videos and return metadata for dataset creation
    
    Args:
        real_dir: Directory with real videos
        fake_dir: Directory with fake videos
        preprocessor: DeepfakePreprocessor instance  
        cache_dir: Base directory for cached files
        num_frames: Number of frames to extract per video
        force_reprocess: Whether to reprocess videos even if cached
        
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
        label=0,  # 0 = real
        num_frames=num_frames,
        force_reprocess=force_reprocess,
        max_videos_per_dir=max_videos_per_dir
    )
    
    # Process fake videos
    logger.info("Processing fake videos...")
    fake_videos = preprocess_video_directory(
        fake_dir, 
        preprocessor, 
        fake_cache_dir, 
        label=1,  # 1 = fake
        num_frames=num_frames,
        force_reprocess=force_reprocess,
        max_videos_per_dir=max_videos_per_dir
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