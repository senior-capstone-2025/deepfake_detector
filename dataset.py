## file: dataset.py
#
# DeepfakeDataset :
# Preprocesses all videos on the first pass.
# https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
##

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import time

# Setup logging
import logging
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, labels, preprocessor, num_frames=32, cache_dir='prerpocessed_cache'):
        """
        Dataset for deepfake detection
        
        Args:
            video_paths: List of paths to video files
            labels: Binary labels (0=real, 1=fake)
            preprocessor: Video preprocessor instance
            num_frames: Number of frames to extract per video
            cache_dir: Directory to store preprocessed video tensors
        """
        self.video_paths = video_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.num_frames = num_frames
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Preprocess and cache all videos upon initialization
        self.cached_data = {}
        self.preprocess_all()
        
    def preprocess_all(self):
        """Preprocess all videos and save to cache"""
        logger.info(f"Preprocessing {len(self.video_paths)} videos...")
        
        # Track stats
        success_count = 0
        fail_count = 0
        cached_count = 0
        
        start_time = time.time()
        
        for i, video_path in enumerate(tqdm(self.video_paths, desc="Preprocessing videos")):
            # Create a unique cache key based on the video path
            cache_key = os.path.basename(video_path).replace('.', '_')
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            
            # Check if already cached
            if os.path.exists(cache_path):
                try:
                    # Load from cache
                    cached_tensors = torch.load(cache_path)
                    self.cached_data[i] = cached_tensors
                    cached_count += 1
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cache for {video_path}: {e}")
            
            # Not cached, need to process
            try:
                # Extract video and face frames
                video_tensor, face_tensor = self.preprocessor.process_video(
                    video_path, num_frames=self.num_frames
                )
                if video_tensor is not None and face_tensor is not None:
                    # Save to memory cache
                    self.cached_data[i] = (video_tensor, face_tensor, True)
                    
                    # Save to disk cache
                    torch.save((video_tensor, face_tensor, True), cache_path)
                    success_count += 1
                else:
                    # Mark as invalid
                    self.cached_data[i] = (None, None, False)
                    fail_count += 1
            except Exception as e:
                logger.error(f"Error preprocessing {video_path}: {e}")
                self.cached_data[i] = (None, None, False)
                fail_count += 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Cache stats: {success_count} successful, {fail_count} failed, {cached_count} loaded from cache")

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if idx in self.cached_data:
            video_tensor, face_tensor, is_valid = self.cached_data[idx]
            if is_valid:
                # Return cached tensors
                return (
                    video_tensor,
                    face_tensor,
                    torch.tensor(label, dtype=torch.float32),
                    True
                )
        
        # Return empty tensors for invalid samples
        logger.warning(f"Invalid sample at index {idx}, returning empty tensors.")
        return (
            torch.zeros((3, self.num_frames, 224, 224)),
            torch.zeros((self.num_frames, 3, 256, 256)),
            torch.tensor(label, dtype=torch.float32),
            False
        )

def create_dataloaders(real_dir, fake_dir, preprocessor, batch_size=8, num_frames=32, cache_dir='prerpocessed_cache'):
    """Create training and validation dataloaders"""
    
    # Collect video paths and labels
    video_paths = []
    labels = []
    
    # Add real videos
    for filename in os.listdir(real_dir):
        if filename.endswith(('.mp4')):
            video_paths.append(os.path.join(real_dir, filename))
            labels.append(0)  # 0 = real
    
    # Add fake videos
    for filename in os.listdir(fake_dir):
        if filename.endswith(('.mp4')):
            video_paths.append(os.path.join(fake_dir, filename))
            labels.append(1)  # 1 = fake

    logger.info(f"Total videos collected: {len(video_paths)}")
    logger.info(f"Total labels collected: {len(labels)}")
    
    # Shuffle and split into train/val
    indices = list(range(len(video_paths)))
    np.random.shuffle(indices)
    
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        [video_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        preprocessor,
        num_frames,
        cache_dir=os.path.join(cache_dir, 'train')
    )
    
    val_dataset = DeepfakeDataset(
        [video_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        preprocessor,
        num_frames,
        cache_dir=os.path.join(cache_dir, 'val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=collate_with_filter
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_with_filter
    )
    
    return train_loader, val_loader

def collate_with_filter(batch):
    """Custom collate function to filter out invalid samples"""
    valid_samples = [item for item in batch if item[3]]  # Check valid flag
    
    if not valid_samples:
        # Return empty batch if no valid samples
        logger.warning("No valid samples in batch, returning empty tensors.")
        return None
    
    # Separate the components
    video_tensors = [item[0] for item in valid_samples]
    face_tensors = [item[1] for item in valid_samples]
    labels = [item[2] for item in valid_samples]
    
    # Stack into batches
    video_batch = torch.stack(video_tensors)
    face_batch = torch.stack(face_tensors)
    label_batch = torch.stack(labels)
    
    return video_batch, face_batch, label_batch