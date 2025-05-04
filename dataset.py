## file: dataset.py
#
# Create dataloader and cached dataset for deepfake detection.
# https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
#
##

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class CachedDeepfakeDataset(Dataset):
    def __init__(self, video_info):
        """
        Dataset for deepfake detection that loads from cache
        
        Args:
            video_info: List of tuples (cache_path, label, is_valid)
        """
        self.video_info = video_info
        
    def __len__(self):
        return len(self.video_info)
    
    def __getitem__(self, idx):
        cache_path, label, is_valid = self.video_info[idx]
        
        if is_valid:
            try:
                # Load cached tensors
                cached_data = torch.load(cache_path)

                video_tensor, style_codes, _ = cached_data

                if video_tensor is not None and style_codes is not None:
                    return (
                        video_tensor,
                        style_codes,
                        torch.tensor(label, dtype=torch.float32),
                        True
                    )
            except Exception as e:
                logger.warning(f"Error loading cache at {cache_path}: {e}")
                is_valid = False
        
        # Return empty tensors for invalid samples
        logger.warning(f"Invalid sample at index {idx}, returning empty tensors.")
        return (
            torch.zeros((3, 32, 224, 224)),
            None,
            torch.tensor(label, dtype=torch.float32),
            False
        )

def create_dataloaders(video_info, batch_size=8, train_split=0.8):
    """
    Create training and validation dataloaders from preprocessed video info
    
    Args:
        video_info: List of tuples (cache_path, label, is_valid)
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        
    Returns:
        train_loader, val_loader
    """
    
    # Shuffle and split into train/val
    indices = list(range(len(video_info)))
    np.random.shuffle(indices)
    
    split = int(train_split * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create train and validation datasets
    train_dataset = CachedDeepfakeDataset(
        [video_info[i] for i in train_indices]
    )
    
    val_dataset = CachedDeepfakeDataset(
        [video_info[i] for i in val_indices]
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    if train_split != 0:
        # Create training dataloader if train_split is not 0
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            collate_fn=collate_with_filter
        )
    else:
        train_loader = None
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_with_filter
    )
    
    return train_loader, val_loader


def collate_with_filter(batch):
    """
    Custom collate function to filter out invalid samples
    
    Args:
        batch: List of tuples (video_tensor, style_codes, label, is_valid)
        
    Returns:
        Tuple of stacked tensors (content_features, style_codes, labels)
    """
    
    valid_samples = [item for item in batch if item[3]]
    
    if not valid_samples:
        # Return empty batch if no valid samples
        logger.warning("No valid samples in batch")
        return None
    
    # Separate the components
    content_features = [item[0] for item in valid_samples]
    style_codes = [item[1] for item in valid_samples]
    labels = [item[2] for item in valid_samples]
    
    # Stack into batches
    content_features_batch = torch.stack(content_features)
    style_codes_batch = torch.stack(style_codes)
    label_batch = torch.stack(labels)
    
    return content_features_batch, style_codes_batch, label_batch