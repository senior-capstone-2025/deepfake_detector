import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from preprocessor import DeepfakePreprocessor

class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, labels, preprocessor, num_frames=32):
        """
        Dataset for deepfake detection
        
        Args:
            video_paths: List of paths to video files
            labels: Binary labels (0=real, 1=fake)
            preprocessor: Video preprocessor instance
            num_frames: Number of frames to extract per video
        """
        self.video_paths = video_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract video and face frames
        video_tensor, face_tensor = self.preprocessor.process_video(
            video_path, num_frames=self.num_frames
        )
        
        # Handle case where processing fails
        if video_tensor is None or face_tensor is None:
            # Return empty tensors and mark as invalid
            return (
                torch.zeros((3, self.num_frames, 224, 224)),
                torch.zeros((self.num_frames, 3, 256, 256)),
                torch.tensor(label, dtype=torch.float32),
                False
            )
        
        return (
            video_tensor,
            face_tensor,
            torch.tensor(label, dtype=torch.float32),
            True
        )

def create_dataloaders(real_dir, fake_dir, preprocessor, batch_size=8, num_frames=32):
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

    print("Total videos collected:", len(video_paths))
    print("Total labels collected:", len(labels))
    for i in range(len(labels)):
        print(f"Label for video {i}: {labels[i]}")
    
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
        num_frames
    )
    
    val_dataset = DeepfakeDataset(
        [video_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        preprocessor,
        num_frames
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=collate_with_filter  # Custom collate function to handle invalid samples
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