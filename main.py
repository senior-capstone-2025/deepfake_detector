import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
# Suppress PyTorch future version warnings
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import os
import sys
# Add the pixel2style2pixel directory to the Python path
pixel2style2pixel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixel2style2pixel')
sys.path.append(pixel2style2pixel_path)


from utils.load_models import load_psp_encoder, load_resnet_model

from preprocessor import DeepfakePreprocessor
from detector import DeepfakeDetector
from dataset import DeepfakeDataset, create_dataloaders

def test_pipeline(detector, preprocessor, real_dir, fake_dir, num_frames=8):
    """
    Test the preprocessing and detector pipeline on a few sample videos
    """
    print("\n=== Testing Pipeline ===")
    
    # Get sample video paths
    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.endswith(('.mp4', '.avi', '.mov'))][:2]  # Just use first 2
    fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                  if f.endswith(('.mp4', '.avi', '.mov'))][:2]  # Just use first 2
    
    if not real_videos or not fake_videos:
        print("No sample videos found in the directories.")
        return
    
    # Test preprocessing
    print("\nTesting preprocessing:")
    for i, video_path in enumerate(real_videos + fake_videos):
        print(f"Processing video {i+1}/{len(real_videos) + len(fake_videos)}: {os.path.basename(video_path)}")
        
        try:
            # Process video with fewer frames for testing
            start_time = time.time()
            video_tensor, face_tensor = preprocessor.process_video(video_path, num_frames=num_frames)
            processing_time = time.time() - start_time
            
            if video_tensor is None or face_tensor is None:
                print("  ❌ Failed to process video")
                continue
                
            print(f"  ✅ Successfully processed video in {processing_time:.2f}s")
            print(f"  Video tensor shape: {video_tensor.shape}")
            print(f"  Face tensor shape: {face_tensor.shape}")
        except Exception as e:
            print(f"  ❌ Error processing video: {e}")
    
    # Test detector forward pass
    print("\nTesting detector forward pass:")
    for i, video_path in enumerate(real_videos + fake_videos):
        try:
            # Process video
            video_tensor, face_tensor = preprocessor.process_video(video_path, num_frames=num_frames)
            
            if video_tensor is None or face_tensor is None:
                continue
                
            # Add batch dimension
            video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
            face_tensor = face_tensor.unsqueeze(0)    # [1, T, C, H, W]
            
            # Run through detector
            print(f"Running detector on {os.path.basename(video_path)}")
            with torch.no_grad():
                start_time = time.time()
                output = detector(video_tensor, face_tensor)
                inference_time = time.time() - start_time
            
            score = output.item()
            prediction = "FAKE" if score > 0.5 else "REAL"
            is_fake = "fake" in video_path.lower()
            correct = (prediction == "FAKE") == is_fake
            
            print(f"  ✅ Forward pass successful in {inference_time:.2f}s")
            print(f"  Score: {score:.4f}, Prediction: {prediction}, {'✓ Correct' if correct else '✗ Incorrect'}")
            
        except Exception as e:
            print(f"  ❌ Error in detector forward pass: {e}")
    
    # Test dataloader
    print("\nTesting dataloaders:")
    try:
        # Create a small test dataloader with just a few samples
        test_dataset = DeepfakeDataset(
            real_videos + fake_videos,
            [0] * len(real_videos) + [1] * len(fake_videos),
            preprocessor,
            num_frames=num_frames
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=True
        )
        
        print(f"Created test dataloader with {len(test_dataset)} samples")
        
        # Try loading a batch
        for batch in test_loader:
            video_batch, face_batch, labels = batch
            print(f"  ✅ Successfully loaded batch")
            print(f"  Video batch shape: {video_batch.shape}")
            print(f"  Face batch shape: {face_batch.shape}")
            print(f"  Labels: {labels}")
            break  # Just test the first batch
            
    except Exception as e:
        print(f"  ❌ Error testing dataloader: {e}")
    
    print("\n=== Pipeline Test Complete ===\n")

def main():

    print("Deepfake Detector")
    # Paths to video files
    real_dir = "videos/original"
    fake_dir = "videos/deepfake"
    # Paths to pretrained pSp encoder and 3D ResNet
    psp_path = "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
    resnet_path = "models/3d_resnet/resnet3d_50_features.pt"
    # Output directory for results
    output_dir = "results"
    # Device configuration (cuda/cpu)
    device = "cpu"
    # Video sizes
    face_size = (256, 256)
    video_size = (224, 224)
    # Hyperparameters
    batch_size = 4
    num_frames = 32
    epochs = 10
    learning_rate = 0.001

    print("Creating preprocessor...")
    # Initialize the video preprocessor
    preprocessor = DeepfakePreprocessor(
        face_size=face_size,
        video_size=video_size,
        device=device
    )

    print("Creating output directory...")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading pretrained models...")
    # Load the models
    psp_model = load_psp_encoder(psp_path)
    content_model = load_resnet_model(resnet_path)
    
    print("Creating deepfake detector model...")
    # Initialize the DeepfakeDetector
    detector = DeepfakeDetector(
        psp_model=psp_model,
        content_model=content_model,
        style_dim=512,          # W+ space dimension
        content_dim=2048,       # ResNet feature dimension
        gru_hidden_size=1024,
        output_dim=512
    )

    print("Creating dataloaders...")
    # Create dataloaders for training and validation
    train_loader, val_loader = create_dataloaders(
        real_dir,
        fake_dir,
        preprocessor,
        batch_size=batch_size,
        num_frames=num_frames
    )

    test_pipeline(detector, preprocessor, real_dir, fake_dir, num_frames=num_frames)

if __name__ == "__main__":
    main()