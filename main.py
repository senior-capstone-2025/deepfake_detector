## file: main.py
#
# Main training script for deepfake detector.
#
##

import time
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import torch

# Suppress PyTorch future version warnings
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import argparse
import os
import sys
# Add the pixel2style2pixel directory to the Python path
pixel2style2pixel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixel2style2pixel')
sys.path.append(pixel2style2pixel_path)

# Import necessary utility functions
from utils.load_models import load_psp_encoder, load_resnet_module

# Import custom modules
from preprocessor import DeepfakePreprocessor
from detector import DeepfakeDetector
from dataset import create_dataloaders
from train import train_model

# Create logger
logger = logging.getLogger(__name__)

def main():

    logger.info("Starting main training loop.")

    # Argument parser to handle device selection
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    # Paths to real and fake video directories: default for testing purposes
    parser.add_argument('--real_dir', type=str, default='videos/original', help='Directory with real videos')
    parser.add_argument('--fake_dir', type=str, default='videos/deepfake', help='Directory with fake videos')
    # Device to use for training: default to cuda if available
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    # Add output directory argument
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    # Add checkpoint directory argument
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()

    logger.info("Arguments parsed: %s", args)

    # Paths to video files
    real_dir = args.real_dir
    fake_dir = args.fake_dir
    # Paths to pretrained pSp encoder and 3D ResNet
    psp_path = "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
    # Output directory for results
    output_dir = args.output_dir
    # Checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    # Device configuration (cuda/cpu)
    device = args.device
    # Video sizes
    face_size = (256, 256)
    video_size = (224, 224)
    # Hyperparameters
    batch_size = 64
    num_frames = 32
    epochs = 2
    learning_rate = 0.001

    logger.info("Using device: %s", device)
    logger.info("Batch size: %d", batch_size)
    logger.info("Number of frames: %d", num_frames)
    logger.info("Number of epochs: %d", epochs)
    logger.info("Learning rate: %.4f", learning_rate)

    # Initialize the video preprocessor
    logger.info("Creating preprocessor.")
    preprocessor = DeepfakePreprocessor(
        face_size=face_size,
        video_size=video_size,
        device=device
    )

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the pretrained pSp encoder and 3D ResNet model
    logger.info("Loading pretrained pSp encoder and 3D ResNet model.")
    psp_model = load_psp_encoder(psp_path, device)
    content_model = load_resnet_module()
    
    # Initialize the DeepfakeDetector with loaded models and hyperparameters
    logger.info("Creating DeepfakeDetector model.")
    detector = DeepfakeDetector(
        psp_model=psp_model,
        content_model=content_model,
        style_dim=512,          # W+ space dimension (only one layer of pSp style)
        content_dim=2048,       # ResNet feature dimension
        gru_hidden_size=1024,
        output_dim=512
    )

    # Create dataloaders for training and validation
    # The preprocessor will transform the videos into tensors on the first pass
    # and cache them for subsequent passes.
    logger.info("Creating traing/validation dataloaders.")
    train_loader, val_loader = create_dataloaders(
        real_dir,
        fake_dir,
        preprocessor,
        batch_size=batch_size,
        num_frames=num_frames,
        cache_dir='preprocessed_cache'
    )

    # Train the deepfake detector
    logger.info("Begin training.")
    trained_model, training_metadata = train_model(
        detector,
        train_loader,
        val_loader,
        device=device,
        num_epochs=epochs,
        lr=learning_rate,
        checkpoint_dir=checkpoint_dir
    )

    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    
    # Save the model state dict along with training metadata
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'training_complete': True,
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'epochs_trained': epochs,
        'best_epoch': training_metadata['best_epoch'],
        'best_val_loss': training_metadata['best_val_loss'],
        'training_time_seconds': training_metadata['training_time']
    }, final_model_path)

    logger.info(f"Training complete. Final model saved to {final_model_path}")
    logger.info(f"Best model was from epoch {training_metadata['best_epoch']} with validation loss: {training_metadata['best_val_loss']:.4f}")
    logger.info(f"Total training time: {training_metadata['training_time']/60:.2f} minutes")

if __name__ == "__main__":
    main()