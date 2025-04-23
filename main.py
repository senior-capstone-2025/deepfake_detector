## file: main.py
#
# Main training script for deepfake detector.
#
##

import time
import datetime
import argparse
import os
import numpy as np
import torch
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)

# Suppress PyTorch future version warnings
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Import necessary utility functions
from preprocess_all_videos import preprocess_all_videos
from evaluate import evaluate_model
# Import custom modules
from preprocessor import DeepfakePreprocessor
from detector import DeepfakeDetector
from dataset import create_dataloaders
from train import train_model

# Create logger
logger = logging.getLogger(__name__)

def create_directories(output_dir):
    # Create a unique directory for the final model
    train_date = datetime.datetime.now().strftime("%m%d_%Hh%Mm%Ss")
    model_dir = os.path.join(output_dir, f"{train_date}_model")
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Final model directory: %s", model_dir)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    return model_dir, checkpoint_dir

def main():

    logger.info("Starting main training loop.")

    # Argument parser to handle device selection
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    # Paths to real and fake video directories: default for testing purposes
    parser.add_argument('--real_dir', type=str, default='./videos/real', help='Directory with real videos')
    parser.add_argument('--fake_dir', type=str, default='./videos/fake', help='Directory with fake videos')
    # Device to use for training: default to cuda if available
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    # Add output directory argument
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Directory to save results')
    # Cache directory
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache', help='Directory for preprocessed data')
    # Maximum number of videos to process from each directory
    parser.add_argument('--max_videos_per_dir', type=int, default=100,
                       help='Maximum number of videos to process from each directory')
    # FLAG: Force reprocessing
    parser.add_argument('--force_reprocess', action='store_true', help='[FLAG] Force reprocessing of all videos (use when making changes to preprocessor)')
    # FLAG: Include evaluation
    parser.add_argument('--include_evaluation', action='store_true', help='[FLAG] Make predictions on validation set after training')
    
    args = parser.parse_args()
    logger.info("Arguments parsed: %s", args)

    # Video sizes
    face_size = (256, 256)
    video_size = (224, 224)

    # Hyperparameters
    batch_size = 64
    num_frames = 32
    epochs = 100
    learning_rate = 0.0001

    # Log hyperparameters
    logger.info("Using device: %s", args.device)
    logger.info("Hyperparameters: batch_size=%d, num_frames=%d, epochs=%d, lr=%.4f", 
    batch_size, num_frames, epochs, learning_rate)

    # Create directories for saving models and results
    model_dir, checkpoint_dir = create_directories(args.output_dir)

    # Initialize the video preprocessor
    logger.info("Creating preprocessor.")
    preprocessor = DeepfakePreprocessor(
        face_size=face_size,
        video_size=video_size,
        device=args.device
    )


    # Preprocess all videos in the specified directories
    logger.info("Starting video preprocessing...")
    video_info = preprocess_all_videos(
        args.real_dir,
        args.fake_dir,
        preprocessor,
        args.cache_dir,
        num_frames=num_frames,
        force_reprocess=args.force_reprocess,
        max_videos_per_dir=args.max_videos_per_dir  # Limit for testing purposes
    )


    # Create dataloaders for training and validation
    # The preprocessor will transform the videos into tensors on the first pass
    # and cache them for subsequent passes.
    logger.info("Creating traing/validation dataloaders.")
    train_loader, val_loader = create_dataloaders(
        video_info,
        batch_size=batch_size
    )


    # Initialize the DeepfakeDetector with loaded models and hyperparameters
    logger.info("Creating DeepfakeDetector model.")
    detector = DeepfakeDetector(
        device=args.device,
        style_dim=512,          # W+ space dimension (only one layer of pSp style)
        content_dim=2048,       # ResNet feature dimension
        gru_hidden_size=1024,
        output_dim=512
    )

    
    # Train the deepfake detector
    logger.info("Begin training.")
    trained_model, training_metadata = train_model(
        detector,
        train_loader,
        val_loader,
        device=args.device,
        num_epochs=epochs,
        lr=learning_rate,
        checkpoint_dir=checkpoint_dir
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pt")
    
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

    # If include_evaluation is set, evaluate the model on the validation set
    if args.include_evaluation:
        logger.info("Evaluating model on validation set.")
        evaluate_model(
            model_dir,
            args.device,
            val_loader
        )

if __name__ == "__main__":
    main()