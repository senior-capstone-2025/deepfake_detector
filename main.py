## file: main.py
#
# Main training and evaluation script for deepfake detector.
#
# This script handles the entire pipeline for the model:
# 1. Preprocessing & caching videos
# 2. Creating dataloaders
# 3. Training the model
# 4. Evaluating the model
#
##

import time
import datetime
import argparse
import os
import sys
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
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Suppress PyTorch future version warnings
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
# Create logger
logger = logging.getLogger(__name__)

# Import necessary utility functions
from utils.preprocess_all_videos import preprocess_all_videos
from utils.model_analyzer import ModelAnalyzer

# Import custom modules
from preprocessor import DeepfakePreprocessor
from detector import DeepfakeDetector
from dataset import create_dataloaders
from train import train_model

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train and analyze deepfake detection model')
    parser.add_argument('--model_name', type=str, default='DeepfakeDetector', help='Name of the model to train')
    parser.add_argument('--real_dir', type=str, default='./videos/real', help='Directory with real videos')
    parser.add_argument('--fake_dir', type=str, default='./videos/fake', help='Directory with fake videos')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Directory to save results')
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache', help='Directory for preprocessed data')
    parser.add_argument('--max_videos_per_dir', type=int, default=100, help='Maximum number of videos to process from each directory')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of all videos')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--num_frames', type=int, default=64, help='Number of frames to extract per video')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='Use cosine annealing scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    return parser.parse_args()
    
def create_directories(output_dir, model_name):
    """Create directories for saving the model, checkpoints, visualization logs, and analysis results."""
    
    # Create a unique directory for the final model based on the current date and time
    train_date = datetime.datetime.now().strftime("%m%d_%Hh%Mm%Ss")
    model_dir = os.path.join(output_dir, f"{train_date}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Final model directory: %s", model_dir)

    # Create directories for checkpoints, visualization logs, and analysis
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    visualizer_dir = os.path.join(model_dir, "visualization_logs")
    analysis_dir = os.path.join(model_dir, "analysis")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualizer_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info(f"Visualization logs directory: {visualizer_dir}")
    logger.info("Analysis directory: %s", analysis_dir)
    
    return model_dir, checkpoint_dir, visualizer_dir, analysis_dir

def main():
    """Main function for training and analyzing deepfake detection model"""
    logger.info("Starting deepfake detection pipeline")
    
    # Parse command-line arguments
    args = parse_arguments()

    # Create directories for output
    model_dir, checkpoint_dir, viz_dir, analysis_dir = create_directories(args.output_dir, args.model_name)

    # Set device
    device = args.device
    logger.info(f"Using device: {device}")

    # Initialize preprocessor
    logger.info("Creating preprocessor")
    face_size = (256, 256)
    video_size = (224, 224)
    preprocessor = DeepfakePreprocessor(
        face_size=face_size,
        video_size=video_size,
        device=device
    )

    # Preprocess all videos, load/store from cache
    logger.info("Starting video preprocessing")
    video_info = preprocess_all_videos(
        args.real_dir,
        args.fake_dir,
        preprocessor,
        args.cache_dir,
        num_frames=args.num_frames,
        force_reprocess=args.force_reprocess,
        max_videos_per_dir=args.max_videos_per_dir
    )
    
    # Create dataloaders for training and validation
    logger.info("Creating dataloaders")
    train_loader, val_loader = create_dataloaders(
        video_info,
        batch_size=args.batch_size
    )

    # Initialize model
    logger.info("Creating DeepfakeDetector model")
    model = DeepfakeDetector(
        device=device,
        style_dim=512,
        content_dim=2048,
        gru_hidden_size=1024,
        output_dim=512
    )

    # Train the model
    trained_model, training_metadata = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
        weight_decay=args.weight_decay,
        use_mixup=args.use_mixup,
        use_cosine_scheduler=args.use_cosine_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        visualizer_dir=viz_dir
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'training_complete': True,
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'epochs_trained': args.num_epochs,
        'best_epoch': training_metadata['best_epoch'],
        'best_val_loss': training_metadata['best_val_loss'],
        'training_time_seconds': training_metadata['training_time'],
        'model_config': {
            'style_dim': 512,
            'content_dim': 2048,
            'gru_hidden_size': 1024,
            'output_dim': 512
        }
    }, final_model_path)

    logger.info(f"Training complete. Final model saved to {final_model_path}")
    logger.info(f"Best model was from epoch {training_metadata['best_epoch']} with validation loss: {training_metadata['best_val_loss']:.4f}")
    logger.info(f"Total training time: {training_metadata['training_time']/60:.2f} minutes")
        
    # Run model analysis
    logger.info("Running model analysis")
    analyzer = ModelAnalyzer(model, device=device, output_dir=analysis_dir)
    analysis_results = analyzer.run_comprehensive_analysis(val_loader, output_dir=analysis_dir)
        
    # Print analysis summary
    metrics = analysis_results['metrics']
    logger.info(f"Analysis complete. Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    logger.info(f"Analysis results saved to {analysis_dir}")
        
    logger.info("Deepfake detection pipeline completed!")

if __name__ == "__main__":
    main()