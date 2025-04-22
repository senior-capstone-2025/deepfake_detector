import os
import torch
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import your custom modules
from detector import DeepfakeDetector
from dataset import create_dataloaders
from preprocess import preprocess_all_videos

import os
from preprocessor import DeepfakePreprocessor

def evaluate_model(model, data_loader, device):

    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for content_features, style_codes, labels in data_loader:
            # Move data to device
            content_features = content_features.to(device)
            style_codes = style_codes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(content_features, style_codes)
            
            # Get predictions
            scores = outputs.squeeze().cpu().numpy()
            preds = (scores >= 0.5).astype(int)
            
            # Collect results
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create classification report
    class_report = classification_report(all_labels, all_preds, 
                                        target_names=['Real', 'Fake'],
                                        zero_division=0)
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{class_report}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_preds,
        'scores': all_scores,
        'labels': all_labels
    }
    
    return results

def plot_confusion_matrix(cm, output_dir):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix array
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Real', 'Fake'],
               yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained deepfake detection model')
    parser.add_argument('--model_dir', type=str, default=None, 
                       help='Path to the trained model directory (selects most recent file)')
    parser.add_argument('--real_dir', type=str, required=True,
                       help='Directory with real videos')
    parser.add_argument('--fake_dir', type=str, required=True,
                       help='Directory with fake videos')
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache',
                       help='Directory for preprocessed data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max_videos_per_dir', type=int, default=100,
                       help='Maximum number of videos to process from each directory')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to extract from each video')
    
    args = parser.parse_args()
    
    # Create output directory
    results_dir = os.path.join(args.model_dir, 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the trained model
    logger.info(f"Loading model from {args.model_dir}")
    model_path = os.path.join(args.model_dir, 'final_model.pt')
    checkpoint = torch.load(model_path, map_location=args.device)
    
    # Create model instance
    model = DeepfakeDetector(
        device=args.device,
        style_dim=512,
        content_dim=2048,
        gru_hidden_size=1024,
        output_dim=512
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Log model info
    logger.info(f"Model loaded. Training metadata:")
    for key in ['training_complete', 'training_date', 'epochs_trained', 'best_epoch', 'best_val_loss']:
        if key in checkpoint:
            logger.info(f"  {key}: {checkpoint[key]}")
    
    # Setup for preprocessing
    logger.info("Setting up preprocessor")
    face_size = (256, 256)
    video_size = (224, 224)
    
    
    # Initialize preprocessor
    preprocessor = DeepfakePreprocessor(
        face_size=face_size,
        video_size=video_size,
        device=args.device
    )
    
    # Preprocess videos (or load from cache)
    logger.info("Preprocessing videos (or loading from cache)")
    video_info = preprocess_all_videos(
        args.real_dir,
        args.fake_dir,
        preprocessor,
        args.cache_dir,
        num_frames=args.num_frames,
        force_reprocess=False,
        max_videos_per_dir=args.max_videos_per_dir
    )
    
    # Create dataloader (using all data as validation)
    _, val_loader = create_dataloaders(
        video_info,
        batch_size=args.batch_size,
        train_split=0  # Use all data for evaluation
    )
    
    eval_date = datetime.datetime.now().strftime("%m%d_%Hh%Mm%Ss")
    eval_results_file = (eval_date + "_results.pt")
    eval_text_file = (eval_date + "_summary.txt")
    logger.info(f"Evaluation results will be saved to {eval_results_file} and {eval_text_file}")

    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(model, val_loader, args.device)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], results_dir)
    
    # Save detailed results
    results_file = os.path.join(results_dir, eval_results_file)
    torch.save(results, results_file)
    logger.info(f"Detailed results saved to {results_file}")
    


    # Save text summary
    summary_file = os.path.join(results_dir, eval_text_file)
    with open(summary_file, 'w') as f:
        f.write("DEEPFAKE DETECTOR EVALUATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation date: {eval_date}\n\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n\n")
        f.write(f"- Fake -")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1 Score:  {results['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{results['confusion_matrix']}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{results['classification_report']}\n")
    
    logger.info(f"Evaluation summary saved to {summary_file}")
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()