## file: model_analysis.py
#
# Model analysis tools for deepfake detection model
# Includes feature importance visualization, failure case analysis, 
# and gradient visualization.
#
##

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
from captum.attr import IntegratedGradients, GradientShap, Occlusion
import logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary utility functions
from utils.preprocess_all_videos import preprocess_all_videos

# Import custom modules
from preprocessor import DeepfakePreprocessor
from detector import DeepfakeDetector
from dataset import create_dataloaders

class ModelAnalyzer:
    def __init__(self, model, device='cuda', output_dir='model_analysis'):
        """
        Initialize the model analyzer
        
        Args:
            model: Trained deepfake detection model
            device: Device to run analysis on
            output_dir: Directory to save analysis results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def analyze_validation_set(self, val_loader, threshold=0.5):
        """
        Run comprehensive analysis on validation set
        
        Args:
            val_loader: Validation data loader
            threshold: Classification threshold
            
        Returns:
            Dictionary of analysis results
        """
        # Collect predictions and ground truths
        all_preds = []
        all_labels = []
        all_content_features = []
        all_style_codes = []
        correct_indices = []
        incorrect_indices = []
        
        logger.info("Collecting predictions on validation set")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                
                content_features, style_codes, labels = batch
                
                content_features = content_features.to(self.device)
                style_codes = style_codes.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(content_features, style_codes)
                outputs = outputs.squeeze(1)
                
                # Collect data
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store content features and style codes for further analysis
                all_content_features.append(content_features.cpu().numpy())
                all_style_codes.append(style_codes.cpu().numpy())
                
                # Record correct and incorrect predictions
                binary_preds = (outputs > threshold).float()
                batch_size = labels.size(0)
                
                for i in range(batch_size):
                    idx = batch_idx * val_loader.batch_size + i
                    if binary_preds[i] == labels[i]:
                        correct_indices.append(idx)
                    else:
                        incorrect_indices.append(idx)
        
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_content_features = np.concatenate(all_content_features, axis=0)
        all_style_codes = np.concatenate(all_style_codes, axis=0)
        
        # Calculate metrics
        binary_preds = (all_preds > threshold).astype(int)
        cm = confusion_matrix(all_labels, binary_preds)
        
        # True/False Positives/Negatives
        tp = cm[1, 1]
        fp = cm[0, 1]
        tn = cm[0, 0]
        fn = cm[1, 0]
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Analysis complete. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Generate analysis results
        results = {
            'predictions': all_preds,
            'labels': all_labels,
            'content_features': all_content_features,
            'style_codes': all_style_codes,
            'correct_indices': correct_indices,
            'incorrect_indices': incorrect_indices,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        }
        
        return results
    
    def visualize_confusion_matrix(self, results, save_path=None):
        """
        Visualize confusion matrix
        
        Args:
            results: Analysis results from analyze_validation_set
            save_path: Path to save the visualization
        """
        # Extract metrics
        metrics = results['metrics']
        
        # Create confusion matrix
        cm = np.array([
            [metrics['tn'], metrics['fp']],
            [metrics['fn'], metrics['tp']]
        ])
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks([0.5, 1.5], ['Real', 'Fake'])
        plt.yticks([0.5, 1.5], ['Real', 'Fake'])
        
        # Save figure
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def _analyze_feature_importance(self, content_feature, style_code):
        """
        Analyze feature importance using integrated gradients
        
        Args:
            content_feature: Content feature tensor
            style_code: Style code tensor
            
        Returns:
            Importance scores for each feature
        """
        try:
            was_training = self.model.training
            self.model.train()
            
            # Define a wrapper function for the model that returns scalar outputs
            def model_wrapper(content, style):
                output = self.model(content, style)
                # Ensure output is squeezed to remove any singleton dimensions
                return output.squeeze()
            
            # Initialize integrated gradients
            ig = IntegratedGradients(model_wrapper)
            
            # Get attributions - the key fix is in the target parameter
            # For binary classification, use None instead of 0 or 1 as target
            attributions, _ = ig.attribute(
                (content_feature, style_code),
                target=None,  # Changed from 0 to None
                return_convergence_delta=True
            )
            
            # Convert to importance scores
            importance_scores = attributions[0].abs().sum(dim=(2, 3)).squeeze().cpu().numpy()
            
            return importance_scores
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            logger.error(f"Content feature shape: {content_feature.shape}, Style code shape: {style_code.shape}")
            # Return dummy importance scores instead of None
            return np.ones((content_feature.shape[1],))  # Return dummy scores based on feature dimension
        finally:
            # Restore to original mode
            if not was_training:
                self.model.eval()
        
    def visualize_decision_boundaries(self, results, save_path=None):
        """
        Visualize decision boundaries in feature space
        
        Args:
            results: Analysis results from analyze_validation_set
            save_path: Path to save the visualization
        """
        from sklearn.decomposition import PCA
        
        # Extract data
        content_features = results['content_features']
        labels = results['labels']
        predictions = results['predictions']
        
        # Flatten features
        if len(content_features.shape) > 2:
            features = content_features.reshape(content_features.shape[0], -1)
        else:
            features = content_features
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        # Plot decision boundaries
        plt.figure(figsize=(12, 10))
        
        # Plot points colored by true label
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                 c=labels, cmap='coolwarm', alpha=0.6, edgecolors='w')
        plt.colorbar(scatter, label='True Label')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('True Labels in Feature Space')
        
        # Plot points colored by prediction
        plt.subplot(2, 1, 2)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                 c=predictions, cmap='coolwarm', alpha=0.6, edgecolors='w')
        plt.colorbar(scatter, label='Prediction')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Predictions in Feature Space')
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Decision boundaries visualization saved to {save_path}")
        
        plt.close()
        
        return reduced_features
    
    def analyze_style_temporal_patterns(self, results, save_path=None):
        """
        Analyze temporal patterns in style codes
        
        Args:
            results: Analysis results from analyze_validation_set
            save_path: Path to save the visualization
        """
        # Extract style codes and labels
        style_codes = results['style_codes']
        labels = results['labels']
        
        # Calculate style flow (difference between consecutive frames)
        style_flow = style_codes[:, 1:] - style_codes[:, :-1]
        
        # Calculate flow magnitude for each sample
        flow_magnitude = np.sqrt(np.sum(style_flow ** 2, axis=2))
        
        # Calculate average flow magnitude per sample
        avg_flow_magnitude = np.mean(flow_magnitude, axis=1)
        
        # Separate by class
        real_magnitude = avg_flow_magnitude[labels == 0]
        fake_magnitude = avg_flow_magnitude[labels == 1]
        
        # Plot distributions
        plt.figure(figsize=(12, 8))
        
        plt.hist(real_magnitude, bins=30, alpha=0.5, label='Real')
        plt.hist(fake_magnitude, bins=30, alpha=0.5, label='Fake')
        
        plt.xlabel('Average Style Flow Magnitude')
        plt.ylabel('Count')
        plt.title('Distribution of Style Flow Magnitude')
        plt.legend()
        
        # Calculate and display statistics
        real_mean = np.mean(real_magnitude)
        fake_mean = np.mean(fake_magnitude)
        real_std = np.std(real_magnitude)
        fake_std = np.std(fake_magnitude)
        
        plt.axvline(real_mean, color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(fake_mean, color='orange', linestyle='dashed', linewidth=1)
        
        plt.text(0.05, 0.95, 
                f"Real: mean={real_mean:.4f}, std={real_std:.4f}\nFake: mean={fake_mean:.4f}, std={fake_std:.4f}", 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Style temporal pattern analysis saved to {save_path}")
        
        plt.close()
        
        # Temporal flow patterns over sequence
        # Plot average flow magnitude over time
        plt.figure(figsize=(12, 8))
        
        # Calculate mean flow magnitude over time for each class
        real_indices = np.where(labels == 0)[0]
        fake_indices = np.where(labels == 1)[0]
        
        if len(real_indices) > 0:
            real_flow_time = np.mean(flow_magnitude[real_indices], axis=0)
            real_flow_std = np.std(flow_magnitude[real_indices], axis=0)
            frames = np.arange(len(real_flow_time))
            plt.plot(frames, real_flow_time, 'b-', label='Real')
            plt.fill_between(frames, 
                          real_flow_time - real_flow_std, 
                          real_flow_time + real_flow_std, 
                          color='blue', alpha=0.2)
        
        if len(fake_indices) > 0:
            fake_flow_time = np.mean(flow_magnitude[fake_indices], axis=0)
            fake_flow_std = np.std(flow_magnitude[fake_indices], axis=0)
            frames = np.arange(len(fake_flow_time))
            plt.plot(frames, fake_flow_time, 'r-', label='Fake')
            plt.fill_between(frames, 
                          fake_flow_time - fake_flow_std, 
                          fake_flow_time + fake_flow_std, 
                          color='red', alpha=0.2)
        
        plt.xlabel('Frame Index')
        plt.ylabel('Average Flow Magnitude')
        plt.title('Style Flow Magnitude Over Time')
        plt.legend()
        
        # Save temporal pattern figure
        if save_path:
            temporal_path = save_path.replace('.png', '_temporal.png')
            plt.savefig(temporal_path)
            logger.info(f"Temporal style pattern analysis saved to {temporal_path}")
        
        plt.close()
    
    def run_comprehensive_analysis(self, val_loader, output_dir=None):
        """
        Run comprehensive analysis on validation set
        
        Args:
            val_loader: Validation data loader
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary of analysis results
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Running comprehensive analysis. Results will be saved to {output_dir}")
        
        # Analyze validation set
        results = self.analyze_validation_set(val_loader)
        
        # Visualize confusion matrix
        self.visualize_confusion_matrix(
            results, 
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # Visualize decision boundaries
        self.visualize_decision_boundaries(
            results,
            save_path=os.path.join(output_dir, 'decision_boundaries.png')
        )
        
        # Analyze style temporal patterns
        self.analyze_style_temporal_patterns(
            results,
            save_path=os.path.join(output_dir, 'style_patterns.png')
        )
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        return results
    
    def _generate_summary_report(self, results, output_dir):
        """
        Generate a summary report of the analysis
        
        Args:
            results: Analysis results
            output_dir: Directory to save the report
        """
        # Extract metrics
        metrics = results['metrics']
        
        # Create report
        report = "# Deepfake Detection Model Analysis Report\n\n"
        
        # Performance metrics
        report += "## Performance Metrics\n\n"
        report += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
        report += f"- **Precision**: {metrics['precision']:.4f}\n"
        report += f"- **Recall**: {metrics['recall']:.4f}\n"
        report += f"- **F1 Score**: {metrics['f1']:.4f}\n\n"
        
        # Confusion matrix
        report += "## Confusion Matrix\n\n"
        report += f"- **True Negatives (Real correctly predicted)**: {metrics['tn']}\n"
        report += f"- **False Positives (Real incorrectly predicted as Fake)**: {metrics['fp']}\n"
        report += f"- **False Negatives (Fake incorrectly predicted as Real)**: {metrics['fn']}\n"
        report += f"- **True Positives (Fake correctly predicted)**: {metrics['tp']}\n\n"
        
        # Error analysis
        report += "## Error Analysis\n\n"
        
        # Real videos misclassified as fake
        fp_rate = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        report += f"- **False Positive Rate**: {fp_rate:.4f} ({metrics['fp']} out of {metrics['fp'] + metrics['tn']} real videos misclassified as fake)\n"
        
        # Fake videos misclassified as real
        fn_rate = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0
        report += f"- **False Negative Rate**: {fn_rate:.4f} ({metrics['fn']} out of {metrics['fn'] + metrics['tp']} fake videos misclassified as real)\n\n"
        
        # Visualizations
        report += "## Visualizations\n\n"
        report += "The following visualizations have been generated:\n\n"
        report += "1. **Confusion Matrix**: `confusion_matrix.png`\n"
        report += "2. **Decision Boundaries**: `decision_boundaries.png`\n"
        report += "3. **Style Temporal Patterns**: `style_patterns.png` and `style_patterns_temporal.png`\n"
        report += "4. **Failure Case Analysis**: `failure_cases/` directory\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        # Add recommendations based on analysis
        if fp_rate > fn_rate:
            report += "- The model has a higher false positive rate than false negative rate. Consider adjusting the classification threshold or retraining with more diverse real video samples.\n"
        else:
            report += "- The model has a higher false negative rate than false positive rate. Consider adjusting the classification threshold or retraining with more diverse fake video samples.\n"
        
        report += "- Review the failure cases to identify patterns in misclassifications.\n"
        report += "- Analyze the feature importance visualizations to understand which features are most important for classification.\n"
        
        # Save report
        report_path = os.path.join(output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis report saved to {report_path}")


def main():
    # Assuming pretrained model
    model_path = '/mnt/d/deepfake_detector/trained_models/0424_20h47m28s_CDF64f16b/final_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 64
    batch_size = 32
    real_dir = '/mnt/d/deepfake_detector/dfdc/dfdc_train_part_00/dfdc_train_part_0/'
    fake_dir = '/mnt/d/deepfake_detector/dfdc/dfdc_train_part_00/dfdc_train_part_0/'
    cache_dir = '/mnt/d/deepfake_detector/eval/cacheCDF64'
    metadata_file = '/mnt/d/deepfake_detector/dfdc/dfdc_train_part_00/dfdc_train_part_0/metadata.json'
    analysis_dir = '/mnt/d/deepfake_detector/evaluating-dfdc/CDF64'
    
    os.makedirs(analysis_dir, exist_ok=True)
    
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
        real_dir,
        fake_dir,
        preprocessor,
        cache_dir,
        num_frames=num_frames,
        force_reprocess=False,
        max_videos_per_dir=40,
        metadata_file=metadata_file
    )
    
    # Create dataloaders for training and validation
    logger.info("Creating dataloaders")
    _, val_loader = create_dataloaders(
        video_info,
        batch_size=batch_size,
        train_split=0
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
    
    model = model.to(device)
    
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    analyzer = ModelAnalyzer(model, device=device, output_dir=analysis_dir)
    
    analysis_results = analyzer.run_comprehensive_analysis(val_loader, output_dir=analysis_dir)
    
    # Print analysis summary
    metrics = analysis_results['metrics']
    logger.info(f"Analysis complete. Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    logger.info(f"Analysis results saved to {analysis_dir}")

if __name__ == "__main__":
    main()