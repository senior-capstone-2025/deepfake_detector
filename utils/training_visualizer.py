## file: training_visualizer.py
#
# Training visualizer : Creates real0time and post-training visualization
#
##

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import seaborn as sns
from tensorboardX import SummaryWriter
import time

class TrainingVisualizer:
    def __init__(self, log_dir='visualization_logs'):
        """
        Initialize the training visualizer
        
        Args:
            log_dir: Directory to save logs and visualizations
        """
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        current_time = time.strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize history dictionary
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch': [],
            'lr': []
        }
        
        # Initialize cache for model outputs
        self.current_epoch_preds = []
        self.current_epoch_labels = []
        self.current_epoch_features = []
        
        print(f"Visualizer initialized. TensorBoard logs saved to {self.log_dir}")
        print("To view TensorBoard, run: tensorboard --logdir=visualization_logs")
        
    def update_batch(self, outputs, labels, features=None):
        """
        Update visualizer with batch results for current epoch
        
        Args:
            outputs: Model outputs (predictions) - tensor
            labels: Ground truth labels - tensor
            features: Optional feature representations for dimensionality reduction
        """
        # Convert tensors to numpy and store
        self.current_epoch_preds.append(outputs.detach().cpu().numpy())
        self.current_epoch_labels.append(labels.detach().cpu().numpy())
        
        if features is not None:
            self.current_epoch_features.append(features.detach().cpu().numpy())
    
    def update_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, learning_rate=None):
        """
        Update visualizer with epoch results
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
            learning_rate: Current learning rate
        """
        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        
        if learning_rate is not None:
            self.history['lr'].append(learning_rate)
        
        # Add to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        if learning_rate is not None:
            self.writer.add_scalar('Learning_rate', learning_rate, epoch)
        
        # Reset batch caches if they were used
        if self.current_epoch_preds:
            # Concatenate all batches
            epoch_preds = np.concatenate(self.current_epoch_preds)
            epoch_labels = np.concatenate(self.current_epoch_labels)
            
            # Calculate AUC-ROC
            try:
                auc = roc_auc_score(epoch_labels, epoch_preds)
                self.writer.add_scalar('AUC-ROC/validation', auc, epoch)
                
                # Add ROC curve every 5 epochs
                if epoch % 5 == 0:
                    self._add_roc_curve(epoch_preds, epoch_labels, epoch)
                    self._add_pr_curve(epoch_preds, epoch_labels, epoch)
                    self._add_confusion_matrix(epoch_preds, epoch_labels, epoch)
            except Exception as e:
                print(f"Error calculating ROC-AUC: {e}")
            
            # Dimensionality reduction visualization if features provided
            if self.current_epoch_features and len(self.current_epoch_features) > 0:
                try:
                    from sklearn.decomposition import PCA
                    features = np.concatenate(self.current_epoch_features)
                    self._visualize_features(features, epoch_labels, epoch, method='pca')
                except Exception as e:
                    print(f"Error in feature visualization: {e}")
            
            # Reset cache
            self.current_epoch_preds = []
            self.current_epoch_labels = []
            self.current_epoch_features = []
    
    def _add_roc_curve(self, preds, labels, epoch):
        """Add ROC curve to TensorBoard"""
        fpr, tpr, _ = roc_curve(labels, preds)
        # Create figure
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(labels, preds):.4f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Epoch {epoch}')
        ax.legend()
        
        self.writer.add_figure('ROC Curve', fig, epoch)
    
    def _add_pr_curve(self, preds, labels, epoch):
        """Add Precision-Recall curve to TensorBoard"""
        precision, recall, _ = precision_recall_curve(labels, preds)
        
        # Create figure
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - Epoch {epoch}')
        
        self.writer.add_figure('PR Curve', fig, epoch)
    
    def _add_confusion_matrix(self, preds, labels, epoch, threshold=0.5):
        """Add confusion matrix to TensorBoard"""
        # Apply threshold to predictions
        binary_preds = (preds > threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, binary_preds)
        
        # Create figure
        fig, ax = plt.figure(figsize=(8, 8)), plt.gca()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Real', 'Fake'])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Real', 'Fake'])
        
        self.writer.add_figure('Confusion Matrix', fig, epoch)
    
    def _visualize_features(self, features, labels, epoch, method='pca'):
        """Visualize feature space using dimensionality reduction"""
        if method == 'pca':
            from sklearn.decomposition import PCA
            # Reshape features if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(features)
            
            # Create figure
            fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
            scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                      c=labels, cmap='coolwarm', alpha=0.6)
            ax.set_title(f'PCA Feature Visualization - Epoch {epoch}')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            plt.colorbar(scatter, ax=ax, label='Class')
            
            self.writer.add_figure('PCA Features', fig, epoch)
        
        elif method == 'tsne':
            # T-SNE is more expensive so we use it less frequently
            from sklearn.manifold import TSNE
            
            # Reshape features if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Apply t-SNE (can be slow for large datasets)
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
            # Create figure
            fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
            scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                      c=labels, cmap='coolwarm', alpha=0.6)
            ax.set_title(f't-SNE Feature Visualization - Epoch {epoch}')
            plt.colorbar(scatter, ax=ax, label='Class')
            
            self.writer.add_figure('t-SNE Features', fig, epoch)
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and accuracy curves)
        
        Args:
            save_path: Path to save the plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot loss
        ax1.plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['epoch'], self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(self.history['epoch'], self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        # Return figure
        return fig
    
    def generate_epoch_report(self, epoch, predictions, true_labels):
        """
        Generate a comprehensive report for the epoch
        
        Args:
            epoch: Current epoch number
            predictions: Model predictions
            true_labels: Ground truth labels
        """
        # Apply threshold to get binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(true_labels, binary_preds)
        precision = precision_score(true_labels, binary_preds)
        recall = recall_score(true_labels, binary_preds)
        f1 = f1_score(true_labels, binary_preds)
        auc = roc_auc_score(true_labels, predictions)
        
        # Create a markdown report
        report = f"# Epoch {epoch} Evaluation Report\n\n"
        report += "## Performance Metrics\n\n"
        report += f"- **Accuracy**: {accuracy:.4f}\n"
        report += f"- **Precision**: {precision:.4f}\n"
        report += f"- **Recall**: {recall:.4f}\n"
        report += f"- **F1 Score**: {f1:.4f}\n"
        report += f"- **AUC-ROC**: {auc:.4f}\n\n"
        
        # Add more sections as needed
        
        # Save report to a file
        report_path = os.path.join(self.log_dir, f'epoch_{epoch}_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Epoch report saved to {report_path}")
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
