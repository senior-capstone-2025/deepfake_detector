## file: improved_train.py
#
# Improved training script with visualization and regularization
#
##

import torch
from tqdm import tqdm
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import logging
import matplotlib.pyplot as plt
from utils.training_visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)

def mixup_data(x1, x2, y, alpha=0.2, device='cuda'):
    """
    Applies mixup augmentation to content and style features
    
    Args:
        x1: Content features
        x2: Style features
        y: Labels
        alpha: Mixup alpha parameter,
        device: Device to perform operations on
        
    Returns:
        Mixed content features, mixed style features, mixed labels
    """
    
    # Sample interpolation coefficient from beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Applies mixup criterion
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a, y_b: Mixed labels
        lam: Mixup lambda value
        
    Returns:
        Mixed loss
    """
    
    # Calculate the loss using the mixup formula
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(
    model, 
    train_loader, 
    val_loader, 
    device, 
    num_epochs=100, 
    lr=0.001, 
    checkpoint_dir='checkpoints',
    weight_decay=1e-4,
    use_mixup=True,
    mixup_alpha=0.4,
    use_cosine_scheduler=True,
    early_stopping_patience=15,
    visualizer_dir='visualization_logs'
):
    """
    Train the deepfake detection model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (e.g., 'cuda' or 'cpu')
        num_epochs: Number of epochs to train
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        weight_decay: Weight decay for optimizer
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Alpha parameter for mixup
        use_cosine_scheduler: Whether to use cosine annealing scheduler
        early_stopping_patience: Patience for early stopping
        visualizer_dir: Directory for visualization logs
        
    Returns:
        model: Trained model
    """
    
    # Log training start information
    logger.info(f"Begin training model on device: {device}")
    logger.info(f"Number of epochs: {num_epochs}, Learning rate: {lr}, Weight decay: {weight_decay}")
    logger.info(f"Using mixup: {use_mixup}, Mixup alpha: {mixup_alpha}")
    logger.info(f"Using cosine scheduler: {use_cosine_scheduler}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    logger.info(f"Training data: {len(train_loader)} batches")
    logger.info(f"Validation data: {len(val_loader)} batches")
    
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()

    # Initialize visualizer
    visualizer = TrainingVisualizer(log_dir=visualizer_dir)

    # Move model to device
    model = model.to(device)
    
    # Create optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCELoss()
    
    # Learning rate scheduler
    if use_cosine_scheduler:
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=lr/100
        )
        scheduler_type = "cosine"
    else:
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        scheduler_type = "plateau"
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    logger.info("Starting training loop")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training phase started")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            
            if batch is None:
                logger.warning(f"Skipping empty batch at index {batch_idx}")
                continue
                
            content_features_batch, style_codes_batch, labels = batch
            
            # Log batch shapes occasionally (every 50 batches)
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}: content_features shape={content_features_batch.shape}, style_codes shape={style_codes_batch.shape}, labels shape={labels.shape}")

            # Move to device
            content_features_batch = content_features_batch.to(device)
            style_codes_batch = style_codes_batch.to(device)
            labels = labels.to(device)
            
            # Apply mixup if enabled (70% batches mixed, 30% not)
            if use_mixup and np.random.random() < 0.7:
                content_features_batch, style_codes_batch, labels_a, labels_b, lam = mixup_data(
                    content_features_batch, style_codes_batch, labels, mixup_alpha, device
                )
                mixup_applied = True
            else:
                mixup_applied = False
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(content_features_batch, style_codes_batch)
            outputs = outputs.squeeze(1)
            
            # Compute loss with mixup if applied
            if mixup_applied:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * content_features_batch.size(0)
            
            if mixup_applied:
                # For accuracy during mixup, use the dominant label
                mixed_labels = lam * labels_a + (1 - lam) * labels_b
                predictions = (outputs > 0.5).float()
                train_correct += ((predictions > 0.5) == (mixed_labels > 0.5)).sum().item()
            else:
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                
            train_total += labels.size(0)
            
            # Compute style hidden state for visualization
            with torch.no_grad():
                # Calculate style flow first
                style_flow = style_codes_batch[:, 1:] - style_codes_batch[:, :-1] if style_codes_batch.shape[1] > 1 else torch.zeros((style_codes_batch.shape[0], 1, style_codes_batch.shape[2]), device=device)
                
                # Process through StyleGRU to get hidden state
                style_hidden, _ = model.style_gru(style_flow)
                
                # Extract features for visualization
                fused_features = model.fusion_layer(torch.cat([
                    model.style_attention(style_hidden, content_features_batch),
                    content_features_batch
                ], dim=1))

            visualizer.update_batch(outputs, labels, features=fused_features)

            # Log batch loss occasionally
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
                
            # Update learning rate for cosine scheduler
            if scheduler_type == "cosine":
                scheduler.step(epoch + batch_idx / len(train_loader))
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total if train_total > 0 else float('inf')
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Log training results
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training completed in {epoch_time:.2f}s: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_labels = []
        all_features = []
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation phase started")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')):
                if batch is None:
                    logger.warning(f"Skipping empty validation batch at index {batch_idx}")
                    continue
                    
                content_features_batch, style_codes_batch, labels = batch
                
                # Move to device
                content_features_batch = content_features_batch.to(device)
                style_codes_batch = style_codes_batch.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(content_features_batch, style_codes_batch)
                outputs = outputs.squeeze(1)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * content_features_batch.size(0)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Store predictions and labels for ROC curve
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Extract features for visualization
                with torch.no_grad():
                    style_hidden, _ = model.style_gru(
                        style_codes_batch[:, 1:] - style_codes_batch[:, :-1] 
                        if style_codes_batch.shape[1] > 1 
                        else torch.zeros((style_codes_batch.shape[0], 1, style_codes_batch.shape[2]), device=device)
                    )
                    fused_features = model.fusion_layer(torch.cat([
                        model.style_attention(style_hidden, content_features_batch),
                        content_features_batch
                    ], dim=1))
                    all_features.extend(fused_features.cpu().numpy())
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate for plateau scheduler
        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update visualizer
        visualizer.update_epoch(
            epoch + 1,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            current_lr
        )
        
        # Print statistics
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation completed: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
        
        # Generate detailed epoch report every 5 epochs
        if (epoch + 1) % 5 == 0 and all_outputs and all_labels:
            visualizer.generate_epoch_report(epoch + 1, np.array(all_outputs), np.array(all_labels))
        
        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_epoch': best_epoch
            }, best_model_path)
            logger.info(f"New best model saved at {best_model_path} with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # After training, plot training history
    history_plot = visualizer.plot_training_history(save_path=os.path.join(checkpoint_dir, 'training_history.png'))
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation loss: {best_checkpoint['val_loss']:.4f}")
    else:
        logger.warning(f"Best model path {best_model_path} not found. Using final model.")
    
    # Close the visualizer
    visualizer.close()
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Return the best model and its metadata
    return model, {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time': total_time
    }
