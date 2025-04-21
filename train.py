## file: train.py
#
# Main training script.
#
##

import torch
from tqdm import tqdm
import os
import time

import logging
# Create logger
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.001, checkpoint_dir='checkpoints'):
    """Train the deepfake detection model"""
    
    # Log training start information
    logger.info(f"Begin training model on device: {device}")
    logger.info(f"Number of epochs: {num_epochs}, Learning rate: {lr}")
    logger.info(f"Training data: {len(train_loader)} batches")
    logger.info(f"Validation data: {len(val_loader)} batches")
    
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()

    # Move model to device
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    logger.info("Starting training loop")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training phase started")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            
            if batch is None:
                logger.warning(f"Skipping empty batch at index {batch_idx}")
                continue
                
            content_features_batch, style_codes_batch, labels = batch
            
            # Log batch shapes occasionally (every 50 batches)
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}: video shape={content_features_batch.shape}, labels shape={labels.shape}")

            # Move to device
            content_features_batch = content_features_batch.to(device)
            style_codes_batch = style_codes_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(content_features_batch, style_codes_batch)
            
            outputs = outputs.squeeze(1)
            
            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * content_features_batch.size(0)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Log batch loss occasionally
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training completed: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
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
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation completed: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_epoch': best_epoch
            }, best_model_path)
            logger.info(f"New best model saved at {best_model_path} with validation loss: {val_loss:.4f}")
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation loss: {best_checkpoint['val_loss']:.4f}")
    else:
        logger.warning(f"Best model path {best_model_path} not found. Using final model.")
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Return the best model and its metadata
    return model, {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time': total_time
    }