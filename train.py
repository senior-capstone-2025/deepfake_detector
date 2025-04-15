import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    """Train the deepfake detection model"""
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            if batch is None:
                continue
                
            video_batch, face_batch, labels = batch
            
            # Move to device
            video_batch = video_batch.to(device)
            face_batch = face_batch.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(video_batch, face_batch)
            outputs = outputs.squeeze(1)  # Remove extra dimension if needed
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * video_batch.size(0)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                if batch is None:
                    continue
                    
                video_batch, face_batch, labels = batch
                
                # Move to device
                video_batch = video_batch.to(device)
                face_batch = face_batch.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(video_batch, face_batch)
                outputs = outputs.squeeze(1)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * video_batch.size(0)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Model saved with validation loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model