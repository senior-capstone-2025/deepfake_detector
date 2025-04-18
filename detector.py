import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import logging
# Create logger
logger = logging.getLogger(__name__)

class StyleGRU(nn.Module):
    """
    Simplified StyleGRU module to process style latent flows
    """
    def __init__(self, input_size=512, hidden_size=1024, num_layers=1, bidirectional=True):
        super(StyleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.dropout(x)
        output, hidden = self.gru(x)
        
        # Get the final hidden state
        if self.num_directions == 2:
            # For bidirectional, concatenate the last hidden states from both directions
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # For unidirectional, just take the last hidden state
            hidden = hidden[-1]
            
        return hidden


class StyleAttention(nn.Module):
    def __init__(self, style_dim, content_dim, output_dim=512):
        super(StyleAttention, self).__init__()
        
        self.query_proj = nn.Linear(content_dim, output_dim)
        self.key_proj = nn.Linear(style_dim, output_dim)
        self.value_proj = nn.Linear(style_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, style_features, content_features):
        # Project features
        query = self.query_proj(content_features)  # [batch_size, output_dim]
        key = self.key_proj(style_features)        # [batch_size, output_dim]
        value = self.value_proj(style_features)    # [batch_size, output_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.t())  # [batch_size, batch_size]
        attn_scores = F.softmax(attn_scores, dim=1)  # Normalize across batch
        
        # Apply attention
        weighted_value = torch.matmul(attn_scores, value)
        
        # Project to output dimension 
        output = self.output_proj(weighted_value)
        
        return output

class DeepfakeDetector(nn.Module):
    """
    Simplified Deepfake Detection model based on style latent flows
    """
    def __init__(self, 
                psp_model,                # Pretrained pSp encoder
                content_model,            # 3D ResNet for content features
                style_dim=512,            # StyleGAN latent dimension
                content_dim=2048,         # Content feature dimension from ResNet
                gru_hidden_size=1024,     # GRU hidden dimension
                output_dim=512):          # Output dimension for attention
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained models
        self.psp_encoder = psp_model
        self.content_model = content_model
        
        # StyleGRU module
        self.style_gru = StyleGRU(
            input_size=style_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            bidirectional=True
        )
        
        # Style Attention Module
        self.style_attention = StyleAttention(
            style_dim=gru_hidden_size * 2,  # Bidirectional GRU
            content_dim=content_dim,
            output_dim=output_dim
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(output_dim + content_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # For faster inference, set models to evaluation mode
        self.psp_encoder.eval()
        self.content_model.eval()

        self.print_resnet_structure()

    def preprocess_video(self, video_tensor):
        """
        Apply the necessary preprocessing for the PyTorchVideo model
        """
        # Expected input: video_tensor with shape [batch_size, C, T, H, W]
        
        # Normalize using the correct mean and std for Kinetics dataset
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)
        
        # Normalize to [0, 1] first if not already done
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        
        # Apply normalization
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor
    
    def print_resnet_structure(self):
        """Print the structure of the 3D ResNet model"""
        print("\nRESNET MODEL STRUCTURE:")
        for name, module in self.content_model.named_children():
            print(f"- {name}: {type(module)}")
        
        # Try a simple forward pass with a random tensor
        test_input = torch.randn(1, 3, 32, 224, 224)  # Same shape as your videos
        with torch.no_grad():
            try:
                output = self.content_model(test_input)
                print(f"Test output shape: {output.shape}")
            except Exception as e:
                print(f"Error in forward pass: {e}")
    def compute_style_flow(self, face_frames):
        """
        Extract style latent vectors and compute style flow using only one latent level
        """
        batch_size, seq_len, c, h, w = face_frames.shape
        
        # Process each frame individually
        latent_codes = []
        
        for i in range(seq_len):
            # Extract a single frame from each batch
            frame = face_frames[:, i, :, :, :]  # [batch_size, c, h, w]
            
            # Extract style latent vector using only the encoder
            with torch.no_grad():
                # This gives us shape [batch_size, 18, 512]
                codes = self.psp_encoder.encoder(frame)
                
                # Use only one latent level (e.g., level 9 which is in the middle)
                # This gives us a 512-dimensional vector per frame
                level_idx = 9  # You can experiment with different levels
                codes = codes[:, level_idx, :]  # Now shape is [batch_size, 512]
                
                latent_codes.append(codes)
        
        # Stack latent codes along sequence dimension
        # Shape becomes [batch_size, seq_len, 512]
        latent_codes = torch.stack(latent_codes, dim=1)
        print(f"Latent codes shape: {latent_codes.shape}")
        
        # Compute style flow (differences between consecutive frames)
        if seq_len > 1:
            style_flow = latent_codes[:, 1:] - latent_codes[:, :-1]
        else:
            # Handle single frame case
            style_flow = torch.zeros((batch_size, 1, latent_codes.size(-1)), device=latent_codes.device)
        
        return style_flow
    
    def extract_content_features(self, video_frames):
        """
        Extract content features using the 3D ResNet model from PyTorchVideo
        """
        # Preprocess video frames
        video_frames = self.preprocess_video(video_frames)
        
        with torch.no_grad():
            # Get features from the model
            features = self.content_model(video_frames)
            
            # Print the feature shape
            logger.debug(f"Raw features shape: {features.shape}")
            
            # Apply global average pooling if needed
            if len(features.shape) > 2:
                # If we have spatial dimensions remaining, apply pooling
                # For SlowFast models, this might be [batch, channels, time, h, w]
                dims_to_pool = list(range(2, len(features.shape)))
                features = torch.mean(features, dim=dims_to_pool)
            
            logger.debug(f"Content features shape: {features.shape}")
        
        return features
    
    def forward(self, video_frames, face_frames):
        """
        Forward pass of the deepfake detection model
        
        Args:
            video_frames: Original video frames [batch_size, channels, time, height, width]
            face_frames: Aligned face frames [batch_size, time, channels, height, width]
        """

        # Check shapes
        print(f"Video frames shape: {video_frames.shape}")
        print(f"Face frames shape: {face_frames.shape}")

        try:

            # Extract content features
            content_features = self.extract_content_features(video_frames)
            logger.debug(f"Content features shape: {content_features.shape}")
            
            # Compute style flow
            style_flow = self.compute_style_flow(face_frames)
            logger.debug(f"Style flow shape: {style_flow.shape}")
            
            # Process style flow with StyleGRU
            style_features = self.style_gru(style_flow)
            logger.debug(f"Style features shape: {style_features.shape}")
            
            # Apply style attention
            attended_features = self.style_attention(style_features, content_features)
            logger.debug(f"Attended features shape: {attended_features.shape}")
            
            # Concatenate with content features for final classification
            combined_features = torch.cat([attended_features, content_features], dim=1)
            logger.debug(f"Combined features shape: {combined_features.shape}")
            
            # Final classification
            output = self.classifier(combined_features)
            logger.debug(f"Output shape: {output.shape}")
            
            return output
        
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise e
