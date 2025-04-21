## file: detector.py
#
# DeepfakeDetector :
# Core model implementation. 
# Extracts and merges style and content features.
# Uses simple classification head for final prediction.
#
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

# Setup logging
import logging
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
                device,
                psp_model,                # Pretrained pSp encoder
                content_model,            # 3D ResNet for content features
                style_dim=512,            # StyleGAN latent dimension
                content_dim=2048,         # Content feature dimension from ResNet
                gru_hidden_size=1024,     # GRU hidden dimension
                output_dim=512):          # Output dimension for attention
        super(DeepfakeDetector, self).__init__()
        
        self.device = device
        
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
    
    
    def forward(self, content_features, style_codes):
        """
        Forward pass of the deepfake detection model
        
        Args:
            video_frames: Original video frames [batch_size, channels, time, height, width]
            face_frames: Aligned face frames [batch_size, time, channels, height, width]
        """

        # Check shapes
        logger.debug(f"Video frames shape: {content_features.shape}")
        logger.debug(f"Style codes shape: {style_codes.shape}")

        try:

            # We have pre-computed style codes
            logger.debug(f"Using pre-computed style codes: {style_codes.shape}")
            # Compute style flow from pre-computed codes
            batch_size, seq_len, dim = style_codes.shape
            if seq_len > 1:
                style_flow = style_codes[:, 1:] - style_codes[:, :-1]
            else:
                # Handle single frame case
                style_flow = torch.zeros((batch_size, 1, dim), device=self.device)
        
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
