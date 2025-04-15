import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import sys
import os
import cv2


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
    """
    Simplified Style Attention Module to integrate style and content features
    """
    def __init__(self, style_dim, content_dim, output_dim=512):
        super(StyleAttention, self).__init__()
        self.query_proj = nn.Linear(content_dim, output_dim)
        self.key_proj = nn.Linear(style_dim, output_dim)
        self.value_proj = nn.Linear(style_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, style_feature, content_feature):
        # Project content features to query
        query = self.query_proj(content_feature)  # [batch_size, output_dim]
        # Project style features to key and value
        key = self.key_proj(style_feature)        # [batch_size, output_dim]
        value = self.value_proj(style_feature)    # [batch_size, output_dim]
        
        # Reshape for attention
        query = query.unsqueeze(1)  # [batch_size, 1, output_dim]
        key = key.unsqueeze(2)      # [batch_size, output_dim, 1]
        
        # Compute attention scores
        attn_scores = torch.bmm(query, key)  # [batch_size, 1, 1]
        attn_scores = F.softmax(attn_scores, dim=2)
        
        # Apply attention
        value = value.unsqueeze(1)  # [batch_size, 1, output_dim]
        output = torch.bmm(attn_scores, value.transpose(1, 2))  # [batch_size, 1, output_dim]
        output = output.squeeze(1)  # [batch_size, output_dim]
        
        # Final projection
        output = self.output_proj(output)
        
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
    
    def compute_style_flow(self, face_frames):
        """
        Extract style latent vectors and compute style flow
        """
        batch_size, seq_len, c, h, w = face_frames.shape
        
        # Reshape for pSp encoder
        frames_flat = face_frames.view(-1, c, h, w)
        
        # Extract style latent vectors (W+ space)
        with torch.no_grad():
            latent_codes = self.psp_encoder(frames_flat)

        # Get the actual dimensions from the output
        latent_dim = latent_codes.shape[-1]
        
        # Reshape back to sequence
        latent_codes = latent_codes.view(batch_size, seq_len, latent_dim)
        
        # Compute style flow (differences between consecutive frames)
        if seq_len > 1:
            style_flow = latent_codes[:, 1:] - latent_codes[:, :-1]
        else:
            # Handle single frame case
            style_flow = torch.zeros((batch_size, 1, latent_dim), device=latent_codes.device)
        
        
        return style_flow
    
    def extract_content_features(self, video_frames):
        """
        Extract content features using 3D ResNet
        """
        with torch.no_grad():
            content_features = self.content_model(video_frames)
        return content_features
    
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
            print(f"Content features shape: {content_features.shape}")
            
            # Compute style flow
            style_flow = self.compute_style_flow(face_frames)
            print(f"Style flow shape: {style_flow.shape}")
            
            # Process style flow with StyleGRU
            style_features = self.style_gru(style_flow)
            print(f"Style features shape: {style_features.shape}")
            
            # Apply style attention
            attended_features = self.style_attention(style_features, content_features)
            print(f"Attended features shape: {attended_features.shape}")
            
            # Concatenate with content features for final classification
            combined_features = torch.cat([attended_features, content_features], dim=1)
            print(f"Combined features shape: {combined_features.shape}")
            
            # Final classification
            output = self.classifier(combined_features)
            print(f"Output shape: {output.shape}")
            
            return output
        
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise e
