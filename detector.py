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

# Enhance the StyleGRU module to capture more temporal dynamics
class StyleGRU(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024, num_layers=2, bidirectional=True):
        super(StyleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Add temporal convolution before GRU
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.dropout = nn.Dropout(0.3)  # Increased dropout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,  # Increased layers
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Apply temporal convolution
        x_t = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        x_t = self.temporal_conv(x_t)
        x = x_t.transpose(1, 2)  # [batch_size, seq_len, input_size]
        
        x = self.dropout(x)
        output, hidden = self.gru(x)
        
        # Return both final hidden state and sequence output
        if self.num_directions == 2:
            # For bidirectional, concatenate the last hidden states from both directions
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # For unidirectional, just take the last hidden state
            final_hidden = hidden[-1]
            
        return final_hidden, output

class MultiHeadStyleAttention(nn.Module):
    def __init__(self, style_dim, content_dim, output_dim=512, num_heads=2):
        super(MultiHeadStyleAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Content dimension projection to match output_dim
        self.content_proj = nn.Linear(content_dim, output_dim)
        
        # Multi-head projections
        self.query_proj = nn.Linear(output_dim, output_dim)  # Query from projected content
        self.key_proj = nn.Linear(style_dim, output_dim)
        self.value_proj = nn.Linear(style_dim, output_dim)
        
        # Output projection and layer norm
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, style_features, content_features, style_seq=None):
        batch_size = content_features.shape[0]
        
        # Project content features to match output_dim
        content_proj = self.content_proj(content_features)
        
        # Project features
        queries = self.query_proj(content_proj).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(style_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(style_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, values).transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        attended = self.output_proj(attended).squeeze(1)
        
        # Residual connection and layer norm (now dimensions match)
        attended = self.layer_norm1(attended + content_proj)
        
        # Feed forward network
        output = self.layer_norm2(attended + self.ffn(attended))
        
        return output
        
class DeepfakeDetector(nn.Module):
    def __init__(self, 
                device,
                style_dim=512,
                content_dim=2048,
                gru_hidden_size=1024,
                output_dim=512,
                num_attn_heads=2):
        super(DeepfakeDetector, self).__init__()
        
        self.device = device
        
        # Improved StyleGRU module
        self.style_gru = StyleGRU(
            input_size=style_dim,
            hidden_size=gru_hidden_size,
            num_layers=2,
            bidirectional=True
        )
        
        # Temporal pooling for content features
        self.content_temporal_pool = nn.Sequential(
            nn.Conv1d(content_dim, content_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Multi-Head Style Attention
        self.style_attention = MultiHeadStyleAttention(
            style_dim=gru_hidden_size * 2,  # Bidirectional GRU
            content_dim=content_dim,
            output_dim=output_dim,
            num_heads=num_attn_heads
        )
        
        # Additional feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim + content_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Simple classification head
        self.classifier = nn.Sequential(
           nn.Linear(output_dim, 128),
           nn.LayerNorm(128),
           nn.ReLU(),
           nn.Dropout(0.4),
           nn.Linear(128, 1),            
           nn.Sigmoid()
        )
    
    def forward(self, content_features, style_codes):
        logger.debug(f"Content features shape: {content_features.shape}")
        logger.debug(f"Style codes shape: {style_codes.shape}")

        try:
            # Compute style flow from pre-computed codes
            batch_size, seq_len, dim = style_codes.shape
            if seq_len > 1:
                style_flow = style_codes[:, 1:] - style_codes[:, :-1]
            else:
                # Handle single frame case
                style_flow = torch.zeros((batch_size, 1, dim), device=self.device)
        
            logger.debug(f"Style flow shape: {style_flow.shape}")
            
            # Process style flow with improved StyleGRU
            style_hidden, style_seq = self.style_gru(style_flow)
            logger.debug(f"Style hidden shape: {style_hidden.shape}")
            logger.debug(f"Style sequence shape: {style_seq.shape}")
            
            # Apply multi-head style attention
            attended_features = self.style_attention(style_hidden, content_features, style_seq)
            logger.debug(f"Attended features shape: {attended_features.shape}")
            
            # Concatenate with content features and apply fusion
            combined_features = torch.cat([attended_features, content_features], dim=1)
            fused_features = self.fusion_layer(combined_features)
            logger.debug(f"Fused features shape: {fused_features.shape}")
            
            # Final classification
            output = self.classifier(fused_features)
            logger.debug(f"Output shape: {output.shape}")
            
            return output
        
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise e