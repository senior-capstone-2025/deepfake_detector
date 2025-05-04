## file: load_models.py
#
# Utility functions to load pretrained models.
#
##

import torch
from argparse import Namespace
import torchvision.transforms as transforms
import pytorchvideo.models.resnet as resnet
import logging

logger = logging.getLogger(__name__)

# Import pSp model
from pixel2style2pixel.models.psp import pSp

def load_psp_encoder(model_path, device):
    """
    Load the pretrained pSp encoder model
    
    Args:
        model_path (str): Path to the pretrained pSp model checkpoint.
        device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
        net (torch.nn.Module): The loaded pSp model.
    """
    
    # Load the checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Get the options used for training
    opts = ckpt['opts']

    # Update the options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    
    # Set the device
    opts['device'] = device

    # Convert to Namespace
    opts = Namespace(**opts)
    
    # Initialize the pSp model
    net = pSp(opts)
    net.eval()  # Set to evaluation mode
    
    logger.info("Pretrained pSp model loaded successfully.")
    
    return net

def load_resnet_module():
    """
    Load the pretrained 3D ResNet model from PyTorchVideo
    """
    
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        class FeatureExtractor(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Store the original head for shapes
                self.original_head = model.blocks[-1].proj
                # Remove the final projection layer
                self.model.blocks[-1].proj = torch.nn.Identity()
            
            def forward(self, x):
                return self.model(x)
        
        feature_extractor = FeatureExtractor(model)
        feature_extractor.eval()
        
        logger.info("Pretrained 3D ResNet model loaded successfully.")
        return feature_extractor
        
    except Exception as e:
        logger.warning(f"Error loading PyTorchVideo model: {e}")
        logger.warning("Falling back to 2D ResNet.")
        
        # Fallback to 2D ResNet
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Remove classification head
        model = torch.nn.Sequential(*list(resnet.children())[:-1])
        model.eval()
        
        logger.info("Pretrained 2D ResNet model loaded successfully.")
        return model
    