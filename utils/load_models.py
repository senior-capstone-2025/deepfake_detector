import torch
from argparse import Namespace
import sys
import os
import torchvision.transforms as transforms
import pytorchvideo.models.resnet as resnet

# Import pSp model
from pixel2style2pixel.models.psp import pSp

def load_psp_encoder(model_path):
    """
    Load the pretrained pSp encoder model
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
    
     # Force CPU mode regardless of what's in the checkpoint
    opts['device'] = 'cpu'

    # Convert to Namespace
    opts = Namespace(**opts)
    
    # Initialize the pSp model
    net = pSp(opts)
    net.eval()  # Set to evaluation mode
    
    # If GPU is available, move to GPU
    if torch.cuda.is_available():
        net = net.cuda()
    
    print('pSp model successfully loaded!')
    
    return net

def load_resnet_model(model_path):
    """
    Load the pretrained ResNet model
    """
    # Initialize the ResNet model first
    net = resnet.create_resnet(
        input_channel=3,
        model_depth=50,
        model_num_class=400,  # Kinetics classes, will be removed
        norm=torch.nn.BatchNorm3d,
        activation=torch.nn.ReLU,
    )
    
    # Remove the final classification layer if needed (similar to save_3d_resnet.py)
    net = torch.nn.Sequential(*list(net.children())[:-1])
    
    # Load the saved state_dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # If you saved just the state_dict (as in save_3d_resnet.py)
    if isinstance(state_dict, dict) and 'state_dict' not in state_dict:
        net.load_state_dict(state_dict)
    # If you saved a checkpoint with 'state_dict' key (common format)
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        net.load_state_dict(state_dict['state_dict'])
    
    net.eval()  # Set to evaluation mode
    
    # If GPU is available, move to GPU
    if torch.cuda.is_available():
        net = net.cuda()
    
    print('ResNet model successfully loaded!')
    
    return net