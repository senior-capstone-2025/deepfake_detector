#!/bin/sh

# Set up conda environment and install dependencies
conda env create -f deepfake_detector_env.yml
conda activate deepfake_detector

# Clone the pSp respository
git clone https://github.com/eladrich/pixel2style2pixel.git

# Get the pretrained pSp encoder weights
cd pixel2style2pixel
mkdir pretrained_models
cd pretrained_models
# Install gdown if not already installed
pip install gdown
# Download the pretrained weights
gdown 1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0

cd ../..
mkdir models
mkdir 3d_resnet

# Get 3D ResNet
python3 save_3d_resnet.py
