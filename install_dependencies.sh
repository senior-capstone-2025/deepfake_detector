#!/bin/sh

# This script install the dependencies for the deepfake detector project.

# It is recommended to run this script in a virtual environment to avoid conflicts with other projects.

# Conda:
# conda env create -f deepfake_detector_env.yml
# conda activate deepfake_detector

# Pip:
python3 -m venv deepfake_detector_env
source deepfake_detector_env/bin/activate

pip install -r requirements.txt

# Clone the pSp respository
git clone https://github.com/eladrich/pixel2style2pixel.git

# Sometimes psp needs this... I think...
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

# Get the pretrained pSp encoder weights
cd pixel2style2pixel
mkdir pretrained_models
cd pretrained_models
# Install gdown if not already installed
pip install gdown
# Download the pretrained weights
gdown 1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0
