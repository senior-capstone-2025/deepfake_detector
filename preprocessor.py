## file: preprocessor.py
#
# Preprocessor class for extracting frames and faces from videos.
#
##

import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from facenet_pytorch import MTCNN
import logging
# Add the pixel2style2pixel directory to the Python path
pixel2style2pixel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixel2style2pixel')
sys.path.append(pixel2style2pixel_path)

logger = logging.getLogger(__name__)

# Load the PSP encoder and 3D ResNet model
from utils.load_models import load_psp_encoder, load_resnet_module

class DeepfakePreprocessor:
    """
    Preprocessor for extracting frames and faces from videos for deepfake detection
    """
    def __init__(self, 
                 face_size=(256, 256), 
                 video_size=(224, 224), 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 psp_path="pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"):
        self.face_size = face_size
        self.video_size = video_size
        self.device = device
        
        # Load psp encoder model and 3D ResNet
        self.psp_model = load_psp_encoder(psp_path, device).to(device)
        self.content_model = load_resnet_module().to(device)
        
        # Initialize face detector (MTCNN)
        self.face_detector = MTCNN(
            image_size=face_size[0],
            margin=40,
            device=device,
            keep_all=False
        )
        
        # Use pSp model normalization for face images
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Use Kinetics dataset normalization for video frames
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

    def _extract_face(self, frame):
        """
        Extract face from a single frame using MTCNN
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            face_img: Extracted face image (numpy array)
        """
        
        # Convert to RGB if needed
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3 and frame.dtype == np.uint8:
            # Check if already RGB or BGR
            if cv2.COLOR_BGR2RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect face
            face = self.face_detector(frame)
            
            # If face detection fails, use center crop
            if face is None:
                logger.debug("No face detected, using center crop")
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                face_img = frame[top:top + min_dim, left:left + min_dim]
                face_img = cv2.resize(face_img, self.face_size)
            # If face is detected, resize and normalize
            else:
                if isinstance(face, torch.Tensor):
                    # MTCNN normalized tensor
                    face_img = face.permute(1, 2, 0).cpu().numpy()
                    # Denormalize tensor
                    face_img = (face_img * 255).astype(np.uint8)
                else:
                    face_img = face
            
            return face_img
            
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            # Fall back to center crop
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            face_img = frame[top:top + min_dim, left:left + min_dim]
            face_img = cv2.resize(face_img, self.face_size)
            
            return face_img
    
    def extract_style_codes(self, face_tensor):
        """
        Extract style latent vectors from face frames
        
        Args:
            face_tensor: Face frames tensor [T, C, H, W]
            
        Returns:
            Style codes tensor [T, 512]
        """
        
        seq_len = face_tensor.shape[0]
        style_codes = []
        
        with torch.no_grad():
            # Loop through each frame
            for i in range(seq_len):
                # Extract a single frame [1, C, H, W]
                frame = face_tensor[i].unsqueeze(0).to(self.device)
                
                # Extract style latent vector using only the encoder
                codes = self.psp_model.encoder(frame)
                
                # Use only one latent level (level 9 which is in the middle)
                level_idx = 9
                # [1, 512]
                codes = codes[:, level_idx, :]
                
                style_codes.append(codes.cpu().squeeze(0))
        
        # Stack style codes along sequence dimension
        if style_codes:
            return torch.stack(style_codes, dim=0)
        else:
            logger.warning("No style codes extracted, returning None")
            return None
        
    def extract_content_features(self, video_tensor):
        """
        Extract content features using the 3D ResNet model from PyTorchVideo
        
        Args:
            video_tensor: Video tensor [T, C, H, W]
            
        Returns:
            Content features tensor [C, T, H, W]
        """
        
        with torch.no_grad():
            # Add batch dimension if it's missing
            if len(video_tensor.shape) == 4:
                video_tensor = video_tensor.unsqueeze(0)
            
            # Move tensor to device
            video_tensor = video_tensor.to(self.device)
            
            # Get features from the model
            content_features = self.content_model(video_tensor)
            
            logger.debug(f"Raw features shape: {content_features.shape}")
            
            # Apply global average pooling if spatial dimensions remaining
            if len(content_features.shape) > 2:
                dims_to_pool = list(range(2, len(content_features.shape)))
                content_features = torch.mean(content_features, dim=dims_to_pool)
            
            # Remove batch dimension and move back to CPU
            content_features = content_features.squeeze(0).cpu()
            
            logger.debug(f"Content features shape: {content_features.shape}")
        
        return content_features

    def process_video(self, video_path, num_frames=32):
        """
        Process video file to extract frames and faces
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract from the video (default: 32)
        
        Returns:
            content_features: Extracted content features tensor
            style_codes: Extracted style codes tensor    
        """
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None, None
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Error opening video file: {video_path}")
            return None, None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.debug(f"Video: {os.path.basename(video_path)}, {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Calculate frame indices to sample
        if total_frames <= num_frames:
            # If video is shorter, use all frames and duplicate
            indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        else:
            # Otherwise sample uniformly
            indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        faces = []
        
        # Process frames
        for i in indices:
            # Set position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Error reading frame {i}")
                # If frame reading fails, duplicate last frame or use blank
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract face
            face_img = self._extract_face(frame_rgb)
            
            # Resize original frame for content model
            resized_frame = cv2.resize(frame_rgb, self.video_size)
            
            # Apply transforms
            frame_tensor = self.video_transform(resized_frame)
            face_tensor = self.face_transform(face_img)
            
            frames.append(frame_tensor)
            faces.append(face_tensor)
        
        cap.release()
        
        # Stack into tensor
        if frames and faces:
            video_tensor = torch.stack(frames, dim=0)
            face_tensor = torch.stack(faces, dim=0)
            
            # Reformat video tensor for 3D ResNet
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            # Extract content features
            content_features = self.extract_content_features(video_tensor)
            
            # Extract style codes
            style_codes = self.extract_style_codes(face_tensor)

            return content_features, style_codes
        else:
            logger.warning(f"Failed to extract frames from {video_path}")
            return None, None
