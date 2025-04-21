## file: preprocessor.py
#
# DeepfakePreprocessor :
# Preprocessing tasks for converting videos to tensors.
#
##

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from facenet_pytorch import MTCNN
import logging

logger = logging.getLogger(__name__)

class DeepfakePreprocessor:
    """
    Preprocessor for extracting frames and faces from videos for deepfake detection
    """
    def __init__(self, 
                 face_size=(256, 256), 
                 video_size=(224, 224), 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 content_model=None,
                 psp_model=None):
        self.face_size = face_size
        self.video_size = video_size
        self.device = device
        self.content_model = content_model.to(device)
        self.psp_model = psp_model.to(device)
        

        # Load content model if provided
        if self.content_model is not None:
            self.content_model.eval()

        # Load PSP model if provided
        if self.psp_model is not None:
            self.psp_model.eval()
        
        # Initialize face detector (MTCNN)
        self.face_detector = MTCNN(
            image_size=face_size[0],
            margin=40,
            device=device,
            keep_all=False  # Only keep the largest face
        )
        
        # Transforms for preprocessing
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # StyleGAN normalization
        ])
        
        # Use Kinetics dataset normalization for video frames (for 3D ResNet)
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])  # Kinetics stats (updated)
        ])

    def _extract_face(self, frame):
        """
        Extract face from a single frame using MTCNN
        """
        # Convert to RGB if needed
        if frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3 and frame.dtype == np.uint8:
            # Check if already RGB or BGR
            if cv2.COLOR_BGR2RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect face
            face = self.face_detector(frame)
            
            if face is None:
                # If face detection fails, use center crop
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                face_img = frame[top:top + min_dim, left:left + min_dim]
                face_img = cv2.resize(face_img, self.face_size)
            else:
                # If face is detected as tensor, convert to numpy
                if isinstance(face, torch.Tensor):
                    # MTCNN returns normalized tensor
                    face_img = face.permute(1, 2, 0).cpu().numpy()
                    # Denormalize
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
            for i in range(seq_len):
                # Extract a single frame
                frame = face_tensor[i].unsqueeze(0).to(self.device)  # [1, C, H, W]
                
                # Extract style latent vector using only the encoder
                codes = self.psp_model.encoder(frame)
                
                # Use only one latent level (e.g., level 9 which is in the middle)
                level_idx = 9
                codes = codes[:, level_idx, :]  # Shape [1, 512]
                
                style_codes.append(codes.cpu().squeeze(0))
        
        # Stack style codes along sequence dimension
        if style_codes:
            return torch.stack(style_codes, dim=0)  # [T, 512]
        else:
            logger.warning("No style codes extracted, returning None")
            return None
        
    def extract_content_features(self, video_tensor):
        """
        Extract content features using the 3D ResNet model from PyTorchVideo
        """
        
        with torch.no_grad():
            # Add batch dimension if it's missing
            if len(video_tensor.shape) == 4:  # [C, T, H, W]
                video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
            
            # Move tensor to device
            video_tensor = video_tensor.to(self.device)
            
            # Get features from the model
            content_features = self.content_model(video_tensor)
            
            # Print the feature shape
            logger.debug(f"Raw features shape: {content_features.shape}")
            
            # Apply global average pooling if needed
            if len(content_features.shape) > 2:
                # If we have spatial dimensions remaining, apply pooling
                # For SlowFast models, this might be [batch, channels, time, h, w]
                dims_to_pool = list(range(2, len(content_features.shape)))
                content_features = torch.mean(content_features, dim=dims_to_pool)
            
            # Remove batch dimension and move back to CPU
            content_features = content_features.squeeze(0).cpu()
            
            logger.debug(f"Content features shape: {content_features.shape}")
        
        return content_features

    def process_video(self, video_path, output_dir=None, num_frames=32, save_frames=False):
        """
        Process video file to extract frames and faces
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None, None
        
        # Create output directory if specified
        if output_dir and save_frames:
            os.makedirs(output_dir, exist_ok=True)
            face_dir = os.path.join(output_dir, 'faces')
            frame_dir = os.path.join(output_dir, 'frames')
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(frame_dir, exist_ok=True)
        
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
            # If video is shorter, use all frames and possibly duplicate
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
            
            # Save frames if requested
            if save_frames and output_dir:
                cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.jpg"), 
                           cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(face_dir, f"face_{i:04d}.jpg"), 
                           cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            # Apply transforms
            frame_tensor = self.video_transform(resized_frame)
            face_tensor = self.face_transform(face_img)
            
            frames.append(frame_tensor)
            faces.append(face_tensor)
        
        cap.release()
        
        # Stack into tensor
        if frames and faces:
            video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
            face_tensor = torch.stack(faces, dim=0)    # [T, C, H, W]
            
            # Reformat video tensor for 3D ResNet
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
            # Extract content features
            content_features = self.extract_content_features(video_tensor)

            # Extract style codes if PSP model is provided
            style_codes = None
            if self.psp_model is not None:
                style_codes = self.extract_style_codes(face_tensor)

            return content_features, style_codes
        else:
            logger.warning(f"Failed to extract frames from {video_path}")
            return None, None
    
    def process_directory(self, input_dir, output_dir=None, num_frames=32, save_frames=False):
        """
        Process all videos in a directory
        """
        if not os.path.exists(input_dir):
            logger.warning(f"Directory not found: {input_dir}")
            return {}
        
        results = {}
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for filename in tqdm(os.listdir(input_dir), desc="Processing videos"):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(input_dir, filename)
                
                # Create video-specific output directory if requested
                video_output_dir = None
                if output_dir and save_frames:
                    video_name = os.path.splitext(filename)[0]
                    video_output_dir = os.path.join(output_dir, video_name)
                
                # Process video
                video_tensor, face_tensor = self.process_video(
                    video_path, 
                    video_output_dir, 
                    num_frames, 
                    save_frames
                )
                
                if video_tensor is not None and face_tensor is not None:
                    results[filename] = {
                        'video_tensor': video_tensor,
                        'face_tensor': face_tensor
                    }
        
        return results
    
    def save_tensors(self, tensors_dict, output_dir):
        """
        Save processed tensors to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for filename, data in tqdm(tensors_dict.items(), desc="Saving tensors"):
            video_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{video_name}.pt")
            
            torch.save({
                'video': data['video_tensor'],
                'face': data['face_tensor']
            }, output_path)
            
        logger.info(f"Saved {len(tensors_dict)} processed videos to {output_dir}")

