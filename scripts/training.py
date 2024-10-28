import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define paths
UCF101_PATH = '/datasets/UCF101/UCF-101'
UCF101_ANNOTATIONS = '/datasets/UCF101/ucfTrainTestlist'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Preprocessing for the UCF101 dataset
logger.info("Setting up data transforms...")
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize video frames
    transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),          # Convert frames to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize values
])

# Load UCF101 dataset using torchvision
logger.info("Loading UCF101 dataset...")
train_dataset = UCF101(
    root=UCF101_PATH,
    annotation_path=UCF101_ANNOTATIONS,
    frames_per_clip=16,  # Number of frames in each video clip (adjustable)
    step_between_clips=2,  # Sampling rate between clips
    train=True,  # Load training split
    transform=transform,
    output_format='TCHW'
)

# 2. Create DataLoader
def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

logger.info("Creating DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate)
i = 0
for batch in train_loader:
    # Assuming the batch contains both data and labels (common in datasets with labeled data)
    data, labels = batch
    i += 1
    # Print the shape of the data and the labels
    print("Data shape:", data.shape)   # Shape of the input batch (e.g., [batch_size, channels, height, width])
    print("Labels shape:", labels.shape)
    if i > 2:
        break
    # Data shape: torch.Size([batch, 16, 240, 320, 3])
    
# # 3. Define the Unet3D model for the video diffusion process
logger.info("Initializing Unet3D model...")
model = Unet3D(
    dim=64,                # Dimension of the model
    dim_mults=(1, 2, 4, 8),   # Multiplier for the model's dimensions
)

# 4. Set up the Gaussian Diffusion Model
logger.info("Setting up Gaussian Diffusion model...")
logger.info(f"Number of parameters in Unet3D model: {sum(p.numel() for p in model.parameters()):,}")
diffusion = GaussianDiffusion(
    model,
    image_size=64,  # Size of each video frame
    num_frames=16,   # Number of video frames (same as frames_per_clip)
    timesteps=1000,  # Number of diffusion steps
    loss_type='l1'   # L1 or L2 loss
).to(device)
logger.info(f"Hyperparameters defined for Gaussian Diffusion model: {diffusion.image_size=}, {diffusion.num_frames=}, {diffusion.num_timesteps=}, {diffusion.loss_type=}")
# 5. Set up the Trainer
logger.info("Setting up the trainer...")
trainer = Trainer(
    diffusion,
    dataloader=train_loader,
    # folder=os.path.join('/datasets/UCF101/UCF-101','PommelHorse'),
    train_batch_size=32,   # Batch size
    train_lr=1e-4,        # Learning rate
    # train_num_steps=700000,  # Total number of training steps
    train_num_steps=10000,
    gradient_accumulate_every=2,  # Accumulate gradients for x steps before updating
    ema_decay=0.995,      # EMA decay rate
    amp=True,             # Use mixed precision for faster training
    save_and_sample_every=1000,  # Save checkpoints and sample generated videos every x steps
    results_folder='./results'   # Directory to save the results
)

# 6. Train the Model
logger.info("Starting training...")
trainer.train()

#7 Sample from the Model
logger.info("Sampling videos after training...")
sampled_videos = diffusion.sample(batch_size = 4)
sampled_videos.shape 
logger.info(f"Sampled videos shape: {sampled_videos.shape}")


def save_video(video_tensor, filename='generated_video.mp4', fps=5):
    logger.info(f"Saving video to {filename}...")
    # video_tensor shape: (batch, channels, frames, height, width)
    batch_size, channels, frames, height, width = video_tensor.shape
    
    # Rescale from [-1, 1] to [0, 255]
    video_tensor = ((video_tensor + 1) * 127.5).clamp(0, 255).byte()
    
    # Convert to numpy
    video_numpy = video_tensor.permute(0, 2, 3, 4, 1).cpu().numpy()  # (batch, frames, height, width, channels)
    
    # Write videos
    for b in range(batch_size):
        # Get a batch of videos
        video = video_numpy[b]  # (frames, height, width, channels)
        
        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(f'{filename}_sample{b}.mp4', fourcc, fps, (width, height))
        
        for frame_idx in range(frames):
            # Extract frame and convert to RGB for OpenCV (if necessary)
            frame = video[frame_idx]
            
            if channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR by default
            
            # Write frame
            out.write(frame)
        
        # Release the video writer object
        out.release()
        logger.info(f"Video {filename}_sample{b}.mp4 saved successfully.")

# Example usage
save_video(sampled_videos, filename='generated_video', fps=5)