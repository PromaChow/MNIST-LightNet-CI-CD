import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from datetime import datetime
from PIL import Image
import numpy as np

class MNISTAugmentation:
    def __init__(self):
        # Augmentations for training
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamp values between 0 and 1
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Transforms for visualization (without normalization)
        self.viz_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))  # Clamp values between 0 and 1
        ])
        
    def get_augmented_sample(self, image):
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image for augmentation
            image = transforms.ToPILImage()(image)
        augmented = self.augmentations(image)
        # Ensure values are in range [-1, 1] after normalization
        return torch.clamp(augmented, -1, 1)
    
    def save_augmented_samples(self, dataset, num_samples=5):
        os.makedirs('augmented_samples', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i in range(num_samples):
            # Get original image without normalization
            image, label = dataset.data[i], dataset.targets[i]
            
            # Convert to PIL Image
            image = Image.fromarray(image.numpy().astype('uint8'), mode='L')
            
            # Save original
            original_tensor = transforms.ToTensor()(image)
            save_image(original_tensor, f'augmented_samples/original_{i}_{timestamp}.png')
            
            # Save augmented versions
            for j in range(3):
                augmented = self.viz_transform(image)
                save_image(augmented, f'augmented_samples/augmented_{i}_{j}_{timestamp}.png') 