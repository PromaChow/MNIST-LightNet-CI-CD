import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from utils.augmentation import MNISTAugmentation
from torchvision import datasets
import os
import glob

def test_augmentation_samples_exist():
    """Test if augmented samples are generated and saved"""
    samples = glob.glob('augmented_samples/*.png')
    assert len(samples) > 0, "No augmented samples found"

def test_augmentation_consistency():
    """Test if augmentation maintains image size and channels"""
    augmenter = MNISTAugmentation()
    dataset = datasets.MNIST('./data', train=True, download=True)
    image, _ = dataset[0]
    
    augmented = augmenter.get_augmented_sample(image)
    assert augmented.shape == (1, 28, 28), "Augmented image shape incorrect"

def test_augmentation_range():
    """Test if augmented values are in normalized range"""
    augmenter = MNISTAugmentation()
    dataset = datasets.MNIST('./data', train=True, download=True)
    image, _ = dataset[0]
    
    augmented = augmenter.get_augmented_sample(image)
    assert torch.min(augmented) >= -1 and torch.max(augmented) <= 1, "Values out of expected range" 