import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Further reduced channels: 1 -> 4 -> 8 (previously 1 -> 8 -> 16)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        
        # Further reduced neurons in fully connected layers
        self.fc1 = nn.Linear(8 * 7 * 7, 32)  # Reduced from 64 to 32
        self.fc2 = nn.Linear(32, 10)  # Output layer remains same (10 classes)
        
        # Pooling and activation functions
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the output
        x = x.view(-1, 8 * 7 * 7)  # Adjusted for new channel size
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 