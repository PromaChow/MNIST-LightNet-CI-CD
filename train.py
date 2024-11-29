import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTModel
from datetime import datetime
import os
import sys
import urllib.request

def download_mnist_fallback():
    """Fallback method to download MNIST dataset from alternative sources"""
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('./data/MNIST/raw', exist_ok=True)
    
    for file_name in files.keys():
        print(f"Downloading {file_name}...")
        url = base_url + file_name
        save_path = os.path.join('./data/MNIST/raw', file_name)
        if not os.path.exists(save_path):
            try:
                urllib.request.urlretrieve(url, save_path)
            except Exception as e:
                print(f"Error downloading {file_name}: {str(e)}")
                return False
    return True

def train():
    try:
        # Set device and print info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        print("Loading and preprocessing data...")
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset with explicit error handling and fallback
        try:
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        except Exception as e:
            print("Failed to download from primary source, trying fallback...")
            if download_mnist_fallback():
                train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
            else:
                print("Failed to download dataset from all sources")
                sys.exit(1)
                
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        print("Dataset loaded successfully")
        
        print("Initializing model...")
        model = MNISTModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        print("Starting training...")
        model.train()
        total_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')
        
        print("Training completed. Saving model...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('models', exist_ok=True)
        save_path = f'models/mnist_model_{timestamp}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    train() 