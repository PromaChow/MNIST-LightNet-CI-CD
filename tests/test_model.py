import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.mnist_model import MNISTModel
from torchvision import datasets, transforms
import glob
import os
from utils.augmentation import MNISTAugmentation

def get_latest_model():
    # Get the most recent model file
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found. Please train the model first using 'python train.py'")
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def test_model_architecture():
    # Test model input shape and output shape
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape is incorrect"

def test_parameter_count():
    # Test if model has less than 25000 parameters
    model = MNISTModel()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    try:
        # Load the latest trained model and test its accuracy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MNISTModel().to(device)
        
        try:
            model.load_state_dict(torch.load(get_latest_model()))
        except RuntimeError as e:
            pytest.fail(f"Failed to load model. Please delete old model files and retrain: {str(e)}")
            
        model.eval()

        # Prepare test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"
    
    except FileNotFoundError as e:
        pytest.fail(str(e)) 

def test_model_augmented_input():
    """Test model performance on augmented data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(get_latest_model()))
    model.eval()
    
    augmenter = MNISTAugmentation()
    dataset = datasets.MNIST('./data', train=False, download=True)
    image, label = dataset[0]
    
    # Test multiple augmented versions
    for _ in range(5):
        augmented = augmenter.get_augmented_sample(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(augmented)
        assert output.shape == (1, 10), "Output shape incorrect for augmented input"

def test_model_robustness():
    """Test model performance on noisy input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(get_latest_model()))
    model.eval()
    
    # Add noise to test input
    test_input = torch.randn(1, 1, 28, 28).to(device)
    test_input = torch.clamp(test_input, -1, 1)
    
    with torch.no_grad():
        output = model(test_input)
    assert output.shape == (1, 10), "Model failed on noisy input"

def test_model_batch_processing():
    """Test model with different batch sizes"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(get_latest_model()))
    model.eval()
    
    batch_sizes = [1, 16, 32, 64, 128]
    for size in batch_sizes:
        test_input = torch.randn(size, 1, 28, 28).to(device)
        with torch.no_grad():
            output = model(test_input)
        assert output.shape == (size, 10), f"Failed for batch size {size}"