# MNIST-LightNet-CI/CD

A lightweight, efficient MNIST classifier with automated CI/CD pipeline. Features a compact CNN (<25K parameters) achieving >95% accuracy with automated testing and deployment.

## Project Structure
```
project/
├── model/
│   ├── __init__.py          # Makes model directory a Python package
│   └── mnist_model.py       # CNN model architecture
├── tests/
│   └── test_model.py        # Automated test suite
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml  # GitHub Actions workflow
├── train.py                 # Training script
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Detailed Component Description

### 1. Model Architecture (`model/mnist_model.py`)
```python
class MNISTModel(nn.Module):
    def __init__(self):
        # Input: 28x28 grayscale images
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 1→4 channels
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # 4→8 channels
        self.fc1 = nn.Linear(8 * 7 * 7, 32)                     # Flattened→32
        self.fc2 = nn.Linear(32, 10)                            # 32→10 (output)
```
- Total Parameters: ~13,242
- Architecture Breakdown:
  - 2 Convolutional layers with MaxPooling
  - 2 Fully connected layers
  - ReLU activation functions
  - Output: 10 classes (digits 0-9)

### 2. Training Script (`train.py`)
Key Features:
- Automatic device selection (CPU/GPU)
- MNIST dataset download with fallback sources
- Data preprocessing and normalization
- One epoch training with Adam optimizer
- Model saving with timestamp
- Comprehensive error handling

Training Process:
```bash
python train.py
```
This will:
1. Download MNIST dataset
2. Initialize model
3. Train for one epoch
4. Save model as `mnist_model_YYYYMMDD_HHMMSS.pth`

### 3. Test Suite (`tests/test_model.py`)
Three main test categories:

1. Architecture Test:
```python
def test_model_architecture():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10)
```

2. Parameter Count Test:
```python
def test_parameter_count():
    model = MNISTModel()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000
```

3. Accuracy Test:
```python
def test_model_accuracy():
    # Tests if model achieves >95% accuracy on test set
```

Run tests with:
```bash
pytest tests/test_model.py -v
```

### 4. CI/CD Pipeline (`.github/workflows/ml-pipeline.yml`)
Automated workflow:
```yaml
name: ML Pipeline
on: [push]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
    - name: Train and Test
      run: |
        python train.py
        pytest tests/test_model.py -v
```

## Setup Instructions

### Local Development Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch torchvision pytest
```

3. Verify installation:
```python
python -c "import torch; print(torch.__version__)"
```

### GitHub Setup
1. Create new repository
2. Initialize git and push:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Error Handling and Troubleshooting

### Common Issues and Solutions:

1. MNIST Download Failure:
```python
# Fallback download mechanism in train.py
def download_mnist_fallback():
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    # Downloads from alternative source
```

2. Model Loading Errors:
- Delete old model files if architecture changes
- Ensure correct model path
- Check device compatibility (CPU/GPU)

3. Memory Issues:
- Batch size can be adjusted in train.py
- Model designed to be memory-efficient
- Uses CPU by default

## Performance Metrics

1. Model Size:
- Parameters: <25,000
- Model file size: ~100KB

2. Accuracy Targets:
- Training accuracy: >95%
- Test accuracy: >95%
- Single epoch training

3. Resource Usage:
- CPU-friendly design
- Minimal memory footprint
- Fast training time

## Best Practices Implemented

1. Code Organization:
- Modular architecture
- Clear separation of concerns
- Comprehensive documentation

2. Testing:
- Automated test suite
- Multiple test categories
- CI/CD integration

3. Error Handling:
- Graceful fallbacks
- Informative error messages
- Robust exception handling

## Contributing Guidelines

1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature
```
3. Commit changes:
```bash
git commit -m "Add your feature"
```
4. Push and create PR:
```bash
git push origin feature/your-feature
```

## License
MIT License - free to use and modify

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest










