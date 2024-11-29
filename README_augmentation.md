# Image Augmentation for MNIST-LightNet

## Overview
This module implements robust image augmentation techniques for the MNIST dataset, enhancing model training through data diversification. The implementation includes automated testing and visualization capabilities.

## Components

### 1. Augmentation Module (`utils/augmentation.py`)

#### Class Structure: `MNISTAugmentation`
```python
class MNISTAugmentation:
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
```

#### Key Features:
1. **Training Augmentations**:
   - Random rotation (±15 degrees)
   - Random affine transformations (translations)
   - Random perspective changes
   - Value normalization and clamping

2. **Visualization Transforms**:
   - Separate pipeline for visualization
   - Preserves image quality for display
   - Maintains original scale

3. **Sample Management**:
   - Automatic sample generation
   - Timestamp-based file naming
   - Organized storage structure

### 2. Testing Suite (`tests/test_augmentation.py`)

#### Test Categories:

1. **Sample Existence Test**:
```python
def test_augmentation_samples_exist():
    """Verifies augmented samples are generated and saved"""
    samples = glob.glob('augmented_samples/*.png')
    assert len(samples) > 0
```

2. **Consistency Test**:
```python
def test_augmentation_consistency():
    """Ensures augmentation maintains image dimensions"""
    augmented = augmenter.get_augmented_sample(image)
    assert augmented.shape == (1, 28, 28)
```

3. **Value Range Test**:
```python
def test_augmentation_range():
    """Validates normalized value range"""
    augmented = augmenter.get_augmented_sample(image)
    assert torch.min(augmented) >= -1 and torch.max(augmented) <= 1
```

## Implementation Details

### Augmentation Parameters
- **Rotation**: ±15 degrees
- **Translation**: Up to 10% in any direction
- **Perspective**: 20% distortion probability of 50%
- **Normalization**: Mean=0.1307, Std=0.3081

### Value Control
```python
transforms.Lambda(lambda x: torch.clamp(x, 0, 1))  # Pre-normalization
torch.clamp(augmented, -1, 1)                      # Post-normalization
```

### Sample Generation
```python
def save_augmented_samples(self, dataset, num_samples=5):
    # Saves both original and augmented versions
    for i in range(num_samples):
        # Original image
        save_image(original_tensor, f'original_{i}_{timestamp}.png')
        # Multiple augmentations
        for j in range(3):
            save_image(augmented, f'augmented_{i}_{j}_{timestamp}.png')
```

## Usage Instructions

### 1. Basic Usage
```python
from utils.augmentation import MNISTAugmentation

# Create augmenter
augmenter = MNISTAugmentation()

# Get augmented sample
augmented = augmenter.get_augmented_sample(image)

# Generate sample set
augmenter.save_augmented_samples(dataset)
```

### 2. Running Tests
```bash
# Run all augmentation tests
pytest tests/test_augmentation.py -v

# Run specific test
pytest tests/test_augmentation.py::test_augmentation_consistency -v
```

### 3. Integration with Training
```python
# In train.py
augmenter = MNISTAugmentation()
train_dataset.transform = augmenter.augmentations
```

## Output Directory Structure
```
project/
├── augmented_samples/
│   ├── original_0_20240329_121530.png
│   ├── augmented_0_0_20240329_121530.png
│   ├── augmented_0_1_20240329_121530.png
│   └── augmented_0_2_20240329_121530.png
```

## Error Handling

### 1. Input Validation
```python
if isinstance(image, torch.Tensor):
    image = transforms.ToPILImage()(image)
```

### 2. Value Range Control
- Pre-normalization clamping: 0 to 1
- Post-normalization clamping: -1 to 1
- Ensures consistent model input

### 3. Directory Management
```python
os.makedirs('augmented_samples', exist_ok=True)
```

## Testing Details

### 1. Sample Generation Test
- Verifies augmented samples exist
- Checks file creation
- Validates directory structure

### 2. Dimensional Consistency Test
- Ensures 28x28 image size
- Validates single channel (grayscale)
- Checks tensor shape integrity

### 3. Value Range Test
- Validates normalization
- Ensures proper scaling
- Checks value boundaries

## Best Practices

1. **Data Consistency**:
   - Consistent image dimensions
   - Normalized value ranges
   - Proper tensor formatting

2. **Error Prevention**:
   - Input type checking
   - Directory existence verification
   - Value range validation

3. **Code Organization**:
   - Modular design
   - Clear class structure
   - Comprehensive documentation

## Integration with CI/CD

The augmentation tests are integrated into the GitHub Actions workflow:
```yaml
- name: Run tests
  run: |
    pytest tests/test_model.py tests/test_augmentation.py -v
```

## Requirements
- PyTorch
- torchvision
- Pillow
- pytest

## Notes
- Augmentation parameters can be adjusted in `MNISTAugmentation.__init__`
- Test thresholds can be modified in test files
- Sample counts are configurable in save_augmented_samples 