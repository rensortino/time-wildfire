# Time-Wildfire 🔥

Wildfire smoke detection based on temporal analysis of sequential frames using multiple deep learning backbones.

## Features

- **Multiple Backbone Architectures**: Choose from 5 different model architectures
  - **EfficientNet**: 2D CNN with motion image processing
  - **3D ResNet**: 3D convolutional network for spatiotemporal features
  - **VideoMAE**: Masked Autoencoder for Videos (self-supervised, 16 frames)
  - **ViViT**: Video Vision Transformer from Google (supervised, 32 frames)
  - **CNN+Transformer**: Hybrid 2D CNN + temporal transformer

  Checkout [BACKBONES](BACKBONES.md) for more details on the architectures

- **Pretrained Models**: All backbones use pretrained weights for better initialization
  - EfficientNet: ImageNet pretrained
  - 3D ResNet: Kinetics-400 pretrained
  - VideoMAE: Kinetics-400 pretrained (self-supervised)
  - ViViT: Kinetics-400 pretrained (supervised)
  - CNN+Transformer: ImageNet pretrained (CNN part)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time-wildfire.git
cd time-wildfire
```

Create the virtual environment from the lockfile 
```bash
uv sync
```

Or install the dependencies manually:
```bash
pip install torch torchvision scikit-image opencv-python matplotlib omegaconf tqdm

# For ViViT backbone, also install transformers
pip install transformers
```

Or simply:
```bash
pip install -r requirements.txt
```

### Training

```bash
# Train with EfficientNet (fast baseline)
python main.py configs/efficientnet.yaml

# Train with 3D ResNet (spatiotemporal features)
python main.py configs/3dresnet.yaml

# Train with VideoMAE (self-supervised transformer, 16 frames)
python main.py configs/videomae.yaml

# Train with ViViT (Google's video transformer, 32 frames)
python main.py configs/vivit.yaml

# Train with CNN+Transformer (balanced approach)
python main.py configs/cnn_transformer.yaml
```

### Testing Models

Test the inference time of all models:
```bash
python benchmark_inference.py
```

## Model Comparison

| Backbone | Memory | Speed | Pretrained | Batch Size | Best For |
|----------|--------|-------|------------|------------|----------|
| EfficientNet | Low | Fast | ✅ ImageNet | 128 | Motion patterns, quick experiments |
| 3D ResNet | High | Medium | ✅ Kinetics-400 | 16 | Short-term spatiotemporal features |
| VideoMAE | High | Medium | ✅ Kinetics-400 (self-sup) | 8 | General video tasks, 16 frames |
| ViViT | Very High | Slow | ✅ Kinetics-400 (supervised) | 4 | Action recognition, 32 frames |
| CNN+Transformer | Medium | Medium | ✅ ImageNet | 32 | Balanced approach, frame + temporal |

## Configuration Examples

### EfficientNet (Motion-based)
```yaml
backbone: efficientnet
batch_size: 128
lr: 1.0e-5
freeze_backbone: true
```

### VideoMAE (Self-supervised)
```yaml
backbone: videomae
videomae_model_name: "MCG-NJU/videomae-base"
num_frames: 16
batch_size: 8
lr: 5.0e-5
freeze_backbone: false  # Fine-tune entire model
```

### ViViT (Google, Supervised)
```yaml
backbone: vivit
vivit_model_name: "google/vivit-b-16x2-kinetics400"
num_frames: 32
batch_size: 4
lr: 5.0e-5
freeze_backbone: false  # Fine-tune entire model
```

### 3D ResNet
```yaml
backbone: 3dresnet
num_frames: 8
batch_size: 16
lr: 1.0e-4
freeze_backbone: true
```

### CNN + Transformer
```yaml
backbone: cnn_transformer
num_frames: 8
batch_size: 32
lr: 3.0e-5
transformer_embed_dim: 512
transformer_num_heads: 8
transformer_num_layers: 4
```

## Project Structure

```
time-wildfire/
├── main.py                  # Main training script
├── models.py                # All model architectures
├── dataset.py               # Dataset classes
├── utils.py                 # Utility functions
├── test_backbones.py        # Test script for all models
├── configs/                 # Configuration files
│   ├── efficientnet.yaml
│   ├── 3dresnet.yaml
│   ├── videomae.yaml
│   ├── vivit.yaml
│   └── cnn_transformer.yaml
├── BACKBONES.md            # Complete backbone guide
├── VIVIT_VS_VIDEOMAE.md    # VideoMAE vs ViViT comparison
├── CHANGES_SUMMARY.md      # Recent changes
└── requirements.txt        # Python dependencies
```

## Usage Examples

### VideoMAE Example
```python
from models import get_model
import torch

# Create VideoMAE model
model = get_model(
    backbone='videomae',
    model_name='MCG-NJU/videomae-base',
    num_frames=16,
    freeze_backbone=True
)

# Forward pass
video = torch.randn(2, 16, 3, 224, 224)  # (batch, 16 frames, channels, H, W)
output = model(video)  # (batch,) - binary classification
```

### ViViT Example
```python
from models import get_model
import torch

# Create ViViT model
model = get_model(
    backbone='vivit',
    model_name='google/vivit-b-16x2-kinetics400',
    num_frames=32,
    freeze_backbone=True
)

# Forward pass
video = torch.randn(2, 32, 3, 224, 224)  # (batch, 32 frames, channels, H, W)
output = model(video)  # (batch,) - binary classification
```

## Highlights: Video Transformers

This project supports **two state-of-the-art video transformers**:

### VideoMAE (Self-supervised)
- ✅ Pretrained with masked autoencoding on Kinetics-400
- ✅ Multiple sizes: small (22M), base (86M), large (304M)
- ✅ Data-efficient learning
- ✅ Works with 16 frames (faster, lower memory)

### ViViT (Supervised)
- ✅ Google's Video Vision Transformer
- ✅ Pretrained on Kinetics-400 with labels
- ✅ Factorized spatiotemporal attention
- ✅ Works with 32 frames (longer temporal context)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- transformers (for ViViT)
- opencv-python
- scikit-image
- omegaconf
- tqdm

## License

See [LICENSE](LICENSE) file for details.