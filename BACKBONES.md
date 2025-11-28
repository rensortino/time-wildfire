# Backbone Selection Guide

This project supports multiple backbone architectures for wildfire detection from image sequences. Each backbone has different characteristics and is suitable for different use cases.

## Available Backbones

### 1. EfficientNet (Motion-based) - `efficientnet`

**Description:** A 2D CNN that processes optical flow/motion images computed from video frames.

**Architecture:**
- Uses EfficientNet V2-S as a feature extractor
- Processes a single motion image (computed from frame differences)
- MLP head for binary classification

**Pros:**
- Lightweight and fast
- Lower memory requirements
- Good for capturing motion patterns
- Pretrained on ImageNet

**Cons:**
- Only uses motion information, not full appearance
- Requires preprocessing to compute motion images
- Single-frame processing

**Dataset:** `FireMotionDataset` (computes optical flow)

**Usage:**
```bash
python main.py configs/efficientnet.yaml
```

**Configuration:**
- Batch size: 128
- Learning rate: 1e-5
- Input: Single RGB motion image (224x224)

---

### 2. 3D ResNet - `3dresnet`

**Description:** A 3D convolutional neural network that processes video sequences directly.

**Architecture:**
- Based on ResNet-18 with 3D convolutions
- Processes spatio-temporal features simultaneously
- Global average pooling followed by MLP classifier

**Pros:**
- Native video understanding
- Captures both spatial and temporal features
- Pretrained on Kinetics-400 dataset
- Proven architecture for action recognition

**Cons:**
- Higher memory consumption
- Slower training
- Requires smaller batch sizes

**Dataset:** `FireSeriesDataset` (returns frame sequences)

**Usage:**
```bash
python main.py configs/3dresnet.yaml
```

**Configuration:**
- Batch size: 16 (smaller due to 3D convolutions)
- Learning rate: 1e-4
- Input: Video sequence (8 frames x 224x224)

---

### 3. ViViT (Video Vision Transformer) - `vivit`

**Description:** Uses pretrained VideoMAE model from HuggingFace for video understanding with fine-tuned classifier.

**Architecture:**
- Based on VideoMAE (Masked Autoencoding for Videos)
- Pretrained on Kinetics-400 dataset
- Patches each frame into tokens
- Applies spatial and temporal self-attention
- CLS token for sequence-level classification
- Custom classifier head for binary classification

**Pros:**
- State-of-the-art for video understanding
- **Pretrained on large-scale video data (Kinetics-400)**
- Long-range temporal dependencies
- Highly flexible attention mechanism
- Captures global spatiotemporal context
- Available in multiple sizes (small/base/large)

**Cons:**
- Highest memory requirements
- Requires most compute resources
- Smallest recommended batch size
- Requires `transformers` library from HuggingFace

**Dataset:** `FireSeriesDataset` (returns frame sequences)

**Usage:**
```bash
python main.py configs/vivit.yaml
```

**Configuration:**
- Batch size: 8 (very memory intensive)
- Learning rate: 5e-5
- Input: Video sequence (16 frames x 224x224)
- Model: "MCG-NJU/videomae-base" (86M params)
- Alternative models: videomae-small (22M), videomae-large (304M)

---

### 4. 2D CNN + Transformer - `cnn_transformer`

**Description:** A hybrid approach combining 2D feature extraction with temporal transformer processing.

**Architecture:**
- EfficientNet V2-S extracts features from each frame independently
- Features are projected to transformer dimension
- Temporal transformer models dependencies across frames
- CLS token for sequence-level classification

**Pros:**
- Best of both worlds (CNN features + transformer temporal modeling)
- Leverages pretrained CNN weights
- More efficient than full ViViT
- Better than 3D CNN for long-range dependencies

**Cons:**
- Two-stage processing (CNN then transformer)
- Medium memory requirements
- More complex architecture

**Dataset:** `FireSeriesDataset` (returns frame sequences)

**Usage:**
```bash
python main.py configs/cnn_transformer.yaml
```

**Configuration:**
- Batch size: 32
- Learning rate: 3e-5
- Input: Video sequence (8 frames x 224x224)
- Embed dim: 512, Heads: 8, Layers: 4

---

## Comparison Table

| Backbone | Memory | Speed | Pretrained | Batch Size | Best For |
|----------|--------|-------|------------|------------|----------|
| EfficientNet | Low | Fast | ✅ ImageNet | 128 | Motion patterns, quick experiments |
| 3D ResNet | High | Medium | ✅ Kinetics-400 | 16 | Short-term spatiotemporal features |
| ViViT | Very High | Slow | ✅ Kinetics-400 (VideoMAE) | 8 | Long-range dependencies, global context |
| CNN+Transformer | Medium | Medium | ✅ ImageNet (CNN) | 32 | Balanced approach, frame + temporal |

---

## Installation Requirements

### Basic Requirements (All backbones except ViViT)
```bash
pip install -r requirements.txt
```

### Additional for ViViT
ViViT requires the HuggingFace transformers library:
```bash
pip install transformers
```

Or install all requirements:
```bash
pip install torch torchvision scikit-image opencv-python matplotlib transformers omegaconf tqdm
```

---

## Running with Different Backbones

### Option 1: Using Configuration Files (Recommended)

```bash
# Train with EfficientNet
python main.py configs/efficientnet.yaml

# Train with 3D ResNet
python main.py configs/3dresnet.yaml

# Train with ViViT
python main.py configs/vivit.yaml

# Train with CNN + Transformer
python main.py configs/cnn_transformer.yaml
```

### Option 2: Default (No Config File)

```bash
# Uses default EfficientNet configuration
python main.py
```

### Option 3: Creating Custom Configs

Create a new YAML file with your custom settings:

```yaml
# my_config.yaml
epochs: 100
lr: 1.0e-4
device: cuda:0
batch_size: 32
num_workers: 4

img_size: 224
train_dir: data/images/train
val_dir: data/images/val
num_frames: 16  # Use more frames

backbone: cnn_transformer
pretrained: true
freeze_backbone: false  # Fine-tune the CNN

transformer_embed_dim: 768
transformer_num_heads: 12
transformer_num_layers: 6
```

Then run:
```bash
python main.py my_config.yaml
```

---

## Recommendations

1. **Start with EfficientNet**: Quick baseline, low resources
2. **Try CNN+Transformer**: Good balance of performance and efficiency
3. **Use 3D ResNet**: If you need proven video understanding
4. **Experiment with ViViT**: If you have GPU resources and want best performance

---

## Model Parameters

When running, the script will print:
- Total model parameters
- Trainable parameters (important when freezing backbones)

Example output:
```
Creating model with backbone: cnn_transformer
Model created with 25.83M parameters
Trainable parameters: 3.14M parameters
```

---

## Implementation Details

All models are defined in `models.py`:
- `EfficientNetFeatureExtractor`: Motion-based 2D CNN
- `ResNet3D`: 3D convolutional network
- `ViViT`: Video vision transformer
- `CNN2DTransformer`: Hybrid 2D CNN + transformer

The `get_model()` factory function creates the appropriate model based on the configuration.

