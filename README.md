# Time-Wildfire

Wildfire smoke detection and object tracking based on temporal analysis of sequential frames. This project combines multiple deep learning backbones for smoke classification with SAM3-based bounding box tracking across video sequences.

## Features

### Smoke Classification (Multiple Backbones)

Choose from 5 different model architectures for binary smoke detection:

- **EfficientNet**: 2D CNN with motion image processing (ImageNet pretrained)
- **3D ResNet**: 3D convolutional network for spatiotemporal features (Kinetics-400 pretrained)
- **VideoMAE**: Masked Autoencoder for Videos, self-supervised (Kinetics-400 pretrained, 16 frames)
- **ViViT**: Video Vision Transformer from Google, supervised (Kinetics-400 pretrained, 32 frames)
- **CNN+Transformer**: Hybrid 2D CNN + temporal transformer (ImageNet pretrained)

See [BACKBONES.md](BACKBONES.md) for architecture details.

### SAM3 Object Tracking

Track objects across video sequences using [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3):

- **Bounding box grounding**: provide a bounding box on the first frame, and SAM3 propagates tracking through the entire sequence
- **Multiple cropping strategies**: `padded`, `resize`, and `balanced` cropping methods to optimize tracking accuracy
- **Evaluation pipeline**: compute AP, precision, recall, F1, and IoU metrics per sequence and per class
- **Visualization tools**: plot per-class metrics and overlay predicted vs. ground truth bounding boxes
- **Gradio demo**: interactive web interface for uploading image sequences and running SAM3 tracking

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/time-wildfire.git
cd time-wildfire
```

Create the virtual environment from the lockfile:
```bash
uv sync
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

### SAM3 Tracking (Gradio Demo)

Launch the interactive tracking demo:
```bash
python app.py
```

This opens a web UI where you can:
1. Upload a sequence of images (frames of a video)
2. Draw or specify a bounding box on the first frame
3. Run SAM3 tracking and see predicted bounding boxes overlaid on every frame

### SAM3 Tracking (CLI — single sequence)

Track an object through a directory of images using a YOLO-format label file:

```bash
# Labels as a directory of per-frame .txt files (YOLO layout)
python run_tracking.py \
    --images-dir data/val_dataset/wildfire/sequence_name/images \
    --labels     data/val_dataset/wildfire/sequence_name/labels \
    --output-dir output/sequence_name/

# Labels as a single .txt file (one line per frame)
python run_tracking.py \
    --images-dir data/sequence_001/ \
    --labels     data/sequence_001/labels.txt \
    --output-dir output/sequence_001/
```

**Inputs:**
- `--images-dir` — directory of images, sorted alphabetically by filename to match sequence order.
- `--labels` — YOLO-format annotations. Either a single `.txt` file (one `class_id x_center y_center width height` line per frame) or a **directory** of per-frame `.txt` files matched by stem name (e.g. `frame_001.txt` ↔ `frame_001.jpg`). The **first frame's** bounding box is used as the SAM3 tracking prompt; the rest are optional ground-truth annotations.
- `--output-dir` — directory where annotated images are saved.

**Outputs:** each frame is saved with a **blue** box (ground truth) and a **yellow** box (SAM3 prediction).

### SAM3 Tracking (batch evaluation)

Run tracking evaluation on a dataset CSV:
```bash
# Full-resolution tracking
python test_sam_tracking.py

# Cropped tracking (balanced method, default)
python sam_tracking_cropped.py --csv-path data/val_dataset/dataset.csv

# Cropped tracking with different methods
python sam_tracking_cropped.py --crop-method padded
python sam_tracking_cropped.py --crop-method resize
python sam_tracking_cropped.py --crop-method balanced
```

Visualize tracking results:
```bash
python visualize_tracking.py --json results/sam3_tracking_results.json --csv data/val_dataset/dataset.csv --sequence <sequence_name>

# List available sequences
python visualize_tracking.py --list
```

Plot per-class metrics:
```bash
python plot_results.py
```

### Training (Smoke Classification)

```bash
python main.py configs/efficientnet.yaml
python main.py configs/3dresnet.yaml
python main.py configs/videomae.yaml
python main.py configs/vivit.yaml
python main.py configs/cnn_transformer.yaml
```

## Project Structure

```
time-wildfire/
├── app.py                      # Gradio web interface for SAM3 tracking
├── run_tracking.py             # CLI script: track a single image sequence
├── sam3_tracker.py             # Core SAM3 tracking logic (shared by app.py & run_tracking.py)
├── main.py                     # Training script for smoke classification
├── models.py                   # Smoke classification model architectures
├── dataset.py                  # Dataset classes
├── utils.py                    # Shared utilities (bbox conversion, IoU, AP metrics, mask-to-bbox)
├── test_sam_tracking.py        # SAM3 tracking evaluation (full resolution)
├── sam_tracking_cropped.py     # SAM3 tracking evaluation (with cropping)
├── visualize_tracking.py       # Visualize tracking predictions vs ground truth
├── plot_results.py             # Plot per-class tracking metrics
├── configs/                    # Training configuration files
│   ├── efficientnet.yaml
│   ├── 3dresnet.yaml
│   ├── videomae.yaml
│   ├── vivit.yaml
│   └── cnn_transformer.yaml
├── sam3/                       # SAM3 model source (facebook/sam3)
├── results/                    # Tracking evaluation results (JSON + plots)
├── BACKBONES.md                # Backbone architecture guide
├── pyproject.toml              # Project metadata and dependencies
└── requirements.txt            # Pip dependencies
```

## Requirements

- Python 3.13+
- PyTorch 2.8+
- torchvision
- transformers (for SAM3, ViViT, VideoMAE)
- gradio (for the web demo)
- opencv-python
- scikit-image
- matplotlib
- omegaconf
- tqdm

## License

See [LICENSE](LICENSE) file for details.
