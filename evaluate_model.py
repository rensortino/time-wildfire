#!/usr/bin/env python3
"""
Script to evaluate all trained models on a dataset.
Runs inference using pretrained checkpoints from the checkpoints directory.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from omegaconf import OmegaConf
import argparse

from models import get_model
from dataset import FireSeriesDataset, FireMotionDataset, get_transforms


def load_checkpoint(checkpoint_path, model, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Loaded checkpoint from epoch {epoch}")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
        print(f"  Loaded checkpoint (no epoch info)")
    
    return model


def evaluate_model(model, dataloader, device, model_name):
    """Evaluate a model on a dataset."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    criterion = nn.BCELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
            sequence, labels = batch
            sequence = sequence.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(sequence)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            probs = logits.cpu().numpy()
            predictions = (logits > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    avg_loss = total_loss / len(dataloader)
    
    # Calculate additional metrics
    true_positives = ((all_predictions == 1) & (all_labels == 1)).sum()
    false_positives = ((all_predictions == 1) & (all_labels == 0)).sum()
    false_negatives = ((all_predictions == 0) & (all_labels == 1)).sum()
    true_negatives = ((all_predictions == 0) & (all_labels == 0)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives),
        'total_samples': len(all_labels)
    }
    
    return metrics




def create_model(config, device):
    """Create model instance based on config, using same logic as main.py."""
    backbone = config.backbone
    
    print(f"Creating model with backbone: {backbone}")
    
    if backbone == 'efficientnet':
        model = get_model('efficientnet', freeze_backbone=config.freeze_backbone)
    elif backbone == '3dresnet':
        model = get_model('3dresnet', pretrained=config.pretrained, freeze_backbone=config.freeze_backbone)
    elif backbone == 'videomae':
        model = get_model('videomae', 
                         model_name=config.videomae_model_name,
                         num_frames=config.num_frames,
                         freeze_backbone=config.freeze_backbone,
                         dropout=config.dropout)
    elif backbone == 'vivit':
        model = get_model('vivit', 
                         model_name=config.vivit_model_name,
                         num_frames=config.num_frames,
                         freeze_backbone=config.freeze_backbone,
                         dropout=config.dropout)
    elif backbone == 'cnn_transformer':
        model = get_model('cnn_transformer',
                         freeze_backbone=config.freeze_backbone,
                         num_frames=config.num_frames,
                         embed_dim=config.transformer_embed_dim,
                         num_heads=config.transformer_num_heads,
                         num_layers=config.transformer_num_layers)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    return model


def main(config):
    """Main evaluation function."""
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get checkpoint path from config
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "final_model.pt"
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {config.backbone}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset CSV: {config.dataset_csv}")
    
    # Create model using same logic as main.py
    model = create_model(config, device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Create transforms
    val_transform = get_transforms(img_size=config.img_size, is_train=False)
    
    # Create appropriate dataset based on backbone
    if config.backbone == 'efficientnet':
        dataset = FireMotionDataset(
            csv_file=config.dataset_csv,
            img_size=config.img_size,
            transform=val_transform,
            max_images_per_sequence=OmegaConf.select(config, 'max_images_per_sequence', default=None)
        )
    else:
        # All video-based models (3dresnet, videomae, vivit, cnn_transformer)
        dataset = FireSeriesDataset(
            csv_file=config.dataset_csv,
            img_size=config.img_size,
            transform=val_transform,
            max_images_per_sequence=OmegaConf.select(config, 'max_images_per_sequence', default=None)
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset size: {len(dataset)} sequences")
    print(f"Number of batches: {len(dataloader)}")
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device, config.backbone)
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}")
    print(f"  Total samples: {metrics['total_samples']}")
    
    # Save results to JSON
    output_file = OmegaConf.select(config, 'output_file', default=None)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {config.backbone: metrics}
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        '--dataset-csv',
        type=str,
        default=None,
        help='Path to dataset CSV file (overrides config)'
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save evaluation results JSON file (overrides config)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)
    
    # Override dataset_csv if provided
    if args.dataset_csv:
        config.dataset_csv = args.dataset_csv
    elif not hasattr(config, 'dataset_csv'):
        config.dataset_csv = 'data/val_dataset/dataset.csv'
    
    # Override output_file if provided
    if args.output_file:
        config.output_file = args.output_file
    
    # Ensure checkpoint_dir is Path
    config.checkpoint_dir = Path(config.checkpoint_dir)


    config.debug = args.debug

    if args.debug:
        config.num_workers = 0

    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    print(OmegaConf.to_yaml(config))
    print("="*50 + "\n")
    
    main(config)

