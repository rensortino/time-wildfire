from tqdm import tqdm
from dataset import FireMotionDataset, FireSeriesDataset, get_transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
from utils import get_motion_image
from omegaconf import OmegaConf
import trackio
from pathlib import Path
import random
from models import get_model
from torch.utils.tensorboard import SummaryWriter

def main(args):
    torch.autograd.set_detect_anomaly(True)
    
    # Create directories for checkpoints and logs
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Choose dataset based on backbone
    # EfficientNet uses motion images, others use frame sequences
    if args.backbone == 'efficientnet':
        train_transforms = get_transforms(img_size=args.img_size, is_train=True)
        train_dataset = FireMotionDataset("data/images/train", img_size=args.img_size, transform=train_transforms)
        val_transform = get_transforms(img_size=args.img_size)
        val_dataset = FireMotionDataset("data/images/val", img_size=args.img_size, transform=val_transform)
    else:
        # All video-based models (3dresnet, videomae, vivit, cnn_transformer)
        train_transforms = get_transforms(img_size=args.img_size, is_train=True)
        train_dataset = FireSeriesDataset("data/images/train", img_size=args.img_size, transform=train_transforms)
        val_transform = get_transforms(img_size=args.img_size)
        val_dataset = FireSeriesDataset("data/images/val", img_size=args.img_size, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        # persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive between epochs
        # pin_memory=True,  # Faster GPU transfer
        # prefetch_factor=2 if args.num_workers > 0 else None,  # Prefetch 2 batches per worker
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        # persistent_workers=True if args.num_workers > 0 else False,
        # pin_memory=True,
        # prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Create model based on backbone choice
    print(f"Creating model with backbone: {args.backbone}")
    
    if args.backbone == 'efficientnet':
        model = get_model('efficientnet', freeze_backbone=args.freeze_backbone)
    elif args.backbone == '3dresnet':
        model = get_model('3dresnet', pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    elif args.backbone == 'videomae':
        model = get_model('videomae', 
                         model_name=args.videomae_model_name,
                         num_frames=args.num_frames,
                         freeze_backbone=args.freeze_backbone,
                         dropout=args.dropout)
    elif args.backbone == 'vivit':
        model = get_model('vivit', 
                         model_name=args.vivit_model_name,
                         num_frames=args.num_frames,
                         freeze_backbone=args.freeze_backbone,
                         dropout=args.dropout)
    elif args.backbone == 'cnn_transformer':
        model = get_model('cnn_transformer',
                         freeze_backbone=args.freeze_backbone,
                         num_frames=args.num_frames,
                         embed_dim=args.transformer_embed_dim,
                         num_heads=args.transformer_num_heads,
                         num_layers=args.transformer_num_layers)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    
    model = model.to(args.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Training phase
        model.train()
        train_acc = 0
        train_loss = 0
        for i, batch in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
            sequence, label = batch
            sequence = sequence.to(args.device)
            label = label.to(args.device)

            logits = model(sequence)
            loss = criterion(logits, label)
            pred = (logits > 0.5).float()
            train_acc += (pred == label).sum() / args.batch_size
            train_loss += loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.debug and i == 10:
                break

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # Validation phase
        model.eval()
        val_acc = 0
        val_loss = 0
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader)):
            sequence, label = batch
            sequence = sequence.to(args.device)
            label = label.to(args.device)
            with torch.inference_mode():
                logits = model(sequence)
                pred = (logits > 0.5).float()
            val_loss += criterion(logits, label)
            val_acc += (pred == label).sum() / args.batch_size

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
        
        # Print metrics
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc,
                'args': OmegaConf.to_container(args),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': OmegaConf.to_container(args),
    }, final_checkpoint_path)
    print(f"\nFinal model saved to: {final_checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)
    config.debug = args.debug
    config.freeze_backbone = args.freeze_backbone

    if args.debug:
        config.epochs = 1
        config.batch_size = 1
        config.num_workers = 0
        config.checkpoint_dir = "checkpoints/debug"
        config.log_dir = "logs/debug"

    config.train_dir = Path(config.train_dir)
    config.val_dir = Path(config.val_dir)
    
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    print(OmegaConf.to_yaml(config))
    print("="*50 + "\n")
    
    main(config)
