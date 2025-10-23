from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from dataset import FireMotionDataset, get_transforms
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


class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet feature extractor with MLP head for binary classification"""
    
    def __init__(self, pretrained_weights=EfficientNet_V2_S_Weights.DEFAULT, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained EfficientNet
        efficientnet = efficientnet_v2_s(weights=pretrained_weights)
        
        # Extract the feature extractor (everything except the classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last conv layer
        # EfficientNet_V2_S has 1280 features
        num_features = 1280
        
        # MLP head for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # MLP classification
        x = self.classifier(x)
        x = x.squeeze(-1)  # Remove last dimension for binary classification
        return x

def main(args):
    torch.autograd.set_detect_anomaly(True)
    train_transforms = get_transforms(img_size=args.img_size, is_train=True)
    train_dataset = FireMotionDataset("data/images/train", img_size=args.img_size, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_transform = get_transforms(img_size=args.img_size)
    val_dataset = FireMotionDataset("data/images/val", img_size=args.img_size, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EfficientNetFeatureExtractor(
        pretrained_weights=EfficientNet_V2_S_Weights.DEFAULT,
        freeze_backbone=True
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        for i, batch in enumerate(train_loader):
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

        avg_loss = train_loss / len(train_loader)
        avg_acc = train_acc / len(train_loader)
        print(avg_loss)
        print(avg_acc)

        for i, batch in val_loader:
            sequence, label = batch
            sequence = sequence.to(args.device)
            with torch.inference_mode():
                logits = model(sequence)
            val_loss += criterion(logits, label)
            val_acc += (pred == label).sum() / args.batch_size

        avg_loss = val_loss / len(val_loader)
        avg_acc = val_acc / len(val_loader)
        print(avg_loss)
        print(avg_acc)


if __name__ == "__main__":
    args = {"epochs": 50, "lr": 1e-5, "device": "cpu", "img_size": 224, "train_dir": "data/images/train", "val_dir": "data/images/val", "batch_size": 8}
    args = OmegaConf.create(args)
    args.train_dir = Path(args.train_dir)
    args.val_dir = Path(args.val_dir)
    main(args)
