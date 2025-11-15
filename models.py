"""
Model architectures for wildfire detection from image sequences.
Supports multiple backbones:
- EfficientNet (2D baseline with motion images)
- 3DResNet (3D CNN for spatiotemporal features)
- ViViT (Vision Transformer for videos)
- 2D CNN + Transformer (hybrid approach)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
import math


class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet feature extractor with MLP head for binary classification.
    Works with single motion images (not sequences)."""
    
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


class ResNet3D(nn.Module):
    """3D ResNet for video classification.
    Processes sequences of frames with 3D convolutions.
    Input shape: (B, T, C, H, W) where T is number of frames.
    """
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained 3D ResNet-18
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        resnet3d = r3d_18(weights=weights)
        
        # Extract feature layers (everything except FC layer)
        self.stem = resnet3d.stem
        self.layer1 = resnet3d.layer1
        self.layer2 = resnet3d.layer2
        self.layer3 = resnet3d.layer3
        self.layer4 = resnet3d.layer4
        self.avgpool = resnet3d.avgpool
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in [*self.stem.parameters(), 
                         *self.layer1.parameters(),
                         *self.layer2.parameters(),
                         *self.layer3.parameters(),
                         *self.layer4.parameters()]:
                param.requires_grad = False
        
        # Number of features from ResNet3D-18
        num_features = 512
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input: (B, T, C, H, W)
        # 3D ResNet expects: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract 3D features
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x


class VideoMAE(nn.Module):
    """VideoMAE (Masked Autoencoder for Videos) for video classification.
    Uses pretrained VideoMAE model from HuggingFace and fine-tunes the classifier.
    Input shape: (B, T, C, H, W) where T is number of frames.
    """
    
    def __init__(self, model_name="MCG-NJU/videomae-base", num_frames=16, 
                 freeze_backbone=False, dropout=0.1):
        super().__init__()
        
        try:
            from transformers import VideoMAEModel, VideoMAEConfig
        except ImportError:
            raise ImportError(
                "transformers library is required for VideoMAE. "
                "Install it with: pip install transformers"
            )
        
        self.num_frames = num_frames
        
        print(f"Loading pretrained VideoMAE model: {model_name}")
        
        # Load pretrained VideoMAE backbone
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        self.config = self.backbone.config
        
        # Get hidden size from config
        hidden_size = self.config.hidden_size  # 768 for base model
        
        # Freeze backbone if specified
        if freeze_backbone:
            print("Freezing VideoMAE backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # VideoMAE expects: (B, T, C, H, W) which is what we have
        # But we need to ensure the number of frames matches
        if T != self.num_frames:
            # Sample or repeat frames to match expected number
            if T > self.num_frames:
                # Sample uniformly
                indices = torch.linspace(0, T - 1, self.num_frames).long()
                x = x[:, indices]
            else:
                # Repeat frames
                repeat_factor = (self.num_frames + T - 1) // T
                x = x.repeat(1, repeat_factor, 1, 1, 1)[:, :self.num_frames]
        
        # Extract features using pretrained backbone
        outputs = self.backbone(pixel_values=x)
        
        # Get the [CLS] token representation
        # Last hidden state shape: (B, num_patches + 1, hidden_size)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]  # Take [CLS] token
        
        # Classification
        logits = self.classifier(cls_output)
        logits = logits.squeeze(-1)
        return logits


class ViViT(nn.Module):
    """ViViT (Video Vision Transformer) for video classification.
    Uses pretrained ViViT model from HuggingFace and fine-tunes the classifier.
    Input shape: (B, T, C, H, W) where T is number of frames.
    """
    
    def __init__(self, model_name="google/vivit-b-16x2-kinetics400", num_frames=32, 
                 freeze_backbone=False, dropout=0.1):
        super().__init__()
        
        try:
            from transformers import VivitModel, VivitConfig
        except ImportError:
            raise ImportError(
                "transformers library is required for ViViT. "
                "Install it with: pip install transformers"
            )
        
        self.num_frames = num_frames
        
        print(f"Loading pretrained ViViT model: {model_name}")
        
        # Load pretrained ViViT backbone
        self.backbone = VivitModel.from_pretrained(model_name)
        self.config = self.backbone.config
        
        # Get hidden size from config
        hidden_size = self.config.hidden_size  # 768 for base model
        
        # Freeze backbone if specified
        if freeze_backbone:
            print("Freezing ViViT backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # ViViT expects: (B, T, C, H, W)
        # But we need to ensure the number of frames matches
        if T != self.num_frames:
            # Sample or repeat frames to match expected number
            if T > self.num_frames:
                # Sample uniformly
                indices = torch.linspace(0, T - 1, self.num_frames).long()
                x = x[:, indices]
            else:
                # Repeat frames
                repeat_factor = (self.num_frames + T - 1) // T
                x = x.repeat(1, repeat_factor, 1, 1, 1)[:, :self.num_frames]
        
        # Extract features using pretrained backbone
        outputs = self.backbone(pixel_values=x)
        
        # Get the [CLS] token representation (or pooled output)
        # Use last_hidden_state and take [CLS] token
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
            cls_output = sequence_output[:, 0]  # Take [CLS] token
        elif hasattr(outputs, 'pooler_output'):
            cls_output = outputs.pooler_output
        else:
            raise ValueError("Model output doesn't have expected attributes")
        
        # Classification
        logits = self.classifier(cls_output)
        logits = logits.squeeze(-1)
        return logits


class CNN2DTransformer(nn.Module):
    """Hybrid model: 2D CNN feature extractor + Transformer for temporal modeling.
    Extracts features from each frame independently, then processes temporal dependencies.
    Input shape: (B, T, C, H, W) where T is number of frames.
    """
    
    def __init__(self, pretrained_weights=EfficientNet_V2_S_Weights.DEFAULT, 
                 freeze_backbone=True, num_frames=8, embed_dim=512, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # 2D Feature extractor (EfficientNet)
        efficientnet = efficientnet_v2_s(weights=pretrained_weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # EfficientNet_V2_S outputs 1280 features
        cnn_features = 1280
        
        # Project CNN features to transformer embedding dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout)
        )
        
        # Temporal positional encoding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        # Extract 2D features from each frame
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B*T, cnn_features)
        
        # Reshape back to sequence: (B, T, cnn_features)
        x = x.reshape(B, T, -1)
        
        # Project to transformer dimension: (B, T, embed_dim)
        x = self.feature_projection(x)
        
        # Add temporal positional encoding
        x = x + self.temporal_pos_embed
        
        # Add CLS token: (B, T+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Take CLS token output: (B, embed_dim)
        x = x[:, 0]
        
        # Layer norm
        x = self.norm(x)
        
        # Classification
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x


def get_model(backbone='efficientnet', **kwargs):
    """
    Factory function to create models based on backbone choice.
    
    Args:
        backbone: One of ['efficientnet', '3dresnet', 'videomae', 'vivit', 'cnn_transformer']
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    if backbone == 'efficientnet':
        return EfficientNetFeatureExtractor(**kwargs)
    elif backbone == '3dresnet':
        return ResNet3D(**kwargs)
    elif backbone == 'videomae':
        return VideoMAE(**kwargs)
    elif backbone == 'vivit':
        return ViViT(**kwargs)
    elif backbone == 'cnn_transformer':
        return CNN2DTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from: "
                        f"['efficientnet', '3dresnet', 'videomae', 'vivit', 'cnn_transformer']")

