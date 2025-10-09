from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from dataset import FireMotionDataset, get_transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import numpy as np
from utils import get_motion_image
from omegaconf import OmegaConf
import trackio
import random

def collate_fn(batch):
    motion_images, labels = [], []
    for el in batch:
        images, label = el
        images_gray = images['gray']
        images_lbp = images['lbp']

        if random.random() > 0.5: # Random flip
            images_gray = images_gray[:,:,::-1]
            images_lbp = images_lbp[:,:,::-1]

        
        prev_idx = random.randint(0,len(images_gray)-2)
        prev = images_gray[prev_idx]
        next_idx = random.randint(prev_idx+1, len(images_gray)-1)
        next = images_gray[next_idx]
        motion_images.append(
            get_motion_image(prev, next, images_lbp[prev_idx])
        )
        labels.append(label)

    motion_images = torch.tensor(motion_images)
    motion_images = motion_images.permute(0, 3, 1, 2)
    motion_images = motion_images / 255
    labels = torch.tensor(labels)
    return motion_images, labels

    

def main(args):
    train_transforms = get_transforms(img_size=args.img_size, is_train=True)
    train_dataset = FireMotionDataset("data/images/train", img_size=args.img_size, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

    val_transform = get_transforms(img_size=args.img_size)
    val_dataset = FireMotionDataset("data/images/val", img_size=args.img_size, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).to(args.device)

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
            pred = torch.nn.functional.softmax(logits)
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
    args = {"epochs": 50, "lr": 1e-5, "device": "cuda:0", "img_size": 224}
    args = OmegaConf.create(args)
    main(args)
