from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from dataset import FireSeriesDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import numpy as np
from utils import get_motion_image
import trackio
import random

def collate_fn(batch):
    motion_images, labels = [], []
    for el in batch:
        images, label = el
        images_gray = images['gray']
        images_lbp = images['lbp']
        prev_idx = random.randint(0,len(images_gray)-1)
        prev = images_gray[prev_idx]
        next_idx = np.random.choice(images_gray[prev_idx:])
        next = images_gray[next_idx]
        motion_images.append(
            get_motion_image(prev, next, images_lbp[prev])
        )
        labels.append(label)

        motion_images = torch.tensor(motion_images)
        # TODO Fix this horrible continuous data parsing and apply the correct normalization of the values
        motion_images = motion_images.permute(0, 3, 1, 2) / 255
        return motion_images, labels

    

def main(args):
    train_dataset = FireSeriesDataset("data/images/train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    val_dataset = FireSeriesDataset("data/images/val")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        for i, batch in train_loader:
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
    args = {"epochs": 50, "lr": 1e-5, "device": "cuda:0"}
    main(args)
