import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from utils import get_lbp, get_motion_image, norm_01


def remove_duplicates(sequence_images):
    unique_images = []
    idxs_to_sample = []
    for i, img, in enumerate(sequence_images):
        path = img['image_path']
        if path not in unique_images:
            unique_images.append(path)
            idxs_to_sample.append(i)

    return [sequence_images[i] for i in idxs_to_sample]

def xywh2xyxy(x: np.array):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def inverse_normalize(x, mean, std):
    return x * std + mean

def get_transforms(img_size, is_train=False):
    # TODO Since the calucation of the optical flow is done in the 
    # collate function, here we manually apply the transforms via the functional API
    # and split them in two. Here we apply resize and randomflip, while in the 
    # collate function we apply to tensor and normalize
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size)
            ),  # Resize to the desired img_size
            # transforms.RandomHorizontalFlip(p=0.5), # TODO Apply random flip to all images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ])

class FireSeriesDataset(Dataset):
    def __init__(self, csv_file, img_size=224, transform=None, crop_margin=1.2, return_torch=True, max_images_per_sequence=None):
        self.transform = transform
        self.sets = []
        # Read CSV using pandas
        df = pd.read_csv(csv_file)
        self.root_dir = str(Path(csv_file).parent)
        
        # Process bbox column: convert string to numpy array and apply xywh2xyxy
        def process_bbox(bbox_str):
            if pd.isna(bbox_str) or bbox_str == '':
                return np.array([]).reshape(0, 4)
            bbox_array = np.array(bbox_str.split(), dtype=np.float32)
            # Reshape to (1, 4) if single bbox, then apply transformation
            if bbox_array.ndim == 1:
                bbox_array = bbox_array.reshape(1, -1)
            return xywh2xyxy(bbox_array)
        
        df['bbox'] = df['bbox'].apply(process_bbox)
        
        # Group by sequence and store sequences
        grouped = df.groupby('sequence', sort=False)
        
        # Store each sequence as a list of image dictionaries
        for sequence_name, group_df in grouped:
            sequence_images = []
            for _, row in group_df.iterrows():
                sequence_images.append({
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "bbox": row["bbox"],
                    "cls_name": row["cls_name"],
                    "sequence": row["sequence"]
                })
            self.sets.append(sequence_images)
        
        random.shuffle(self.sets)
        self.img_size = img_size
        self.crop_margin = crop_margin
        self.max_images_per_sequence = max_images_per_sequence
        self.return_torch = return_torch

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        # idx refers to a sequence
        sequence_images = self.sets[idx]
        sequence_images = remove_duplicates(sequence_images)

        # Sample max_images_per_sequence images from the sequence
        if self.max_images_per_sequence is not None and len(sequence_images) > self.max_images_per_sequence:
            sampled_images = random.sample(sequence_images, self.max_images_per_sequence)
            # Sort by image path to maintain temporal order if needed
        else:
            sampled_images = sequence_images
        
        sampled_images = sorted(sampled_images, key=lambda x: x["image_path"])
        # Load first image to get dimensions and compute crop coordinates
        first_image_path = sampled_images[0]["image_path"]
        first_image = Image.open(os.path.join(self.root_dir, first_image_path))
        w, h = first_image.size
        
        # Collect all bboxes from sampled images
        all_bboxes = []
        for img_data in sampled_images:
            bbox = img_data["bbox"]
            if bbox.size > 0:  # Only add non-empty bboxes
                all_bboxes.append(bbox)
        
        if len(all_bboxes) == 0:
            # No bboxes, use full image
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            # Combine all bboxes to find overall bounding box
            combined_bboxes = np.vstack(all_bboxes)
            x0, y0 = np.min(combined_bboxes[:, :2], axis=0)
            x1, y1 = np.max(combined_bboxes[:, 2:], axis=0)
        
        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        xc = x0 + (x1 - x0) / 2
        yc = y0 + (y1 - y0) / 2
        crop_size = max(x1 - x0, y1 - y0) * self.crop_margin

        if crop_size < self.img_size:
            crop_x0 = max(int(xc - self.img_size / 2), 0)
            crop_x1 = min(int(xc + self.img_size / 2), w)
            crop_y0 = max(int(yc - self.img_size / 2), 0)
            crop_y1 = min(int(yc + self.img_size / 2), h)

        else:
            crop_x0 = max(int(xc - crop_size / 2), 0)
            crop_x1 = min(int(xc + crop_size / 2), w)
            crop_y0 = max(int(yc - crop_size / 2), 0)
            crop_y1 = max(int(yc + crop_size / 2), h)

        img_sequence = []
        labels = []

        # Crop, resize, and transform each image in the sequence
        for img_data in sampled_images:
            image_path = img_data["image_path"]
            image = Image.open(os.path.join(self.root_dir, image_path))
            
            cropped_image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            if crop_size > self.img_size:
                cropped_image = cropped_image.resize((self.img_size, self.img_size))

            if self.transform:
                cropped_image = self.transform(cropped_image)

            img_sequence.append(cropped_image)
            
            # Collect label (use first non-empty label if available)
            label = img_data["label"]
            if label != '':
                labels.append(int(label))
        
        is_wildfire = labels[0] == 0

        # Stack the images into a tensor with shape (sequence_length, C, H, W)
        if self.return_torch:
            img_sequence = torch.stack(img_sequence, dim=0)

        return img_sequence, torch.tensor(is_wildfire, dtype=torch.float32)


class FireMotionDataset(FireSeriesDataset):
    def __init__(self, csv_file, img_size=224, transform=None, crop_margin=1.2, max_images_per_sequence=None):
        super().__init__(csv_file, img_size, transform=None, crop_margin=crop_margin, return_torch=False, max_images_per_sequence=max_images_per_sequence)
        self.motion_transform = transform

    def __getitem__(self, idx):
        img_sequence, label = super().__getitem__(idx)

        images = [np.array(img) for img in img_sequence]

        images_gray = []
        images_lbp = []
        for image in images:
            gs_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lbp = get_lbp(gs_image.squeeze())

            if random.random() > 0.5: # Random flip
                gs_image = gs_image[:,::-1]
                lbp = lbp[:,::-1]
            
            images_gray.append(gs_image)
            images_lbp.append(lbp)

        prev_idx = random.randint(0,len(images_gray)-2)
        prev = images_gray[prev_idx]
        next_idx = random.randint(prev_idx+1, len(images_gray)-1)
        next = images_gray[next_idx]
        motion_image = get_motion_image(prev, next, images_lbp[prev_idx])

        motion_image = self.motion_transform(Image.fromarray(motion_image))
        label = torch.tensor(label).float()
        return motion_image, label


if __name__ == "__main__":
    train_transforms = get_transforms(img_size=224, is_train=True)
    ds = FireMotionDataset("data/val_dataset/dataset.csv", max_images_per_sequence=10, transform=train_transforms)
    from torchvision.utils import save_image

    for el in ds:
        imgs, label = el
        print(imgs.shape, label)
        # imgs = imgs / 255
        save_image(imgs, "motion_image.png", normalize=True)
        break
    
    train_transforms = get_transforms(img_size=224, is_train=True)
    ds = FireSeriesDataset("data/val_dataset/dataset.csv", max_images_per_sequence=10, transform=train_transforms)
    from torchvision.utils import save_image

    for el in ds:
        imgs, label = el
        print(imgs.shape, label)
        # imgs = imgs / 255
        save_image(imgs, "series_image.png", normalize=True)
        break
