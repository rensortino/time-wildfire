import glob
import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_lbp, get_motion_image, norm_01


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
    def __init__(self, root_dir, img_size=224, transform=None, crop_margin=1.2, return_torch=True):
        self.transform = transform
        self.sets = glob.glob(f"{root_dir}/**/*")
        random.shuffle(self.sets)
        self.img_size = img_size
        self.crop_margin = crop_margin
        self.label2name = {
            0: "no_fire",
            1: "fire",
        }
        self.return_torch = return_torch

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        img_folder = self.sets[idx]
        img_list = glob.glob(f"{img_folder}/*.jpg")
        img_list.sort()

        cls_label = int(img_folder.split(os.path.sep)[-2]) 

        images = [Image.open(file) for file in img_list]
        w, h = images[0].size

        # Collect labels for bounding boxes (assuming one label per image)
        labels = []
        for file in img_list:
            label_file = file.replace("images", "labels").replace(".jpg", ".txt")
            with open(label_file, "r") as f:
                lines = f.readlines()

            # Assuming the first line in each label file contains the necessary bounding box info
            labels.append(np.array(lines[0].split(" ")[1:5]).astype("float"))

        labels = np.array(labels)

        labels = xywh2xyxy(labels)

        x0, y0 = np.min(labels[:, :2], 0)
        x1, y1 = np.max(labels[:, 2:], 0)

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

        # Crop, resize, and transform each image in the sequence
        for im in images:
            cropped_image = im.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            if crop_size > self.img_size:
                cropped_image = cropped_image.resize((self.img_size, self.img_size))

            if self.transform:
                cropped_image = self.transform(cropped_image)

            img_sequence.append(cropped_image)

        # Stack the images into a tensor with shape (sequence_length, C, H, W)
        if self.return_torch:
            img_sequence = torch.stack(img_sequence, dim=0)

        return img_sequence, cls_label  # Adjust label as necessary


class FireMotionDataset(FireSeriesDataset):
    def __init__(self, root_dir, img_size=224, transform=None, crop_margin=1.2):
        super().__init__(root_dir, img_size, transform=None, crop_margin=crop_margin, return_torch=False)
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
    ds = FireMotionDataset("data/images/train")
    from torchvision.utils import save_image

    for el in ds:
        imgs, label = el
        print(imgs.shape, label)
        # imgs = imgs / 255
        save_image(imgs, "tmp.png", normalize=True)
        break
