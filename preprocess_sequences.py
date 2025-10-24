import glob
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from itertools import combinations

from utils import get_lbp, get_motion_image


def xywh2xyxy(x: np.array):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def preprocess_sequence(img_folder, output_folder, img_size=224, crop_margin=1.2):
    """
    Process a single sequence folder and generate motion image.
    
    Args:
        img_folder: Path to the sequence folder containing images
        output_folder: Path to save the motion image
        img_size: Target image size
        crop_margin: Margin for cropping around bounding box
    """
    # Get list of images
    img_list = glob.glob(f"{img_folder}/*.jpg")
    img_list.sort()
    
    if len(img_list) < 2:
        print(f"Warning: Sequence {img_folder} has less than 2 images, skipping")
        return
    
    # Load images
    images = [Image.open(file) for file in img_list]
    w, h = images[0].size
    
    # Collect labels for bounding boxes
    labels = []
    for file in img_list:
        label_file = file.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} not found, skipping sequence")
            return
        
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print(f"Warning: Empty label file {label_file}, skipping sequence")
            return
        
        # Assuming the first line in each label file contains the necessary bounding box info
        labels.append(np.array(lines[0].split(" ")[1:5]).astype("float"))
    
    labels = np.array(labels)
    labels = xywh2xyxy(labels)
    
    # Calculate crop region based on all bounding boxes in sequence
    x0, y0 = np.min(labels[:, :2], 0)
    x1, y1 = np.max(labels[:, 2:], 0)
    
    x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    xc = x0 + (x1 - x0) / 2
    yc = y0 + (y1 - y0) / 2
    crop_size = max(x1 - x0, y1 - y0) * crop_margin
    
    if crop_size < img_size:
        crop_x0 = max(int(xc - img_size / 2), 0)
        crop_x1 = min(int(xc + img_size / 2), w)
        crop_y0 = max(int(yc - img_size / 2), 0)
        crop_y1 = min(int(yc + img_size / 2), h)
    else:
        crop_x0 = max(int(xc - crop_size / 2), 0)
        crop_x1 = min(int(xc + crop_size / 2), w)
        crop_y0 = max(int(yc - crop_size / 2), 0)
        crop_y1 = min(int(yc + crop_size / 2), h)
    
    # Crop and resize all images in sequence
    processed_images = []
    for im in images:
        cropped_image = im.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        if crop_size > img_size:
            cropped_image = cropped_image.resize((img_size, img_size))
        processed_images.append(np.array(cropped_image))
    
    # Convert to grayscale and compute LBP
    images_gray = []
    images_lbp = []
    for image in processed_images:
        gs_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = get_lbp(gs_image.squeeze())
        images_gray.append(gs_image)
        images_lbp.append(lbp)
    
    # Generate all possible temporally ordered frame combinations
    num_frames = len(images_gray)
    frame_indices = list(range(num_frames))
    
    # Get all pairs (i, j) where i < j
    all_combinations = list(combinations(frame_indices, 2))
    
    # Limit to maximum of 10 combinations
    if len(all_combinations) > 10:
        all_combinations = all_combinations[:10]
    
    # Generate and save motion images for each combination
    os.makedirs(output_folder, exist_ok=True)
    for prev_idx, next_idx in all_combinations:
        prev = images_gray[prev_idx]
        next = images_gray[next_idx]
        motion_image = get_motion_image(prev, next, images_lbp[prev_idx])
        
        # Save with unique filename indicating frame indices
        output_path = os.path.join(output_folder, f"motion_{prev_idx}_{next_idx}.jpg")
        Image.fromarray(motion_image).save(output_path)


def preprocess_dataset(root_dir, output_root, img_size=224, crop_margin=1.2):
    """
    Preprocess entire dataset (train or val).
    
    Args:
        root_dir: Root directory containing class folders (e.g., data/images/train)
        output_root: Output root directory (e.g., data/motion_images/train)
        img_size: Target image size
        crop_margin: Margin for cropping around bounding box
    """
    # Get all sequence folders
    sequence_folders = glob.glob(f"{root_dir}/**/*")
    sequence_folders = [f for f in sequence_folders if os.path.isdir(f)]
    
    print(f"Processing {len(sequence_folders)} sequences from {root_dir}")
    
    for seq_folder in tqdm(sequence_folders, desc=f"Processing {root_dir}"):
        # Construct output path maintaining the same structure
        relative_path = os.path.relpath(seq_folder, root_dir)
        output_folder = os.path.join(output_root, relative_path)
        
        try:
            preprocess_sequence(seq_folder, output_folder, img_size, crop_margin)
        except Exception as e:
            print(f"Error processing {seq_folder}: {str(e)}")
            continue


def main():
    """Main function to preprocess both train and val datasets."""
    img_size = 224
    crop_margin = 1.2
    
    # Process training set
    print("=" * 80)
    print("Processing Training Set")
    print("=" * 80)
    train_dir = "data/images/train"
    train_output = "data/motion_images/train"
    preprocess_dataset(train_dir, train_output, img_size, crop_margin)
    
    # Process validation set
    print("\n" + "=" * 80)
    print("Processing Validation Set")
    print("=" * 80)
    val_dir = "data/images/val"
    val_output = "data/motion_images/val"
    preprocess_dataset(val_dir, val_output, img_size, crop_margin)
    
    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"Motion images saved to:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")


if __name__ == "__main__":
    main()

