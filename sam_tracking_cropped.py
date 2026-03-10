#!/usr/bin/env python3
"""
Test SAM3 object tracking capabilities on the validation dataset with cropped inputs.
For each sequence, uses the first frame's bounding box as grounding and tracks the 
object in subsequent frames. Images are cropped around the bounding box to 1008x1008
before feeding to SAM3.

Three cropping methods are available:
1. 'padded': Pad equally in all directions from the bounding box to reach 1008x1008,
   shifting to avoid going outside image boundaries while keeping the bbox fully contained.
2. 'resize': Crop the bounding box + 20% padding, then resize to 1008x1008.
3. 'balanced': Crop with at least 200px padding, max 512x512 crop, then resize to 1008x1008.
   This is the default method, designed for 1280x720 images to enlarge the object while
   maintaining sufficient context.

Usage:
    python sam_tracking_cropped.py --csv-path data/val_dataset/sampled.csv
    python sam_tracking_cropped.py --crop-method balanced --target-size 1008
    python sam_tracking_cropped.py --crop-method padded
    python sam_tracking_cropped.py --crop-method resize
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from typing import List, Dict
from utils import xywh2xyxy_normalized, mask_to_bbox, calculate_ap_metrics

try:
    from transformers import Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
    import torch
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 transformers not available. Please install: pip install transformers")




def crop_around_bbox_padded(image: Image.Image, bbox: np.ndarray, 
                             target_size: int = 1008) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Crop image around bounding box with equal padding to reach target_size x target_size.
    Shifts the crop window to avoid going outside image boundaries while keeping bbox fully contained.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixel coordinates
        target_size: Target size for the crop (default: 1008)
    
    Returns:
        Tuple of (cropped_image, transformed_bbox, crop_coords)
        - cropped_image: PIL Image of size target_size x target_size
        - transformed_bbox: Bounding box in the cropped image coordinates
        - crop_coords: [crop_x_min, crop_y_min, crop_x_max, crop_y_max] for reverse transformation
    """
    img_width, img_height = image.size
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate bbox center
    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2
    
    # Calculate initial crop window centered on bbox
    half_size = target_size / 2
    crop_x_min = bbox_center_x - half_size
    crop_y_min = bbox_center_y - half_size
    crop_x_max = bbox_center_x + half_size
    crop_y_max = bbox_center_y + half_size
    
    # Shift to avoid going outside image boundaries
    if crop_x_min < 0:
        shift_x = -crop_x_min
        crop_x_min += shift_x
        crop_x_max += shift_x
    elif crop_x_max > img_width:
        shift_x = img_width - crop_x_max
        crop_x_min += shift_x
        crop_x_max += shift_x
    
    if crop_y_min < 0:
        shift_y = -crop_y_min
        crop_y_min += shift_y
        crop_y_max += shift_y
    elif crop_y_max > img_height:
        shift_y = img_height - crop_y_max
        crop_y_min += shift_y
        crop_y_max += shift_y
    
    # Clamp to image boundaries (in case image is smaller than target_size)
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(img_width, crop_x_max)
    crop_y_max = min(img_height, crop_y_max)
    
    # Crop the image
    crop_coords = np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max])
    cropped = image.crop((int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)))
    
    # If cropped image is smaller than target_size (edge case for small images), pad it
    if cropped.size[0] < target_size or cropped.size[1] < target_size:
        # Create a black canvas of target_size
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        # Paste the cropped image in the center
        paste_x = (target_size - cropped.size[0]) // 2
        paste_y = (target_size - cropped.size[1]) // 2
        padded.paste(cropped, (paste_x, paste_y))
        cropped = padded
        # Adjust crop_coords to account for padding
        crop_x_min -= paste_x
        crop_y_min -= paste_y
        crop_coords = np.array([crop_x_min, crop_y_min, crop_x_min + target_size, crop_y_min + target_size])
    
    # Transform bbox to cropped image coordinates
    transformed_bbox = bbox - np.array([crop_x_min, crop_y_min, crop_x_min, crop_y_min])
    
    return cropped, transformed_bbox, crop_coords


def crop_around_bbox_resize(image: Image.Image, bbox: np.ndarray, 
                             padding_percent: float = 0.20, 
                             target_size: int = 1008) -> tuple[Image.Image, np.ndarray, np.ndarray, float]:
    """
    Crop around bounding box with fixed padding percentage and resize to target_size x target_size.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixel coordinates
        padding_percent: Percentage of bbox size to add as padding (default: 0.20 = 20%)
        target_size: Target size for the resized crop (default: 1008)
    
    Returns:
        Tuple of (cropped_image, transformed_bbox, crop_coords, scale_factor)
        - cropped_image: PIL Image of size target_size x target_size
        - transformed_bbox: Bounding box in the resized image coordinates
        - crop_coords: [crop_x_min, crop_y_min, crop_x_max, crop_y_max] for reference
        - scale_factor: Ratio of target_size to original crop size
    """
    img_width, img_height = image.size
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate bbox dimensions
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Add padding
    padding_x = bbox_width * padding_percent
    padding_y = bbox_height * padding_percent
    
    crop_x_min = x_min - padding_x
    crop_y_min = y_min - padding_y
    crop_x_max = x_max + padding_x
    crop_y_max = y_max + padding_y
    
    # Clamp to image boundaries
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(img_width, crop_x_max)
    crop_y_max = min(img_height, crop_y_max)
    
    # Crop the image
    crop_coords = np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max])
    cropped = image.crop((int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)))
    
    # Calculate scale factor for resizing
    crop_width = crop_x_max - crop_x_min
    crop_height = crop_y_max - crop_y_min
    scale_factor = target_size / max(crop_width, crop_height)
    
    # Resize to target_size x target_size (may not be exactly square if aspect ratio differs)
    new_width = int(crop_width * scale_factor)
    new_height = int(crop_height * scale_factor)
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # If not exactly target_size x target_size, pad to make it square
    if new_width != target_size or new_height != target_size:
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))
        resized = padded
        
        # Transform bbox to resized image coordinates (accounting for padding)
        transformed_bbox = (bbox - np.array([crop_x_min, crop_y_min, crop_x_min, crop_y_min])) * scale_factor
        transformed_bbox += np.array([paste_x, paste_y, paste_x, paste_y])
    else:
        # Transform bbox to resized image coordinates
        transformed_bbox = (bbox - np.array([crop_x_min, crop_y_min, crop_x_min, crop_y_min])) * scale_factor
    
    return resized, transformed_bbox, crop_coords, scale_factor


def crop_around_bbox_balanced(image: Image.Image, bbox: np.ndarray,
                               min_padding: int = 200,
                               max_crop_size: int = 512,
                               target_size: int = 1008) -> tuple[Image.Image, np.ndarray, np.ndarray, float]:
    """
    Crop around bounding box with balanced padding to maintain context while enlarging the object.
    Uses at least min_padding pixels of padding, but caps the crop size at max_crop_size.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixel coordinates
        min_padding: Minimum padding in pixels around the bbox (default: 200)
        max_crop_size: Maximum size of the crop before resizing (default: 512)
        target_size: Target size for the resized crop (default: 1008)
    
    Returns:
        Tuple of (cropped_image, transformed_bbox, crop_coords, scale_factor)
        - cropped_image: PIL Image of size target_size x target_size
        - transformed_bbox: Bounding box in the resized image coordinates
        - crop_coords: [crop_x_min, crop_y_min, crop_x_max, crop_y_max] for reference
        - scale_factor: Ratio of target_size to original crop size
    """
    img_width, img_height = image.size
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate bbox center and dimensions
    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Start with bbox + minimum padding
    crop_width = bbox_width + 2 * min_padding
    crop_height = bbox_height + 2 * min_padding
    
    # Cap at max_crop_size while maintaining aspect ratio
    if crop_width > max_crop_size or crop_height > max_crop_size:
        # Use max_crop_size for both dimensions (square crop)
        crop_width = max_crop_size
        crop_height = max_crop_size
    else:
        # Make it square by using the larger dimension
        crop_size = max(crop_width, crop_height)
        crop_width = crop_size
        crop_height = crop_size
    
    # Calculate crop window centered on bbox
    half_width = crop_width / 2
    half_height = crop_height / 2
    crop_x_min = bbox_center_x - half_width
    crop_y_min = bbox_center_y - half_height
    crop_x_max = bbox_center_x + half_width
    crop_y_max = bbox_center_y + half_height
    
    # Shift to avoid going outside image boundaries
    if crop_x_min < 0:
        shift_x = -crop_x_min
        crop_x_min += shift_x
        crop_x_max += shift_x
    elif crop_x_max > img_width:
        shift_x = img_width - crop_x_max
        crop_x_min += shift_x
        crop_x_max += shift_x
    
    if crop_y_min < 0:
        shift_y = -crop_y_min
        crop_y_min += shift_y
        crop_y_max += shift_y
    elif crop_y_max > img_height:
        shift_y = img_height - crop_y_max
        crop_y_min += shift_y
        crop_y_max += shift_y
    
    # Final clamp to image boundaries
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(img_width, crop_x_max)
    crop_y_max = min(img_height, crop_y_max)
    
    # Crop the image
    crop_coords = np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max])
    cropped = image.crop((int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)))
    
    # Calculate actual crop dimensions
    actual_crop_width = crop_x_max - crop_x_min
    actual_crop_height = crop_y_max - crop_y_min
    
    # Calculate scale factor for resizing
    scale_factor = target_size / max(actual_crop_width, actual_crop_height)
    
    # Resize to target_size x target_size
    new_width = int(actual_crop_width * scale_factor)
    new_height = int(actual_crop_height * scale_factor)
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Pad to make it exactly target_size x target_size if needed
    if new_width != target_size or new_height != target_size:
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))
        resized = padded
        
        # Transform bbox to resized image coordinates (accounting for padding)
        transformed_bbox = (bbox - np.array([crop_x_min, crop_y_min, crop_x_min, crop_y_min])) * scale_factor
        transformed_bbox += np.array([paste_x, paste_y, paste_x, paste_y])
    else:
        # Transform bbox to resized image coordinates
        transformed_bbox = (bbox - np.array([crop_x_min, crop_y_min, crop_x_min, crop_y_min])) * scale_factor
    
    return resized, transformed_bbox, crop_coords, scale_factor


def track_sequence_sam3(image_paths: List[str], first_frame_bbox: np.ndarray, 
                        root_dir: str, processor, model, device: str,
                        crop_method: str = 'balanced', target_size: int = 1008) -> List[np.ndarray]:
    """
    Track an object in a sequence of images using SAM3 from Hugging Face.
    Images are cropped around the bounding box before feeding to SAM3.
    
    Args:
        image_paths: List of relative image paths
        first_frame_bbox: Bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
        root_dir: Root directory for images
        processor: SAM3TrackerVideoProcessor instance
        model: SAM3TrackerVideoModel instance
        device: Device to run inference on ('cuda' or 'cpu')
        crop_method: Cropping method ('padded', 'resize', or 'balanced')
            - 'padded': Pad equally in all directions to reach target_size
            - 'resize': Crop with 20% padding and resize to target_size
            - 'balanced': Crop with min 200px padding, max 512x512, then resize (default)
        target_size: Target size for cropped images (default: 1008)
    
    Returns:
        List of predicted bounding boxes for each frame (in original image coordinates)
    """
    # Load all images as PIL Images
    full_image_paths = [os.path.join(root_dir, img_path) for img_path in image_paths]
    original_images = []
    original_sizes = []
    
    for img_path in full_image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            original_sizes.append(img.size)  # (width, height)
            original_images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
    
    if len(original_images) == 0:
        return [np.array([]) for _ in range(len(image_paths))]
    
    # Crop all images around the first frame bounding box
    video_frames = []
    crop_transforms = []  # Store transform info for each frame
    
    for idx, img in enumerate(original_images):
        if crop_method == 'padded':
            cropped_img, transformed_bbox, crop_coords = crop_around_bbox_padded(
                img, first_frame_bbox, target_size=target_size
            )
            crop_transforms.append({
                'crop_coords': crop_coords,
                'scale_factor': 1.0,  # No scaling in padded method
                'transformed_bbox': transformed_bbox
            })
        elif crop_method == 'resize':
            cropped_img, transformed_bbox, crop_coords, scale_factor = crop_around_bbox_resize(
                img, first_frame_bbox, padding_percent=0.20, target_size=target_size
            )
            crop_transforms.append({
                'crop_coords': crop_coords,
                'scale_factor': scale_factor,
                'transformed_bbox': transformed_bbox
            })
        elif crop_method == 'balanced':
            cropped_img, transformed_bbox, crop_coords, scale_factor = crop_around_bbox_balanced(
                img, first_frame_bbox, min_padding=200, max_crop_size=512, target_size=target_size
            )
            crop_transforms.append({
                'crop_coords': crop_coords,
                'scale_factor': scale_factor,
                'transformed_bbox': transformed_bbox
            })
        else:
            raise ValueError(f"Unknown crop_method: {crop_method}. Use 'padded', 'resize', or 'balanced'")
        
        video_frames.append(cropped_img)
    
    # Use the transformed bbox from the first frame
    first_frame_bbox_transformed = crop_transforms[0]['transformed_bbox']
    
    # Initialize video session with cropped frames
    try:
        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            dtype=torch.float32,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SAM3 video session: {e}")
    
    # Convert transformed bbox to box prompt format
    # Format: input_boxes needs 3 levels: image level, box level, box coordinates
    # Box coordinates: x_min, y_min, x_max, y_max
    x_min, y_min, x_max, y_max = first_frame_bbox_transformed
    # 3 levels: image level, box level, box coordinates
    box_coords = [float(x_min), float(y_min), float(x_max), float(y_max)]
    input_boxes = [[box_coords]]
    obj_ids = [1]  # Single object to track
    
    # Add bounding box input to inference session for first frame (condition only, no prediction)
    try:
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add inputs to SAM3 session: {e}")
    
    # Store results for all frames (in cropped coordinates)
    predicted_bboxes_cropped = []
    
    # Store the first frame bbox directly (it's the condition, not a prediction)
    predicted_bboxes_cropped.append(first_frame_bbox_transformed.copy())
    
    # Propagate tracking through remaining frames (starting from frame 1, skipping frame 0)
    try:
        for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session, start_frame_idx=0):
            frame_idx = sam3_tracker_video_output.frame_idx
            
            # Skip frame 0 since we only added the bounding box condition, no prediction
            if frame_idx == 0:
                continue
            
            video_res_masks = processor.post_process_masks(
                [sam3_tracker_video_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=True
            )[0]
            
            # Convert mask to bbox (in cropped image coordinates)
            if len(video_res_masks) > 0:
                mask = video_res_masks[0]
                bbox = mask_to_bbox(mask.squeeze())
            else:
                bbox = np.array([])
            
            # Ensure we have the right number of bboxes
            while len(predicted_bboxes_cropped) <= frame_idx:
                predicted_bboxes_cropped.append(np.array([]))
            
            predicted_bboxes_cropped[frame_idx] = bbox
    except Exception as e:
        raise RuntimeError(f"Failed to propagate tracking with SAM3: {e}")
    
    # Ensure we have a bbox for each frame
    while len(predicted_bboxes_cropped) < len(image_paths):
        predicted_bboxes_cropped.append(np.array([]))
    
    # Transform predicted bboxes back to original image coordinates
    predicted_bboxes = []
    for idx, bbox_cropped in enumerate(predicted_bboxes_cropped):
        if bbox_cropped.size == 0:
            predicted_bboxes.append(np.array([]))
            continue
        
        transform_info = crop_transforms[idx]
        crop_coords = transform_info['crop_coords']
        scale_factor = transform_info['scale_factor']
        
        if crop_method == 'padded':
            # Simply add back the crop offset
            bbox_original = bbox_cropped + np.array([crop_coords[0], crop_coords[1], 
                                                      crop_coords[0], crop_coords[1]])
        elif crop_method in ['resize', 'balanced']:
            # First, handle potential padding in the resized image
            crop_width = crop_coords[2] - crop_coords[0]
            crop_height = crop_coords[3] - crop_coords[1]
            new_width = int(crop_width * scale_factor)
            new_height = int(crop_height * scale_factor)
            
            if new_width != target_size or new_height != target_size:
                paste_x = (target_size - new_width) // 2
                paste_y = (target_size - new_height) // 2
                # Remove padding offset
                bbox_cropped = bbox_cropped - np.array([paste_x, paste_y, paste_x, paste_y])
            
            # Scale back to original crop size
            bbox_original = bbox_cropped / scale_factor
            # Add back the crop offset
            bbox_original = bbox_original + np.array([crop_coords[0], crop_coords[1], 
                                                       crop_coords[0], crop_coords[1]])
        
        predicted_bboxes.append(bbox_original)
    
    return predicted_bboxes


def main(csv_path, crop_method: str = 'balanced', target_size: int = 1008):
    """
    Main function to test SAM3 tracking.
    
    Args:
        csv_path: Path to the CSV file containing the dataset
        crop_method: Cropping method ('padded', 'resize', or 'balanced')
            - 'padded': Pad equally in all directions to reach target_size
            - 'resize': Crop with 20% padding and resize to target_size
            - 'balanced': Crop with min 200px padding, max 512x512, then resize (default)
        target_size: Target size for cropped images (default: 1008)
    """
    if not SAM3_AVAILABLE:
        print("Error: SAM3 transformers is not available. Please install it first:")
        print("  pip install transformers")
        return
    
    # Configuration
    root_dir = os.path.dirname(csv_path)
    
    print(f"Loading dataset from: {csv_path}")
    print(f"Using crop method: {crop_method}")
    print(f"Target size: {target_size}x{target_size}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Process bbox column: convert string to numpy array
    def process_bbox(bbox_str):
        if pd.isna(bbox_str) or bbox_str == '':
            return np.array([])
        bbox_array = np.array(bbox_str.split(), dtype=np.float32)
        return bbox_array  # Keep as normalized xywh
    
    df['bbox'] = df['bbox'].apply(process_bbox)
    
    # Group by sequence
    grouped = df.groupby('sequence', sort=False)
    
    print(f"Found {len(grouped)} sequences")
    
    # Initialize SAM3 from Hugging Face
    print("Initializing SAM3 from Hugging Face...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        processor = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
        model = Sam3TrackerVideoModel.from_pretrained('facebook/sam3')
        model = model.to(device)
        model.eval()
        print("SAM3 model loaded successfully")
    except Exception as e:
        print(f"Error loading SAM3 model: {e}")
        print("Make sure you have transformers installed: pip install transformers")
        return
    
    # Process each sequence
    all_metrics = []
    sequence_results = []
    
    for sequence_name, group_df in tqdm(grouped, desc="Processing sequences"):
        # Sort by image path to maintain temporal order
        sequence_df = group_df.sort_values('image_path').reset_index(drop=True)
        
        if len(sequence_df) < 2:
            print(f"Skipping sequence {sequence_name}: less than 2 frames")
            continue
        
        # Get first frame info
        first_row = sequence_df.iloc[0]
        first_image_path = first_row['image_path']
        first_bbox_normalized = first_row['bbox']
        
        if first_bbox_normalized.size == 0:
            print(f"Skipping sequence {sequence_name}: no bbox in first frame")
            continue
        
        # Load first image to get dimensions
        first_image_full_path = os.path.join(root_dir, first_image_path)
        try:
            first_image = Image.open(first_image_full_path)
            img_width, img_height = first_image.size
        except Exception as e:
            print(f"Error loading first image {first_image_full_path}: {e}")
            continue
        
        # Convert first frame bbox to pixel coordinates
        first_bbox_px = xywh2xyxy_normalized(first_bbox_normalized, img_width, img_height)
        
        # Get all image paths in sequence
        image_paths = sequence_df['image_path'].tolist()
        
        # Get ground truth bboxes for all frames
        ground_truth_bboxes = []
        for _, row in sequence_df.iterrows():
            bbox_normalized = row['bbox']
            if bbox_normalized.size > 0:
                # Load image to get dimensions (assuming same size)
                img_path = os.path.join(root_dir, row['image_path'])
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    bbox_px = xywh2xyxy_normalized(bbox_normalized, w, h)
                    ground_truth_bboxes.append(bbox_px)
                except Exception:
                    ground_truth_bboxes.append(np.array([]))
            else:
                ground_truth_bboxes.append(np.array([]))
        
        # Track using SAM3
        try:
            predicted_bboxes = track_sequence_sam3(
                image_paths, 
                first_bbox_px, 
                root_dir, 
                processor,
                model,
                device,
                crop_method=crop_method,
                target_size=target_size
            )
        except Exception as e:
            print(f"Error tracking sequence {sequence_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Ensure predicted and ground truth bboxes have the same length
        num_frames = len(image_paths)
        if len(predicted_bboxes) != num_frames:
            print(f"Warning: predicted_bboxes length ({len(predicted_bboxes)}) != num_frames ({num_frames})")
            # Pad or truncate to match
            if len(predicted_bboxes) < num_frames:
                predicted_bboxes.extend([np.array([])] * (num_frames - len(predicted_bboxes)))
            else:
                predicted_bboxes = predicted_bboxes[:num_frames]
        
        if len(ground_truth_bboxes) != num_frames:
            print(f"Warning: ground_truth_bboxes length ({len(ground_truth_bboxes)}) != num_frames ({num_frames})")
            # Pad or truncate to match
            if len(ground_truth_bboxes) < num_frames:
                ground_truth_bboxes.extend([np.array([])] * (num_frames - len(ground_truth_bboxes)))
            else:
                ground_truth_bboxes = ground_truth_bboxes[:num_frames]
        
        # Calculate metrics
        metrics = calculate_ap_metrics(predicted_bboxes, ground_truth_bboxes)
        metrics['sequence'] = sequence_name
        metrics['num_frames'] = len(image_paths)
        
        all_metrics.append(metrics)
        sequence_results.append({
            'sequence': sequence_name,
            'metrics': metrics,
            'predictions': [bbox.tolist() if bbox.size > 0 else [] for bbox in predicted_bboxes],
            'ground_truths': [bbox.tolist() if bbox.size > 0 else [] for bbox in ground_truth_bboxes]
        })
    
    # Calculate overall metrics
    if len(all_metrics) > 0:
        overall_metrics = {
            'mean_ap': np.mean([m['ap'] for m in all_metrics]),
            'mean_precision': np.mean([m['precision'] for m in all_metrics]),
            'mean_recall': np.mean([m['recall'] for m in all_metrics]),
            'mean_f1': np.mean([m['f1'] for m in all_metrics]),
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
            'total_sequences': len(all_metrics),
            'total_tp': sum([m['tp'] for m in all_metrics]),
            'total_fp': sum([m['fp'] for m in all_metrics]),
            'total_fn': sum([m['fn'] for m in all_metrics])
        }
        
        print("\n" + "="*60)
        print("SAM3 TRACKING RESULTS")
        print("="*60)
        print(f"Total sequences processed: {overall_metrics['total_sequences']}")
        print(f"Mean AP: {overall_metrics['mean_ap']:.4f}")
        print(f"Mean Precision: {overall_metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {overall_metrics['mean_recall']:.4f}")
        print(f"Mean F1: {overall_metrics['mean_f1']:.4f}")
        print(f"Mean IoU: {overall_metrics['mean_iou']:.4f}")
        print(f"\nTotal TP: {overall_metrics['total_tp']}")
        print(f"Total FP: {overall_metrics['total_fp']}")
        print(f"Total FN: {overall_metrics['total_fn']}")
        
        # Save results to JSON
        output_file = f"results/sam3_tracking_{crop_method}_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results = {
            'overall_metrics': overall_metrics,
            'per_sequence_metrics': all_metrics,
            'detailed_results': sequence_results,
            'config': {
                'crop_method': crop_method,
                'target_size': target_size
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("No sequences were successfully processed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM3 tracking with cropped images")
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default="data/val_dataset/onesequence.csv",
        help="Path to the CSV file containing the dataset"
    )
    parser.add_argument(
        "--crop-method", 
        type=str, 
        default="balanced", 
        choices=["padded", "resize", "balanced"],
        help="Cropping method: 'padded' (pad equally to reach target size), 'resize' (crop with 20%% padding and resize), or 'balanced' (min 200px padding, max 512x512 crop, then resize - default)"
    )
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=1008,
        help="Target size for cropped images (default: 1008)"
    )
    
    args = parser.parse_args()
    main(args.csv_path, args.crop_method, args.target_size)
