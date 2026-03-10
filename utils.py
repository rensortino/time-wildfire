import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import cv2
from typing import List, Dict


def norm_01(img):
    return (img - img.min()) / (img.max() - img.min())

def to_uint8(image):
    return (image.permute(1,2,0).numpy()*255).astype(np.uint8)

def to_grayscale(image):
    image = norm_01(image)
    image = to_uint8(image)
    return np.array(Image.fromarray(image).convert("L"))

def get_lbp(image, radius=3, n_points=8):
    n_points = radius * n_points
    lbp = local_binary_pattern(image, n_points, radius)
    return norm_01(lbp)

def get_optical_flow_map(prev, next):
    """prev and next should be np.array with shape (H,W) and range [0,255], dtype=np.uint8
    
    # """
    prev = cv2.GaussianBlur(prev, (5,5), 1)
    next = cv2.GaussianBlur(next, (5,5), 1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    prev = clahe.apply(prev)
    next = clahe.apply(next)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnite and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    ang_map = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    mag_map = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to rgb
    # rgb_representation = cv2.cvtColor(motion_image, cv2.COLOR_HSV2RGB)
    return ang_map, mag_map


def get_motion_image(prev, next, lbp_map):
    motion_image = np.zeros(lbp_map.shape, dtype=np.uint8)
    ang, mag = get_optical_flow_map(prev, next)
    motion_image = np.concatenate((ang[...,None], mag[...,None], (lbp_map[...,None]*255)), axis=2)
    motion_image = motion_image.astype(np.uint8)

    return cv2.cvtColor(motion_image, cv2.COLOR_HSV2RGB)

def xywh2xyxy_normalized(bbox_xywh: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized xywh bbox to pixel coordinates xyxy format.
    
    Args:
        bbox_xywh: Normalized bbox [x_center, y_center, width, height] in [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Bbox in pixel coordinates [x_min, y_min, x_max, y_max]
    """
    x_center, y_center, width, height = bbox_xywh
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to xyxy format
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2
    
    return np.array([x_min, y_min, x_max, y_max])


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: Bounding box [x_min, y_min, x_max, y_max]
        bbox2: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using the 11-point interpolation method.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
    
    Returns:
        Average Precision value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def calculate_ap_metrics(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                         iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate AP metrics for tracking results.
    
    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        Dictionary with AP metrics
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'ap': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Calculate IoU for each prediction
    ious = []
    fn = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.size > 0 and gt.size > 0:
            iou = calculate_iou(pred, gt)
            ious.append(iou)
        elif pred.size == 0 and gt.size > 0:
            ious.append(0.0)
            fn += 1
        else:
            ious.append(0.0)
    
    ious = np.array(ious)
    
    # Calculate precision and recall
    tp = np.sum(ious >= iou_threshold)
    fp = len(predictions) - tp - fn
    # fn = len(ground_truths) - tp
    # fn = abs(len(ground_truths) - len(predictions))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AP using different IoU thresholds
    ap_values = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        tp_at_threshold = np.sum(ious >= threshold)
        fp_at_threshold = len(predictions) - tp_at_threshold
        
        prec_at_threshold = tp_at_threshold / (tp_at_threshold + fp_at_threshold) if (tp_at_threshold + fp_at_threshold) > 0 else 0.0
        
        ap_values.append(prec_at_threshold)
    
    ap = np.mean(ap_values) if ap_values else 0.0
    
    return {
        'ap': float(ap),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_iou': float(np.mean(ious)) if len(ious) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def mask_to_bbox(mask) -> np.ndarray:
    """
    Convert a mask to a bounding box.
    
    Args:
        mask: Mask can be a numpy array, dict with mask data, or RLE encoded
    
    Returns:
        Bounding box [x_min, y_min, x_max, y_max] or empty array
    """
    if mask is None:
        return np.array([])
    
    # If mask is a dict, try to extract the actual mask data
    if isinstance(mask, dict):
        if "mask" in mask:
            mask = mask["mask"]
        elif "segmentation" in mask:
            mask = mask["segmentation"]
        elif "rle" in mask:
            # Handle RLE encoding if needed (would require pycocotools)
            return np.array([])
    
    # Convert mask to numpy array if it's not already
    if not isinstance(mask, np.ndarray):
        try:
            # Handle torch tensors
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif hasattr(mask, 'numpy'):
                mask = mask.numpy()
            else:
                mask = np.array(mask)
        except Exception:
            return np.array([])
    
    # Handle different mask formats
    if mask.ndim == 2:
        # Binary mask
        coords = np.where(mask > 0)
    elif mask.ndim == 3:
        # Multi-channel mask, use first channel or combine
        if mask.shape[2] == 1:
            coords = np.where(mask[:, :, 0] > 0)
        else:
            # Combine all channels
            mask_combined = np.any(mask > 0, axis=2)
            coords = np.where(mask_combined)
    else:
        return np.array([])
    
    if len(coords[0]) > 0:
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        return np.array([x_min, y_min, x_max, y_max])
    else:
        return np.array([])


def bbox_to_points(bbox: np.ndarray) -> List[List[int]]:
    """
    Convert bounding box to center point for SAM3 tracking.
    SAM3 uses points as prompts, so we'll use the center of the bbox.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        List of point coordinates [[x, y]]
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    return [[center_x, center_y]]
