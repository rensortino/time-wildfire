import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import cv2


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