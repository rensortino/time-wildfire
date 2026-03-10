#!/usr/bin/env python3
"""
Test script to validate cropping functions for SAM3 tracking.
Tests both 'padded' and 'resize' cropping methods.
"""

import numpy as np
from PIL import Image, ImageDraw
import sys

# Import the cropping functions
from sam_tracking_cropped import crop_around_bbox_padded, crop_around_bbox_resize, crop_around_bbox_balanced


def create_test_image(width=1920, height=1080, bbox=None):
    """
    Create a test image with a bounding box drawn on it.
    
    Args:
        width: Image width
        height: Image height
        bbox: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        PIL Image with bbox drawn
    """
    # Create a gradient image
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels[x, y] = (r, g, b)
    
    if bbox is not None:
        draw = ImageDraw.Draw(img)
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    
    return img


def test_padded_method():
    """Test the padded cropping method."""
    print("Testing padded cropping method...")
    
    # Create test image and bbox
    img_width, img_height = 1920, 1080
    bbox = np.array([800.0, 400.0, 1000.0, 600.0])  # [x_min, y_min, x_max, y_max]
    
    img = create_test_image(img_width, img_height, bbox)
    
    # Apply padded cropping
    cropped_img, transformed_bbox, crop_coords = crop_around_bbox_padded(
        img, bbox, target_size=1008
    )
    
    print(f"  Original image size: {img.size}")
    print(f"  Original bbox: {bbox}")
    print(f"  Cropped image size: {cropped_img.size}")
    print(f"  Transformed bbox: {transformed_bbox}")
    print(f"  Crop coords: {crop_coords}")
    
    # Verify cropped image size
    assert cropped_img.size == (1008, 1008), f"Expected (1008, 1008), got {cropped_img.size}"
    
    # Transform bbox back to original coordinates
    bbox_back = transformed_bbox + np.array([crop_coords[0], crop_coords[1], 
                                              crop_coords[0], crop_coords[1]])
    
    print(f"  Bbox transformed back: {bbox_back}")
    
    # Verify the transformation is reversible (within floating point tolerance)
    assert np.allclose(bbox, bbox_back, atol=1.0), f"Transformation not reversible: {bbox} vs {bbox_back}"
    
    # Verify bbox is within cropped image
    assert transformed_bbox[0] >= 0 and transformed_bbox[1] >= 0, \
        f"Transformed bbox has negative coordinates: {transformed_bbox}"
    assert transformed_bbox[2] <= 1008 and transformed_bbox[3] <= 1008, \
        f"Transformed bbox exceeds image bounds: {transformed_bbox}"
    
    print("  ✓ Padded method test passed!")
    return cropped_img, transformed_bbox


def test_resize_method():
    """Test the resize cropping method."""
    print("\nTesting resize cropping method...")
    
    # Create test image and bbox
    img_width, img_height = 1920, 1080
    bbox = np.array([800.0, 400.0, 1000.0, 600.0])  # [x_min, y_min, x_max, y_max]
    
    img = create_test_image(img_width, img_height, bbox)
    
    # Apply resize cropping
    cropped_img, transformed_bbox, crop_coords, scale_factor = crop_around_bbox_resize(
        img, bbox, padding_percent=0.20, target_size=1008
    )
    
    print(f"  Original image size: {img.size}")
    print(f"  Original bbox: {bbox}")
    print(f"  Cropped image size: {cropped_img.size}")
    print(f"  Transformed bbox: {transformed_bbox}")
    print(f"  Crop coords: {crop_coords}")
    print(f"  Scale factor: {scale_factor}")
    
    # Verify cropped image size
    assert cropped_img.size == (1008, 1008), f"Expected (1008, 1008), got {cropped_img.size}"
    
    # Transform bbox back to original coordinates
    # First, handle potential padding
    crop_width = crop_coords[2] - crop_coords[0]
    crop_height = crop_coords[3] - crop_coords[1]
    new_width = int(crop_width * scale_factor)
    new_height = int(crop_height * scale_factor)
    
    bbox_back = transformed_bbox.copy()
    if new_width != 1008 or new_height != 1008:
        paste_x = (1008 - new_width) // 2
        paste_y = (1008 - new_height) // 2
        bbox_back = bbox_back - np.array([paste_x, paste_y, paste_x, paste_y])
    
    # Scale back
    bbox_back = bbox_back / scale_factor
    # Add back crop offset
    bbox_back = bbox_back + np.array([crop_coords[0], crop_coords[1], 
                                       crop_coords[0], crop_coords[1]])
    
    print(f"  Bbox transformed back: {bbox_back}")
    
    # Verify the transformation is reversible (within floating point tolerance)
    assert np.allclose(bbox, bbox_back, atol=1.0), f"Transformation not reversible: {bbox} vs {bbox_back}"
    
    # Verify bbox is within cropped image
    assert transformed_bbox[0] >= 0 and transformed_bbox[1] >= 0, \
        f"Transformed bbox has negative coordinates: {transformed_bbox}"
    assert transformed_bbox[2] <= 1008 and transformed_bbox[3] <= 1008, \
        f"Transformed bbox exceeds image bounds: {transformed_bbox}"
    
    print("  ✓ Resize method test passed!")
    return cropped_img, transformed_bbox


def test_balanced_method():
    """Test the balanced cropping method."""
    print("\nTesting balanced cropping method...")
    
    # Create test image and bbox (simulating 1280x720 image)
    img_width, img_height = 1280, 720
    bbox = np.array([400.0, 200.0, 500.0, 300.0])  # 100x100 bbox
    
    img = create_test_image(img_width, img_height, bbox)
    
    # Apply balanced cropping
    cropped_img, transformed_bbox, crop_coords, scale_factor = crop_around_bbox_balanced(
        img, bbox, min_padding=200, max_crop_size=512, target_size=1008
    )
    
    print(f"  Original image size: {img.size}")
    print(f"  Original bbox: {bbox}")
    print(f"  Cropped image size: {cropped_img.size}")
    print(f"  Transformed bbox: {transformed_bbox}")
    print(f"  Crop coords: {crop_coords}")
    print(f"  Scale factor: {scale_factor}")
    
    # Verify cropped image size
    assert cropped_img.size == (1008, 1008), f"Expected (1008, 1008), got {cropped_img.size}"
    
    # Verify crop size is at most 512x512
    crop_width = crop_coords[2] - crop_coords[0]
    crop_height = crop_coords[3] - crop_coords[1]
    assert crop_width <= 512 and crop_height <= 512, \
        f"Crop size exceeds 512x512: {crop_width}x{crop_height}"
    
    # Verify at least 200px padding (when possible)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    expected_min_crop_width = bbox_width + 2 * 200
    expected_min_crop_height = bbox_height + 2 * 200
    
    # Since we cap at 512, the crop should be the minimum of (bbox + 400px) and 512
    expected_crop_size = min(max(expected_min_crop_width, expected_min_crop_height), 512)
    print(f"  Expected crop size: ~{expected_crop_size}x{expected_crop_size}")
    print(f"  Actual crop size: {crop_width}x{crop_height}")
    
    # Transform bbox back to original coordinates
    crop_width = crop_coords[2] - crop_coords[0]
    crop_height = crop_coords[3] - crop_coords[1]
    new_width = int(crop_width * scale_factor)
    new_height = int(crop_height * scale_factor)
    
    bbox_back = transformed_bbox.copy()
    if new_width != 1008 or new_height != 1008:
        paste_x = (1008 - new_width) // 2
        paste_y = (1008 - new_height) // 2
        bbox_back = bbox_back - np.array([paste_x, paste_y, paste_x, paste_y])
    
    # Scale back
    bbox_back = bbox_back / scale_factor
    # Add back crop offset
    bbox_back = bbox_back + np.array([crop_coords[0], crop_coords[1], 
                                       crop_coords[0], crop_coords[1]])
    
    print(f"  Bbox transformed back: {bbox_back}")
    
    # Verify the transformation is reversible (within floating point tolerance)
    assert np.allclose(bbox, bbox_back, atol=1.0), f"Transformation not reversible: {bbox} vs {bbox_back}"
    
    # Verify bbox is within cropped image
    assert transformed_bbox[0] >= 0 and transformed_bbox[1] >= 0, \
        f"Transformed bbox has negative coordinates: {transformed_bbox}"
    assert transformed_bbox[2] <= 1008 and transformed_bbox[3] <= 1008, \
        f"Transformed bbox exceeds image bounds: {transformed_bbox}"
    
    print("  ✓ Balanced method test passed!")
    return cropped_img, transformed_bbox


def test_edge_cases():
    """Test edge cases like small images and bboxes near boundaries."""
    print("\nTesting edge cases...")
    
    # Test 1: Small image (smaller than target_size)
    print("  Test 1: Small image")
    img_small = create_test_image(500, 500)
    bbox_small = np.array([100.0, 100.0, 200.0, 200.0])
    
    cropped, transformed_bbox, _ = crop_around_bbox_padded(img_small, bbox_small, target_size=1008)
    assert cropped.size == (1008, 1008), f"Expected (1008, 1008), got {cropped.size}"
    print("    ✓ Small image handled correctly")
    
    # Test 2: Bbox near image boundary
    print("  Test 2: Bbox near boundary")
    img = create_test_image(1920, 1080)
    bbox_edge = np.array([10.0, 10.0, 100.0, 100.0])  # Near top-left corner
    
    cropped, transformed_bbox, _ = crop_around_bbox_padded(img, bbox_edge, target_size=1008)
    assert cropped.size == (1008, 1008), f"Expected (1008, 1008), got {cropped.size}"
    assert transformed_bbox[0] >= 0 and transformed_bbox[1] >= 0, "Bbox should be inside cropped image"
    print("    ✓ Boundary bbox handled correctly")
    
    # Test 3: Large bbox
    print("  Test 3: Large bbox")
    bbox_large = np.array([100.0, 100.0, 900.0, 700.0])
    
    cropped, transformed_bbox, _ = crop_around_bbox_padded(img, bbox_large, target_size=1008)
    assert cropped.size == (1008, 1008), f"Expected (1008, 1008), got {cropped.size}"
    print("    ✓ Large bbox handled correctly")
    
    print("  ✓ All edge case tests passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Cropping Functions")
    print("="*60)
    
    try:
        test_padded_method()
        test_resize_method()
        test_balanced_method()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
