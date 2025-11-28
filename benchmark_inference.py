"""
Get inference times for each backbone.
This script instantiates each model and runs some warmup passes and then some inference passes with dummy data.
"""

import torch
from models import get_model
import time

NUM_FRAMES = 32
BATCH_SIZE = 2


def test_backbone(backbone_name, input_shape, model_kwargs=None, num_warmup_passes=10, num_inference_passes=30):
    """Test a single backbone with dummy data."""
    print(f"\n{'='*60}")
    print(f"Testing {backbone_name.upper()}")
    print('='*60)
    
    if model_kwargs is None:
        model_kwargs = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model
        model = get_model(backbone_name, **model_kwargs)
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params/1e6:.2f}M")
        print(f"  - Trainable parameters: {trainable_params/1e6:.2f}M")
        
        inference_times = []

        total_passes = num_warmup_passes + num_inference_passes
        for i in range(total_passes):
            print(f"✓ Pass {i} of {total_passes}\r", end="")

            # Create dummy input
            dummy_input = torch.randn(input_shape, device=device)
            
            start = time.time()
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)

            end = time.time()
            if i >= num_warmup_passes:
                inference_times.append(end - start)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"✓ Average inference time: {avg_inference_time:.4f} seconds for model {backbone_name}")
        return avg_inference_time
        
    except Exception as e:
        print(f"✗ Error testing {backbone_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all backbones."""
    print("\n" + "="*60)
    print("BACKBONE ARCHITECTURE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: EfficientNet (2D motion images)
    print("\n[1/5] Testing EfficientNet (Motion-based)")
    results['efficientnet'] = test_backbone(
        backbone_name='efficientnet',
        input_shape=(BATCH_SIZE, 3, 224, 224),  # (B, C, H, W)
        model_kwargs={'freeze_backbone': True}
    )
    
    # Test 2: 3D ResNet (video sequences)
    print("\n[2/5] Testing 3D ResNet")
    results['3dresnet'] = test_backbone(
        backbone_name='3dresnet',
        input_shape=(BATCH_SIZE, NUM_FRAMES, 3, 224, 224),  # (B, T, C, H, W)
        model_kwargs={'pretrained': True, 'freeze_backbone': True}
    )
    
    # Test 3: VideoMAE (video sequences)
    print("\n[3/5] Testing VideoMAE from HuggingFace")
    print("Note: This will download pretrained weights on first run (~90MB for small)")
    try:
        results['videomae'] = test_backbone(
            backbone_name='videomae',
            input_shape=(BATCH_SIZE, NUM_FRAMES, 3, 224, 224),  # (B, T, C, H, W)
            model_kwargs={
                'model_name': 'MCG-NJU/videomae-base',  # Use small model for testing
                'num_frames': 16,
                'freeze_backbone': True,
                'dropout': 0.1
            }
        )
    except ImportError as e:
        print(f"✗ Skipping VideoMAE test - transformers library not installed")
        print(f"  Install with: pip install transformers")
        results['videomae'] = False
    
    # Test 4: ViViT (video sequences)
    print("\n[4/5] Testing ViViT from HuggingFace (Google)")
    print("Note: This will download pretrained weights on first run (~350MB)")
    try:
        results['vivit'] = test_backbone(
            backbone_name='vivit',
            input_shape=(BATCH_SIZE, NUM_FRAMES, 3, 224, 224),  # (B, T, C, H, W)
            model_kwargs={
                'model_name': 'google/vivit-b-16x2-kinetics400',
                'num_frames': 32,
                'freeze_backbone': True,
                'dropout': 0.1
            }
        )
    except ImportError as e:
        print(f"✗ Skipping ViViT test - transformers library not installed")
        print(f"  Install with: pip install transformers")
        results['vivit'] = False
    except Exception as e:
        print(f"✗ ViViT test failed: {str(e)}")
        print(f"  This might be due to model availability. Marking as failed.")
        results['vivit'] = False
    
    # Test 5: CNN + Transformer (video sequences)
    print("\n[5/5] Testing CNN + Transformer")
    results['cnn_transformer'] = test_backbone(
        backbone_name='cnn_transformer',
        input_shape=(BATCH_SIZE, NUM_FRAMES, 3, 224, 224),  # (B, T, C, H, W)
        model_kwargs={
            'freeze_backbone': True,
            'num_frames': NUM_FRAMES,
            'embed_dim': 256,     # Smaller for testing
            'num_heads': 4,
            'num_layers': 2
        }
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for backbone, inference_time in results.items():
        print(f"  {backbone:20s}: {inference_time:.4f} seconds")


if __name__ == "__main__":
    main()

