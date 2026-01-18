#!/usr/bin/env python3
"""
Script to plot SAM3 tracking results by class.
Plots: average number of frames, mean IoU, and precision for each class.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_sequence_to_class_mapping(csv_path):
    """Load the mapping from sequence name to class name from dataset CSV."""
    df = pd.read_csv(csv_path)
    # Create a mapping from sequence to class name
    sequence_to_class = df[['sequence', 'cls_name']].drop_duplicates().set_index('sequence')['cls_name'].to_dict()
    return sequence_to_class


def load_results(json_path):
    """Load the tracking results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_center_distance(bbox1, bbox2):
    """
    Calculate Euclidean distance between centers of two bounding boxes.
    
    Args:
        bbox1: Bounding box [x_min, y_min, x_max, y_max]
        bbox2: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        Distance between centers in pixels
    """
    # Calculate centers
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Euclidean distance
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: Bounding box [x_min, y_min, x_max, y_max]
        bbox2: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_box_area(bbox):
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        Area in pixels squared
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return width * height


def calculate_per_frame_metrics(detailed_results):
    """
    Calculate IoU, center distance, and box size for each frame.
    
    Args:
        detailed_results: List of detailed results with predictions and ground truths
    
    Returns:
        Dict mapping sequence name to metrics
    """
    sequence_metrics = {}
    
    for result in detailed_results:
        sequence = result['sequence']
        predictions = result.get('predictions', [])
        ground_truths = result.get('ground_truths', [])
        
        ious = []
        center_distances = []
        box_sizes = []
        
        # Calculate metrics for each frame where we have both pred and gt
        for pred, gt in zip(predictions, ground_truths):
            # Calculate box size from ground truth
            if gt is not None and len(gt) == 4:
                box_area = calculate_box_area(gt)
                box_sizes.append(box_area)
            
            # Check if prediction exists (not None/empty and has valid coordinates)
            if pred is not None and len(pred) == 4:
                # Check if ground truth exists
                if gt is not None and len(gt) == 4:
                    # Calculate IoU
                    iou = calculate_iou(pred, gt)
                    ious.append(iou)
                    
                    # Calculate center distance
                    distance = calculate_center_distance(pred, gt)
                    center_distances.append(distance)
        
        sequence_metrics[sequence] = {
            'mean_iou_with_pred': np.mean(ious) if ious else 0.0,
            'mean_center_distance': np.mean(center_distances) if center_distances else 0.0,
            'mean_box_size': np.mean(box_sizes) if box_sizes else 0.0,
            'num_predictions': len(ious)
        }
    
    return sequence_metrics


def aggregate_metrics_by_class(results, sequence_to_class, per_frame_metrics):
    """Aggregate metrics by class."""
    class_metrics = {}
    
    for seq_data in results['per_sequence_metrics']:
        sequence = seq_data['sequence']
        
        # Get the class for this sequence
        cls_name = sequence_to_class.get(sequence)
        if cls_name is None:
            print(f"Warning: No class found for sequence {sequence}, skipping...")
            continue
        
        # Initialize class entry if needed
        if cls_name not in class_metrics:
            class_metrics[cls_name] = {
                'num_frames': [],
                'mean_iou': [],
                'precision': [],
                'tp': [],
                'fp': [],
                'fn': [],
                'mean_iou_with_pred': [],
                'mean_center_distance': [],
                'mean_box_size': []
            }
        
        # Collect metrics
        class_metrics[cls_name]['num_frames'].append(seq_data['num_frames'])
        class_metrics[cls_name]['mean_iou'].append(seq_data['mean_iou'])
        class_metrics[cls_name]['precision'].append(seq_data['precision'])
        class_metrics[cls_name]['tp'].append(seq_data['tp'])
        class_metrics[cls_name]['fp'].append(seq_data['fp'])
        class_metrics[cls_name]['fn'].append(seq_data['fn'])
        
        # Add per-frame metrics (only for frames with predictions)
        if sequence in per_frame_metrics:
            class_metrics[cls_name]['mean_iou_with_pred'].append(
                per_frame_metrics[sequence]['mean_iou_with_pred']
            )
            class_metrics[cls_name]['mean_center_distance'].append(
                per_frame_metrics[sequence]['mean_center_distance']
            )
            class_metrics[cls_name]['mean_box_size'].append(
                per_frame_metrics[sequence]['mean_box_size']
            )
    
    # Calculate averages and totals, keeping raw data for box plots
    aggregated = {}
    for cls_name, metrics in class_metrics.items():
        aggregated[cls_name] = {
            'avg_num_frames': np.mean(metrics['num_frames']),
            'mean_iou': np.mean(metrics['mean_iou']),
            'mean_precision': np.mean(metrics['precision']),
            'num_sequences': len(metrics['num_frames']),
            'total_tp': sum(metrics['tp']),
            'total_fp': sum(metrics['fp']),
            'total_fn': sum(metrics['fn']),
            'mean_iou_with_pred': np.mean(metrics['mean_iou_with_pred']) if metrics['mean_iou_with_pred'] else 0.0,
            'mean_center_distance': np.mean(metrics['mean_center_distance']) if metrics['mean_center_distance'] else 0.0,
            'mean_box_size': np.mean(metrics['mean_box_size']) if metrics['mean_box_size'] else 0.0,
            # Keep raw data for box plots
            'num_frames_raw': metrics['num_frames'],
            'mean_iou_raw': metrics['mean_iou'],
            'precision_raw': metrics['precision'],
            'mean_iou_with_pred_raw': metrics['mean_iou_with_pred'],
            'mean_center_distance_raw': metrics['mean_center_distance'],
            'mean_box_size_raw': metrics['mean_box_size']
        }
    
    return aggregated


def plot_metrics(aggregated_metrics, output_dir):
    """Create and save plots for each metric."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort classes alphabetically
    classes = sorted(aggregated_metrics.keys())
    
    # Extract metrics (means for summary)
    avg_frames = [aggregated_metrics[cls]['avg_num_frames'] for cls in classes]
    mean_ious = [aggregated_metrics[cls]['mean_iou'] for cls in classes]
    mean_precisions = [aggregated_metrics[cls]['mean_precision'] for cls in classes]
    num_sequences = [aggregated_metrics[cls]['num_sequences'] for cls in classes]
    total_tps = [aggregated_metrics[cls]['total_tp'] for cls in classes]
    total_fps = [aggregated_metrics[cls]['total_fp'] for cls in classes]
    total_fns = [aggregated_metrics[cls]['total_fn'] for cls in classes]
    mean_ious_with_pred = [aggregated_metrics[cls]['mean_iou_with_pred'] for cls in classes]
    mean_center_distances = [aggregated_metrics[cls]['mean_center_distance'] for cls in classes]
    
    # Extract raw data for box plots
    num_frames_data = [aggregated_metrics[cls]['num_frames_raw'] for cls in classes]
    mean_iou_data = [aggregated_metrics[cls]['mean_iou_raw'] for cls in classes]
    precision_data = [aggregated_metrics[cls]['precision_raw'] for cls in classes]
    mean_iou_with_pred_data = [aggregated_metrics[cls]['mean_iou_with_pred_raw'] for cls in classes]
    mean_center_distance_data = [aggregated_metrics[cls]['mean_center_distance_raw'] for cls in classes]
    mean_box_size_data = [aggregated_metrics[cls]['mean_box_size_raw'] for cls in classes]
    mean_box_sizes = [aggregated_metrics[cls]['mean_box_size'] for cls in classes]
    
    # Set up color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('SAM3 Tracking Results by Class', fontsize=18, fontweight='bold')
    
    # Plot 1: Number of Frames Distribution (Box Plot)
    ax1 = axes[0, 0]
    bp1 = ax1.boxplot(num_frames_data, labels=classes, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    ax1.set_title('Number of Frames Distribution per Class', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, num_frames_data)):
        median_val = np.median(data)
        ax1.text(i+1, ax1.get_ylim()[1]*0.95, f'n={n_seq}\nmed={median_val:.1f}',
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Plot 2: Mean IoU Distribution (Box Plot)
    ax2 = axes[0, 1]
    bp2 = ax2.boxplot(mean_iou_data, labels=classes, patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
    ax2.set_title('Mean IoU Distribution per Class', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, mean_iou_data)):
        median_val = np.median(data)
        ax2.text(i+1, 0.95, f'n={n_seq}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Plot 2b: Box Size Distribution (Box Plot)
    ax2b = axes[0, 2]
    bp2b = ax2b.boxplot(mean_box_size_data, labels=classes, patch_artist=True,
                        boxprops=dict(facecolor='lightpink', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp2b['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2b.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2b.set_ylabel('Box Size (pixels²)', fontsize=12, fontweight='bold')
    ax2b.set_title('Box Size Distribution per Class', fontsize=13, fontweight='bold')
    ax2b.set_xticklabels(classes, rotation=45, ha='right')
    ax2b.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, mean_box_size_data)):
        median_val = np.median(data)
        ax2b.text(i+1, ax2b.get_ylim()[1]*0.95, f'n={n_seq}\nmed={median_val:.0f}',
                 ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Plot 3: Precision Distribution (Box Plot)
    ax3 = axes[1, 0]
    bp3 = ax3.boxplot(precision_data, labels=classes, patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision Distribution per Class', fontsize=13, fontweight='bold')
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, precision_data)):
        median_val = np.median(data)
        ax3.text(i+1, 0.95, f'n={n_seq}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Plot 4: Number of Sequences per Class
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(classes)), num_sequences, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax4.set_title('Number of Sequences per Class', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(classes)))
    ax4.set_xticklabels(classes, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars4, num_sequences)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 5: TP/FP/FN Stacked Bar Chart (Percentages)
    ax5 = axes[1, 2]
    x_pos = np.arange(len(classes))
    width = 0.6
    
    # Calculate percentages
    total_counts = np.array(total_tps) + np.array(total_fps) + np.array(total_fns)
    tp_percentages = (np.array(total_tps) / total_counts * 100).tolist()
    fp_percentages = (np.array(total_fps) / total_counts * 100).tolist()
    fn_percentages = (np.array(total_fns) / total_counts * 100).tolist()
    
    # Create stacked bars with percentages
    ax5.bar(x_pos, tp_percentages, width, label='True Positives', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax5.bar(x_pos, fp_percentages, width, bottom=tp_percentages, label='False Positives', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax5.bar(x_pos, fn_percentages, width, bottom=np.array(tp_percentages) + np.array(fp_percentages), 
            label='False Negatives', color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax5.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax5.set_title('TP/FP/FN Distribution by Class (%)', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(classes, rotation=45, ha='right')
    ax5.set_ylim(0, 100)
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 6: Mean IoU (with predictions only) Distribution (Box Plot)
    ax6 = axes[2, 0]
    bp6 = ax6.boxplot(mean_iou_with_pred_data, labels=classes, patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp6['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax6.set_ylabel('IoU (predictions only)', fontsize=12, fontweight='bold')
    ax6.set_title('IoU Distribution per Class (Predictions Only)', fontsize=13, fontweight='bold')
    ax6.set_xticklabels(classes, rotation=45, ha='right')
    ax6.set_ylim(0, 1)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, mean_iou_with_pred_data)):
        median_val = np.median(data)
        ax6.text(i+1, 0.95, f'n={n_seq}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Plot 7: Center Distance Distribution (Box Plot)
    ax7 = axes[2, 1]
    bp7 = ax7.boxplot(mean_center_distance_data, labels=classes, patch_artist=True,
                      boxprops=dict(facecolor='lightyellow', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    # Color each box differently
    for patch, color in zip(bp7['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax7.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Center Distance (pixels)', fontsize=12, fontweight='bold')
    ax7.set_title('Center Distance Distribution per Class', fontsize=13, fontweight='bold')
    ax7.set_xticklabels(classes, rotation=45, ha='right')
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size and median annotations
    for i, (n_seq, data) in enumerate(zip(num_sequences, mean_center_distance_data)):
        median_val = np.median(data)
        ax7.text(i+1, ax7.get_ylim()[1]*0.95, f'n={n_seq}\nmed={median_val:.1f}',
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Hide the unused subplot (2, 2)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the combined plot
    output_path = output_dir / 'sam3_tracking_metrics_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")
    
    # Create individual plots for each metric
    # Individual plot 1: Number of Frames Distribution (Box Plot)
    fig1, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(num_frames_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    ax.set_title('Number of Frames Distribution per Sequence by Class', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, mean, and median annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, avg_frames, num_frames_data)):
        median_val = np.median(data)
        y_pos = ax.get_ylim()[1] * 0.97
        ax.text(i+1, y_pos, f'n={n_seq}\nμ={mean_val:.1f}\nmed={median_val:.1f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'avg_frames_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved frames distribution plot to: {output_path}")
    plt.close()
    
    # Individual plot 2: Mean IoU Distribution (Box Plot)
    fig2, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(mean_iou_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
    ax.set_title('Mean IoU Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, mean, and median annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, mean_ious, mean_iou_data)):
        median_val = np.median(data)
        ax.text(i+1, 0.97, f'n={n_seq}\nμ={mean_val:.3f}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'mean_iou_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved IoU distribution plot to: {output_path}")
    plt.close()
    
    # Individual plot 3: Precision Distribution (Box Plot)
    fig3, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(precision_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, mean, and median annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, mean_precisions, precision_data)):
        median_val = np.median(data)
        ax.text(i+1, 0.97, f'n={n_seq}\nμ={mean_val:.3f}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'mean_precision_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved precision distribution plot to: {output_path}")
    plt.close()
    
    # Individual plot 4: TP/FP/FN Stacked Bar Chart (Percentages)
    fig4, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(len(classes))
    width = 0.6
    
    # Calculate percentages
    total_counts = np.array(total_tps) + np.array(total_fps) + np.array(total_fns)
    tp_percentages = (np.array(total_tps) / total_counts * 100).tolist()
    fp_percentages = (np.array(total_fps) / total_counts * 100).tolist()
    fn_percentages = (np.array(total_fns) / total_counts * 100).tolist()
    
    # Create stacked bars with percentages
    ax.bar(x_pos, tp_percentages, width, label='True Positives', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, fp_percentages, width, bottom=tp_percentages, label='False Positives', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, fn_percentages, width, bottom=np.array(tp_percentages) + np.array(fp_percentages), 
           label='False Negatives', color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('TP/FP/FN Distribution by Class (Percentages)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on each segment with percentages and counts
    for i in range(len(classes)):
        # TP labels
        if tp_percentages[i] > 3:  # Only show label if segment is large enough
            label = f'{tp_percentages[i]:.1f}%\n({total_tps[i]})'
            ax.text(i, tp_percentages[i]/2, label,
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        # FP labels
        if fp_percentages[i] > 3:
            label = f'{fp_percentages[i]:.1f}%\n({total_fps[i]})'
            ax.text(i, tp_percentages[i] + fp_percentages[i]/2, label,
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        # FN labels
        if fn_percentages[i] > 3:
            label = f'{fn_percentages[i]:.1f}%\n({total_fns[i]})'
            ax.text(i, tp_percentages[i] + fp_percentages[i] + fn_percentages[i]/2, label,
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    plt.tight_layout()
    output_path = output_dir / 'tp_fp_fn_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved TP/FP/FN plot to: {output_path}")
    plt.close()
    
    # Individual plot 5: Grouped Bar Chart for TP/FP/FN
    fig5, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(len(classes))
    width = 0.25
    
    bars_tp = ax.bar(x_pos - width, total_tps, width, label='True Positives', 
                     color='#2ecc71', alpha=0.8, edgecolor='black')
    bars_fp = ax.bar(x_pos, total_fps, width, label='False Positives', 
                     color='#e74c3c', alpha=0.8, edgecolor='black')
    bars_fn = ax.bar(x_pos + width, total_fns, width, label='False Negatives', 
                     color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('TP/FP/FN Distribution by Class (Grouped)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars_tp, bars_fp, bars_fn]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'tp_fp_fn_grouped_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved TP/FP/FN grouped plot to: {output_path}")
    plt.close()
    
    # Individual plot 6: IoU Distribution (with predictions only) (Box Plot)
    fig6, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(mean_iou_with_pred_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU (predictions only)', fontsize=12, fontweight='bold')
    ax.set_title('IoU Distribution per Class (Predictions Only - Excludes False Negatives)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, mean, and median annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, mean_ious_with_pred, mean_iou_with_pred_data)):
        median_val = np.median(data)
        ax.text(i+1, 0.97, f'n={n_seq}\nμ={mean_val:.3f}\nmed={median_val:.3f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'mean_iou_with_pred_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved IoU (predictions only) distribution plot to: {output_path}")
    plt.close()
    
    # Individual plot 7: Center Distance Distribution (Box Plot)
    fig7, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(mean_center_distance_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightyellow', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Center Distance (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Center Distance Distribution Between Predicted and Ground Truth Boxes by Class', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, mean, and median annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, mean_center_distances, mean_center_distance_data)):
        median_val = np.median(data)
        y_pos = ax.get_ylim()[1] * 0.97
        ax.text(i+1, y_pos, f'n={n_seq}\nμ={mean_val:.1f}px\nmed={median_val:.1f}px',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'mean_center_distance_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved center distance distribution plot to: {output_path}")
    plt.close()
    
    # Individual plot 8: Box Size Distribution (Box Plot)
    fig8, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(mean_box_size_data, labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='lightpink', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Box Size (pixels²)', fontsize=12, fontweight='bold')
    ax.set_title('Box Size Distribution (Area) by Class', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size, median, and mean annotations
    for i, (n_seq, mean_val, data) in enumerate(zip(num_sequences, mean_box_sizes, mean_box_size_data)):
        median_val = np.median(data)
        y_pos = ax.get_ylim()[1] * 0.97
        ax.text(i+1, y_pos, f'n={n_seq}\nμ={mean_val:.0f}\nmed={median_val:.0f}',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / 'mean_box_size_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved box size distribution plot to: {output_path}")
    plt.close()
    
    plt.show()


def print_summary(aggregated_metrics):
    """Print a summary table of the metrics."""
    print("\n" + "="*175)
    print("SAM3 TRACKING RESULTS SUMMARY BY CLASS")
    print("="*175)
    print(f"{'Class':<20} {'Seqs':>6} {'Avg Frames':>11} {'Mean IoU':>10} {'Mean Prec':>11} "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'IoU(pred)':>11} {'CenterDist':>11} {'BoxSize':>11}")
    print("-"*175)
    
    # Sort by class name
    for cls in sorted(aggregated_metrics.keys()):
        metrics = aggregated_metrics[cls]
        print(f"{cls:<20} {metrics['num_sequences']:>6} "
              f"{metrics['avg_num_frames']:>11.2f} "
              f"{metrics['mean_iou']:>10.3f} "
              f"{metrics['mean_precision']:>11.3f} "
              f"{metrics['total_tp']:>6} "
              f"{metrics['total_fp']:>6} "
              f"{metrics['total_fn']:>6} "
              f"{metrics['mean_iou_with_pred']:>11.3f} "
              f"{metrics['mean_center_distance']:>11.1f} "
              f"{metrics['mean_box_size']:>11.0f}")
    
    print("="*175)
    print("IoU(pred) = Mean IoU for predictions only (excludes false negatives)")
    print("CenterDist = Mean center distance in pixels between predicted and ground truth boxes")
    print("BoxSize = Mean bounding box area in pixels²")
    print("="*175 + "\n")


def main():
    # Paths
    project_root = Path(__file__).parent
    csv_path = project_root / 'dataset.csv'
    json_path = project_root / 'results' / 'sam3_tracking_results.json'
    output_dir = project_root / 'results' / 'class_visualizations'
    
    print("Loading data...")
    # Load sequence to class mapping
    sequence_to_class = load_sequence_to_class_mapping(csv_path)
    print(f"Loaded {len(sequence_to_class)} sequence-to-class mappings")
    
    # Load tracking results
    results = load_results(json_path)
    print(f"Loaded results for {len(results['per_sequence_metrics'])} sequences")
    
    print("\nCalculating per-frame metrics (IoU and center distance for predictions)...")
    # Calculate per-frame metrics
    per_frame_metrics = calculate_per_frame_metrics(results.get('detailed_results', []))
    print(f"Calculated metrics for {len(per_frame_metrics)} sequences with detailed results")
    
    print("\nAggregating metrics by class...")
    # Aggregate metrics by class
    aggregated_metrics = aggregate_metrics_by_class(results, sequence_to_class, per_frame_metrics)
    print(f"Found {len(aggregated_metrics)} classes")
    
    # Print summary
    print_summary(aggregated_metrics)
    
    print("Creating plots...")
    # Create plots
    plot_metrics(aggregated_metrics, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
