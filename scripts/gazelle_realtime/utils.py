#!/usr/bin/env python3
"""
Utility functions for the Gazelle real-time gaze estimation system.
"""

import numpy as np
import cv2
from pathlib import Path


def create_directories(paths):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(exist_ok=True)


def clear_output_directories(dirs):
    """Clear all files in output directories for privacy."""
    import shutil
    for dir_path in dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            # Remove all contents
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"[PRIVACY] Cleared directory: {dir_path}")


def normalize_bounding_boxes(boxes, width, height):
    """Normalize bounding boxes to [0,1] range."""
    return [[np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]]


def get_hef_input_dimensions(hef_model):
    """Extract input dimensions from HEF model."""
    input_vs = hef_model.get_input_vstream_infos()
    if not input_vs:
        raise ValueError("Failed to get HEF input information")
    
    shape = input_vs[0].shape
    if len(shape) == 3:  # HWC format
        return shape[0], shape[1]
    else:  # NHWC format
        return shape[1], shape[2]


def process_dino_features(feat_raw):
    """Process DINOv2 raw features to correct tensor format."""
    if feat_raw.ndim == 3:
        # [H, W, C] -> [1, C, H, W] format conversion
        feat_processed = np.transpose(feat_raw, (2, 0, 1))
        feat_processed = np.expand_dims(feat_processed, 0)
    elif feat_raw.ndim == 4:
        if feat_raw.shape[-1] == 384:  # [N, H, W, C] format
            # [N, H, W, C] -> [N, C, H, W] format conversion
            feat_processed = np.transpose(feat_raw, (0, 3, 1, 2))
        else:  # Already [N, C, H, W] format
            feat_processed = feat_raw
    else:
        raise ValueError(f"Unexpected feature shape: {feat_raw.shape}")
    
    return feat_processed


def find_gaze_point(heatmap):
    """Find gaze point in heatmap (maximum value location)."""
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    gaze_y, gaze_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return gaze_x, gaze_y


def compute_gaze_targets(heatmaps, object_detections, img_width, img_height):
    """Determine which objects are being gazed at based on heatmaps and detections."""
    gaze_targets = []
    
    for heatmap in heatmaps:
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()
        
        # Resize heatmap to image size if needed
        if heatmap.shape != (img_height, img_width):
            heatmap = cv2.resize(heatmap, (img_width, img_height))
        
        # Find object with highest gaze probability
        best_object = None
        best_score = 0.0
        
        for detection in object_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract heatmap region for this bounding box
            bbox_heatmap = heatmap[y1:y2, x1:x2]
            
            if bbox_heatmap.size > 0:
                # Calculate average and max gaze probability in this box
                avg_gaze_prob = np.mean(bbox_heatmap)
                max_gaze_prob = np.max(bbox_heatmap)
                
                # Use combination of average and max for robustness
                combined_score = 0.7 * max_gaze_prob + 0.3 * avg_gaze_prob
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_object = {
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'bbox': bbox,
                        'gaze_score': combined_score,
                        'max_gaze_prob': max_gaze_prob,
                        'avg_gaze_prob': avg_gaze_prob
                    }
        
        # Get peak gaze point
        gaze_x, gaze_y = find_gaze_point(heatmap)
        
        # Add gaze target information
        gaze_targets.append({
            'gaze_point': (gaze_x, gaze_y),
            'gaze_object': best_object,
            'heatmap_max': np.max(heatmap)
        })
    
    return gaze_targets


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def softmax(x, axis=-1):
    """Compute softmax values for array x along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)