#!/usr/bin/env python3
"""
Real-time gaze estimation using Hailo-8 infrastructure with GStreamer.
Uses hailo_apps_infra for camera capture and processing pipeline.
"""
import argparse
import sys
import time
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import hailo
from hailo_platform import (
    VDevice, HailoStreamInterface, InferVStreams, 
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)

# Add parent directory to import gazelle module
sys.path.append(str(Path(__file__).parent.parent))
from gazelle.model import GazeLLE
from gazelle.backbone import DinoV2Backbone

# Import hailo infrastructure
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    get_default_parser,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_app import GStreamerApp
from hailo_apps_infra.gstreamer_helper_pipelines import (
    QUEUE,
    SOURCE_PIPELINE,
    DISPLAY_PIPELINE,
)


# ============================================================================
# Configuration and Constants
# ============================================================================

DEFAULT_CONFIG = {
    'output_dir': './output_frames',
    'inference_output_dir': './inference_results',
    'device': 'cpu',
    'save_interval': 1.0,
    'max_frames': 10,
    'skip_frames': 0,
    'save_mode': 'time',
    'nominal_fps': 30,
}

DEBUG_INTERVALS = {
    'frame_info': 10,      # Print frame info every N frames
    'heartbeat': 5,        # Heartbeat message every N seconds
    'processing_avg': 10,  # Average processing time over last N frames
}


# ============================================================================
# Utility Functions
# ============================================================================

def create_directories(paths):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(exist_ok=True)


def normalize_bounding_boxes(boxes, width, height):
    """Normalize bounding boxes to [0,1] range."""
    return [[np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]]


def get_hef_input_dimensions(hef_model):
    """Extract input dimensions from HEF model."""
    input_vs = hef_model.get_input_vstream_infos()
    if not input_vs:
        raise ValueError("Failed to get HEF input info")
    
    shape = input_vs[0].shape
    if len(shape) == 3:  # HWC format
        return shape[0], shape[1]
    else:  # NHWC format
        return shape[1], shape[2]


def process_dino_features(feat_raw):
    """Process raw DINOv2 features into correct tensor format."""
    print(f"[DEBUG] Raw feature shape from Hailo: {feat_raw.shape}")
    
    if feat_raw.ndim == 3:
        # [H, W, C] -> [1, C, H, W]
        feat_processed = np.transpose(feat_raw, (2, 0, 1))
        feat_processed = np.expand_dims(feat_processed, 0)
    elif feat_raw.ndim == 4:
        if feat_raw.shape[-1] == 384:  # [N, H, W, C] format
            # [N, H, W, C] -> [N, C, H, W]
            feat_processed = np.transpose(feat_raw, (0, 3, 1, 2))
        else:
            # Already in [N, C, H, W] format
            feat_processed = feat_raw
    else:
        raise ValueError(f"Unexpected feature shape: {feat_raw.shape}")
    
    print(f"[DEBUG] Processed feature shape: {feat_processed.shape}")
    return feat_processed


def find_gaze_point(heatmap):
    """Find gaze point from heatmap (maximum value location)."""
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    gaze_y, gaze_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return gaze_x, gaze_y


def compute_gaze_targets(heatmaps, object_detections, img_width, img_height):
    """Determine which objects are being gazed at based on heatmap and detections."""
    gaze_targets = []
    
    for heatmap in heatmaps:
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()
        
        # Resize heatmap to match image dimensions if needed
        if heatmap.shape != (img_height, img_width):
            heatmap = cv2.resize(heatmap, (img_width, img_height))
        
        # Find the object with highest gaze probability
        best_object = None
        best_score = 0.0
        
        for detection in object_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract heatmap region for this bounding box
            bbox_heatmap = heatmap[y1:y2, x1:x2]
            
            if bbox_heatmap.size > 0:
                # Compute average gaze probability in this box
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
        
        # Also get the peak gaze point
        gaze_x, gaze_y = find_gaze_point(heatmap)
        
        gaze_targets.append({
            'gaze_point': (gaze_x, gaze_y),
            'gaze_object': best_object,
            'heatmap_max': np.max(heatmap)
        })
    
    return gaze_targets


# ============================================================================
# Hailo Inference Manager
# ============================================================================

class HailoInferenceManager:
    """Manages Hailo AI accelerator for DINOv2 inference."""
    
    def __init__(self, hef_path, vdevice=None):
        self.hef_path = hef_path
        self.vdevice = vdevice
        self._init_device()
    
    def _init_device(self):
        """Initialize Hailo device and load model."""
        import hailo_platform as hpf
        
        # Create Virtual Device if not provided
        if self.vdevice is None:
            self.vdevice = VDevice()
        
        # Load HEF model
        self.hef = hpf.HEF(self.hef_path)
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        network_group = self.vdevice.configure(self.hef, configure_params)[0]
        
        # Setup streams
        self.input_vstreams_params = InputVStreamParams.make(
            network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )
        
        self.network_group = network_group
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model input/output specifications."""
        print(f"[DEBUG] Initialized Hailo inference with "
              f"{len(self.input_vstreams_params)} inputs and "
              f"{len(self.output_vstreams_params)} outputs")
        
        # Print input info
        for info in self.hef.get_input_vstream_infos():
            print(f"[DEBUG] Input: {info.name}, shape: {info.shape}, format: {info.format}")
        
        # Print output info
        for info in self.hef.get_output_vstream_infos():
            print(f"[DEBUG] Output: {info.name}, shape: {info.shape}, format: {info.format}")
    
    def run_inference(self, frame):
        """Run inference on input frame."""
        # Get input dimensions
        input_info = self.hef.get_input_vstream_infos()[0]
        input_shape = input_info.shape
        
        if len(input_shape) == 3:  # HWC
            h, w = input_shape[0], input_shape[1]
        else:  # NHWC
            h, w = input_shape[1], input_shape[2]
        
        # Log original frame info
        if hasattr(self, '_first_frame_logged'):
            pass
        else:
            print(f"[DEBUG] Original frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"[DEBUG] Frame pixel range: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
            self._first_frame_logged = True
        
        # Store original size and model input size for DETR
        if isinstance(self, DETRInferenceManager) and self.detr_input_size is None:
            self.detr_input_size = (h, w)
        
        # Resize frame
        resized = cv2.resize(frame, (w, h))
        
        # Log resized frame info
        if not hasattr(self, '_resize_logged'):
            print(f"[DEBUG] Resized from {frame.shape[:2]} to {resized.shape[:2]}")
            print(f"[DEBUG] Scale factors: x={frame.shape[1]/w:.3f}, y={frame.shape[0]/h:.3f}")
            print(f"[DEBUG] Resized pixel range: min={resized.min()}, max={resized.max()}, mean={resized.mean():.2f}")
            self._resize_logged = True
        
        # Run inference
        with InferVStreams(self.network_group, 
                          self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            input_data = {
                list(self.input_vstreams_params.keys())[0]: np.expand_dims(resized, axis=0)
            }
            
            with self.network_group.activate(None):
                results = infer_pipeline.infer(input_data)
        
        return results


class SCRFDInferenceManager(HailoInferenceManager):
    """Specialized manager for SCRFD object detection."""
    
    def __init__(self, hef_path, vdevice=None):
        # Call parent class constructor with shared vdevice
        super().__init__(hef_path, vdevice)
    
    def process_detections(self, outputs, threshold=0.5):
        """Process SCRFD outputs to get face detections."""
        # SCRFD typically outputs multiple scales of predictions
        # This is a simplified version - adjust based on actual model outputs
        detections = []
        
        for output_name, output_data in outputs.items():
            if 'bbox' in output_name or 'loc' in output_name:
                # Process bounding box predictions
                # Shape depends on SCRFD variant
                continue
            elif 'conf' in output_name or 'cls' in output_name:
                # Process confidence scores
                continue
        
        # For now, return empty list - implement based on actual SCRFD output format
        return detections


class DETRInferenceManager(HailoInferenceManager):
    """Specialized manager for DETR object detection."""
    
    def __init__(self, hef_path, vdevice=None, confidence_threshold=0.7, nms_threshold=0.3):
        # Call parent class constructor with shared vdevice
        super().__init__(hef_path, vdevice)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        print(f"[DEBUG] DETRInferenceManager initialized with confidence_threshold={confidence_threshold}, nms_threshold={nms_threshold}")
        # COCO class names for DETR - 92 classes including placeholders
        # The model outputs 92 logits: 91 object classes (0-90) + 1 "no-object" class (91)
        self.COCO_CLASSES_92 = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",}
        # Store original image size and DETR input size for proper scaling
        self.detr_input_size = None
    
    def process_detections(self, outputs, img_width, img_height, threshold=None):
        """Process DETR outputs to get object detections."""
        if threshold is None:
            threshold = self.confidence_threshold
        detections = []
        
        # DETR outputs: conv113 (class logits), conv116 (boxes)
        boxes_output = None
        scores_output = None
        
        for output_name, output_data in outputs.items():
            if not hasattr(self, '_output_shapes_logged'):
                print(f"[DEBUG] DETR output: {output_name}, shape: {output_data.shape}")
            # Based on your output shapes:
            # conv116: (1, 1, 100, 4) - boxes
            # conv113: (1, 1, 100, 92) - class logits
            if 'conv116' in output_name or (output_data.shape[-1] == 4 and len(output_data.shape) >= 3):
                boxes_output = output_data
            elif 'conv113' in output_name or (output_data.shape[-1] > 4 and len(output_data.shape) >= 3):
                scores_output = output_data
        
        if not hasattr(self, '_output_shapes_logged'):
            self._output_shapes_logged = True
        
        if boxes_output is not None and scores_output is not None:
            # Log raw output ranges
            if not hasattr(self, '_raw_outputs_logged'):
                print(f"\n[DEBUG] Raw boxes output range: min={boxes_output.min():.3f}, max={boxes_output.max():.3f}")
                print(f"[DEBUG] Raw scores output range: min={scores_output.min():.3f}, max={scores_output.max():.3f}")
                self._raw_outputs_logged = True
            
            # Reshape to remove extra dimensions
            # From (1, 1, 100, 4) to (100, 4)
            if len(boxes_output.shape) == 4:
                boxes_output = boxes_output[0, 0]  # Remove batch and extra dim
            elif len(boxes_output.shape) == 3:
                boxes_output = boxes_output[0]  # Remove batch
                
            # From (1, 1, 100, 92) to (100, 92)
            if len(scores_output.shape) == 4:
                scores_output = scores_output[0, 0]  # Remove batch and extra dim
            elif len(scores_output.shape) == 3:
                scores_output = scores_output[0]  # Remove batch
            
            # Log first few box values to check format
            if not hasattr(self, '_box_format_logged'):
                print(f"\n[DEBUG] First 5 boxes (raw):")
                for i in range(min(5, boxes_output.shape[0])):
                    print(f"  Box {i}: {boxes_output[i]}")
                
                # Check if boxes need sigmoid
                if boxes_output.min() < 0 or boxes_output.max() > 1:
                    print(f"[DEBUG] Boxes appear to be logits (range: {boxes_output.min():.3f} to {boxes_output.max():.3f})")
                    print(f"[DEBUG] Applying sigmoid to boxes...")
                    boxes_output = 1 / (1 + np.exp(-boxes_output))
                    print(f"[DEBUG] After sigmoid - range: {boxes_output.min():.3f} to {boxes_output.max():.3f}")
                    print(f"[DEBUG] First 5 boxes (after sigmoid):")
                    for i in range(min(5, boxes_output.shape[0])):
                        print(f"  Box {i}: {boxes_output[i]}")
                
                self._box_format_logged = True
            
            # Apply sigmoid to boxes if needed (for subsequent frames)
            if boxes_output.min() < 0 or boxes_output.max() > 1:
                boxes_output = 1 / (1 + np.exp(-boxes_output))
            
            # Apply softmax to get probabilities from logits
            scores_probs = self._softmax(scores_output, axis=-1)
            
            # Sanity check for first frame - verify class mapping fix
            if not hasattr(self, '_sanity_check_logged') and scores_probs.shape[0] > 0:
                print("\n[DEBUG] Sanity Check - Top predictions for first query (after class mapping fix):")
                # Get all class probabilities including background
                all_probs = scores_probs[0]  # Shape: (92,)
                topk_indices = np.argsort(all_probs)[-5:][::-1]  # Top 5 indices
                for idx_model in topk_indices:
                    if idx_model < len(self.COCO_CLASSES_92):
                        class_name_check = self.COCO_CLASSES_92[idx_model]
                    elif idx_model == 91:
                        class_name_check = "background/no-object"
                    else:
                        class_name_check = f"unknown_class_{idx_model}"
                    prob_check = all_probs[idx_model]
                    print(f"  Model Index: {idx_model}, Class: {class_name_check}, Prob: {prob_check:.3f}")
                self._sanity_check_logged = True
            
            # Log softmax results
            if not hasattr(self, '_softmax_logged'):
                print(f"\n[DEBUG] After softmax - max prob: {scores_probs.max():.3f}")
                print(f"[DEBUG] Top 5 predictions from first query:")
                top5_idx = np.argsort(scores_probs[0])[-5:][::-1]
                for idx in top5_idx:
                    # Handle all 92 classes (91 object classes + 1 background)
                    if idx < len(self.COCO_CLASSES_92):
                        class_name = self.COCO_CLASSES_92[idx]
                    elif idx == 91:
                        class_name = 'background'
                    else:
                        class_name = f'class_{idx}'
                    print(f"  {class_name}: {scores_probs[0][idx]:.3f}")
                self._softmax_logged = True
            
            # Process each detection
            num_queries = min(boxes_output.shape[0], 100)
            detection_count = 0
            
            for i in range(num_queries):
                box = boxes_output[i]
                class_probs = scores_probs[i]
                
                # Get best class including background
                best_class_idx = np.argmax(class_probs)
                best_score = class_probs[best_class_idx]
                
                # Skip if best class is background (index 91 in DETR output)
                if best_class_idx == 91:
                    continue
                
                # Filter: above threshold
                if best_score > threshold:
                    # DETR uses center_x, center_y, width, height format (normalized)
                    cx, cy, w, h = box
                    
                    # Get class name from the 92-class list, with bounds checking
                    if best_class_idx < len(self.COCO_CLASSES_92):
                        class_name = self.COCO_CLASSES_92[best_class_idx]
                    else:
                        # Unknown class index - skip this detection
                        if not hasattr(self, '_unknown_classes_logged'):
                            self._unknown_classes_logged = set()
                        if best_class_idx not in self._unknown_classes_logged:
                            print(f"[DEBUG] Warning: Unknown class index {best_class_idx} detected (confidence: {best_score:.3f})")
                            self._unknown_classes_logged.add(best_class_idx)
                        continue
                    
                    # Skip "N/A" placeholder classes or background
                    if class_name == "N/A" or class_name == "__background__":
                        continue
                    
                    # Log first detection details
                    if detection_count == 0 and not hasattr(self, '_first_detection_logged'):
                        print(f"\n[DEBUG] First detection details:")
                        print(f"  Query index: {i}")
                        print(f"  Box (cx,cy,w,h): {cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f}")
                        print(f"  Class: {class_name} (idx={best_class_idx})")
                        print(f"  Confidence: {best_score:.3f}")
                        print(f"  DETR input size: {getattr(self, 'detr_input_size', 'unknown')}")
                        print(f"  Output image size: {img_width}x{img_height}")
                    
                    # Convert to corner format
                    # Box coordinates are normalized to [0,1] on the DETR input size (800x800)
                    # We need to scale them to the output image size (1280x720)
                    x1 = (cx - w/2) * img_width
                    y1 = (cy - h/2) * img_height
                    x2 = (cx + w/2) * img_width
                    y2 = (cy + h/2) * img_height
                    
                    # Debug: Check for suspiciously large or small boxes
                    if not hasattr(self, '_box_size_logged'):
                        box_w = x2 - x1
                        box_h = y2 - y1
                        if box_w > img_width * 0.8 or box_h > img_height * 0.8:
                            print(f"[DEBUG] WARNING: Very large box detected - width={box_w:.1f} ({box_w/img_width*100:.1f}%), height={box_h:.1f} ({box_h/img_height*100:.1f}%)")
                            print(f"  Original: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
                        self._box_size_logged = True
                    
                    # Log coordinate conversion
                    if detection_count == 0 and not hasattr(self, '_first_detection_logged'):
                        print(f"  Before clamping: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                        print(f"  Image size: {img_width}x{img_height}")
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y2 = max(0, min(y2, img_height - 1))
                    
                    if detection_count == 0 and not hasattr(self, '_first_detection_logged'):
                        print(f"  After clamping: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                        self._first_detection_logged = True
                    
                    # Convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Skip very small boxes (likely noise)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # Minimum absolute size
                    if box_width < 20 or box_height < 20:
                        continue
                    
                    # Minimum relative size (0.1% of image area)
                    min_relative_area = 0.001 * img_width * img_height
                    if box_area < min_relative_area:
                        continue
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': float(best_score)
                    })
                    detection_count += 1
        
        # Apply Non-Maximum Suppression (NMS) to remove duplicates
        if len(detections) > 0:
            num_before_nms = len(detections)
            detections = self._apply_nms(detections, iou_threshold=self.nms_threshold)
            num_after_nms = len(detections)
            
            # Log NMS results for every frame (or every N frames)
            if not hasattr(self, '_nms_frame_count'):
                self._nms_frame_count = 0
            self._nms_frame_count += 1
            
            # Log every 10th frame or first frame
            if self._nms_frame_count == 1 or self._nms_frame_count % 10 == 0:
                print(f"\n[DEBUG] Frame {self._nms_frame_count} - NMS: {num_before_nms} -> {num_after_nms} detections (removed {num_before_nms - num_after_nms})")
                print(f"[DEBUG] NMS threshold: {self.nms_threshold}")
        
        if len(detections) > 0 and not hasattr(self, '_detection_summary_logged'):
            print(f"\n[DEBUG] DETR found {len(detections)} detections (after NMS)")
            print(f"[DEBUG] Detection summary:")
            for i, det in enumerate(detections[:5]):  # Show first 5
                print(f"  {i}: {det['class']} @ [{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}] conf={det['confidence']:.3f}")
            self._detection_summary_logged = True
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return detections
        
        # Debug: Log initial state for first few frames
        if not hasattr(self, '_nms_debug_count'):
            self._nms_debug_count = 0
        self._nms_debug_count += 1
        
        if self._nms_debug_count <= 3:  # Debug first 3 frames
            print(f"\n[DEBUG] _apply_nms called with {len(detections)} detections, iou_threshold={iou_threshold}")
            # Show class distribution
            class_counts = {}
            for det in detections:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            print(f"[DEBUG] Detection distribution: {class_counts}")
        
        # Group detections by class for class-specific NMS
        detections_by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        # Debug: Log class distribution
        if not hasattr(self, '_nms_class_dist_logged'):
            print(f"[DEBUG] Detection distribution by class:")
            for cls, dets in detections_by_class.items():
                print(f"  {cls}: {len(dets)} detections")
            self._nms_class_dist_logged = True
        
        # Apply NMS per class
        kept_detections = []
        nms_stats = {}  # Track removals per class
        
        for cls, class_detections in detections_by_class.items():
            # Sort by confidence (descending)
            class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS within this class
            kept_class_detections = []
            removed_count = 0
            
            for i, det in enumerate(class_detections):
                # Check if this detection overlaps too much with any kept detection of same class
                should_keep = True
                for j, kept in enumerate(kept_class_detections):
                    iou = self._compute_iou(det['bbox'], kept['bbox'])
                    
                    # Debug: Log high IoU cases
                    if iou > 0.1 and not hasattr(self, f'_nms_iou_logged_{cls}'):
                        print(f"[DEBUG] Class '{cls}' - IoU between detection {i} (conf={det['confidence']:.3f}) and kept {j} (conf={kept['confidence']:.3f}): {iou:.3f}")
                        if iou > iou_threshold:
                            print(f"  -> Detection {i} will be REMOVED (IoU {iou:.3f} > threshold {iou_threshold})")
                        setattr(self, f'_nms_iou_logged_{cls}', True)
                    
                    if iou > iou_threshold:
                        should_keep = False
                        removed_count += 1
                        break
                
                if should_keep:
                    kept_class_detections.append(det)
            
            nms_stats[cls] = {'original': len(class_detections), 'kept': len(kept_class_detections), 'removed': removed_count}
            kept_detections.extend(kept_class_detections)
        
        # Debug: Log NMS statistics for first few frames
        if not hasattr(self, '_nms_stats_count'):
            self._nms_stats_count = 0
        self._nms_stats_count += 1
        
        if self._nms_stats_count <= 3:
            print(f"\n[DEBUG] NMS statistics for frame {self._nms_stats_count}:")
            for cls, stats in nms_stats.items():
                if stats['removed'] > 0:
                    print(f"  {cls}: {stats['original']} -> {stats['kept']} (removed {stats['removed']})")
        
        # Sort all kept detections by confidence and limit to top N
        kept_detections = sorted(kept_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 20 detections (reduced from 25)
        return kept_detections[:20]
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Debug: Check for suspicious box coordinates
        if not hasattr(self, '_iou_debug_count'):
            self._iou_debug_count = 0
        self._iou_debug_count += 1
        
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
        
        # Debug: Log suspicious boxes or high IoU values for first few calculations
        if self._iou_debug_count <= 10:
            if box1_area > 50000 or box2_area > 50000:  # Large boxes (might indicate coordinate issues)
                print(f"[DEBUG] Large box detected: box1_area={box1_area:.1f}, box2_area={box2_area:.1f}")
                print(f"  box1: [{x1_1:.1f},{y1_1:.1f},{x2_1:.1f},{y2_1:.1f}]")
                print(f"  box2: [{x1_2:.1f},{y1_2:.1f},{x2_2:.1f},{y2_2:.1f}]")
        
        # Compute IoU
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        
        # Debug: Log IoU calculations for first few frames
        if self._iou_debug_count <= 5 and iou > 0.1:
            print(f"[DEBUG] IoU={iou:.3f} (intersection={intersection_area:.1f}, union={union_area:.1f})")
        
        return iou
    
    def _softmax(self, x, axis=-1):
        """Compute softmax values for array x along specified axis."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ============================================================================
# Frame Processor
# ============================================================================

class FrameProcessor:
    """Handles frame processing logic."""
    
    def __init__(self, gazelle_model, hailo_manager, device='cpu', scrfd_manager=None, detr_manager=None):
        self.gazelle_model = gazelle_model
        self.hailo_manager = hailo_manager
        self.scrfd_manager = scrfd_manager
        self.detr_manager = detr_manager
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        print("Initialized MTCNN face detector")
        if self.scrfd_manager:
            print("Initialized SCRFD face detector")
        if self.detr_manager:
            print("Initialized DETR object detector")
    
    def process_frame(self, frame, width, height):
        """Process a single frame and return gaze results."""
        # Run Hailo inference for features
        infer_results = self.hailo_manager.run_inference(frame)
        
        # Extract and process features
        output_name = list(infer_results.keys())[0]
        feat_raw = infer_results[output_name]
        feat_processed = process_dino_features(feat_raw)
        feat_tensor = torch.from_numpy(feat_processed).to(self.device)
        
        # Detect faces using SCRFD if available, otherwise use MTCNN
        face_detections = []
        object_detections = []
        
        if self.scrfd_manager:
            # Run SCRFD for face detection
            scrfd_results = self.scrfd_manager.run_inference(frame)
            scrfd_detections = self.scrfd_manager.process_detections(scrfd_results)
            face_detections.extend(scrfd_detections)
            
            # Extract face boxes from SCRFD detections
            face_boxes = [d['bbox'] for d in scrfd_detections if d.get('class') == 'face']
            if face_boxes:
                boxes = np.array(face_boxes)
            else:
                # Fallback to MTCNN if no faces detected by SCRFD
                boxes, probs = self.mtcnn.detect(frame)
        else:
            # Use MTCNN if SCRFD not available
            boxes, probs = self.mtcnn.detect(frame)
        
        if boxes is None or len(boxes) == 0:
            # Use center region if no faces detected
            boxes = np.array([[width*0.25, height*0.25, width*0.75, height*0.75]])
        
        # Run DETR for general object detection if available
        if self.detr_manager:
            detr_results = self.detr_manager.run_inference(frame)
            detr_detections = self.detr_manager.process_detections(detr_results, width, height)
            object_detections.extend(detr_detections)
        
        # Combine all detections
        all_detections = face_detections + object_detections
        
        # Normalize bounding boxes
        norm_bboxes = normalize_bounding_boxes(boxes, width, height)
        
        # Run GazeLLE inference
        with torch.no_grad():
            out = self.gazelle_model({
                "extracted_features": feat_tensor, 
                "bboxes": norm_bboxes
            })
        
        heatmaps = out["heatmap"][0].cpu().numpy()
        
        # Compute gaze targets
        gaze_targets = compute_gaze_targets(heatmaps, all_detections, width, height)
        
        return {
            'features': feat_tensor,
            'boxes': boxes,
            'heatmaps': heatmaps,
            'object_detections': all_detections,
            'gaze_targets': gaze_targets
        }


# ============================================================================
# Result Saver
# ============================================================================

class ResultSaver:
    """Handles saving of frames and inference results."""
    
    def __init__(self, output_dir, inference_output_dir=None):
        self.output_dir = Path(output_dir)
        self.inference_output_dir = Path(inference_output_dir) if inference_output_dir else None
        self.saved_frames = 0
        self.saved_inference = 0
        
        # Create directories
        dirs = [self.output_dir]
        if self.inference_output_dir:
            dirs.append(self.inference_output_dir)
        create_directories(dirs)
    
    def save_visualization(self, frame, boxes, heatmaps, object_detections=None, gaze_targets=None):
        """Save frame with gaze visualization and object detections."""
        try:
            frame_pil = Image.fromarray(frame)
            
            # Use first heatmap for visualization
            heatmap = heatmaps[0] if len(heatmaps) > 0 else np.zeros((frame.shape[0], frame.shape[1]))
            
            if heatmap.ndim == 3:
                heatmap = heatmap.squeeze()
            
            # Normalize and resize heatmap
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            if heatmap.shape != (frame.shape[0], frame.shape[1]):
                heatmap_norm = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]))
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(frame_pil)
            # Reduce heatmap opacity to ensure objects are visible
            plt.imshow(heatmap_norm, alpha=0.3, cmap='jet')
            
            # Get current axes for drawing
            ax = plt.gca()
            
            # Draw face bounding boxes
            for i, bbox in enumerate(boxes):
                xmin, ymin, xmax, ymax = bbox
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   fill=False, edgecolor='lime', linewidth=3)
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f'Face {i+1}',
                       color='lime', fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # Get gazed objects for highlighting
            gazed_objects = []
            if gaze_targets:
                for target in gaze_targets:
                    if target['gaze_object']:
                        gazed_objects.append(target['gaze_object'])
            
            # Draw object detections if available
            if object_detections:
                # Debug: Print what we're about to draw
                non_face_detections = [d for d in object_detections if d.get('class') != 'face']
                if non_face_detections:
                    print(f"[DEBUG] Drawing {len(non_face_detections)} object detections")
                    for i, d in enumerate(non_face_detections[:3]):
                        print(f"  - {d['class']} at [{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]")
                
                for detection in object_detections:
                    if detection.get('class') != 'face':  # Skip faces as they're already drawn
                        bbox = detection['bbox']
                        xmin, ymin, xmax, ymax = bbox
                        
                        # Check if this object is being gazed at
                        is_gazed = False
                        gaze_score = 0
                        for gazed_obj in gazed_objects:
                            if (gazed_obj['bbox'] == bbox and 
                                gazed_obj['class'] == detection['class']):
                                is_gazed = True
                                gaze_score = gazed_obj['gaze_score']
                                break
                        
                        # Use different colors and styles for gazed objects
                        if is_gazed:
                            linewidth = 5
                            linestyle = '-'
                            # Red color for gazed object
                            color = 'red'
                            label = f"[GAZE] {detection['class']} {detection.get('confidence', 0):.2f} (gaze: {gaze_score:.2f})"
                        else:
                            linewidth = 3
                            linestyle = '-'  # Changed from '--' to '-' for solid lines
                            # More visible color scheme
                            if 'person' in detection.get('class', '').lower():
                                color = 'cyan'
                            elif any(x in detection.get('class', '').lower() for x in ['chair', 'couch', 'bed']):
                                color = 'orange'
                            elif any(x in detection.get('class', '').lower() for x in ['mouse', 'keyboard', 'laptop', 'tv']):
                                color = 'magenta'
                            elif 'plant' in detection.get('class', '').lower():
                                color = 'green'
                            else:
                                color = 'yellow'
                            label = f"{detection['class']} {detection.get('confidence', 0):.2f}"
                        
                        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           fill=False, edgecolor=color, linewidth=linewidth,
                                           linestyle=linestyle)
                        ax.add_patch(rect)
                        
                        # Draw text label
                        text_y = ymin-5 if ymin > 20 else ymax+20  # Adjust text position if too close to top
                        ax.text(xmin, text_y, label,
                               color=color, fontsize=10 if not is_gazed else 12, 
                               weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='black' if not is_gazed else 'darkred', 
                                       alpha=0.7))
            
            # Draw gaze points
            if gaze_targets:
                for i, target in enumerate(gaze_targets):
                    gaze_x, gaze_y = target['gaze_point']
                    # Draw crosshair at gaze point
                    ax.plot(gaze_x, gaze_y, 'r+', markersize=20, markeredgewidth=3)
                    ax.add_patch(plt.Circle((gaze_x, gaze_y), 10, 
                                          fill=False, edgecolor='red', linewidth=2))
            
            # Update title with gaze information
            title = f"Gaze Estimation - Frame {self.saved_frames + 1}"
            if gaze_targets and gaze_targets[0]['gaze_object']:
                gazed_class = gaze_targets[0]['gaze_object']['class']
                title += f" | Looking at: {gazed_class}"
            
            plt.title(title)
            plt.axis('off')
            
            # Force redraw to ensure all patches are rendered
            plt.draw()
            
            # Save
            output_path = self.output_dir / f"frame_{self.saved_frames + 1:04d}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            print(f"Saved frame {self.saved_frames + 1} to {output_path}")
            if gaze_targets and gaze_targets[0]['gaze_object']:
                print(f"  User is looking at: {gaze_targets[0]['gaze_object']['class']} "
                      f"(confidence: {gaze_targets[0]['gaze_object']['confidence']:.2f}, "
                      f"gaze score: {gaze_targets[0]['gaze_object']['gaze_score']:.2f})")
            self.saved_frames += 1
            
        except Exception as e:
            print(f"[DEBUG] Error saving frame: {e}")
    
    def save_inference_data(self, features, boxes, heatmaps, frame_count, object_detections=None, gaze_targets=None):
        """Save raw inference results."""
        if not self.inference_output_dir:
            return
        
        try:
            result_dict = {
                'frame_number': frame_count,
                'timestamp': time.time(),
                'features': features.cpu().numpy(),
                'boxes': boxes,
                'heatmaps': heatmaps,
            }
            
            if object_detections:
                result_dict['object_detections'] = object_detections
            
            if gaze_targets:
                result_dict['gaze_targets'] = gaze_targets
            
            output_path = self.inference_output_dir / f"inference_{self.saved_inference + 1:04d}.npz"
            np.savez_compressed(output_path, **result_dict)
            
            print(f"Saved inference results {self.saved_inference + 1} to {output_path}")
            self.saved_inference += 1
            
        except Exception as e:
            print(f"[DEBUG] Error saving inference results: {e}")


# ============================================================================
# Timing Manager
# ============================================================================

class TimingManager:
    """Manages frame timing and save intervals."""
    
    def __init__(self, save_mode='time', save_interval=1.0, skip_frames=0):
        self.save_mode = save_mode
        self.save_interval = save_interval
        self.skip_frames = skip_frames
        
        # Timing tracking
        self.first_pts = None
        self.nominal_fps = DEFAULT_CONFIG['nominal_fps']
        self.last_save_ts = -1e9
        self.last_saved_orig_idx = -1e9
        self.last_inference_save_ts = -1e9
        self.last_inference_saved_orig_idx = -1e9
    
    def should_skip_frame(self, frame_count):
        """Check if frame should be skipped."""
        if self.skip_frames > 0:
            return frame_count % (self.skip_frames + 1) != 0
        return False
    
    def update_timing(self, pts_ms):
        """Update timing information from PTS."""
        if self.first_pts is None:
            self.first_pts = pts_ms
        return (pts_ms - self.first_pts) / 1000.0  # seconds since start
    
    def should_save_frame(self, rel_t):
        """Check if frame should be saved based on timing."""
        if self.save_mode == 'time':
            if rel_t - self.last_save_ts >= self.save_interval:
                self.last_save_ts = rel_t
                return True
        else:  # frame mode
            orig_idx = int(rel_t * self.nominal_fps)
            if orig_idx - self.last_saved_orig_idx >= self.save_interval:
                self.last_saved_orig_idx = orig_idx
                return True
        return False
    
    def should_save_inference(self, rel_t):
        """Check if inference results should be saved."""
        if self.save_mode == 'time':
            if rel_t - self.last_inference_save_ts >= self.save_interval:
                self.last_inference_save_ts = rel_t
                return True
        else:  # frame mode
            orig_idx = int(rel_t * self.nominal_fps)
            if orig_idx - self.last_inference_saved_orig_idx >= self.save_interval:
                self.last_inference_saved_orig_idx = orig_idx
                return True
        return False


# ============================================================================
# GazeLLE Callback Class
# ============================================================================

class GazeLLECallbackClass(app_callback_class):
    """Simplified callback class for GazeLLE processing."""
    
    def __init__(self, gazelle_model, device='cpu', output_dir='./output_frames',
                 save_interval=1.0, max_frames=10, hef_path=None, skip_frames=0,
                 save_inference_results=False, inference_output_dir='./inference_results',
                 save_mode='time', scrfd_hef_path=None, detr_hef_path=None,
                 detr_confidence=0.7, detr_nms=0.3):
        super().__init__()
        
        # Configuration
        self.device = device
        self.max_frames = max_frames
        self.save_inference_results = save_inference_results
        
        # Initialize components with shared VDevice
        shared_vdevice = None
        self.hailo_manager = HailoInferenceManager(hef_path, shared_vdevice) if hef_path else None
        if self.hailo_manager:
            shared_vdevice = self.hailo_manager.vdevice
        self.scrfd_manager = SCRFDInferenceManager(scrfd_hef_path, shared_vdevice) if scrfd_hef_path else None
        self.detr_manager = DETRInferenceManager(detr_hef_path, shared_vdevice, detr_confidence, detr_nms) if detr_hef_path else None
        self.frame_processor = FrameProcessor(gazelle_model, self.hailo_manager, device, self.scrfd_manager, self.detr_manager)
        self.result_saver = ResultSaver(output_dir, inference_output_dir if save_inference_results else None)
        self.timing_manager = TimingManager(save_mode, save_interval, skip_frames)
        
        # Tracking
        self.frame_count = 0
        self.processing_times = []
        
        # Print configuration
        self._print_config(output_dir, save_mode, save_interval, inference_output_dir)
        if scrfd_hef_path:
            print(f"SCRFD face detection model loaded from: {scrfd_hef_path}")
        if detr_hef_path:
            print(f"DETR object detection model loaded from: {detr_hef_path}")
    
    def _print_config(self, output_dir, save_mode, save_interval, inference_output_dir):
        """Print configuration summary."""
        print(f"Output frames will be saved to: {output_dir}")
        if save_mode == 'time':
            print(f"Saving frames every {save_interval} seconds")
        else:
            print(f"Saving frames every {save_interval} frames")
        if self.save_inference_results:
            print(f"Inference results will be saved to: {inference_output_dir}")
    
    def should_continue(self):
        """Check if processing should continue."""
        return True


# ============================================================================
# Main Callback Function
# ============================================================================

def gazelle_callback(pad, info, user_data):
    """Main GStreamer callback for processing frames."""
    user_data.frame_count += 1
    
    # Get buffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Get timing
    pts_ms = buffer.pts / Gst.SECOND * 1000
    rel_t = user_data.timing_manager.update_timing(pts_ms)
    
    # Check if should skip
    if user_data.timing_manager.should_skip_frame(user_data.frame_count):
        return Gst.PadProbeReturn.OK
    
    # Check if already saved max frames
    if user_data.result_saver.saved_frames >= user_data.max_frames:
        return Gst.PadProbeReturn.OK
    
    # Measure processing time
    start_time = time.time()
    
    # Debug logging
    if user_data.frame_count % DEBUG_INTERVALS['frame_info'] == 1:
        if user_data.processing_times:
            avg_time = sum(user_data.processing_times[-DEBUG_INTERVALS['processing_avg']:]) / \
                      len(user_data.processing_times[-DEBUG_INTERVALS['processing_avg']:])
            print(f"[DEBUG] Frame {user_data.frame_count}, Avg processing: {avg_time*1000:.1f}ms")
    
    # Get frame info
    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        return Gst.PadProbeReturn.OK
    
    # Get frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        return Gst.PadProbeReturn.OK
    
    # Debug first frame
    if user_data.frame_count == 1:
        print(f"[FORMAT CHECK] Buffer format: {format}, shape: {frame.shape}, dtype: {frame.dtype}")
    
    try:
        # Process frame
        results = user_data.frame_processor.process_frame(frame, width, height)
        
        # Add to GStreamer ROI (for pipeline integration)
        _add_roi_to_buffer(buffer, results['boxes'], results['heatmaps'], width, height)
        
        # Save visualizations
        if user_data.timing_manager.should_save_frame(rel_t):
            user_data.result_saver.save_visualization(
                frame, results['boxes'], results['heatmaps'], 
                results.get('object_detections'), results.get('gaze_targets')
            )
        
        # Save inference results
        if user_data.save_inference_results and user_data.timing_manager.should_save_inference(rel_t):
            user_data.result_saver.save_inference_data(
                results['features'], results['boxes'], results['heatmaps'], 
                user_data.frame_count, results.get('object_detections'), 
                results.get('gaze_targets')
            )
            
    except Exception as e:
        print(f"[DEBUG] Processing error: {e}")
        import traceback
        traceback.print_exc()
    
    # Record processing time
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    if len(user_data.processing_times) > 100:
        user_data.processing_times.pop(0)
    
    return Gst.PadProbeReturn.OK


def _add_roi_to_buffer(buffer, boxes, heatmaps, width, height):
    """Add ROI data to GStreamer buffer."""
    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        main_bbox = hailo.HailoBBox(0, 0, 1, 1)
        roi = hailo.HailoROI(main_bbox)
        hailo.add_roi_to_buffer(buffer, roi)
    
    for i, (bbox, heatmap) in enumerate(zip(boxes, heatmaps)):
        xmin, ymin, xmax, ymax = bbox
        norm_xmin = xmin / width
        norm_ymin = ymin / height
        norm_width = (xmax - xmin) / width
        norm_height = (ymax - ymin) / height
        
        face_bbox = hailo.HailoBBox(norm_xmin, norm_ymin, norm_width, norm_height)
        detection = hailo.HailoDetection(face_bbox, "face_with_gaze", 0.9)
        roi.add_object(detection)


# ============================================================================
# Command Line Parser
# ============================================================================

def get_gazelle_parser():
    """Create argument parser."""
    parser = get_default_parser()
    
    # Model arguments
    parser.add_argument("--hef", required=True, help="Path to compiled HEF backbone model file")
    parser.add_argument("--pth", required=True, help="Path to GazeLLE head checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="Torch device for GazeLLE head (cpu or cuda)")
    parser.add_argument("--scrfd-hef", help="Path to SCRFD HEF model for face detection")
    parser.add_argument("--detr-hef", help="Path to DETR HEF model for object detection")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./output_frames", help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to save")
    
    # Saving configuration
    parser.add_argument("--save-mode", choices=['time', 'frame'], default='time',
                       help="Save mode: 'time' for time-based intervals, 'frame' for frame count-based")
    parser.add_argument("--save-interval", type=float, default=1.0,
                       help="Save interval: seconds (if save-mode=time) or frames (if save-mode=frame)")
    
    # Processing options
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Skip N frames between processing (0=process all)")
    
    # Inference saving
    parser.add_argument("--save-inference", action="store_true",
                       help="Save raw inference results")
    parser.add_argument("--inference-output-dir", default="./inference_results",
                       help="Directory to save inference results")
    
    # Detection parameters
    parser.add_argument("--detr-confidence", type=float, default=0.7,
                       help="Confidence threshold for DETR object detection (default: 0.7)")
    parser.add_argument("--detr-nms", type=float, default=0.3,
                       help="NMS IoU threshold for DETR object detection (default: 0.3)")
    
    return parser


# ============================================================================
# GStreamer Application
# ============================================================================

class GStreamerGazeLLEApp(GStreamerApp):
    """GStreamer application for real-time gaze estimation."""
    
    def __init__(self, args, user_data):
        parser = get_gazelle_parser()
        super().__init__(parser, user_data)
        
        self.app_callback = gazelle_callback
        self._auto_detect_arch()
        self.hef_path = self.options_menu.hef
        self.create_pipeline()
    
    def _auto_detect_arch(self):
        """Auto-detect Hailo architecture if not specified."""
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
    
    def get_pipeline_string(self):
        """Build GStreamer pipeline string."""
        source = SOURCE_PIPELINE(
            self.video_source,
            self.video_width,
            self.video_height,
            self.video_format
        )
        
        callback = (
            f'queue name=hailo_pre_callback_q leaky=downstream '
            f'max-size-buffers=60 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback'
        )
        
        if self.options_menu.headless:
            display = 'fakesink name=hailo_display'
        else:
            display = DISPLAY_PIPELINE(
                video_sink=self.video_sink,
                sync=self.sync,
                show_fps=self.show_fps
            )
        
        pipeline = f'{source} ! {QUEUE("pre_callback_q")} ! {callback} ! {display}'
        print(f"Pipeline string: {pipeline}")
        return pipeline
    
    def setup_callback(self):
        """Setup callback on identity element."""
        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            pad = identity.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)
                print("Callback probe added successfully")
        else:
            print("Warning: Could not find identity element for callback")
    
    def on_pipeline_state_changed(self, bus, msg):
        """Handle pipeline state changes."""
        old_state, new_state, pending = msg.parse_state_changed()
        if msg.src == self.pipeline and new_state == Gst.State.PLAYING:
            self.setup_callback()
        super().on_pipeline_state_changed(bus, msg)


# ============================================================================
# Model Loading
# ============================================================================

def load_gazelle_model(pth_path, hef_path, device='cpu'):
    """Load and configure GazeLLE model."""
    import hailo_platform as hpf
    
    # Get HEF dimensions
    hef_model = hpf.HEF(hef_path)
    hef_h, hef_w = get_hef_input_dimensions(hef_model)
    print(f"HEF input resolution: {hef_w}x{hef_h}")
    
    # Create model
    backbone = DinoV2Backbone("dinov2_vits14")
    gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights (backbone excluded - using Hailo)
    gazelle_model.load_gazelle_state_dict(state_dict, include_backbone=False)
    gazelle_model.to(device)
    gazelle_model.eval()
    
    print("GazeLLE model loaded successfully")
    return gazelle_model


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    parser = get_gazelle_parser()
    args = parser.parse_args()
    
    # Load model
    print("Loading GazeLLE model...")
    gazelle_model = load_gazelle_model(args.pth, args.hef, args.device)
    
    # Create callback handler
    user_data = GazeLLECallbackClass(
        gazelle_model=gazelle_model,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        max_frames=args.max_frames,
        hef_path=args.hef,
        skip_frames=args.skip_frames,
        save_inference_results=args.save_inference,
        inference_output_dir=args.inference_output_dir,
        save_mode=args.save_mode,
        scrfd_hef_path=args.scrfd_hef,
        detr_hef_path=args.detr_hef,
        detr_confidence=args.detr_confidence,
        detr_nms=args.detr_nms
    )
    
    # Create and run application
    app = GStreamerGazeLLEApp(args, user_data)
    
    # Add heartbeat
    def heartbeat():
        print(f"[DEBUG] Heartbeat - Frames processed: {user_data.frame_count}")
        return True
    
    GLib.timeout_add_seconds(DEBUG_INTERVALS['heartbeat'], heartbeat)
    
    print("[DEBUG] Starting GStreamer app...")
    app.run()


if __name__ == "__main__":
    main()