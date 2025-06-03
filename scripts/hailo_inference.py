#!/usr/bin/env python3
"""
Hailo inference managers for different models.
"""

import numpy as np
import cv2
from hailo_platform import (
    VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)
from config import COCO_CLASSES_92, DETECTION_THRESHOLDS
from utils import compute_iou, softmax


class HailoInferenceManager:
    """Manages Hailo AI accelerator for DINOv2 inference."""
    
    def __init__(self, hef_path, vdevice=None):
        """Initialize Hailo inference manager."""
        self.hef_path = hef_path
        self.vdevice = vdevice
        self._init_device()
    
    def _init_device(self):
        """Initialize Hailo device and load model."""
        import hailo_platform as hpf
        
        # Create virtual device if not provided
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
        pass  # Implementation for debugging if needed
    
    def run_inference(self, frame):
        """Run inference on input frame."""
        # Get input dimensions
        input_info = self.hef.get_input_vstream_infos()[0]
        input_shape = input_info.shape
        
        if len(input_shape) == 3:  # HWC format
            h, w = input_shape[0], input_shape[1]
        else:  # NHWC format
            h, w = input_shape[1], input_shape[2]
        
        # Store DETR input size for proper scaling
        if isinstance(self, DETRInferenceManager) and self.detr_input_size is None:
            self.detr_input_size = (h, w)
        
        # Resize frame
        resized = cv2.resize(frame, (w, h))
        
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
    
    def cleanup(self):
        """Clean up Hailo resources and clear cached data."""
        try:
            # Force network group deactivation if still active
            if hasattr(self, 'network_group') and self.network_group:
                # Try to safely deactivate any active configurations
                pass
            
            # Release VDevice if we own it
            if hasattr(self, 'vdevice') and self.vdevice:
                try:
                    # Only release if we created the device
                    if not hasattr(self, '_shared_vdevice'):
                        self.vdevice.release()
                        print("[PRIVACY] Hailo VDevice released")
                except Exception as e:
                    print(f"[PRIVACY] Warning: VDevice release error: {e}")
            
            # Clear references to large objects
            self.network_group = None
            self.hef = None
            print("[PRIVACY] Hailo inference manager cleaned up")
            
        except Exception as e:
            print(f"[PRIVACY] Hailo cleanup error: {e}")
    
    def flush_memory(self):
        """Force flush any cached inference data."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            print("[PRIVACY] Hailo memory flushed")
        except Exception as e:
            print(f"[PRIVACY] Memory flush warning: {e}")


class SCRFDInferenceManager(HailoInferenceManager):
    """Specialized manager for SCRFD object detection."""
    
    def __init__(self, hef_path, vdevice=None):
        """Initialize SCRFD inference manager."""
        super().__init__(hef_path, vdevice)
    
    def process_detections(self, outputs, threshold=0.5):
        """Process SCRFD outputs to get face detection results."""
        # SCRFD typically outputs multi-scale predictions
        # This is a simplified version - adjust for actual model output
        detections = []
        
        for output_name, output_data in outputs.items():
            if 'bbox' in output_name or 'loc' in output_name:
                # Process bounding box predictions
                # Shape depends on SCRFD variant
                continue
            elif 'conf' in output_name or 'cls' in output_name:
                # Process confidence scores
                continue
        
        # Currently returns empty list - implement based on actual SCRFD output format
        return detections


class DETRInferenceManager(HailoInferenceManager):
    """Specialized manager for DETR object detection."""
    
    def __init__(self, hef_path, vdevice=None, confidence_threshold=0.7, nms_threshold=0.3):
        """Initialize DETR inference manager."""
        super().__init__(hef_path, vdevice)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.COCO_CLASSES_92 = COCO_CLASSES_92
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
            # Based on output shapes:
            # conv116: (1, 1, 100, 4) - boxes
            # conv113: (1, 1, 100, 92) - class logits
            if 'conv116' in output_name or (output_data.shape[-1] == 4 and len(output_data.shape) >= 3):
                boxes_output = output_data
            elif 'conv113' in output_name or (output_data.shape[-1] > 4 and len(output_data.shape) >= 3):
                scores_output = output_data
        
        if boxes_output is not None and scores_output is not None:
            # Reshape to remove extra dimensions
            if len(boxes_output.shape) == 4:
                boxes_output = boxes_output[0, 0]  # Remove batch and extra dim
            elif len(boxes_output.shape) == 3:
                boxes_output = boxes_output[0]  # Remove batch
                
            if len(scores_output.shape) == 4:
                scores_output = scores_output[0, 0]  # Remove batch and extra dim
            elif len(scores_output.shape) == 3:
                scores_output = scores_output[0]  # Remove batch
            
            # Apply sigmoid to boxes if needed
            if boxes_output.min() < 0 or boxes_output.max() > 1:
                boxes_output = 1 / (1 + np.exp(-boxes_output))
            
            # Apply softmax to get probabilities from logits
            scores_probs = softmax(scores_output, axis=-1)
            
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
                    
                    # Get class name with bounds checking
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
                    
                    # Skip placeholder classes or background
                    if class_name == "N/A" or class_name == "__background__":
                        continue
                    
                    # Convert to corner format and scale to output image size
                    x1 = (cx - w/2) * img_width
                    y1 = (cy - h/2) * img_height
                    x2 = (cx + w/2) * img_width
                    y2 = (cy + h/2) * img_height
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y2 = max(0, min(y2, img_height - 1))
                    
                    # Convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Skip very small boxes
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # Minimum absolute size
                    if box_width < DETECTION_THRESHOLDS['min_box_size'] or box_height < DETECTION_THRESHOLDS['min_box_size']:
                        continue
                    
                    # Minimum relative size
                    min_relative_area = DETECTION_THRESHOLDS['min_relative_area'] * img_width * img_height
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
            detections = self._apply_nms(detections, iou_threshold=self.nms_threshold)
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return detections
        
        # Group detections by class for class-specific NMS
        detections_by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        # Apply NMS per class
        kept_detections = []
        
        for cls, class_detections in detections_by_class.items():
            # Sort by confidence (descending)
            class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS within this class
            kept_class_detections = []
            
            for i, det in enumerate(class_detections):
                # Check if this detection overlaps too much with any kept detection
                should_keep = True
                for j, kept in enumerate(kept_class_detections):
                    iou = compute_iou(det['bbox'], kept['bbox'])
                    
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    kept_class_detections.append(det)
            
            kept_detections.extend(kept_class_detections)
        
        # Sort all kept detections by confidence and limit to top N
        kept_detections = sorted(kept_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 20 detections
        return kept_detections[:20]