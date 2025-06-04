#!/usr/bin/env python3
"""
Frame processing logic for unified gazelle and object detection.
"""

import numpy as np
import torch
from facenet_pytorch import MTCNN
from utils import process_dino_features, normalize_bounding_boxes, compute_gaze_targets
from config import DETECTION_THRESHOLDS


class FrameProcessor:
    """Handles frame processing logic."""
    
    def __init__(self, gazelle_model, hailo_manager, device='cpu', scrfd_manager=None, detr_manager=None):
        self.gazelle_model = gazelle_model
        self.hailo_manager = hailo_manager
        self.scrfd_manager = scrfd_manager
        self.detr_manager = detr_manager
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
    
    def process_frame(self, frame, width, height):
        """Process a single frame with unified gazelle and DETR processing."""
        processing_results = {
            'features': None,
            'boxes': [],
            'heatmaps': [],
            'object_detections': [],
            'gaze_targets': [],
            'highest_probability_target': None,
            'should_save': False
        }
        
        # Step 1: Face detection first to avoid unnecessary processing
        face_detections = []
        if self.scrfd_manager:
            scrfd_results = self.scrfd_manager.run_inference(frame)
            scrfd_detections = self.scrfd_manager.process_detections(scrfd_results)
            face_detections.extend(scrfd_detections)
            face_boxes = [d['bbox'] for d in scrfd_detections if d.get('class') == 'face']
            if face_boxes:
                boxes = np.array(face_boxes)
            else:
                boxes, probs = self.mtcnn.detect(frame)
        else:
            boxes, probs = self.mtcnn.detect(frame)
        
        # Early exit if no faces detected - skip all processing including Hailo inference
        if boxes is None or len(boxes) == 0:
            processing_results['no_faces_detected'] = True
            return processing_results
        
        # Step 2: Run Hailo inference for DINOv2 features (only if faces detected)
        if not self.hailo_manager:
            return processing_results
            
        infer_results = self.hailo_manager.run_inference(frame)
        output_name = list(infer_results.keys())[0]
        feat_raw = infer_results[output_name]
        feat_processed = process_dino_features(feat_raw)
        feat_tensor = torch.from_numpy(feat_processed).to(self.device)
        processing_results['features'] = feat_tensor
        
        processing_results['boxes'] = boxes
        
        # Step 3: DETR object detection
        object_detections = []
        if self.detr_manager:
            detr_results = self.detr_manager.run_inference(frame)
            detr_detections = self.detr_manager.process_detections(detr_results, width, height)
            object_detections.extend(detr_detections)
        
        processing_results['object_detections'] = face_detections + object_detections
        
        # Step 4: Only proceed with gazelle processing if we have both features and objects
        if len(processing_results['object_detections']) == 0:
            return processing_results
        
        # Step 5: Run GazeLLE inference
        norm_bboxes = normalize_bounding_boxes(boxes, width, height)
        with torch.no_grad():
            out = self.gazelle_model({
                "extracted_features": feat_tensor, 
                "bboxes": norm_bboxes
            })
        
        heatmaps = out["heatmap"][0].cpu().numpy()
        processing_results['heatmaps'] = heatmaps
        
        # Step 6: Compute gaze targets and find highest probability
        gaze_targets = compute_gaze_targets(heatmaps, processing_results['object_detections'], width, height)
        processing_results['gaze_targets'] = gaze_targets
        
        # Step 7: Determine highest probability target across all results
        highest_prob_target = self._find_highest_probability_target(gaze_targets, processing_results['object_detections'])
        processing_results['highest_probability_target'] = highest_prob_target
        
        # Step 8: Only save if we have a high confidence result
        if highest_prob_target and highest_prob_target['probability'] > DETECTION_THRESHOLDS['gaze_confidence']:
            processing_results['should_save'] = True
        
        return processing_results
    
    def _find_highest_probability_target(self, gaze_targets, object_detections):
        """Find the highest probability target across all results."""
        highest_prob_target = None
        highest_prob_score = 0.0
        
        # Check gaze targets
        for target in gaze_targets:
            if target['gaze_object'] and target['gaze_object']['gaze_score'] > highest_prob_score:
                highest_prob_score = target['gaze_object']['gaze_score']
                highest_prob_target = {
                    'type': 'gaze_target',
                    'object': target['gaze_object'],
                    'probability': target['gaze_object']['gaze_score'],
                    'source': 'gazelle_gaze'
                }
        
        # Check object detection confidence scores
        for detection in object_detections:
            if detection['confidence'] > highest_prob_score:
                highest_prob_score = detection['confidence']
                highest_prob_target = {
                    'type': 'object_detection',
                    'object': detection,
                    'probability': detection['confidence'],
                    'source': 'detr_detection'
                }
        
        return highest_prob_target