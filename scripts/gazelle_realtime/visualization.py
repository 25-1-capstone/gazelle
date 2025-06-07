#!/usr/bin/env python3
"""
Visualization and result saving functionality for GazeLLE.

This module handles:
- Saving processed frames with gaze heatmap overlays
- Drawing bounding boxes for detected faces and objects
- Highlighting gaze targets
- Saving raw inference data for analysis
"""

import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from config import OBJECT_COLORS
from utils import create_directories, clear_output_directories


class ResultSaver:
    """Handles saving of frames and inference results."""
    
    def __init__(self, output_dir, inference_output_dir=None):
        self.output_dir = Path(output_dir)
        self.inference_output_dir = Path(inference_output_dir) if inference_output_dir else None
        
        # Clear and create directories for privacy
        dirs = [self.output_dir]
        if self.inference_output_dir:
            dirs.append(self.inference_output_dir)
        clear_output_directories(dirs)
        create_directories(dirs)
        
        # Initialize counters based on existing files
        self.saved_frames = self._get_next_frame_number()
        self.saved_inference = self._get_next_inference_number()
    
    def _get_next_frame_number(self):
        """Get the next frame number to avoid overwriting existing files."""
        existing_frames = list(self.output_dir.glob("frame_*.png"))
        if not existing_frames:
            return 0
        
        # Extract frame numbers and find the maximum
        frame_numbers = []
        for frame_file in existing_frames:
            try:
                # Extract number from filename like "frame_0001.png"
                number_str = frame_file.stem.split('_')[1]
                frame_numbers.append(int(number_str))
            except (IndexError, ValueError):
                continue
        
        return max(frame_numbers) if frame_numbers else 0
    
    def _get_next_inference_number(self):
        """Get the next inference number to avoid overwriting existing files."""
        if not self.inference_output_dir:
            return 0
            
        existing_inferences = list(self.inference_output_dir.glob("inference_*.npz"))
        if not existing_inferences:
            return 0
        
        # Extract inference numbers and find the maximum
        inference_numbers = []
        for inference_file in existing_inferences:
            try:
                # Extract number from filename like "inference_0001.npz"
                number_str = inference_file.stem.split('_')[1]
                inference_numbers.append(int(number_str))
            except (IndexError, ValueError):
                continue
        
        return max(inference_numbers) if inference_numbers else 0
    
    def save_visualization(self, frame, boxes, heatmaps, object_detections=None, gaze_targets=None, highest_prob_target=None):
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
            self._draw_face_boxes(ax, boxes)
            
            # Get gazed objects for highlighting
            gazed_objects = self._get_gazed_objects(gaze_targets)
            
            # Draw object detections if available
            if object_detections:
                self._draw_object_detections(ax, object_detections, gazed_objects)
            
            # Draw gaze points
            if gaze_targets:
                self._draw_gaze_points(ax, gaze_targets)
            
            # Update title with unified processing information
            title = self._create_title(highest_prob_target, gaze_targets)
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
            print(f"Error saving frame: {e}")
    
    def _draw_face_boxes(self, ax, boxes):
        """Draw face bounding boxes."""
        for i, bbox in enumerate(boxes):
            xmin, ymin, xmax, ymax = bbox
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               fill=False, edgecolor=OBJECT_COLORS['face'], linewidth=3)
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f'Face {i+1}',
                   color=OBJECT_COLORS['face'], fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    def _get_gazed_objects(self, gaze_targets):
        """Get list of objects being gazed at."""
        gazed_objects = []
        if gaze_targets:
            for target in gaze_targets:
                if target['gaze_object']:
                    gazed_objects.append(target['gaze_object'])
        return gazed_objects
    
    def _draw_object_detections(self, ax, object_detections, gazed_objects):
        """Draw object detection bounding boxes."""
        for detection in object_detections:
            if detection.get('class') != 'face':  # Skip faces as they're already drawn
                bbox = detection['bbox']
                xmin, ymin, xmax, ymax = bbox
                
                # Check if this object is being gazed at
                is_gazed, gaze_score = self._check_if_gazed(detection, gazed_objects)
                
                # Set visual properties based on gaze status
                if is_gazed:
                    color = OBJECT_COLORS['gazed']
                    linewidth = 5
                    label = f"[GAZE] {detection['class']} {detection.get('confidence', 0):.2f} (gaze: {gaze_score:.2f})"
                else:
                    color = self._get_object_color(detection['class'])
                    linewidth = 3
                    label = f"{detection['class']} {detection.get('confidence', 0):.2f}"
                
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   fill=False, edgecolor=color, linewidth=linewidth)
                ax.add_patch(rect)
                
                # Draw text label
                text_y = ymin-5 if ymin > 20 else ymax+20
                ax.text(xmin, text_y, label,
                       color=color, fontsize=10 if not is_gazed else 12, 
                       weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='black' if not is_gazed else 'darkred', 
                               alpha=0.7))
    
    def _check_if_gazed(self, detection, gazed_objects):
        """Check if an object is being gazed at."""
        for gazed_obj in gazed_objects:
            if (gazed_obj['bbox'] == detection['bbox'] and 
                gazed_obj['class'] == detection['class']):
                return True, gazed_obj['gaze_score']
        return False, 0
    
    def _get_object_color(self, class_name):
        """Get color for object based on its class."""
        class_lower = class_name.lower()
        
        if 'person' in class_lower:
            return OBJECT_COLORS['person']
        elif any(x in class_lower for x in ['chair', 'couch', 'bed']):
            return OBJECT_COLORS['chair']
        elif any(x in class_lower for x in ['mouse', 'keyboard', 'laptop', 'tv']):
            return OBJECT_COLORS['mouse']
        elif any(x in class_lower for x in ['plant', 'potted']):
            return OBJECT_COLORS['plant']
        else:
            return OBJECT_COLORS['default']
    
    def _draw_gaze_points(self, ax, gaze_targets):
        """Draw gaze points on the visualization."""
        for i, target in enumerate(gaze_targets):
            gaze_x, gaze_y = target['gaze_point']
            # Draw crosshair at gaze point
            ax.plot(gaze_x, gaze_y, 'r+', markersize=20, markeredgewidth=3)
            ax.add_patch(plt.Circle((gaze_x, gaze_y), 10, 
                                  fill=False, edgecolor='red', linewidth=2))
    
    def _create_title(self, highest_prob_target, gaze_targets):
        """Create title for the visualization."""
        title = f"Unified Processing - Frame {self.saved_frames + 1}"
        if highest_prob_target:
            target_obj = highest_prob_target['object']
            title += f" | Highest Prob: {target_obj['class']} ({highest_prob_target['probability']:.2f}) [{highest_prob_target['source']}]"
        elif gaze_targets and gaze_targets[0]['gaze_object']:
            gazed_class = gaze_targets[0]['gaze_object']['class']
            title += f" | Looking at: {gazed_class}"
        return title
    
    def save_inference_data(self, features, boxes, heatmaps, frame_count, object_detections=None, gaze_targets=None, highest_prob_target=None):
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
            
            if highest_prob_target:
                result_dict['highest_probability_target'] = highest_prob_target
            
            output_path = self.inference_output_dir / f"inference_{self.saved_inference + 1:04d}.npz"
            np.savez_compressed(output_path, **result_dict)
            
            print(f"Saved inference results {self.saved_inference + 1} to {output_path}")
            self.saved_inference += 1
            
        except Exception as e:
            print(f"Error saving inference results: {e}")