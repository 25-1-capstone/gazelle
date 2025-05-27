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
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

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
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp


class GazeLLEDetectionCallbackClass(app_callback_class):
    """Custom callback class for combined GazeLLE and object detection processing"""
    def __init__(self, gazelle_model, device='cpu', output_dir='./output_frames', 
                 save_interval=1.0, max_frames=10, hef_path=None, skip_frames=0):
        super().__init__()
        self.gazelle_model = gazelle_model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
        self.max_frames = max_frames
        self.last_save_time = time.time()
        self.saved_frames = 0
        self.frame_count = 0  # Add frame counter
        self.last_tensors = None  # Store tensors from hailonet
        self.skip_frames = skip_frames  # Process every N+1 frames
        self.last_process_time = 0
        self.processing_times = []  # Track processing times
        self.last_detections = []  # Store object detections for visualization
        
        # Initialize Hailo device for direct inference
        if hef_path:
            self.init_hailo_inference(hef_path)
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        print(f"Initialized MTCNN face detector")
        print(f"Output frames will be saved to: {self.output_dir}")
    
    def init_hailo_inference(self, hef_path):
        """Initialize Hailo device for direct inference"""
        import hailo_platform as hpf
        # Create VDevice
        self.vdevice = VDevice()
        
        # Configure network
        self.hef = hpf.HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.vdevice.configure(self.hef, configure_params)[0]
        
        # Get input/output info with format type
        self.input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            
        # Create vstreams
        self.network_group = network_group
        print(f"[DEBUG] Initialized Hailo inference with {len(self.input_vstreams_params)} inputs and {len(self.output_vstreams_params)} outputs")
        
        # Get and print stream info from HEF
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        
        for info in input_vstream_info:
            print(f"[DEBUG] Input stream: {info.name}, shape: {info.shape}, format: {info.format}")
            
        for info in output_vstream_info:
            print(f"[DEBUG] Output stream: {info.name}, shape: {info.shape}, format: {info.format}")
    
    def run_hailo_inference(self, frame):
        """Run inference on frame using Hailo device"""
        # Get input shape from stored info
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        input_shape = input_vstream_info.shape
        
        # Determine H,W based on shape length
        if len(input_shape) == 3:  # HWC format
            h, w = input_shape[0], input_shape[1]
        else:  # NHWC format
            h, w = input_shape[1], input_shape[2]
        
        # Resize frame to model input size
        resized = cv2.resize(frame, (w, h))
        
        # Normalize to [0, 1] if needed
        normalized = resized.astype(np.float32) / 255.0
        
        # Run inference
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {list(self.input_vstreams_params.keys())[0]: np.expand_dims(normalized, axis=0)}
            
            with self.network_group.activate(None):
                infer_results = infer_pipeline.infer(input_data)
                
        return infer_results
        
    def should_continue(self):
        """Check if we should continue processing"""
        return self.saved_frames < self.max_frames


def gazelle_detection_callback(pad, info, user_data):
    """Callback function for processing frames with GazeLLE and object detection"""
    user_data.frame_count += 1
    
    # Skip frames if configured
    if user_data.skip_frames > 0 and user_data.frame_count % (user_data.skip_frames + 1) != 0:
        return Gst.PadProbeReturn.OK
    
    # Measure processing time
    start_time = time.time()
    
    # Only print debug every 10 frames to reduce noise
    if user_data.frame_count % 10 == 1:
        if user_data.processing_times:
            avg_time = sum(user_data.processing_times[-10:]) / len(user_data.processing_times[-10:])
            print(f"[DEBUG] Frame {user_data.frame_count}, Avg processing: {avg_time*1000:.1f}ms")
    
    # Get buffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Check if we should continue
    if not user_data.should_continue():
        return Gst.PadProbeReturn.OK
    
    # Get video frame
    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        return Gst.PadProbeReturn.OK
    
    # Get the frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        return Gst.PadProbeReturn.OK
    
    # Run Hailo inference directly on the frame
    if hasattr(user_data, 'run_hailo_inference'):
        try:
            # Run inference to get DINOv2 features
            infer_results = user_data.run_hailo_inference(frame)
            
            # Get the output tensor (DINOv2 features)
            output_name = list(infer_results.keys())[0]
            feat_raw = infer_results[output_name]
            
            # Convert to torch tensor and process shape
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
                print(f"[DEBUG] Unexpected feature shape: {feat_raw.shape}")
                return Gst.PadProbeReturn.OK
            
            print(f"[DEBUG] Processed feature shape: {feat_processed.shape}")
                
            feat_tensor = torch.from_numpy(feat_processed).to(user_data.device)
            
            # Face detection for bounding boxes
            boxes_original, probs = user_data.mtcnn.detect(frame)
            
            if boxes_original is not None and len(boxes_original) > 0:
                boxes = boxes_original
            else:
                # No faces detected, use center region
                boxes = np.array([[width*0.25, height*0.25, width*0.75, height*0.75]])
            
            # Normalize bounding boxes
            norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]]
            
            # GazeLLE inference
            with torch.no_grad():
                out = user_data.gazelle_model({"extracted_features": feat_tensor, "bboxes": norm_bboxes})
            
            # Get heatmaps
            heatmaps = out["heatmap"][0].cpu().numpy()
            
            # Get ROI from buffer - it should already exist from detection pipeline
            roi = hailo.get_roi_from_buffer(buffer)
            if roi is None:
                print("[WARNING] No ROI found in buffer from detection pipeline")
                return Gst.PadProbeReturn.OK
            
            # Get existing object detections from the buffer
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            user_data.last_detections = detections  # Store for visualization
            
            # Add gaze results to ROI as sub-objects
            for i, (bbox, heatmap) in enumerate(zip(boxes, heatmaps)):
                # Create a detection object for each face with gaze
                xmin, ymin, xmax, ymax = bbox
                # Normalize to [0,1]
                norm_xmin = xmin / width
                norm_ymin = ymin / height
                norm_width = (xmax - xmin) / width
                norm_height = (ymax - ymin) / height
                
                # Create bbox for face
                face_bbox = hailo.HailoBBox(norm_xmin, norm_ymin, norm_width, norm_height)
                
                # Add as detection with gaze info
                detection = hailo.HailoDetection(face_bbox, "face_with_gaze", 0.9)
                
                # Find gaze point (max value in heatmap)
                if heatmap.ndim == 3:
                    heatmap = heatmap.squeeze()
                gaze_y, gaze_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                # Convert gaze point to frame coordinates
                gaze_x_frame = int(gaze_x * width / heatmap.shape[1])
                gaze_y_frame = int(gaze_y * height / heatmap.shape[0])
                
                # Add gaze info as metadata (simplified - just save the point)
                # Note: Hailo landmarks API might be different, so we'll skip for now
                
                # Add detection to ROI
                roi.add_object(detection)
            
            # Save frame if interval has passed
            current_time = time.time()
            if current_time - user_data.last_save_time >= user_data.save_interval:
                save_gazelle_detection_frame(frame, boxes, heatmaps, user_data)
                user_data.last_save_time = current_time
                
        except Exception as e:
            print(f"[DEBUG] Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    # Record processing time
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    if len(user_data.processing_times) > 100:  # Keep only last 100 times
        user_data.processing_times.pop(0)
    
    return Gst.PadProbeReturn.OK


def save_gazelle_detection_frame(frame, boxes, heatmaps, user_data):
    """Save frame with gaze and object detection visualization"""
    try:
        frame_pil = Image.fromarray(frame)
        
        # Use first face's heatmap for visualization
        heatmap = heatmaps[0] if len(heatmaps) > 0 else np.zeros((frame.shape[0], frame.shape[1]))
        
        # Handle different heatmap dimensions
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()
        
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Resize heatmap if needed
        if heatmap.shape != (frame.shape[0], frame.shape[1]):
            heatmap_norm = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(frame_pil)
        plt.imshow(heatmap_norm, alpha=0.4, cmap='jet')
        
        # Draw bounding boxes
        ax = plt.gca()
        
        # Draw face/gaze boxes
        for i, bbox in enumerate(boxes):
            xmin, ymin, xmax, ymax = bbox
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               fill=False, edgecolor='lime', linewidth=3)
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f'Face {i+1}',
                   color='lime', fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Draw object detection boxes
        if hasattr(user_data, 'last_detections') and user_data.last_detections:
            height, width = frame.shape[:2]
            for detection in user_data.last_detections:
                label = detection.get_label()
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()
                
                # Convert normalized bbox to pixel coordinates
                xmin = int(bbox.xmin() * width)
                ymin = int(bbox.ymin() * height)
                bbox_width = int(bbox.width() * width)
                bbox_height = int(bbox.height() * height)
                
                # Different color for objects
                rect = plt.Rectangle((xmin, ymin), bbox_width, bbox_height,
                                   fill=False, edgecolor='cyan', linewidth=2)
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f'{label} ({confidence:.2f})',
                       color='cyan', fontsize=10, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.title(f"Gaze Estimation + Object Detection - Frame {user_data.saved_frames + 1}")
        plt.axis('off')
        
        # Save frame
        output_path = user_data.output_dir / f"frame_{user_data.saved_frames + 1:04d}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        print(f"Saved frame {user_data.saved_frames + 1} to {output_path}")
        user_data.saved_frames += 1
        
    except Exception as e:
        print(f"[DEBUG] Error saving frame: {e}")


def get_gazelle_detection_parser():
    """Get argument parser with GazeLLE and detection-specific arguments"""
    parser = get_default_parser()
    
    # Add GazeLLE-specific arguments
    parser.add_argument("--gazelle-hef", required=True, help="Path to compiled HEF backbone model file for GazeLLE")
    parser.add_argument("--pth", required=True, help="Path to GazeLLE head checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="Torch device for GazeLLE head (cpu or cuda)")
    parser.add_argument("--output-dir", default="./output_frames", help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to save")
    parser.add_argument("--save-interval", type=float, default=1.0, help="Interval between saving frames")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processing (0=process all)")
    
    # Detection model path will be handled by parent class via --hef argument
    
    return parser


class GStreamerGazeLLEDetectionApp(GStreamerDetectionApp):
    """GStreamer application for combined GazeLLE gaze estimation and object detection"""
    
    def __init__(self, args, user_data):
        # Initialize parent with custom parser
        parser = get_gazelle_parser()
        super().__init__(parser, user_data)
        
        # Set parameters
        self.app_callback = gazelle_detection_callback
        
        # Detect architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
        
        # Set HEF path
        self.hef_path = self.options_menu.hef
        
        # Create the pipeline after initialization
        self.create_pipeline()
    
    # Remove the get_pipeline_string method - parent class handles it
    # Remove setup_callback and on_pipeline_state_changed - parent class handles them


def main():
    # Create argument parser with GazeLLE and detection arguments
    parser = get_gazelle_detection_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load GazeLLE model
    print("Loading GazeLLE model...")
    
    # Get HEF input dimensions
    import hailo_platform as hpf
    hef_model = hpf.HEF(args.gazelle_hef)
    input_vs = hef_model.get_input_vstream_infos()
    if not input_vs:
        print("Failed to get HEF input info", file=sys.stderr)
        sys.exit(1)
    
    shape = input_vs[0].shape
    if len(shape) == 3:
        hef_h, hef_w = shape[0], shape[1]
    else:
        hef_h, hef_w = shape[1], shape[2]
    print(f"HEF input resolution: {hef_w}x{hef_h}")
    
    # Create GazeLLE model
    backbone = DinoV2Backbone("dinov2_vits14")
    gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
    
    # Load checkpoint
    checkpoint = torch.load(args.pth, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    gazelle_model.load_gazelle_state_dict(state_dict, include_backbone=False)
    gazelle_model.to(args.device)
    gazelle_model.eval()
    print("GazeLLE model loaded successfully")
    
    # Create user data with GazeLLE model
    user_data = GazeLLEDetectionCallbackClass(
        gazelle_model=gazelle_model,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        max_frames=args.max_frames,
        hef_path=args.gazelle_hef,
        skip_frames=args.skip_frames
    )
    
    # Create and run the app
    app = GStreamerGazeLLEDetectionApp(gazelle_detection_callback, user_data)
    
    # Add heartbeat to show the app is running
    def heartbeat():
        print(f"[DEBUG] Heartbeat - Frames processed: {user_data.frame_count}")
        return True
    
    GLib.timeout_add_seconds(5, heartbeat)
    
    print("[DEBUG] Starting GStreamer app...")
    app.run()


if __name__ == "__main__":
    main()
