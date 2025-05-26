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
from hailo_apps_infra.gstreamer_app import GStreamerApp
from hailo_apps_infra.gstreamer_helper_pipelines import (
    QUEUE,
    SOURCE_PIPELINE,
    DISPLAY_PIPELINE,
)


class GazeLLECallbackClass(app_callback_class):
    """
    Custom callback class for GazeLLE processing
    
    This class handles the main processing pipeline:
    1. Receives video frames from GStreamer
    2. Runs DINOv2 feature extraction using Hailo accelerator
    3. Detects faces using MTCNN
    4. Performs gaze estimation using GazeLLE model
    5. Saves results and visualizations
    """
    def __init__(self, gazelle_model, device='cpu', output_dir='./output_frames', 
                 save_interval=1.0, max_frames=10, hef_path=None, skip_frames=0,
                 save_inference_results=False, inference_output_dir='./inference_results',
                 save_mode='time'):
        # Initialize parent callback class
        super().__init__()
        
        # Core models and processing configuration
        self.gazelle_model = gazelle_model  # Pre-trained GazeLLE gaze estimation model
        self.device = device  # PyTorch device (cpu/cuda) for GazeLLE head processing
        
        # Output directory setup for saving processed frames
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Frame saving configuration
        self.save_interval = save_interval  # How often to save frames (time/frame based)
        self.max_frames = max_frames  # Maximum number of frames to save before stopping
        self.last_save_time = time.time()  # Track last save time for time-based saving
        self.saved_frames = 0  # Counter for saved frames
        
        # Frame processing tracking
        self.frame_count = 0  # Total frames processed counter
        self.last_tensors = None  # Store feature tensors from Hailo inference
        self.skip_frames = skip_frames  # Skip N frames between processing (0=process all)
        self.last_process_time = 0  # Track timing for performance monitoring
        self.processing_times = []  # List to track processing times for averaging
        
        # Save mode configuration - determines how saving intervals are calculated
        self.save_mode = save_mode  # 'time' or 'frame' based saving
        # save_interval interpretation:
        # - 'time': interval in seconds (float) - save every N seconds
        # - 'frame': interval in frames (int) - save every N frames
        self.save_interval = int(save_interval) if save_mode == 'frame' else float(save_interval)
        self.last_saved_frame_count = 0  # Track last saved frame number for frame-based saving
        
        # Inference results saving configuration - saves raw model outputs for analysis
        self.save_inference_results = save_inference_results  # Enable saving raw inference data
        self.inference_output_dir = Path(inference_output_dir)  # Directory for .npz files
        if self.save_inference_results:
            self.inference_output_dir.mkdir(exist_ok=True)
        self.last_inference_save_time = time.time()  # Track timing for inference result saving
        self.last_inference_saved_frame_count = 0  # Track last saved inference frame
        self.saved_inference_results = 0  # Counter for saved inference result files
        
        # Initialize Hailo AI accelerator for DINOv2 feature extraction
        if hef_path:
            self.init_hailo_inference(hef_path)  # Setup Hailo device with compiled HEF model
        
        # Initialize MTCNN for face detection (runs on CPU)
        # keep_all=True ensures we detect multiple faces in a frame
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        
        # Print configuration summary
        print(f"Initialized MTCNN face detector")
        print(f"Output frames will be saved to: {self.output_dir}")
        if self.save_mode == 'time':
            print(f"Saving frames every {self.save_interval} seconds")
        else:
            print(f"Saving frames every {self.save_interval} frames")
        if self.save_inference_results:
            print(f"Inference results will be saved to: {self.inference_output_dir}")
    
    def init_hailo_inference(self, hef_path):
        """
        Initialize Hailo AI accelerator for DINOv2 backbone inference
        
        Sets up the Hailo device with the compiled HEF (Hailo Executable Format) model
        for accelerated DINOv2 feature extraction. This replaces the PyTorch DINOv2
        backbone with hardware-accelerated inference.
        
        Args:
            hef_path (str): Path to the compiled .hef model file
        """
        import hailo_platform as hpf
        
        # Create Virtual Device - represents the Hailo AI processor
        self.vdevice = VDevice()
        
        # Load and configure the compiled neural network model
        self.hef = hpf.HEF(hef_path)  # Load HEF (Hailo Executable Format) file
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.vdevice.configure(self.hef, configure_params)[0]
        
        # Setup input/output stream parameters for inference
        # Input: UINT8 quantized format (0-255 pixel values)
        # Output: FLOAT32 format for DINOv2 features
        self.input_vstreams_params = InputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            
        # Store network group for inference execution
        self.network_group = network_group
        print(f"[DEBUG] Initialized Hailo inference with {len(self.input_vstreams_params)} inputs and {len(self.output_vstreams_params)} outputs")
        
        # Debug: Print model input/output specifications from HEF file
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        
        # Display input specifications (image dimensions, data format)
        for info in input_vstream_info:
            print(f"[DEBUG] Input stream: {info.name}, shape: {info.shape}, format: {info.format}")
            
        # Display output specifications (feature tensor dimensions)
        for info in output_vstream_info:
            print(f"[DEBUG] Output stream: {info.name}, shape: {info.shape}, format: {info.format}")
    
    def run_hailo_inference(self, frame):
        """
        Run DINOv2 feature extraction on input frame using Hailo accelerator
        
        Takes a video frame, preprocesses it for the DINOv2 model, runs inference
        on the Hailo AI processor, and returns the extracted features.
        
        Args:
            frame (np.ndarray): Input video frame in RGB format [H,W,C]
            
        Returns:
            dict: Dictionary containing feature tensors from DINOv2 backbone
        """
        # Get required input dimensions from the compiled model
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        input_shape = input_vstream_info.shape
        
        # Extract height and width from shape (handle different tensor formats)
        if len(input_shape) == 3:  # HWC format (Height, Width, Channels)
            h, w = input_shape[0], input_shape[1]
        else:  # NHWC format (Batch, Height, Width, Channels)
            h, w = input_shape[1], input_shape[2]
        
        # Resize input frame to match model's expected input size
        resized = cv2.resize(frame, (w, h))
        
        # Debug: Check frame format after preprocessing (first inference only)
        if hasattr(self, '_first_inference'):
            if not self._first_inference:
                print(f"[FORMAT CHECK] After cv2.resize - shape: {resized.shape}, sample pixel: {resized[h//2, w//2]}")
                self._first_inference = True
        else:
            self._first_inference = False
        
        # Keep original UINT8 pixel values (0-255 range) as expected by quantized HEF model
        # No normalization needed - the compiled model handles quantization internally
        
        # Execute inference on Hailo AI processor
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            # Prepare input data: add batch dimension [1, H, W, C]
            input_data = {list(self.input_vstreams_params.keys())[0]: np.expand_dims(resized, axis=0)}
            
            # Activate network and run inference
            with self.network_group.activate(None):
                infer_results = infer_pipeline.infer(input_data)
                
        return infer_results
        
    def should_continue(self):
        """
        Check if the processing pipeline should continue running
        
        Currently always returns True to keep the pipeline running continuously.
        Frame saving limits are handled separately in the callback function.
        
        Returns:
            bool: True to continue processing, False to stop
        """
        # Keep the GStreamer pipeline running continuously
        # Frame saving is limited separately by max_frames parameter
        return True


def gazelle_callback(pad, info, user_data):
    """
    Main GStreamer callback function for real-time gaze estimation
    
    This function is called for each video frame in the GStreamer pipeline and performs:
    1. Frame preprocessing and format conversion
    2. Face detection using MTCNN
    3. DINOv2 feature extraction using Hailo accelerator
    4. Gaze estimation using GazeLLE model
    5. Result visualization and saving
    
    Args:
        pad: GStreamer pad (pipeline element connection point)
        info: GStreamer probe info containing the video buffer
        user_data: GazeLLECallbackClass instance with models and configuration
        
    Returns:
        Gst.PadProbeReturn.OK: Signal to continue pipeline processing
    """
    user_data.frame_count += 1
    
    # Get buffer and timestamp
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Get PTS (presentation timestamp) for accurate timing
    pts_ms = buffer.pts / Gst.SECOND * 1000  # nanoseconds to milliseconds
    if not hasattr(user_data, 'first_pts'):
        user_data.first_pts = pts_ms
        user_data.nominal_fps = 30  # Default to 30fps, could be detected from caps
    rel_t = (pts_ms - user_data.first_pts) / 1000.0  # seconds since start
    
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
    
    # Check if we should continue processing (but not necessarily saving)
    if not user_data.should_continue():
        return Gst.PadProbeReturn.OK
    
    # Check if we've already saved max frames
    if user_data.saved_frames >= user_data.max_frames:
        return Gst.PadProbeReturn.OK
    
    # Get video frame
    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        return Gst.PadProbeReturn.OK
    
    # Debug: Print format
    if user_data.frame_count == 1:
        print(f"[FORMAT CHECK] GStreamer buffer format: {format}")
    
    # Get the frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        return Gst.PadProbeReturn.OK
    
    # Debug: Check frame shape and sample pixel
    if user_data.frame_count == 1:
        print(f"[FORMAT CHECK] Frame shape: {frame.shape}")
        print(f"[FORMAT CHECK] Frame dtype: {frame.dtype}")
        # Sample a pixel to check channel order (center pixel)
        center_y, center_x = frame.shape[0]//2, frame.shape[1]//2
        pixel = frame[center_y, center_x]
        print(f"[FORMAT CHECK] Center pixel RGB values: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")
    
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
            # Debug: MTCNN expects RGB format
            if user_data.frame_count == 1:
                print(f"[FORMAT CHECK] Sending frame to MTCNN with shape: {frame.shape}")
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
            
            # Get or create ROI from buffer
            roi = hailo.get_roi_from_buffer(buffer)
            if roi is None:
                # Create main ROI covering the whole frame
                main_bbox = hailo.HailoBBox(0, 0, 1, 1)
                roi = hailo.HailoROI(main_bbox)
                hailo.add_roi_to_buffer(buffer, roi)
            
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
            
            # Save frame based on configured mode using original timestamps
            should_save_frame = False
            
            if user_data.save_mode == 'time':
                # Time-based saving using PTS
                if rel_t - getattr(user_data, 'last_save_ts', -1e9) >= user_data.save_interval:
                    should_save_frame = True
                    user_data.last_save_ts = rel_t
            elif user_data.save_mode == 'frame':
                # Frame count-based saving using original frame index
                orig_frame_idx = int(rel_t * user_data.nominal_fps)
                if orig_frame_idx - getattr(user_data, 'last_saved_orig_idx', -1e9) >= user_data.save_interval:
                    should_save_frame = True
                    user_data.last_saved_orig_idx = orig_frame_idx
            
            if should_save_frame:
                save_gazelle_frame(frame, boxes, heatmaps, user_data)
            
            # Save inference results if enabled (using same timestamp-based logic)
            if user_data.save_inference_results:
                should_save_inference = False
                
                if user_data.save_mode == 'time':
                    if rel_t - getattr(user_data, 'last_inference_save_ts', -1e9) >= user_data.save_interval:
                        should_save_inference = True
                        user_data.last_inference_save_ts = rel_t
                elif user_data.save_mode == 'frame':
                    orig_frame_idx = int(rel_t * user_data.nominal_fps)
                    if orig_frame_idx - getattr(user_data, 'last_inference_saved_orig_idx', -1e9) >= user_data.save_interval:
                        should_save_inference = True
                        user_data.last_inference_saved_orig_idx = orig_frame_idx
                
                if should_save_inference:
                    save_inference_results(feat_tensor, boxes, heatmaps, user_data)
                
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


def save_gazelle_frame(frame, boxes, heatmaps, user_data):
    """
    Save video frame with gaze estimation visualization overlay
    
    Creates a visualization showing:
    - Original video frame
    - Gaze heatmap overlay (color-coded attention map)
    - Face bounding boxes with labels
    
    Args:
        frame (np.ndarray): Original video frame [H,W,C]
        boxes (np.ndarray): Face bounding boxes [N,4] (xmin,ymin,xmax,ymax)
        heatmaps (np.ndarray): Gaze attention heatmaps [N,H,W]
        user_data: GazeLLECallbackClass instance with save configuration
    """
    try:
        # Debug: PIL expects RGB format
        if user_data.saved_frames == 0:
            print(f"[FORMAT CHECK] Creating PIL image from array with shape: {frame.shape}")
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
        for i, bbox in enumerate(boxes):
            xmin, ymin, xmax, ymax = bbox
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               fill=False, edgecolor='lime', linewidth=3)
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f'Face {i+1}',
                   color='lime', fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.title(f"Gaze Estimation - Frame {user_data.saved_frames + 1}")
        plt.axis('off')
        
        # Save frame
        output_path = user_data.output_dir / f"frame_{user_data.saved_frames + 1:04d}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        print(f"Saved frame {user_data.saved_frames + 1} to {output_path}")
        user_data.saved_frames += 1
        
    except Exception as e:
        print(f"[DEBUG] Error saving frame: {e}")


def save_inference_results(feat_tensor, boxes, heatmaps, user_data):
    """
    Save raw inference results to disk for offline analysis
    
    Saves all intermediate processing results in compressed NumPy format:
    - DINOv2 feature tensors from Hailo accelerator
    - Face bounding boxes from MTCNN
    - Gaze heatmaps from GazeLLE model
    - Frame metadata (timestamp, frame number)
    
    Args:
        feat_tensor (torch.Tensor): DINOv2 features [1,C,H,W]
        boxes (np.ndarray): Face bounding boxes [N,4]
        heatmaps (np.ndarray): Gaze heatmaps [N,H,W]
        user_data: GazeLLECallbackClass instance with save configuration
    """
    try:
        # Create result dictionary
        result_dict = {
            'frame_number': user_data.frame_count,
            'timestamp': time.time(),
            'features': feat_tensor.cpu().numpy(),  # DINOv2 features [1, C, H, W]
            'boxes': boxes,  # Face bounding boxes
            'heatmaps': heatmaps,  # Gaze heatmaps
        }
        
        # Save as numpy compressed file
        output_path = user_data.inference_output_dir / f"inference_{user_data.saved_inference_results + 1:04d}.npz"
        np.savez_compressed(output_path, **result_dict)
        
        print(f"Saved inference results {user_data.saved_inference_results + 1} to {output_path}")
        user_data.saved_inference_results += 1
        
    except Exception as e:
        print(f"[DEBUG] Error saving inference results: {e}")


def get_gazelle_parser():
    """
    Create command-line argument parser for GazeLLE application
    
    Builds upon the default Hailo infrastructure parser and adds
    GazeLLE-specific configuration options for models, saving, and processing.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = get_default_parser()
    
    # Add GazeLLE-specific arguments
    parser.add_argument("--hef", required=True, help="Path to compiled HEF backbone model file")
    parser.add_argument("--pth", required=True, help="Path to GazeLLE head checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="Torch device for GazeLLE head (cpu or cuda)")
    parser.add_argument("--output-dir", default="./output_frames", help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to save")
    parser.add_argument("--save-mode", choices=['time', 'frame'], default='time', help="Save mode: 'time' for time-based intervals, 'frame' for frame count-based intervals")
    parser.add_argument("--save-interval", type=float, default=1.0, help="Save interval: seconds (if save-mode=time) or frames (if save-mode=frame)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processing (0=process all)")
    parser.add_argument("--save-inference", action="store_true", help="Save raw inference results (features, boxes, heatmaps)")
    parser.add_argument("--inference-output-dir", default="./inference_results", help="Directory to save inference results")
    
    return parser


class GStreamerGazeLLEApp(GStreamerApp):
    """
    Main GStreamer application for real-time gaze estimation using GazeLLE
    
    This class extends the base GStreamerApp with GazeLLE-specific functionality:
    - Sets up video capture pipeline with Hailo infrastructure
    - Configures callback for frame processing
    - Manages pipeline state and error handling
    - Supports both live camera and file input
    """
    
    def __init__(self, args, user_data):
        """Initialize GStreamer application with GazeLLE-specific configuration"""
        # Initialize parent GStreamer application
        parser = get_gazelle_parser()
        super().__init__(parser, user_data)
        
        # Set main processing callback function
        self.app_callback = gazelle_callback
        
        # Auto-detect or use specified Hailo architecture
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
        
        # Store HEF model path for Hailo inference
        self.hef_path = self.options_menu.hef
        
        # Build and initialize the GStreamer pipeline
        self.create_pipeline()
    
    def get_pipeline_string(self):
        """
        Build the GStreamer pipeline string for video processing
        
        Creates a pipeline that:
        1. Captures video from camera or file
        2. Processes frames through identity element with callback
        3. Displays or discards output (headless mode)
        
        Returns:
            str: Complete GStreamer pipeline string
        """
        # === Pipeline Components ===
        
        # 1. Video input (camera or file source)
        source_pipeline = SOURCE_PIPELINE(
            self.video_source,  # Camera device or video file path
            self.video_width,   # Frame width
            self.video_height,  # Frame height
            self.video_format   # Pixel format (RGB, BGR, etc.)
        )
        
        # 2. Frame processing with callback
        # Buffer management for real-time processing:
        # - leaky=downstream: drops new frames when queue is full (preserves old frames)
        # - max-size-buffers=60: allows 2 seconds of buffering at 30fps
        user_callback_pipeline = (
            f'queue name=hailo_pre_callback_q leaky=downstream max-size-buffers=60 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback '  # Identity element where we attach frame processing callback
        )
        
        # 3. Output handling (display or headless)
        if self.options_menu.headless:
            # Headless mode: discard frames after processing
            display_pipeline = 'fakesink name=hailo_display'
        else:
            # GUI mode: display processed frames
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink,  # Display method (autovideosink, etc.)
                sync=self.sync,              # Synchronize with clock
                show_fps=self.show_fps       # Show FPS overlay
            )
        
        # === Assemble Complete Pipeline ===
        # Flow: Source -> Queue -> Processing Callback -> Display/Sink
        pipeline_string = (
            f'{source_pipeline} ! '          # Video input
            f'{QUEUE("pre_callback_q")} ! '   # Buffer queue
            f'{user_callback_pipeline} ! '   # Frame processing
            f'{display_pipeline}'            # Output
        )
        
        print(f"Pipeline string: {pipeline_string}")
        return pipeline_string
    
    def setup_callback(self):
        """
        Attach frame processing callback to the identity element
        
        This method finds the identity element in the pipeline and adds
        a probe that calls our processing function for each video frame.
        """
        # Locate the identity element in the pipeline
        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            # Get the source pad (output) of the identity element
            pad = identity.get_static_pad("src")
            if pad:
                # Attach our processing callback to intercept each video buffer
                pad.add_probe(
                    Gst.PadProbeType.BUFFER,  # Trigger on each video buffer
                    self.app_callback,         # Our processing function
                    self.user_data            # User data (GazeLLECallbackClass)
                )
                print("Callback probe added successfully")
        else:
            print("Warning: Could not find identity element for callback")
        
        print("[DEBUG] Pipeline state is PLAYING, waiting for frames...")
    
    def on_pipeline_state_changed(self, bus, msg):
        """
        Handle GStreamer pipeline state transitions
        
        When the pipeline transitions to PLAYING state, we set up
        the frame processing callback.
        
        Args:
            bus: GStreamer message bus
            msg: State change message
        """
        old_state, new_state, pending = msg.parse_state_changed()
        if msg.src == self.pipeline:
            if new_state == Gst.State.PLAYING:
                # Pipeline is now running - attach our callback
                self.setup_callback()
        
        # Call parent class handler for other state changes
        super().on_pipeline_state_changed(bus, msg)


def main():
    """
    Main entry point for real-time GazeLLE gaze estimation application
    
    This function:
    1. Parses command-line arguments
    2. Loads and configures the GazeLLE model
    3. Sets up Hailo AI accelerator
    4. Creates the GStreamer processing pipeline
    5. Runs the real-time inference loop
    """
    # === Command Line Configuration ===
    parser = get_gazelle_parser()
    args = parser.parse_args()
    
    # === Model Loading and Setup ===
    print("Loading GazeLLE model...")
    
    # === Hailo Model Configuration ===
    # Load HEF (Hailo Executable Format) to get model specifications
    import hailo_platform as hpf
    hef_model = hpf.HEF(args.hef)
    
    # Extract input stream information from compiled model
    input_vs = hef_model.get_input_vstream_infos()
    
    # Validate that model has input specifications
    if not input_vs:
        print("Failed to get HEF input info", file=sys.stderr)
        sys.exit(1)
    
    # Extract input dimensions from HEF model
    shape = input_vs[0].shape
    if len(shape) == 3:  # HWC format
        hef_h, hef_w = shape[0], shape[1]
    else:  # NHWC format
        hef_h, hef_w = shape[1], shape[2]
    print(f"HEF input resolution: {hef_w}x{hef_h}")
    
    # === GazeLLE Model Initialization ===
    # Create DINOv2 backbone (will be replaced by Hailo accelerator)
    backbone = DinoV2Backbone("dinov2_vits14")
    # Initialize GazeLLE model with input/output dimensions matching HEF
    gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
    
    # Load pre-trained GazeLLE weights from checkpoint
    checkpoint = torch.load(args.pth, map_location="cpu")
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # Assume direct state dict
    else:
        state_dict = checkpoint
    
    # Load only the head weights (backbone will use Hailo accelerator)
    gazelle_model.load_gazelle_state_dict(state_dict, include_backbone=False)
    gazelle_model.to(args.device)  # Move to specified device (CPU/GPU)
    gazelle_model.eval()  # Set to evaluation mode
    print("GazeLLE model loaded successfully")
    
    # === Application Configuration ===
    # Create callback handler with all models and configuration
    user_data = GazeLLECallbackClass(
        gazelle_model=gazelle_model,                    # Loaded GazeLLE model
        device=args.device,                             # PyTorch device
        output_dir=args.output_dir,                     # Output directory for visualizations
        save_interval=args.save_interval,               # Frame/time saving interval
        max_frames=args.max_frames,                     # Maximum frames to save
        hef_path=args.hef,                             # Path to Hailo model
        skip_frames=args.skip_frames,                   # Frame skipping for performance
        save_inference_results=args.save_inference,     # Save raw inference data
        inference_output_dir=args.inference_output_dir, # Raw data output directory
        save_mode=args.save_mode                        # Saving mode (time/frame)
    )
    
    # === Application Execution ===
    # Create GStreamer application with our configuration
    app = GStreamerGazeLLEApp(args, user_data)
    
    # Add periodic status updates (every 5 seconds)
    def heartbeat():
        print(f"[DEBUG] Heartbeat - Frames processed: {user_data.frame_count}")
        return True  # Continue periodic calls
    
    GLib.timeout_add_seconds(5, heartbeat)
    
    # Start the real-time processing loop
    print("[DEBUG] Starting GStreamer app...")
    app.run()  # This blocks until the application is stopped


if __name__ == "__main__":
    main()