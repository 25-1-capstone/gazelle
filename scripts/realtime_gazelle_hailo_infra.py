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


# ============================================================================
# Hailo Inference Manager
# ============================================================================

class HailoInferenceManager:
    """Manages Hailo AI accelerator for DINOv2 inference."""
    
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self._init_device()
    
    def _init_device(self):
        """Initialize Hailo device and load model."""
        import hailo_platform as hpf
        
        # Create Virtual Device
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


# ============================================================================
# Frame Processor
# ============================================================================

class FrameProcessor:
    """Handles frame processing logic."""
    
    def __init__(self, gazelle_model, hailo_manager, device='cpu'):
        self.gazelle_model = gazelle_model
        self.hailo_manager = hailo_manager
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        print("Initialized MTCNN face detector")
    
    def process_frame(self, frame, width, height):
        """Process a single frame and return gaze results."""
        # Run Hailo inference for features
        infer_results = self.hailo_manager.run_inference(frame)
        
        # Extract and process features
        output_name = list(infer_results.keys())[0]
        feat_raw = infer_results[output_name]
        feat_processed = process_dino_features(feat_raw)
        feat_tensor = torch.from_numpy(feat_processed).to(self.device)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(frame)
        if boxes is None or len(boxes) == 0:
            # Use center region if no faces detected
            boxes = np.array([[width*0.25, height*0.25, width*0.75, height*0.75]])
        
        # Normalize bounding boxes
        norm_bboxes = normalize_bounding_boxes(boxes, width, height)
        
        # Run GazeLLE inference
        with torch.no_grad():
            out = self.gazelle_model({
                "extracted_features": feat_tensor, 
                "bboxes": norm_bboxes
            })
        
        heatmaps = out["heatmap"][0].cpu().numpy()
        
        return {
            'features': feat_tensor,
            'boxes': boxes,
            'heatmaps': heatmaps
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
    
    def save_visualization(self, frame, boxes, heatmaps):
        """Save frame with gaze visualization."""
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
            
            plt.title(f"Gaze Estimation - Frame {self.saved_frames + 1}")
            plt.axis('off')
            
            # Save
            output_path = self.output_dir / f"frame_{self.saved_frames + 1:04d}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            print(f"Saved frame {self.saved_frames + 1} to {output_path}")
            self.saved_frames += 1
            
        except Exception as e:
            print(f"[DEBUG] Error saving frame: {e}")
    
    def save_inference_data(self, features, boxes, heatmaps, frame_count):
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
                 save_mode='time'):
        super().__init__()
        
        # Configuration
        self.device = device
        self.max_frames = max_frames
        self.save_inference_results = save_inference_results
        
        # Initialize components
        self.hailo_manager = HailoInferenceManager(hef_path) if hef_path else None
        self.frame_processor = FrameProcessor(gazelle_model, self.hailo_manager, device)
        self.result_saver = ResultSaver(output_dir, inference_output_dir if save_inference_results else None)
        self.timing_manager = TimingManager(save_mode, save_interval, skip_frames)
        
        # Tracking
        self.frame_count = 0
        self.processing_times = []
        
        # Print configuration
        self._print_config(output_dir, save_mode, save_interval, inference_output_dir)
    
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
            user_data.result_saver.save_visualization(frame, results['boxes'], results['heatmaps'])
        
        # Save inference results
        if user_data.save_inference_results and user_data.timing_manager.should_save_inference(rel_t):
            user_data.result_saver.save_inference_data(
                results['features'], results['boxes'], results['heatmaps'], user_data.frame_count
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
        save_mode=args.save_mode
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