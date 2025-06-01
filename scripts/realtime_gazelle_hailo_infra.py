#!/usr/bin/env python3
"""
Real-time gaze estimation using Hailo-8 infrastructure and GStreamer.
Uses hailo_apps_infra for camera capture and processing pipeline.
"""
import argparse
import sys
import time
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import torch
import hailo

# Add parent directory to path for gazelle modules
sys.path.append(str(Path(__file__).parent.parent))
from gazelle.model import GazeLLE
from gazelle.backbone import DinoV2Backbone

# Hailo infrastructure modules
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

# Local modules
from config import DEFAULT_CONFIG
from hailo_inference import HailoInferenceManager, SCRFDInferenceManager, DETRInferenceManager
from frame_processing import FrameProcessor
from visualization import ResultSaver
from timing_manager import TimingManager
from utils import get_hef_input_dimensions


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
        
        # Performance monitoring
        self.consecutive_slow_frames = 0
        self.slow_frame_threshold = 0.1  # 100ms processing time threshold
        self.adaptive_skip_multiplier = 1
        
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
        self.dropped_frame_count = 0
        
        print(f"[CONFIG] Output: {output_dir}, Save mode: {save_mode}, Interval: {save_interval}")
        if inference_output_dir:
            print(f"[CONFIG] Inference output: {inference_output_dir}")
    
    def should_continue(self):
        """Check if processing should continue."""
        return True


# ============================================================================
# Main Callback Function
# ============================================================================

def gazelle_callback(pad, info, user_data):
    """Main GStreamer callback for processing frames."""
    user_data.frame_count += 1
    
    # Get buffer and basic checks
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Get timing
    pts_ms = buffer.pts / Gst.SECOND * 1000
    rel_t = user_data.timing_manager.update_timing(pts_ms)
    
    # Early exit conditions with real-time timing
    current_time = time.time()
    if (user_data.timing_manager.should_skip_frame(user_data.frame_count, current_time, user_data.adaptive_skip_multiplier) or
        user_data.result_saver.saved_frames >= user_data.max_frames):
        user_data.dropped_frame_count += 1
        if user_data.frame_count % 300 == 0:  # Log every 10 seconds at 30fps
            drop_rate = user_data.dropped_frame_count / user_data.frame_count * 100
            avg_time = sum(user_data.processing_times[-30:]) / len(user_data.processing_times[-30:]) if user_data.processing_times else 0
            print(f"[PERF] Frame {user_data.frame_count}: Drop rate: {drop_rate:.1f}%, Avg processing: {avg_time*1000:.1f}ms")
        return Gst.PadProbeReturn.OK
    
    # Get frame data
    format, width, height = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None or format is None:
        return Gst.PadProbeReturn.OK
    
    # Process frame with error handling
    start_time = time.time()
    try:
        results = user_data.frame_processor.process_frame(frame, width, height)
        
        # Add ROI to buffer for pipeline integration
        _add_roi_to_buffer(buffer, results['boxes'], results['heatmaps'], width, height)
        
        # Save frame if conditions are met
        if results.get('should_save', False) and user_data.timing_manager.should_save_frame(rel_t):
            user_data.result_saver.save_visualization(
                frame, results['boxes'], results['heatmaps'], 
                results.get('object_detections'), results.get('gaze_targets'),
                highest_prob_target=results.get('highest_probability_target')
            )
            
            # Log result
            if results.get('highest_probability_target'):
                target = results['highest_probability_target']
                print(f"[FRAME {user_data.frame_count}] {target['source']}: "
                      f"{target['object']['class']} (prob: {target['probability']:.3f})")
        
        # Save inference data if enabled
        if (user_data.save_inference_results and results.get('should_save', False) and 
            user_data.timing_manager.should_save_inference(rel_t)):
            user_data.result_saver.save_inference_data(
                results['features'], results['boxes'], results['heatmaps'], 
                user_data.frame_count, results.get('object_detections'), 
                results.get('gaze_targets'), 
                highest_prob_target=results.get('highest_probability_target')
            )
            
    except Exception as e:
        print(f"Frame {user_data.frame_count} processing error: {e}")
    
    # Track processing time and adapt if needed
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    if len(user_data.processing_times) > 100:
        user_data.processing_times.pop(0)
    
    # Adaptive performance monitoring
    if processing_time > user_data.slow_frame_threshold:
        user_data.consecutive_slow_frames += 1
        if user_data.consecutive_slow_frames > 5:
            # Temporarily increase frame skipping for performance
            user_data.adaptive_skip_multiplier = min(3, user_data.adaptive_skip_multiplier + 1)
            print(f"[ADAPT] Slow processing detected, increasing skip multiplier to {user_data.adaptive_skip_multiplier}")
            user_data.consecutive_slow_frames = 0
    else:
        user_data.consecutive_slow_frames = max(0, user_data.consecutive_slow_frames - 1)
        if user_data.consecutive_slow_frames == 0 and user_data.adaptive_skip_multiplier > 1:
            user_data.adaptive_skip_multiplier = max(1, user_data.adaptive_skip_multiplier - 1)
            print(f"[ADAPT] Performance improved, reducing skip multiplier to {user_data.adaptive_skip_multiplier}")
    
    return Gst.PadProbeReturn.OK


def _add_roi_to_buffer(buffer, boxes, heatmaps, width, height):
    """Add ROI data to GStreamer buffer."""
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        if roi is None:
            main_bbox = hailo.HailoBBox(0, 0, 1, 1)
            roi = hailo.HailoROI(main_bbox)
            hailo.add_roi_to_buffer(buffer, roi)
        
        for i, (bbox, heatmap) in enumerate(zip(boxes, heatmaps)):
            xmin, ymin, xmax, ymax = bbox
            norm_xmin = max(0, min(1, xmin / width))
            norm_ymin = max(0, min(1, ymin / height))
            norm_width = max(0, min(1, (xmax - xmin) / width))
            norm_height = max(0, min(1, (ymax - ymin) / height))
            
            face_bbox = hailo.HailoBBox(norm_xmin, norm_ymin, norm_width, norm_height)
            detection = hailo.HailoDetection(face_bbox, "face_with_gaze", 0.9)
            roi.add_object(detection)
    except Exception as e:
        print(f"Warning: Failed to add ROI to buffer: {e}")


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
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG['output_dir'], 
                       help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_CONFIG['max_frames'], 
                       help="Maximum number of frames to save")
    
    # Saving configuration
    parser.add_argument("--save-mode", choices=['time', 'frame'], default=DEFAULT_CONFIG['save_mode'],
                       help="Save mode: 'time' for time-based intervals, 'frame' for frame count-based")
    parser.add_argument("--save-interval", type=float, default=DEFAULT_CONFIG['save_interval'],
                       help="Save interval: seconds (if save-mode=time) or frames (if save-mode=frame)")
    
    # Processing options
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--skip-frames", type=int, default=DEFAULT_CONFIG['skip_frames'],
                       help="Skip N frames between processing (0=process all)")
    
    # Inference saving
    parser.add_argument("--save-inference", action="store_true",
                       help="Save raw inference results")
    parser.add_argument("--inference-output-dir", default=DEFAULT_CONFIG['inference_output_dir'],
                       help="Directory to save inference results")
    
    # Detection parameters
    parser.add_argument("--detr-confidence", type=float, default=0.7,
                       help="Confidence threshold for DETR object detection")
    parser.add_argument("--detr-nms", type=float, default=0.3,
                       help="NMS IoU threshold for DETR object detection")
    
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
            f'max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
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
        
        pipeline = f'{source} ! queue name=pre_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! {callback} ! {display}'
        return pipeline
    
    def setup_callback(self):
        """Setup callback on identity element."""
        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            pad = identity.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)
    
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
    try:
        import hailo_platform as hpf
        
        # Validate file paths
        if not Path(pth_path).exists():
            raise FileNotFoundError(f"PTH file not found: {pth_path}")
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
        
        # Get HEF dimensions
        hef_model = hpf.HEF(hef_path)
        hef_h, hef_w = get_hef_input_dimensions(hef_model)
        print(f"[MODEL] HEF input dimensions: {hef_h}x{hef_w}")
        
        # Create model
        backbone = DinoV2Backbone("dinov2_vits14")
        gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
        
        # Load checkpoint
        print(f"[MODEL] Loading checkpoint: {pth_path}")
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
        
        print(f"[MODEL] GazeLLE model loaded successfully on device: {device}")
        return gazelle_model
        
    except Exception as e:
        print(f"[ERROR] Failed to load GazeLLE model: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Parse arguments
        parser = get_gazelle_parser()
        args = parser.parse_args()
        
        print("[INIT] Starting Gazelle real-time gaze estimation...")
        print(f"[INIT] Models: HEF={args.hef}, PTH={args.pth}")
        print(f"[INIT] Output: {args.output_dir}, Max frames: {args.max_frames}")
        
        # Load model
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
        
        print("[INIT] Starting GStreamer pipeline...")
        app.run()
        
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()