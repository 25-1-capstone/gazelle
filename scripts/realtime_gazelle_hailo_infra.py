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
        
        # Save mode configuration
        self.save_mode = save_mode  # 'time' or 'frame'
        # save_interval is interpreted based on save_mode:
        # - 'time': interval in seconds (float)
        # - 'frame': interval in frames (int)
        self.save_interval = int(save_interval) if save_mode == 'frame' else float(save_interval)
        self.last_saved_frame_count = 0  # Track last saved frame for frame-based saving
        
        # Inference results saving
        self.save_inference_results = save_inference_results
        self.inference_output_dir = Path(inference_output_dir)
        if self.save_inference_results:
            self.inference_output_dir.mkdir(exist_ok=True)
        self.last_inference_save_time = time.time()
        self.last_inference_saved_frame_count = 0  # Track last saved inference frame
        self.saved_inference_results = 0
        
        # Initialize Hailo device for direct inference
        if hef_path:
            self.init_hailo_inference(hef_path)
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        print(f"Initialized MTCNN face detector")
        print(f"Output frames will be saved to: {self.output_dir}")
        if self.save_mode == 'time':
            print(f"Saving frames every {self.save_interval} seconds")
        else:
            print(f"Saving frames every {self.save_interval} frames")
        if self.save_inference_results:
            print(f"Inference results will be saved to: {self.inference_output_dir}")
    
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
        self.input_vstreams_params = InputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
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
        
        # Debug: Check if cv2.resize changes format
        if hasattr(self, '_first_inference'):
            if not self._first_inference:
                print(f"[FORMAT CHECK] After cv2.resize - shape: {resized.shape}, sample pixel: {resized[h//2, w//2]}")
                self._first_inference = True
        else:
            self._first_inference = False
        
        # No normalization - keep UINT8 values as expected by HEF
        # The HEF expects UINT8 input in 0-255 range
        
        # Run inference
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {list(self.input_vstreams_params.keys())[0]: np.expand_dims(resized, axis=0)}
            
            with self.network_group.activate(None):
                infer_results = infer_pipeline.infer(input_data)
                
        return infer_results
        
    def should_continue(self):
        """Check if we should continue processing"""
        # Let the pipeline keep running; limit only the saving
        return True


def gazelle_callback(pad, info, user_data):
    """Callback function for processing frames with GazeLLE"""
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
    """Save frame with gaze visualization"""
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
    """Save raw inference results (features, boxes, heatmaps) to disk"""
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
    """Get argument parser with GazeLLE-specific arguments"""
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
    """GStreamer application for GazeLLE gaze estimation"""
    
    def __init__(self, args, user_data):
        # Initialize parent with custom parser
        parser = get_gazelle_parser()
        super().__init__(parser, user_data)
        
        # Set parameters
        self.app_callback = gazelle_callback
        
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
    
    def get_pipeline_string(self):
        """Build the GStreamer pipeline string"""
        # Source pipeline
        source_pipeline = SOURCE_PIPELINE(
            self.video_source, 
            self.video_width, 
            self.video_height,
            self.video_format
        )
        
        # User callback - process frames directly
        # leaky=downstream drops new frames when full (keeps old frames)
        # leaky=upstream drops old frames when full (keeps new frames) - better for real-time
        # max-size-buffers=60 provides 2 seconds of buffer at 30fps
        user_callback_pipeline = (
            f'queue name=hailo_pre_callback_q leaky=downstream max-size-buffers=60 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback '
        )
        
        # Display pipeline - use fakesink if headless
        if self.options_menu.headless:
            display_pipeline = 'fakesink name=hailo_display'
        else:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink, 
                sync=self.sync, 
                show_fps=self.show_fps
            )
        
        # Complete pipeline - simplified without hailonet
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{QUEUE("pre_callback_q")} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        
        print(f"Pipeline string: {pipeline_string}")
        return pipeline_string
    
    def setup_callback(self):
        """Setup the callback on the identity element"""
        # Get the identity element
        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            # Add probe callback
            pad = identity.get_static_pad("src")
            if pad:
                pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self.app_callback,
                    self.user_data
                )
                print("Callback probe added successfully")
        else:
            print("Warning: Could not find identity element for callback")
        
        print("[DEBUG] Pipeline state is PLAYING, waiting for frames...")
    
    def on_pipeline_state_changed(self, bus, msg):
        """Handle pipeline state changes"""
        old_state, new_state, pending = msg.parse_state_changed()
        if msg.src == self.pipeline:
            if new_state == Gst.State.PLAYING:
                self.setup_callback()
        
        # Call parent handler
        super().on_pipeline_state_changed(bus, msg)


def main():
    # Create argument parser with GazeLLE arguments
    parser = get_gazelle_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load GazeLLE model
    print("Loading GazeLLE model...")
    
    # Get HEF input dimensions
    import hailo_platform as hpf
    hef_model = hpf.HEF(args.hef)
    
    # hef 모델의 입력 정보에 대해서 가져오기
    input_vs = hef_model.get_input_vstream_infos()
    
    # 입력 정보가 없으면 오류 출력하고 종료
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
    
    # Create and run the app
    app = GStreamerGazeLLEApp(args, user_data)
    
    # Add heartbeat to show the app is running
    def heartbeat():
        print(f"[DEBUG] Heartbeat - Frames processed: {user_data.frame_count}")
        return True
    
    GLib.timeout_add_seconds(5, heartbeat)
    
    print("[DEBUG] Starting GStreamer app...")
    app.run()


if __name__ == "__main__":
    main()