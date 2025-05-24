#!/usr/bin/env python3
"""
Real-time gaze estimation using Hailo-8 HEF for DINOv2 backbone and GazeLLE head.
Captures video via GStreamer pipeline or webcam, runs Hailo-8 inference, then feeds extracted features into GazeLLE head.
Displays heatmap overlay in real-time.
"""
import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path to import gazelle module
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import hailo_platform as hpf
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

from gazelle.model import GazeLLE
from gazelle.backbone import DinoV2Backbone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time gaze estimation pipeline using Hailo-8 and GazeLLE head."
    )
    parser.add_argument("--hef", required=True, help="Path to compiled HEF backbone model file")
    parser.add_argument(
        "--pth", required=True, help="Path to GazeLLE head checkpoint (.pth) to load head weights"
    )
    parser.add_argument(
        "--gst",
        default=None,
        help=(
            "GStreamer pipeline string for cv2.VideoCapture. "
            "If not provided, default libcamerasrc pipeline is used."
        ),
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to video file to process instead of camera",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for GazeLLE head inference (cpu or cuda)",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_frames",
        help="Directory to save output frames instead of displaying",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="Maximum number of frames to process before stopping",
    )
    parser.add_argument(
        "--save_interval",
        type=float,
        default=1.0,
        help="Interval in seconds between saving frames (default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output frames will be saved to: {output_dir}")

    hef_model = hpf.HEF(args.hef)
    input_vs = hef_model.get_input_vstream_infos()
    output_vs = hef_model.get_output_vstream_infos()
    if not input_vs or not output_vs:
        print("Failed to get HEF input/output vstream infos", file=sys.stderr)
        sys.exit(1)
    shape = input_vs[0].shape
    if len(shape) == 3:
        hef_h, hef_w = shape[0], shape[1]
    else:
        hef_h, hef_w = shape[1], shape[2]
    print(f"HEF input resolution: {hef_w}x{hef_h}")

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device="cpu")  # Use CPU for face detection to avoid GPU memory issues
    print("Initialized MTCNN face detector")

    backbone = DinoV2Backbone("dinov2_vits14")
    head_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
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
    head_model.load_gazelle_state_dict(state_dict, include_backbone=False)
    head_model.to(args.device)
    head_model.eval()

    if args.video:
        print(f"Processing video file: {args.video}")
        # Debug: Check if file exists and is readable
        video_path = Path(args.video)
        print(f"Debug: Video file path: {video_path}")
        print(f"Debug: File exists: {video_path.exists()}")
        if video_path.exists():
            print(f"Debug: File size: {video_path.stat().st_size} bytes")
            print(f"Debug: File is readable: {video_path.is_file()}")
        
        # Try to handle potential encoding issues
        try:
            # Use the string path directly
            cap = cv2.VideoCapture(str(video_path))
            print(f"Debug: VideoCapture created successfully")
            print(f"Debug: VideoCapture.isOpened(): {cap.isOpened()}")
            if cap.isOpened():
                print(f"Debug: Video FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                print(f"Debug: Video frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                print(f"Debug: Video width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                print(f"Debug: Video height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            else:
                # Try with FFMPEG backend explicitly
                print("Debug: Trying with FFMPEG backend...")
                cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
                print(f"Debug: FFMPEG VideoCapture.isOpened(): {cap.isOpened()}")
        except Exception as e:
            print(f"Debug: Exception creating VideoCapture: {e}")
            cap = None
    elif args.gst:
        print(f"Using custom GStreamer pipeline: {args.gst}")
        cap = cv2.VideoCapture(args.gst, cv2.CAP_GSTREAMER)
    else:
        # For Raspberry Pi with libcamera, use a subprocess approach
        print("Using libcamera through FIFO pipe...")
        import subprocess
        import os
        
        # Create named pipe
        fifo_path = "/tmp/camera_fifo"
        if os.path.exists(fifo_path):
            os.remove(fifo_path)
        os.mkfifo(fifo_path)
        
        # Start libcamera-vid in background
        cmd = [
            "libcamera-vid",
            "-t", "0",  # Run indefinitely
            "--width", str(hef_w),
            "--height", str(hef_h),
            "--framerate", "30",
            "--codec", "yuv420",
            "-o", fifo_path
        ]
        print(f"Starting libcamera-vid: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Open FIFO for reading
        cap = cv2.VideoCapture(fifo_path)
        
    if not cap or not cap.isOpened():
        print("ERROR: Could not open any video capture source", file=sys.stderr)
        if args.video:
            print(f"Debug: Trying alternative methods for video file...")
            # Try with absolute path
            abs_path = Path(args.video).absolute()
            print(f"Debug: Absolute path: {abs_path}")
            cap = cv2.VideoCapture(str(abs_path))
            if not cap.isOpened():
                # Try with encoded path
                import urllib.parse
                encoded_path = urllib.parse.quote(str(abs_path))
                print(f"Debug: URL encoded path: {encoded_path}")
                cap = cv2.VideoCapture(encoded_path)
            if not cap.isOpened():
                print("ERROR: All video opening methods failed", file=sys.stderr)
                sys.exit(1)
            else:
                print("Debug: Successfully opened video with alternative method")
        else:
            sys.exit(1)

    in_name = input_vs[0].name
    out_name = output_vs[0].name

    try:
        with hpf.VDevice() as target:
            network_group = target.configure(hef_model)[0]
            in_params = hpf.InputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=hpf.FormatType.UINT8
            )
            out_params = hpf.OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
            )
            with network_group.activate():
                with hpf.InferVStreams(network_group, in_params, out_params) as infer_pipeline:
                    frame_count = 0
                    empty_reads = 0
                    max_empty_reads = 50  # Allow up to 5 seconds of empty reads
                    last_save_time = time.time()  # Track time for interval-based saving
                    
                    print(f"Debug: Starting frame capture loop, max_frames={args.max_frames}")
                    while frame_count < args.max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            empty_reads += 1
                            if empty_reads == 1:
                                print(f"Debug: First empty read, cap.isOpened()={cap.isOpened()}")
                            if empty_reads % 10 == 0:
                                print(f"Debug: Empty reads count: {empty_reads}")
                            if empty_reads > max_empty_reads:
                                print("ERROR: No frames received for 5 seconds. Camera may be blocked.", file=sys.stderr)
                                break
                            time.sleep(0.1)
                            continue
                        
                        # Debug successful frame read
                        if empty_reads > 0:
                            print(f"Debug: Frame read successful after {empty_reads} empty reads")
                        
                        
                        empty_reads = 0  # Reset counter on successful read

                        # Detect faces on the original frame (before resizing)
                        mtcnn_start = time.time()
                        try:
                            # Convert BGR to RGB for MTCNN
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            boxes_original, probs = mtcnn.detect(frame_rgb)
                            if boxes_original is not None and len(boxes_original) > 0:
                                print(f"Face detection: Found {len(boxes_original)} face(s) with probabilities: {probs}")
                        except Exception as e:
                            print(f"Face detection error: {e}")
                            import traceback
                            traceback.print_exc()
                            boxes_original = None
                        mtcnn_time = time.time() - mtcnn_start
                        print(f"MTCNN inference time: {mtcnn_time*1000:.2f}ms")
                        
                        if boxes_original is None or len(boxes_original) == 0:
                            # No faces detected, use default center box
                            # Save a debug frame to check what the video contains
                            if frame_count == 0:
                                debug_path = output_dir / "debug_original_frame.png"
                                cv2.imwrite(str(debug_path), frame)
                                print(f"Saved debug frame to {debug_path} for inspection")
                            print(f"No faces detected in frame, using default center region")
                            # Scale default box to resized frame dimensions
                            boxes = np.array([[hef_w*0.25, hef_h*0.25, hef_w*0.75, hef_h*0.75]])
                        else:
                            # Scale detected boxes to resized frame dimensions
                            scale_x = hef_w / frame.shape[1]
                            scale_y = hef_h / frame.shape[0]
                            boxes = []
                            for box in boxes_original:
                                x1, y1, x2, y2 = box
                                boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                            boxes = np.array(boxes)
                            print(f"Detected {len(boxes)} face(s) in frame")

                        if frame.shape[1] != hef_w or frame.shape[0] != hef_h:
                            frame_in = cv2.resize(frame, (hef_w, hef_h))
                        else:
                            frame_in = frame

                        # HEF (DINOv2 backbone) inference
                        hef_start = time.time()
                        hef_input = np.expand_dims(frame_in, axis=0)
                        output_dict = infer_pipeline.infer({in_name: hef_input})
                        feat_raw = output_dict[out_name]
                        if feat_raw.ndim == 4 and feat_raw.shape[0] == 1:
                            feat_processed = np.transpose(feat_raw, (0, 3, 1, 2))
                        elif feat_raw.ndim == 3:
                            feat_processed = np.transpose(np.expand_dims(feat_raw, 0), (0, 3, 1, 2))
                        else:
                            feat_processed = feat_raw
                        feat_tensor = torch.from_numpy(feat_processed).to(args.device)
                        hef_time = time.time() - hef_start
                        print(f"HEF (DINOv2) inference time: {hef_time*1000:.2f}ms")
                        
                        # Normalize bounding boxes
                        norm_bboxes = [[np.array(bbox) / np.array([hef_w, hef_h, hef_w, hef_h]) for bbox in boxes]]
                        
                        # GazeLLE head inference
                        gazelle_start = time.time()
                        with torch.no_grad():
                            out = head_model({"extracted_features": feat_tensor, "bboxes": norm_bboxes})
                        
                        # Get heatmaps for all detected faces
                        heatmaps = out["heatmap"][0].cpu().numpy()  # Shape: [num_faces, H, W] or [num_faces, 1, H, W]
                        gazelle_time = time.time() - gazelle_start
                        print(f"GazeLLE head inference time: {gazelle_time*1000:.2f}ms")
                        
                        # Handle different heatmap dimensions
                        if heatmaps.ndim == 4:  # [num_faces, 1, H, W]
                            heatmaps = heatmaps.squeeze(1)  # Remove channel dimension -> [num_faces, H, W]
                        elif heatmaps.ndim == 2:  # Single face case [H, W]
                            heatmaps = np.expand_dims(heatmaps, 0)  # Add face dimension -> [1, H, W]
                        
                        # For visualization, we'll use the first face's heatmap or combine them
                        if len(heatmaps) > 0:
                            # Option 1: Use the first face's heatmap
                            heatmap = heatmaps[0]
                            # Option 2: Combine all heatmaps (uncomment if preferred)
                            # heatmap = np.max(heatmaps, axis=0)  # Take maximum across all faces
                        else:
                            # Fallback to zeros if no heatmaps
                            heatmap = np.zeros((hef_h, hef_w))

                        # Check if enough time has passed since last save
                        current_time = time.time()
                        if current_time - last_save_time >= args.save_interval:
                            save_start = time.time()
                            # Create heatmap overlay using matplotlib instead of cv2
                            frame_pil = Image.fromarray(cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB))
                            
                            # Normalize heatmap to 0-1 range
                            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                            
                            # Create figure and save
                            plt.figure(figsize=(10, 8))
                            plt.imshow(frame_pil)
                            plt.imshow(heatmap_norm, alpha=0.4, cmap='jet')
                            
                            # Draw bounding boxes for all detected faces
                            ax = plt.gca()
                            for i, bbox in enumerate(boxes):
                                xmin, ymin, xmax, ymax = bbox
                                box_x = xmin
                                box_y = ymin
                                box_width = xmax - xmin
                                box_height = ymax - ymin
                                rect = plt.Rectangle((box_x, box_y), box_width, box_height, 
                                                   fill=False, edgecolor='lime', linewidth=3)
                                ax.add_patch(rect)
                                
                                # Add face number label
                                ax.text(box_x, box_y - 5, f'Face {i+1}', 
                                       color='lime', fontsize=12, weight='bold', 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                            
                            plt.title(f"Gaze Estimation - Frame {frame_count + 1}")
                            plt.axis('off')
                            
                            # Save frame
                            output_path = output_dir / f"frame_{frame_count + 1:04d}.png"
                            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
                            plt.close()
                            
                            save_time = time.time() - save_start
                            total_frame_time = time.time() - last_save_time
                            print(f"\nFrame {frame_count + 1} processing summary:")
                            print(f"  - Total frame processing time: {total_frame_time*1000:.2f}ms")
                            print(f"  - Image save time: {save_time*1000:.2f}ms")
                            print(f"  - Saved to {output_path}")
                            frame_count += 1
                            last_save_time = current_time
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        print(f"Processing complete. Frames saved to {output_dir}")


if __name__ == "__main__":
    main()