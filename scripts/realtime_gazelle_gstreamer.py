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
import matplotlib.pyplot as plt

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
        "--device",
        default="cpu",
        help="Torch device for GazeLLE head inference (cpu or cuda)",
    )
    parser.add_argument(
        "--bbox",
        default="0.45,0.45,0.55,0.55",
        help=(
            "Normalized face bbox as xmin,ymin,xmax,ymax for head prompt injection; "
            "default is center region"
        ),
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

    xmin, ymin, xmax, ymax = map(float, args.bbox.split(","))
    bboxes = [[(xmin, ymin, xmax, ymax)]]

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

    if args.gst:
        cap = cv2.VideoCapture(args.gst, cv2.CAP_GSTREAMER)
    else:
        gst_default = (
            f"libcamerasrc ! video/x-raw,width={hef_w},height={hef_h},format=BGR ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(gst_default, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open video capture with GStreamer, trying default camera", file=sys.stderr)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, hef_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hef_h)

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
                    while frame_count < args.max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.1)
                            continue

                        if frame.shape[1] != hef_w or frame.shape[0] != hef_h:
                            frame_in = cv2.resize(frame, (hef_w, hef_h))
                        else:
                            frame_in = frame

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

                        with torch.no_grad():
                            out = head_model({"extracted_features": feat_tensor, "bboxes": bboxes})
                        heatmap = out["heatmap"][0].cpu().numpy()

                        # Create heatmap overlay using matplotlib instead of cv2
                        frame_pil = Image.fromarray(cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB))
                        
                        # Normalize heatmap to 0-1 range
                        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        # Create figure and save
                        plt.figure(figsize=(10, 8))
                        plt.imshow(frame_pil)
                        plt.imshow(heatmap_norm, alpha=0.4, cmap='jet')
                        plt.title(f"Gaze Estimation - Frame {frame_count + 1}")
                        plt.axis('off')
                        
                        # Save frame
                        output_path = output_dir / f"frame_{frame_count + 1:04d}.png"
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
                        plt.close()
                        
                        print(f"Saved frame {frame_count + 1}/{args.max_frames} to {output_path}")
                        frame_count += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        print(f"Processing complete. Frames saved to {output_dir}")


if __name__ == "__main__":
    main()