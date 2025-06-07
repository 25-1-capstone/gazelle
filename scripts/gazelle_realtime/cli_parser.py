#!/usr/bin/env python3
"""
Command line argument parser for GazeLLE real-time application.
"""
from hailo_apps_infra.hailo_rpi_common import get_default_parser
from config import DEFAULT_CONFIG


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = get_default_parser()
    
    # Model arguments
    parser.add_argument("--hef", required=True, 
                       help="Path to compiled HEF backbone model file")
    parser.add_argument("--pth", required=True, 
                       help="Path to GazeLLE head checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", 
                       help="Torch device for GazeLLE head (cpu or cuda)")
    parser.add_argument("--scrfd-hef", 
                       help="Path to SCRFD HEF model for face detection")
    parser.add_argument("--detr-hef", 
                       help="Path to DETR HEF model for object detection")
    
    # Output arguments
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG['output_dir'], 
                       help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_CONFIG['max_frames'], 
                       help="Maximum number of frames to save")
    
    # Saving configuration
    parser.add_argument("--save-mode", choices=['time', 'frame'], 
                       default=DEFAULT_CONFIG['save_mode'],
                       help="Save mode: 'time' for time-based intervals, 'frame' for frame count-based")
    parser.add_argument("--save-interval", type=float, default=DEFAULT_CONFIG['save_interval'],
                       help="Save interval: seconds (if save-mode=time) or frames (if save-mode=frame)")
    
    # Processing options
    parser.add_argument("--headless", action="store_true", 
                       help="Run in headless mode (no display)")
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