#!/usr/bin/env python3
"""
Main application module for GazeLLE real-time gaze estimation.

This module serves as the primary entry point and coordinates all components
for real-time gaze tracking using Hailo-8 inference accelerator.
"""
import sys
import traceback
from pathlib import Path

# Add paths for gazelle modules
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent
project_root = scripts_dir.parent
sys.path.extend([str(project_root), str(scripts_dir)])

# Import application components
from gazelle_app import GazeLLEApplication
from gazelle_callback import GazeLLECallbackHandler
from model_loader import load_gazelle_model
from cli_parser import create_argument_parser
from system_cleanup import cleanup_system


def run_gazelle_realtime():
    """
    Main function that initializes and runs the GazeLLE real-time application.
    
    This function:
    1. Parses command line arguments
    2. Loads the GazeLLE model
    3. Sets up the callback handler with all necessary components
    4. Creates and runs the GStreamer pipeline
    5. Handles cleanup on exit
    """
    try:
        # Clear system cache for privacy before starting
        cleanup_system()
        
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Display initialization info
        print("[INIT] Starting GazeLLE real-time gaze estimation...")
        print(f"[INIT] Backbone model (HEF): {args.hef}")
        print(f"[INIT] Head model (PTH): {args.pth}")
        print(f"[INIT] Output directory: {args.output_dir}")
        print(f"[INIT] Maximum frames to save: {args.max_frames}")
        
        # Load the GazeLLE model with Hailo backend
        gazelle_model = load_gazelle_model(args.pth, args.hef, args.device)
        
        # Create callback handler with all processing components
        callback_handler = GazeLLECallbackHandler(
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
            detr_nms=args.detr_nms,
            mqtt_host='18.208.62.86'  # TODO: Make this configurable
        )
        
        # Create GStreamer application
        app = GazeLLEApplication(args, callback_handler)
        
        # Clear any existing pipeline buffers before starting
        print("[PRIVACY] Clearing any existing pipeline buffers...")
        app.flush_pipeline_buffers()
        
        # Run the application
        print("[INIT] Starting GStreamer pipeline...")
        try:
            app.run()
        finally:
            # Always perform cleanup, even if errors occur
            print("[PRIVACY] Starting comprehensive cleanup...")
            app.cleanup_resources()
            cleanup_system()
            print("[EXIT] Application stopped and all memory cleared")
        
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user")
        cleanup_system()
    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        traceback.print_exc()
        cleanup_system()
        sys.exit(1)


# Allow this module to be run directly
if __name__ == "__main__":
    run_gazelle_realtime()