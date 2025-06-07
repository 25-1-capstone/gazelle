#!/usr/bin/env python3
"""
Entry point script for real-time gaze estimation using Hailo-8 infrastructure.
This script handles import paths and launches the main application.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for gazelle modules
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent
project_root = scripts_dir.parent

sys.path.append(str(project_root))
sys.path.append(str(scripts_dir))

# Now import and run the main application
from gazelle_realtime_app import run_gazelle_realtime

if __name__ == "__main__":
    run_gazelle_realtime()