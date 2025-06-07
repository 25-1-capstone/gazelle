#!/usr/bin/env python3
"""
Launch script for the GazeLLE real-time gaze estimation application.

This script serves as the entry point for users, setting up the proper
import paths and launching the main application from the gazelle_realtime package.
"""
import sys
from pathlib import Path

# Add current directory to path for proper module imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import and run the main application
from gazelle_realtime.gazelle_realtime_app import run_gazelle_realtime

if __name__ == "__main__":
    run_gazelle_realtime()