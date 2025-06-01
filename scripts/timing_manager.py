#!/usr/bin/env python3
"""
Timing management for frame processing and saving intervals.
"""

from config import DEFAULT_CONFIG


class TimingManager:
    """Manages frame timing and save intervals with real-time prioritization."""
    
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
        
        # Real-time tracking
        self.last_processed_time = 0
        self.target_processing_interval = 1.0 / (30 / (skip_frames + 1)) if skip_frames > 0 else 1.0 / 30
        self.processing_overdue_threshold = self.target_processing_interval * 1.5
    
    def should_skip_frame(self, frame_count, current_time=None, adaptive_multiplier=1):
        """Check if frame should be skipped with real-time priority."""
        import time
        current_time = current_time or time.time()
        
        # If we're behind schedule, always process the latest frame
        time_since_last = current_time - self.last_processed_time
        if time_since_last >= self.processing_overdue_threshold:
            self.last_processed_time = current_time
            return False
        
        # Apply adaptive skip multiplier for performance
        effective_skip_frames = self.skip_frames * adaptive_multiplier
        
        # Standard frame skipping for normal operation
        if effective_skip_frames > 0:
            should_skip = frame_count % (int(effective_skip_frames) + 1) != 0
            if not should_skip:
                self.last_processed_time = current_time
            return should_skip
        
        self.last_processed_time = current_time
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