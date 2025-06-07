#!/usr/bin/env python3
"""
Main GazeLLE application class and GStreamer pipeline management.
"""
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_app import GStreamerApp
from hailo_apps_infra.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    DISPLAY_PIPELINE,
)

from gazelle_callback import GazeLLECallbackHandler
from cli_parser import create_argument_parser


class GazeLLEApplication(GStreamerApp):
    """Main GStreamer application for real-time gaze estimation."""
    
    def __init__(self, args, callback_handler):
        parser = create_argument_parser()
        super().__init__(parser, callback_handler)
        
        self.app_callback = gazelle_frame_callback
        self._setup_pipeline_config()
        self.create_pipeline()
    
    def _setup_pipeline_config(self):
        """Setup pipeline configuration."""
        self._auto_detect_arch()
        self.hef_path = self.options_menu.hef
        self.video_position = 0
        self.video_duration = 0
        self.eos_count = 0
    
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
        # Use custom source with proper video file handling
        if self.video_source.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            source = (f'filesrc location={self.video_source} ! decodebin ! '
                     f'videoconvert ! videoscale ! '
                     f'video/x-raw,format=RGB,width={self.video_width},height={self.video_height}')
            print(f"[PIPELINE] Using video file source: {self.video_source}")
        else:
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
        
        pipeline = (f'{source} ! queue name=pre_callback_q leaky=downstream '
                   f'max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! '
                   f'{callback} ! {display}')
        print(f"[PIPELINE] Full pipeline: {pipeline}")
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
        print(f"[PIPELINE] State change: {old_state.value_name} -> {new_state.value_name}")
        if msg.src == self.pipeline and new_state == Gst.State.PLAYING:
            self.setup_callback()
            self.query_video_duration()
        super().on_pipeline_state_changed(bus, msg)
    
    def on_eos(self):
        """Handle End-of-Stream events."""
        self.eos_count += 1
        current_pos = self.query_video_position()
        print(f"[EOS] End-of-stream #{self.eos_count} at position {current_pos:.3f}s/{self.video_duration:.3f}s")
        
        if self.video_duration > 0 and current_pos < self.video_duration - 0.1:
            print(f"[EOS] Premature EOS detected! Expected: {self.video_duration:.3f}s, actual: {current_pos:.3f}s")
        
        # For video files, seek back to beginning
        if self.video_source.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print("[EOS] Seeking back to beginning of video file...")
            success = self.pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                0  # seek to beginning
            )
            if success:
                print("[EOS] Video seek successful")
            else:
                print("[EOS] Video seek failed, falling back to parent method")
                super().on_eos()
        else:
            super().on_eos()
    
    def query_video_duration(self):
        """Query video duration and log it."""
        try:
            success, duration = self.pipeline.query_duration(Gst.Format.TIME)
            if success:
                self.video_duration = duration / Gst.SECOND
                print(f"[VIDEO] Duration: {self.video_duration:.2f} seconds")
            else:
                print(f"[VIDEO] Could not query duration")
        except Exception as e:
            print(f"[VIDEO] Error querying duration: {e}")
    
    def query_video_position(self):
        """Query current video position."""
        try:
            success, position = self.pipeline.query_position(Gst.Format.TIME)
            if success:
                self.video_position = position / Gst.SECOND
                return self.video_position
            return 0
        except Exception as e:
            print(f"[VIDEO] Error querying position: {e}")
            return 0
    
    def flush_pipeline_buffers(self):
        """Flush all buffers in the pipeline to clear cached frames."""
        if self.pipeline:
            # Send flush events to clear all buffers
            self.pipeline.send_event(Gst.Event.new_flush_start())
            self.pipeline.send_event(Gst.Event.new_flush_stop(True))
            
            # Clear queue buffers specifically
            for queue_name in ["pre_callback_q", "hailo_pre_callback_q"]:
                queue = self.pipeline.get_by_name(queue_name)
                if queue:
                    # Force queue to drop all buffers
                    queue.set_property("flush-on-eos", True)
            
            print("[PRIVACY] Pipeline buffers flushed")
    
    def cleanup_resources(self):
        """Comprehensive memory clearing for privacy."""
        # Flush GStreamer buffers
        self.flush_pipeline_buffers()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clean up user data
        if hasattr(self, 'user_data'):
            self.user_data.cleanup()
        
        print("[PRIVACY] All memory cleared")


def gazelle_frame_callback(pad, info, user_data):
    """Main GStreamer callback for processing frames."""
    user_data.frame_count += 1
    
    # Get buffer and basic checks
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Process buffer directly without copying to avoid memory accumulation
    buffer_to_use = buffer
    
    # Get timing and log video position
    pts_ms = buffer_to_use.pts / Gst.SECOND * 1000
    pts_seconds = pts_ms / 1000.0
    print(f"[VIDEO] Frame {user_data.frame_count}: PTS={pts_seconds:.3f}s")
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
    
    # Get frame data directly without copying
    format, width, height = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer_to_use, format, width, height)
    if frame is None or format is None:
        return Gst.PadProbeReturn.OK
    
    # Process frame with error handling
    start_time = time.time()
    try:
        results = user_data.frame_processor.process_frame(frame, width, height)
        
        # Check if no faces were detected - skip all processing except logging
        if results.get('no_faces_detected', False):
            print(f"[FRAME {user_data.frame_count}] No faces detected - skipping processing")
            frame = None  # Clear frame reference immediately
            return Gst.PadProbeReturn.OK
        
        # Add ROI to original buffer for pipeline integration
        _add_roi_to_buffer(buffer, results['boxes'], results['heatmaps'], width, height)
        
        # Save frame if conditions are met (before clearing frame reference)
        if results.get('should_save', False) and user_data.timing_manager.should_save_frame(rel_t):
            user_data.result_saver.save_visualization(
                frame, results['boxes'], results['heatmaps'], 
                results.get('object_detections'), results.get('gaze_targets'),
                highest_prob_target=results.get('highest_probability_target')
            )
        
        # Log result and send MQTT message
        if results.get('highest_probability_target'):
            target = results['highest_probability_target']
            log_message = f"[FRAME {user_data.frame_count}] {target['source']}: " \
                         f"{target['object']['class']} (prob: {target['probability']:.3f})"
            print(log_message)
            
            # Send MQTT message when user is looking at something
            if "user is looking at something" in log_message.lower() or target['probability'] > 0.5:
                mqtt_message = f"user looking at {target['object']['class']} with probability {target['probability']:.3f}"
                user_data.send_mqtt_message("state/caps1", mqtt_message)
        
        # Clear frame reference immediately after use
        frame = None
        
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
        frame = None  # Clear frame reference on error
    
    # Track processing time and adapt if needed
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    if len(user_data.processing_times) > 100:
        user_data.processing_times.pop(0)
    
    # Adaptive performance monitoring
    if processing_time > user_data.slow_frame_threshold:
        user_data.consecutive_slow_frames += 1
        if user_data.consecutive_slow_frames > 5:
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