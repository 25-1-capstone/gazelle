#!/usr/bin/env python3
"""
Callback handler for GazeLLE frame processing.
"""
import time
import paho.mqtt.client as mqtt
from hailo_apps_infra.hailo_rpi_common import app_callback_class

from hailo_inference import HailoInferenceManager, SCRFDInferenceManager, DETRInferenceManager
from frame_processing import FrameProcessor
from visualization import ResultSaver
from timing_manager import TimingManager
from config import DETECTION_THRESHOLDS


class GazeLLECallbackHandler(app_callback_class):
    """Callback handler for GazeLLE processing."""
    
    def __init__(self, gazelle_model, device='cpu', output_dir='./output_frames',
                 save_interval=1.0, max_frames=10, hef_path=None, skip_frames=0,
                 save_inference_results=False, inference_output_dir='./inference_results',
                 save_mode='time', scrfd_hef_path=None, detr_hef_path=None,
                 detr_confidence=0.7, detr_nms=0.3, mqtt_host='18.208.62.86'):
        super().__init__()
        
        # Configuration
        self.device = device
        self.max_frames = max_frames
        self.save_inference_results = save_inference_results
        
        # MQTT configuration
        self.mqtt_host = mqtt_host
        self.mqtt_client = None
        self.setup_mqtt()
        
        # Reset all state for fresh start
        self.reset_processing_state()
        
        # Initialize components with shared VDevice
        shared_vdevice = None
        self.hailo_manager = HailoInferenceManager(hef_path, shared_vdevice) if hef_path else None
        if self.hailo_manager:
            shared_vdevice = self.hailo_manager.vdevice
        self.scrfd_manager = SCRFDInferenceManager(scrfd_hef_path, shared_vdevice) if scrfd_hef_path else None
        self.detr_manager = DETRInferenceManager(detr_hef_path, shared_vdevice, detr_confidence, detr_nms) if detr_hef_path else None
        
        # Initialize processors
        self.frame_processor = FrameProcessor(gazelle_model, self.hailo_manager, device, self.scrfd_manager, self.detr_manager)
        self.result_saver = ResultSaver(output_dir, inference_output_dir if save_inference_results else None)
        self.timing_manager = TimingManager(save_mode, save_interval, skip_frames)
        
        print(f"[CONFIG] Output: {output_dir}, Save mode: {save_mode}, Interval: {save_interval}")
        if inference_output_dir:
            print(f"[CONFIG] Inference output: {inference_output_dir}")
    
    def setup_mqtt(self):
        """Initialize MQTT client and connect to broker."""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            self.mqtt_client.connect(self.mqtt_host, 1883, 60)
            self.mqtt_client.loop_start()
            print(f"[MQTT] Connecting to broker at {self.mqtt_host}")
        except Exception as e:
            print(f"[MQTT] Failed to setup MQTT client: {e}")
            self.mqtt_client = None
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for when MQTT client connects."""
        if rc == 0:
            print("[MQTT] Connected successfully")
            self.send_mqtt_message("init/caps1", "initialize connection")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for when MQTT client disconnects."""
        print(f"[MQTT] Disconnected with code {rc}")
    
    def send_mqtt_message(self, topic, message):
        """Send MQTT message to specified topic."""
        if self.mqtt_client and self.mqtt_client.is_connected():
            try:
                self.mqtt_client.publish(topic, message)
                print(f"[MQTT] Sent to {topic}: {message}")
            except Exception as e:
                print(f"[MQTT] Failed to send message: {e}")
        else:
            print(f"[MQTT] Client not connected, message not sent: {topic} - {message}")
    
    def reset_processing_state(self):
        """Reset all processing state for a fresh start."""
        # Performance monitoring
        self.consecutive_slow_frames = 0
        self.slow_frame_threshold = 0.1  # 100ms processing time threshold
        self.adaptive_skip_multiplier = 1
        
        # Tracking - start fresh
        self.frame_count = 0
        self.processing_times = []
        self.dropped_frame_count = 0
    
    def should_continue(self):
        """Check if processing should continue."""
        return True
    
    def cleanup(self):
        """Clean up resources and reset state."""
        print("[PRIVACY] Starting comprehensive cleanup...")
        
        # Flush memory first
        if hasattr(self.hailo_manager, 'flush_memory'):
            self.hailo_manager.flush_memory()
        if hasattr(self.scrfd_manager, 'flush_memory'):
            self.scrfd_manager.flush_memory()
        if hasattr(self.detr_manager, 'flush_memory'):
            self.detr_manager.flush_memory()
        
        # Clean up Hailo managers
        if hasattr(self.hailo_manager, 'cleanup'):
            self.hailo_manager.cleanup()
        if hasattr(self.scrfd_manager, 'cleanup'):
            self.scrfd_manager.cleanup()
        if hasattr(self.detr_manager, 'cleanup'):
            self.detr_manager.cleanup()
        
        # Clean up MQTT client
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("[MQTT] Client disconnected")
        
        # Clear any cached data
        self.processing_times.clear()
        
        # Reset timing
        if hasattr(self.timing_manager, 'reset_timing'):
            self.timing_manager.reset_timing()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("[PRIVACY] All resources cleaned up")