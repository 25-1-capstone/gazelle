#!/usr/bin/env python3
"""
System cleanup utilities for privacy and resource management.
"""
import subprocess
import gc


def cleanup_hailort_service():
    """Clear HailoRT service cache for privacy."""
    try:
        print("[PRIVACY] Clearing HailoRT service cache...")
        
        # Stop HailoRT service
        subprocess.run(['sudo', 'systemctl', 'stop', 'hailort'], 
                      check=True, capture_output=True)
        
        # Remove temp files
        subprocess.run(['sudo', 'rm', '-f', '/tmp/hailort*'], 
                      shell=True, capture_output=True)
        
        # Restart HailoRT service
        subprocess.run(['sudo', 'systemctl', 'start', 'hailort'], 
                      check=True, capture_output=True)
        
        print("[PRIVACY] HailoRT service cache cleared successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"[PRIVACY] Warning: Could not clear HailoRT cache: {e}")
    except Exception as e:
        print(f"[PRIVACY] Error during HailoRT cleanup: {e}")


def cleanup_system():
    """Perform comprehensive system cleanup."""
    print("[PRIVACY] Starting comprehensive system cleanup...")
    
    # Clear HailoRT cache
    cleanup_hailort_service()
    
    # Force garbage collection
    gc.collect()
    
    print("[PRIVACY] System cleanup completed")