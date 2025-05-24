#!/usr/bin/env python3
"""Test camera capture with different methods"""
import cv2
import sys

print("Testing camera capture methods...")

# Test 1: Direct V4L2
print("\n1. Testing V4L2 capture on /dev/video0...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   SUCCESS: Got frame with shape {frame.shape}")
    else:
        print("   FAILED: Could not read frame")
    cap.release()
else:
    print("   FAILED: Could not open device")

# Test 2: libcamera with different formats
print("\n2. Testing libcamera with YUV format...")
gst_yuv = "libcamerasrc ! video/x-raw,format=YUV420,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(gst_yuv, cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   SUCCESS: Got frame with shape {frame.shape}")
    else:
        print("   FAILED: Could not read frame")
    cap.release()
else:
    print("   FAILED: Could not open pipeline")

# Test 3: Simple v4l2src
print("\n3. Testing v4l2src...")
gst_v4l2 = "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(gst_v4l2, cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   SUCCESS: Got frame with shape {frame.shape}")
    else:
        print("   FAILED: Could not read frame")
    cap.release()
else:
    print("   FAILED: Could not open pipeline")

print("\nDone testing.")