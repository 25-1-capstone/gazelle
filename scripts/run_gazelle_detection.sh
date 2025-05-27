#!/bin/bash
# Example script to run the combined gaze + object detection pipeline

# Path to the detection model (e.g., YOLOv5)
DETECTION_HEF="/path/to/yolov5m_wo_spp_60p.hef"

# Path to the GazeLLE backbone model (DINOv2)
GAZELLE_HEF="/path/to/dinov2_vits14_518x518_uint8.hef"

# Path to the GazeLLE head checkpoint
GAZELLE_PTH="/path/to/gazelle_dinov2_vitsmall_gazefollow_XYhead_inout518_aug1_bs8_lr1e-04.pth"

# Run the combined pipeline
python3 realtime_gazelle_plus_object_detection.py \
    --hef "$DETECTION_HEF" \
    --gazelle-hef "$GAZELLE_HEF" \
    --pth "$GAZELLE_PTH" \
    --input /dev/video0 \
    --output-dir ./output_frames \
    --max-frames 30 \
    --save-interval 0.5 \
    --skip-frames 2