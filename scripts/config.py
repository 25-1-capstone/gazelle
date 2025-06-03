#!/usr/bin/env python3
"""
Configuration settings for the Gazelle real-time gaze estimation system.
"""

# Default configuration values
DEFAULT_CONFIG = {
    'output_dir': './output_frames',
    'inference_output_dir': './inference_results',
    'device': 'cpu',
    'save_interval': 1.0,
    'max_frames': 10,
    'skip_frames': 0,
    'save_mode': 'time',
    'nominal_fps': 30,
}

# Debug output intervals
DEBUG_INTERVALS = {
    'frame_info': 10,      # N frames per frame info output
    'heartbeat': 5,        # N seconds per heartbeat message
    'processing_avg': 10,  # Average processing time for last N frames
}

# DETR COCO class labels (92 classes including background)
COCO_CLASSES_92 = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",
}

# Color scheme for object visualization
OBJECT_COLORS = {
    'face': 'lime',
    'person': 'cyan',
    'chair': 'orange',
    'couch': 'orange',
    'bed': 'orange',
    'mouse': 'magenta',
    'keyboard': 'magenta',
    'laptop': 'magenta',
    'tv': 'magenta',
    'plant': 'green',
    'potted': 'green',
    'gazed': 'red',
    'default': 'yellow'
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    'detr_confidence': 0.7,
    'detr_nms': 0.3,
    'gaze_confidence': 0.5,
    'min_box_size': 20,
    'min_relative_area': 0.001
}