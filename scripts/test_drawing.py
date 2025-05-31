#!/usr/bin/env python3
"""Test script to verify matplotlib drawing of object detections."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create a test image
img = np.ones((720, 1280, 3), dtype=np.uint8) * 100  # Gray background
img[100:200, 100:300] = [255, 0, 0]  # Red rectangle

# Test detections
detections = [
    {'bbox': [402, 18, 900, 715], 'class': 'person', 'confidence': 0.999},
    {'bbox': [338, 274, 710, 719], 'class': 'chair', 'confidence': 0.998},
    {'bbox': [769, 298, 995, 410], 'class': 'mouse', 'confidence': 0.993},
]

# Create figure and draw
plt.figure(figsize=(10, 8))
plt.imshow(img)

ax = plt.gca()

# Draw detections
for detection in detections:
    bbox = detection['bbox']
    xmin, ymin, xmax, ymax = bbox
    
    color = 'cyan' if detection['class'] == 'person' else 'orange' if detection['class'] == 'chair' else 'magenta'
    
    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                       fill=False, edgecolor=color, linewidth=3)
    ax.add_patch(rect)
    
    label = f"{detection['class']} {detection['confidence']:.2f}"
    ax.text(xmin, ymin-5, label,
           color=color, fontsize=12, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

plt.title("Test Object Detection Drawing")
plt.axis('off')
plt.savefig('test_drawing.png', bbox_inches='tight', pad_inches=0, dpi=100)
plt.close()

print("Test image saved to test_drawing.png")
print("Detections drawn:")
for d in detections:
    print(f"  - {d['class']} at [{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]")