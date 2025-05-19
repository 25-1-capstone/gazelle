import sys
import os

# Add the project root directory to sys.path
# This allows importing 'gazelle' as a top-level package
# __file__ is gazelle/scripts/gazelle_demo_test.py
# os.path.dirname(__file__) is gazelle/scripts
# os.path.dirname(os.path.dirname(__file__)) is gazelle
# os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) is the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np

device = "cuda" if torch.cuda.is_available() else "mps"
from gazelle.model import GazeLLE
from gazelle.backbone import DinoV2Backbone

backbone_s14 = DinoV2Backbone("dinov2_vits14")
print(
    f"Created DinoV2Backbone with 'dinov2_vits14'. Pretrained DINOv2 weights for backbone are loaded via torch.hub."
)

# 2. GazeLLE 모델을 s14 백본으로 직접 초기화
# GazeLLE의 다른 파라미터는 기본값을 사용합니다.
# (이는 gazelle_dinov2_vitb14() 또는 gazelle_dinov2_vits14() 함수가 GazeLLE를 인스턴스화하는 방식과 동일합니다.)
# inout=False가 GazeLLE의 기본값입니다.
model = GazeLLE(backbone=backbone_s14)
print(f"GazeLLE model initialized with '{backbone_s14.model}' backbone.")

# 3. 변환(transform)은 생성된 s14 백본에서 가져오기
# GazeLLE 모델은 일반적으로 (448, 448) 크기의 입력을 기대합니다.
transform = backbone_s14.get_transform(in_size=(448, 448))

model.eval()
model.to(
    device
)  # 4. GazeLLE 헤드 가중치를 'gazelle_dinov2_vitb14' 모델로부터 가져와 로드

# 4. Load GazeLLE weights from a local checkpoint file
# This replaces the original logic of loading from torch.hub
checkpoint_path = "/Users/ggrrm/Documents/embeded-ai/gazelle/scripts/experiments/train_gazefollow/epoch_14.pt"
print(
    f"Attempting to load GazeLLE head weights from local checkpoint: {checkpoint_path}"
)
try:
    # Load the checkpoint. It could be the state_dict itself or a dictionary wrapping it.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    actual_state_dict = None
    if isinstance(checkpoint, dict):
        # Check for common wrapper keys
        if "model_state_dict" in checkpoint:
            actual_state_dict = checkpoint["model_state_dict"]
            print("Extracted 'model_state_dict' from checkpoint.")
        elif "state_dict" in checkpoint:
            actual_state_dict = checkpoint["state_dict"]
            print("Extracted 'state_dict' from checkpoint.")
        elif "model" in checkpoint:
            # Assuming if 'model' key exists, its value is a state_dict or compatible.
            actual_state_dict = checkpoint["model"]
            print("Extracted 'model' from checkpoint, assuming it's a state_dict.")
        else:
            # If no known wrapper key, assume the dictionary itself is the state_dict.
            # This is common if torch.save(model.state_dict(), path) was used.
            print(
                "Checkpoint is a dictionary without standard wrapper keys. Assuming it is the state_dict itself."
            )
            actual_state_dict = checkpoint
    else:
        # If not a dictionary, assume it's the state_dict directly.
        print("Checkpoint is not a dictionary. Assuming it is the state_dict directly.")
        actual_state_dict = checkpoint

    if actual_state_dict is None:
        # This case implies the checkpoint variable itself was None before any logic, which is unlikely with torch.load unless file is empty/corrupt in a specific way.
        raise ValueError("Loaded checkpoint is None, cannot extract state_dict.")

    # Load the GazeLLE head weights using the model's dedicated method.
    # include_backbone=False ensures that only head weights are loaded from actual_state_dict,
    # preserving the script's initialized 'dinov2_vits14' backbone.
    model.load_gazelle_state_dict(actual_state_dict, include_backbone=False)

    print(
        f"Successfully loaded GazeLLE head weights from local checkpoint: {checkpoint_path}"
    )
    print(
        "The model is now using the 'dinov2_vits14' backbone (initialized in this script) with the head weights from your checkpoint."
    )
    print(
        "If the checkpoint contained backbone weights, they were ignored due to 'include_backbone=False'."
    )

except Exception as e:
    print(
        f"Error loading GazeLLE head weights from local checkpoint {checkpoint_path}: {e}"
    )
    print(
        "Please ensure the checkpoint path is correct and the file contains a compatible GazeLLE state_dict "
        "(either directly, or wrapped in a dictionary with keys like 'model_state_dict' or 'state_dict')."
    )
    print(
        "Proceeding with an uninitialized GazeLLE head (the DINOv2 s14 backbone itself is pretrained)."
    )

# load an input image

image_url = "https://www.looper.com/img/gallery/the-office-funniest-moments-ranked/jim-and-dwights-customer-service-training-1627594561.jpg"
# image_url = "https://ew.com/thmb/n5b8Asz4Y5Lp0sSEF7WgS-ESyFc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/succession-finale-052923-ec304aabcbf24a7b9fad891a87f49b74.jpg"
# image_url = "https://i.kym-cdn.com/entries/icons/original/000/045/575/blackcatzoningout_meme.jpg"

try:
    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    width, height = image.size

    plt.imshow(image)
    plt.axis("off")
    plt.show()

except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")

from facenet_pytorch import MTCNN
import numpy as np

# initialize once (on GPU if available)
mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")

# detect
boxes, probs = mtcnn.detect(np.array(image))

# boxes is an N×4 array of [x1, y1, x2, y2]
print(boxes)


# prepare gazelle input
img_tensor = transform(image).unsqueeze(0).to(device)
norm_bboxes = [
    [np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]
]

input = {
    "images": img_tensor,  # [num_images, 3, 448, 448]
    "bboxes": norm_bboxes,  # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
}

with torch.no_grad():
    output = model(input)

img1_person1_heatmap = output["heatmap"][0][0]  # [64, 64] heatmap
print(img1_person1_heatmap.shape)
if model.inout:
    img1_person1_inout = output["inout"][0][
        0
    ]  # gaze in frame score (if model supports inout prediction)
    print(img1_person1_inout.item())

# visualize predicted gaze heatmap for each person and gaze in/out of frame score


def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    heatmap = plt.cm.jet(np.array(heatmap) / 255.0)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle(
            [xmin * width, ymin * height, xmax * width, ymax * height],
            outline="lime",
            width=int(min(width, height) * 0.01),
        )

        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text(
                (text_x, text_y),
                text,
                fill="lime",
                font=ImageFont.load_default(size=int(min(width, height) * 0.05)),
            )
    return overlay_image


for i in range(len(boxes)):
    plt.figure()
    plt.imshow(
        visualize_heatmap(
            image,
            output["heatmap"][0][i],
            norm_bboxes[0][i],
            inout_score=output["inout"][0][i] if output["inout"] is not None else None,
        )
    )
    plt.axis("off")
    plt.show()
