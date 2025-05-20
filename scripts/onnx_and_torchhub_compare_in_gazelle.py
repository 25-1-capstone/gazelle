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

# --- 프로젝트 루트를 sys.path에 추가 ---
# gazelle_demo_test.py 파일의 현재 위치를 기준으로 project_root 계산
# /Users/ggrrm/Documents/embeded-ai/gazelle/scripts/gazelle_demo_test.py
# -> /Users/ggrrm/Documents/embeded-ai/gazelle
# 이 경로에 gazelle 패키지 (gazelle/model.py 등)가 있다고 가정
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root {project_root} added to sys.path")
    # gazelle 패키지 임포트 시도 (sys.path 설정 후)
    from gazelle.model import GazeLLE
    from gazelle.backbone import DinoV2Backbone

    print("Successfully imported GazeLLE and DinoV2Backbone from gazelle package.")
except ImportError as e:
    print(f"Error importing from gazelle package after sys.path modification: {e}")
    print(
        "Please ensure the 'gazelle' directory (containing model.py, backbone.py, etc.) is directly under the project_root."
    )
    # 오류 발생 시, 대체 경로로 다시 시도 (만약 gazelle/gazelle 구조라면)
    # project_root_alt = os.path.dirname(project_root) # /Users/ggrrm/Documents/embeded-ai
    # if project_root_alt not in sys.path:
    # sys.path.insert(0, project_root_alt)
    # print(f"Attempting import with alternative project root {project_root_alt}")
    # from gazelle.model import GazeLLE
    # from gazelle.backbone import DinoV2Backbone
    # print("Successfully imported GazeLLE and DinoV2Backbone with alternative project root.")
    # pass # 필요한 경우 여기서 exit() 또는 다른 처리
    exit()  # 임포트 실패 시 종료

# ONNX Runtime 및 기타 필요한 라이브러리 임포트
import onnxruntime as ort
import torchvision.transforms as T
import torch.nn as nn  # DinoV2FeatureExtractor를 위해


# --- DinoV2FeatureExtractor 클래스 정의 (export_onnx_dinov2_vits14.py에서 가져옴) ---
class DinoV2FeatureExtractor(nn.Module):
    def __init__(
        self, dino_model, patch_size=14
    ):  # patch_size 인자 추가 및 기본값 설정
        super().__init__()
        self.dino_model = dino_model
        self.patch_size = patch_size  # DinoV2Backbone에서 가져온 patch_size 사용 가능

    def get_out_size(self, input_size):
        h, w = input_size
        out_h = h // self.patch_size
        out_w = w // self.patch_size
        return out_h, out_w

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        features_dict = self.dino_model.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"]
        reshaped_tokens = patch_tokens.view(b, out_h, out_w, -1)
        output_features = reshaped_tokens.permute(0, 3, 1, 2)
        return output_features


# --- 기본 설정 ---
device = "cuda" if torch.cuda.is_available() else "mps"
ONNX_MODEL_PATH = (
    "/Users/ggrrm/Documents/embeded-ai/dinov2_vits14_feature_extractor.onnx"
)
# ONNX 모델은 448x448 입력을 기준으로 변환되었을 가능성이 높음
ONNX_INPUT_SIZE = (448, 448)


# 1. DinoV2 s14 백본 로드
# 이 백본은 GazeLLE 모델에 사용되며, PyTorch 기반의 특징 추출을 담당
backbone_s14 = DinoV2Backbone("dinov2_vits14")
print(
    f"Created DinoV2Backbone with dino_vits14. Pretrained DINOv2 weights for backbone are loaded via torch.hub."
)
# backbone_s14.model은 torch.hub에서 로드된 원본 DINOv2 모델임

# 2. GazeLLE 모델을 s14 백본으로 직접 초기화
model = GazeLLE(backbone=backbone_s14)
print(f"GazeLLE model initialized with dino_vits14 backbone.")

# 3. 변환(transform)은 생성된 s14 백본에서 가져오기 (GazeLLE용: 448x448)
gazelle_transform = backbone_s14.get_transform(in_size=(448, 448))

model.eval()
model.to(device)

# 4. 로컬 체크포인트에서 GazeLLE 헤드 가중치 로드
checkpoint_path = "/Users/ggrrm/Documents/embeded-ai/gazelle/scripts/experiments/train_gazefollow/epoch_14.pt"
print(
    f"Attempting to load GazeLLE head weights from local checkpoint: {checkpoint_path}"
)
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actual_state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            actual_state_dict = checkpoint["model_state_dict"]
            print("Extracted 'model_state_dict' from checkpoint.")
        elif "state_dict" in checkpoint:
            actual_state_dict = checkpoint["state_dict"]
            print("Extracted 'state_dict' from checkpoint.")
        elif "model" in checkpoint:
            actual_state_dict = checkpoint["model"]
            print("Extracted 'model' from checkpoint, assuming it's a state_dict.")
        else:
            print(
                "Checkpoint is a dictionary without standard wrapper keys. Assuming it is the state_dict itself."
            )
            actual_state_dict = checkpoint
    else:
        print("Checkpoint is not a dictionary. Assuming it is the state_dict directly.")
        actual_state_dict = checkpoint

    if actual_state_dict is None:
        raise ValueError("Loaded checkpoint is None, cannot extract state_dict.")

    model.load_gazelle_state_dict(actual_state_dict, include_backbone=False)
    print(
        f"Successfully loaded GazeLLE head weights from local checkpoint: {checkpoint_path}"
    )

except Exception as e:
    print(
        f"Error loading GazeLLE head weights from local checkpoint {checkpoint_path}: {e}"
    )
    print("Proceeding with an uninitialized GazeLLE head.")


# --- ONNX 백본과 PyTorch 백본 비교 로직 ---
print("\\n--- Comparing PyTorch DinoV2Backbone with ONNX Feature Extractor ---")
# 테스트용 이미지 로드 (기존 이미지 재사용)
image_url = "https://www.looper.com/img/gallery/the-office-funniest-moments-ranked/jim-and-dwights-customer-service-training-1627594561.jpg"
try:
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    pil_image_for_comparison = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Loaded image for ONNX comparison from: {image_url}")

    # ONNX 모델용 입력 변환 (448x448)
    # DINOv2 표준 변환 (ImageNet 통계 사용, from DINOv2_repo/dinov2/data/transforms.py)
    # 또는 torch.hub.load 시 반환되는 transform 객체를 사용할 수도 있음.
    # 여기서는 간단히 torchvision의 표준 변환을 사용.
    # 실제 DINOv2 학습 시 사용된 정확한 변환을 적용하는 것이 가장 좋음.
    onnx_input_transform = T.Compose(
        [
            T.Resize(ONNX_INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    onnx_input_tensor = (
        onnx_input_transform(pil_image_for_comparison).unsqueeze(0).to(device)
    )
    print(f"ONNX input tensor shape: {onnx_input_tensor.shape}")

    # 1. PyTorch 백본 특징 추출 (DinoV2FeatureExtractor 래퍼 사용)
    # GazeLLE의 DinoV2Backbone 내부 모델(self.model)과 패치 크기를 사용
    pytorch_feature_extractor = DinoV2FeatureExtractor(
        dino_model=backbone_s14.model,  # 원본 DINOv2 모델
        # patch_size=backbone_s14.patch_size, # 사용자가 이 줄을 삭제했음
    ).eval()  # .to(device)를 여기서 제거

    with torch.no_grad():
        # --- MPS 오류 회피 로직 시작 ---
        if device == "mps":
            try:
                print("Attempting PyTorch feature extraction on MPS...")
                pytorch_features = pytorch_feature_extractor.to(device)(
                    onnx_input_tensor.to(device)
                )
            except RuntimeError as e:
                if "Adaptive pool MPS" in str(e) or "Non-divisible input sizes" in str(
                    e
                ):
                    print(
                        "MPS AdaptivePool error detected for PyTorch features. Switching to CPU."
                    )
                    # 모델과 입력을 CPU로 이동하여 실행
                    pytorch_features = pytorch_feature_extractor.to("cpu")(
                        onnx_input_tensor.to("cpu")
                    )
                    # 비교는 CPU에서 수행하므로, 결과를 다시 MPS로 옮길 필요는 없음 (필요시 주석 해제)
                    # pytorch_features = pytorch_features.to(device)
                    print("PyTorch features extracted on CPU.")
                else:
                    raise e  # 다른 RuntimeError이면 다시 발생
        else:
            # MPS가 아니면 원래 설정된 device에서 실행
            pytorch_features = pytorch_feature_extractor.to(device)(
                onnx_input_tensor.to(device)
            )
        # --- MPS 오류 회피 로직 끝 ---
    print(f"PyTorch backbone (via wrapper) output shape: {pytorch_features.shape}")

    # 2. ONNX 런타임으로 특징 추출
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ONNX model file not found at: {ONNX_MODEL_PATH}")
        print("Skipping ONNX comparison.")
    else:
        print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
        ort_session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CPUExecutionProvider"],  # CUDA가 없는 경우 CPU만 사용
        )
        # ['CUDAExecutionProvider', 'CPUExecutionProvider'] 순서로 시도, CUDA 없으면 CPU

        ort_input_name = ort_session.get_inputs()[0].name
        ort_inputs = {
            ort_input_name: onnx_input_tensor.cpu().numpy()
        }  # ONNX Runtime은 numpy 배열 입력

        onnx_features_list = ort_session.run(None, ort_inputs)
        onnx_features = torch.from_numpy(onnx_features_list[0]).to(
            device
        )  # 첫 번째 출력을 텐서로 변환
        print(f"ONNX backbone output shape: {onnx_features.shape}")

        # 3. 결과 비교
        try:
            np.testing.assert_allclose(
                pytorch_features.cpu().numpy(),  # .cpu()를 호출하여 CPU에서 비교
                onnx_features.cpu().numpy(),  # ONNX 결과도 CPU로 (ONNX 세션이 CPU 사용 가정)
                rtol=1e-3,  # 허용 오차 범위는 실험적으로 조정 필요
                atol=1e-5,
            )
            print("SUCCESS: PyTorch backbone features and ONNX features are close.")
        except AssertionError as e:
            print("FAILURE: PyTorch backbone features and ONNX features differ.")
            # print(e) # 상세 오류 메시지
            # 차이 계산 (옵션)
            abs_diff = torch.abs(pytorch_features - onnx_features)
            print(f"  Max absolute difference: {abs_diff.max().item()}")
            print(f"  Mean absolute difference: {abs_diff.mean().item()}")


except requests.exceptions.RequestException as e:
    print(f"Error downloading image for ONNX comparison: {e}")
except ImportError:
    print(
        "ONNX Runtime (onnxruntime) or torchvision not fully available. Skipping ONNX comparison."
    )
    print("Please install them: pip install onnxruntime torchvision")
except Exception as e:
    print(f"An error occurred during ONNX comparison: {e}")
    import traceback

    traceback.print_exc()

print("--- End ONNX Comparison ---\\n")

# --- 이하 GazeLLE 추론 및 시각화 로직 (기존 코드 유지) ---

# 데모용 이미지 로드 (GazeLLE용)
# image_url = "https://www.looper.com/img/gallery/the-office-funniest-moments-ranked/jim-and-dwights-customer-service-training-1627594561.jpg"
# ... (기존 이미지 로드 로직은 이미 위에서 한 번 실행됨, pil_image_for_comparison을 사용하거나 다시 로드)
# 여기서는 비교에 사용된 pil_image_for_comparison을 GazeLLE 데모에도 사용하도록 수정
# 단, GazeLLE는 (448,448) 입력을 기대하므로, gazelle_transform을 적용해야 함.

try:
    # pil_image_for_comparison 변수가 위 ONNX 비교 블록에서 성공적으로 로드되었다고 가정
    # 만약 ONNX 비교가 실패했거나 이미지가 로드되지 않았다면, 여기서 다시 로드해야 할 수 있음.
    if "pil_image_for_comparison" not in locals() or pil_image_for_comparison is None:
        print(
            "Re-loading image for GazeLLE demo as it was not available from ONNX comparison."
        )
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        pil_image_for_gazelle = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        pil_image_for_gazelle = (
            pil_image_for_comparison  # ONNX 비교에 사용된 이미지 재활용
        )

    width, height = pil_image_for_gazelle.size

    # plt.imshow(pil_image_for_gazelle) # 이미지 표시는 후반부 시각화에서 처리
    # plt.axis("off")
    # plt.show()

except requests.exceptions.RequestException as e:
    print(f"Error downloading image for GazeLLE demo: {e}")
    exit()
except NameError:  # pil_image_for_comparison이 정의되지 않은 경우 대비
    print(
        "Error: pil_image_for_comparison not defined. Cannot proceed with GazeLLE demo."
    )
    exit()


from facenet_pytorch import MTCNN

# import numpy as np # 이미 위에서 임포트됨

# initialize once (on GPU if available)
mtcnn = MTCNN(keep_all=True, device=device)  # device 변수 사용

# detect faces
# GazeLLE 데모용 이미지(pil_image_for_gazelle)에 대해 얼굴 검출 수행
# MTCNN은 PIL 이미지를 직접 받거나 numpy 배열로 변환해야 함
boxes, probs = mtcnn.detect(pil_image_for_gazelle)  # PIL 이미지 직접 전달

# boxes is an N×4 array of [x1, y1, x2, y2]
# Handle case where no faces are detected
if boxes is None:
    print("No faces detected in the image for GazeLLE demo. Exiting.")
    exit()
print("Detected face boxes for GazeLLE:", boxes)


# prepare gazelle input
# GazeLLE용 변환(gazelle_transform)과 이미지(pil_image_for_gazelle) 사용
img_tensor = gazelle_transform(pil_image_for_gazelle).unsqueeze(0).to(device)

# Normalize bboxes (ensure boxes are not None)
norm_bboxes = (
    [[np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]]
    if boxes is not None
    else []
)


input_data = {  # 변수명 변경: input -> input_data (파이썬 내장 함수와 충돌 방지)
    "images": img_tensor,  # [num_images, 3, 448, 448]
    "bboxes": norm_bboxes,  # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
}

if (
    not norm_bboxes or not norm_bboxes[0]
):  # Check if norm_bboxes is empty or contains an empty list
    print("No bounding boxes to process for GazeLLE. Skipping inference.")
else:
    with torch.no_grad():
        output = model(input_data)

    # Check if output is valid before trying to access its elements
    if (
        output
        and "heatmap" in output
        and output["heatmap"] is not None
        and output["heatmap"].numel() > 0
    ):
        img1_person1_heatmap = output["heatmap"][0][0]  # [64, 64] heatmap
        print("GazeLLE output heatmap shape:", img1_person1_heatmap.shape)
        if (
            model.inout and "inout" in output and output["inout"] is not None
        ):  # Check model.inout and key existence
            img1_person1_inout = output["inout"][0][0]
            print("GazeLLE in/out score:", img1_person1_inout.item())
        else:
            print("In/out score not available or model.inout is False.")
    else:
        print(
            "GazeLLE model did not produce valid output or heatmap. Skipping visualization."
        )
        output = None  # Ensure output is None if invalid


# visualize predicted gaze heatmap for each person and gaze in/out of frame score
def visualize_heatmap(
    pil_image,
    heatmap_tensor,
    bbox_normalized=None,
    inout_score_tensor=None,
    text_font_size_factor=0.03,
):  # font size factor 추가
    if isinstance(heatmap_tensor, torch.Tensor):
        heatmap_np = heatmap_tensor.detach().cpu().numpy()
    else:  # Already numpy
        heatmap_np = heatmap_tensor

    # Resize heatmap to PIL image size
    heatmap_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    # Apply colormap
    colored_heatmap_np = plt.cm.jet(np.array(heatmap_pil) / 255.0)[
        :, :, :3
    ]  # Take only RGB
    colored_heatmap_pil = Image.fromarray(
        (colored_heatmap_np * 255).astype(np.uint8)
    ).convert("RGBA")

    # Set alpha for heatmap overlay
    colored_heatmap_pil.putalpha(90)  # Transparency

    # Composite image with heatmap
    overlay_image = Image.alpha_composite(
        pil_image.convert("RGBA"), colored_heatmap_pil
    )

    if bbox_normalized is not None:
        img_width, img_height = pil_image.size
        # Denormalize bbox coordinates
        xmin, ymin, xmax, ymax = bbox_normalized
        abs_bbox = [
            xmin * img_width,
            ymin * img_height,
            xmax * img_width,
            ymax * img_height,
        ]

        draw = ImageDraw.Draw(overlay_image)
        # Draw rectangle for face bbox
        draw.rectangle(
            abs_bbox,
            outline="lime",
            width=int(min(img_width, img_height) * 0.01),  # Relative width
        )

        if inout_score_tensor is not None:
            inout_score_val = (
                inout_score_tensor.item()
                if isinstance(inout_score_tensor, torch.Tensor)
                else inout_score_tensor
            )
            text = f"in-frame: {inout_score_val:.2f}"

            # Determine text size based on image dimensions
            font_size = int(min(img_width, img_height) * text_font_size_factor)
            try:
                font = ImageFont.truetype(
                    "arial.ttf", font_size
                )  # Try loading a common font
            except IOError:
                font = ImageFont.load_default(
                    size=font_size
                )  # Fallback to default font

            # Calculate text position more robustly
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = abs_bbox[0]  # Start text at left edge of bbox
            text_y = abs_bbox[3] + int(
                text_height * 0.2
            )  # Position text below bbox with small padding

            # Ensure text is within image bounds (simple check)
            if text_y + text_height > img_height:
                text_y = (
                    abs_bbox[1] - text_height - int(text_height * 0.2)
                )  # Try above if below is out of bounds
            if text_x + text_width > img_width:
                text_x = img_width - text_width - 5  # Adjust if too wide

            draw.text((text_x, text_y), text, fill="lime", font=font)

    return overlay_image.convert("RGB")  # Convert back to RGB for imshow


if (
    output and boxes is not None and norm_bboxes and norm_bboxes[0]
):  # Check if there are outputs and boxes
    num_persons_to_visualize = len(norm_bboxes[0])
    # Ensure output["heatmap"] and output["inout"] have enough entries
    if (
        output["heatmap"] is not None
        and output["heatmap"].size(1) >= num_persons_to_visualize
    ):
        for i in range(num_persons_to_visualize):
            current_heatmap = output["heatmap"][0][i]
            current_inout_score = None
            if (
                model.inout
                and output["inout"] is not None
                and output["inout"].size(1) > i
            ):
                current_inout_score = output["inout"][0][i]

            visualized_img = visualize_heatmap(
                pil_image_for_gazelle,  # Original PIL image for GazeLLE demo
                current_heatmap,
                norm_bboxes[0][i],
                inout_score_tensor=current_inout_score,
            )
            plt.figure(figsize=(8, 6))  # Adjust figure size
            plt.imshow(visualized_img)
            plt.title(f"Gaze Heatmap for Person {i+1}")
            plt.axis("off")
            plt.show()
    else:
        print(
            "Not enough heatmap/inout entries in GazeLLE output for the number of detected faces."
        )
elif boxes is None:
    print("Skipping GazeLLE visualization as no faces were detected.")
else:
    print(
        "Skipping GazeLLE visualization as GazeLLE model output is not available or no valid bboxes."
    )

print("\\nDemo finished.")
