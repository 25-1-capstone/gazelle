import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import torch
import torchvision.transforms as T
import onnxruntime as ort
import hailo_platform as hpf

# ============ configurable paths via CLI ============

parser = argparse.ArgumentParser()
parser.add_argument("--hef", required=True, help="Path to compiled HEF backbone")
parser.add_argument(
    "--onnx",
    required=True,
    help="ONNX export of the same backbone (used only for shape check)",
)
parser.add_argument(
    "--pth",
    default=None,
    required=True,
    help="(Optional) .pth checkpoint used by GazeLLE – proves we share weights",
)
parser.add_argument(
    "--size", type=int, default=448, required=True, help="Input resolution (224 or 448)"
)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument(
    "--img",
    default="https://www.looper.com/img/gallery/the-office-funniest-moments-ranked/jim-and-dwights-customer-service-training-1627594561.jpg",
)
args = parser.parse_args()

IMG_SIZE = (args.size, args.size)
device = args.device
# ----------------------------------------------------------------------------
# ---------- helper functions ------------------------------------------------

MEAN_255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def hailo_normalize(img_np: np.ndarray) -> np.ndarray:
    """Hailo parser expects float32 RGB 0-255 followed by (x-mean)/std"""
    img = img_np.copy()
    for c in range(3):
        img[..., c] = (img[..., c] - MEAN_255[c]) / STD_255[c]
    return img


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    )


# ----------------------------------------------------------------------------


# ---------- import gazelle package ------------------------------------------

# project_root를 계산하고, gazelle 패키지를 임포트

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


# ----------------------------------------------------------------------------


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


# ---------- 1. acquire test image ------------------------------------------

print(f"\nDownloading sample image …")
from io import BytesIO

pil = download_image(args.img)
pil = pil.resize(IMG_SIZE, Image.Resampling.BICUBIC)

# keep both PIL (for PyTorch) and np.float32 HWC (for HEF)
np_img_255 = np.array(pil).astype(np.float32)  # [H, W, C] 0-255


# --- 기본 설정 ---

ONNX_MODEL_PATH = args.onnx
# ONNX 모델은 448x448 입력을 기준으로 변환되었을 가능성이 높음
ONNX_INPUT_SIZE = IMG_SIZE


# 1. DinoV2 s14 백본 로드
# 이 백본은 GazeLLE 모델에 사용되며, PyTorch 기반의 특징 추출을 담당
backbone_s14 = DinoV2Backbone("dinov2_vits14")
print(
    f"Created DinoV2Backbone with dino_vits14. Pretrained DINOv2 weights for backbone are loaded via torch.hub."
)
# backbone_s14.model은 torch.hub에서 로드된 원본 DINOv2 모델임

# 2. GazeLLE 모델을 s14 백본으로 직접 초기화
model = GazeLLE(backbone=backbone_s14, in_size=IMG_SIZE)
print(f"GazeLLE model initialized with dino_vits14 backbone.")

# 3. 변환(transform)은 생성된 s14 백본에서 가져오기 (GazeLLE용: 448x448)
gazelle_transform = backbone_s14.get_transform(in_size=IMG_SIZE)

model.eval()
model.to(device)

# 4. 로컬 체크포인트에서 GazeLLE 헤드 가중치 로드
checkpoint_path = args.pth
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
        # export_onnx_dinov2_vits14.py와 동일한 허용 오차 사용
        rtol_val = 1e-02
        atol_val = 1e-04
        print(f"Comparing with rtol={rtol_val}, atol={atol_val}")
        try:
            np.testing.assert_allclose(
                pytorch_features.cpu().numpy(),  # .cpu()를 호출하여 CPU에서 비교
                onnx_features.cpu().numpy(),  # ONNX 결과도 CPU로 (ONNX 세션이 CPU 사용 가정)
                rtol=rtol_val,  # 허용 오차 범위는 export_onnx_dinov2_vits14.py와 일치
                atol=atol_val,
            )
            abs_diff = torch.abs(pytorch_features - onnx_features)
            print(f"  Max absolute difference: {abs_diff.max().item()}")
            print(f"  Mean absolute difference: {abs_diff.mean().item()}")
            print("SUCCESS: PyTorch backbone features and ONNX features are close.")
        except AssertionError as e:
            print("FAILURE: PyTorch backbone features and ONNX features differ.")
            print(e)  # 상세 오류 메시지 출력 활성화
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

# --- Comparing PyTorch Backbone with HEF ---
print("\\n--- Comparing PyTorch Backbone with HEF ---")
if not args.hef or not os.path.exists(args.hef):
    print(f"HEF model file not provided or not found at: {args.hef}")
    print("Skipping HEF comparison.")
else:
    try:
        # Ensure pil_image_for_comparison and pytorch_features are available from ONNX block
        if (
            "pil_image_for_comparison" not in locals()
            or pil_image_for_comparison is None
        ):
            print(
                "Error: pil_image_for_comparison not available for HEF test. Skipping HEF comparison."
            )
            raise RuntimeError("pil_image_for_comparison not found for HEF test.")

        if "pytorch_features" not in locals() or pytorch_features is None:
            print(
                "Error: pytorch_features not defined from ONNX block. Cannot compare with HEF. Skipping HEF comparison."
            )
            raise RuntimeError("pytorch_features not found for HEF comparison.")

        print(f"Using HEF model from: {args.hef}")

        # 1. Prepare HEF input
        hef_model_obj = hpf.HEF(args.hef)
        input_vstream_infos = hef_model_obj.get_input_vstream_infos()
        output_vstream_infos = hef_model_obj.get_output_vstream_infos()

        if not input_vstream_infos:
            raise RuntimeError("No input vstream info found in HEF.")
        if not output_vstream_infos:
            raise RuntimeError("No output vstream info found in HEF.")

        # Get HEF expected input shape (H, W, C)
        # input_vstream_infos[0].shape can be (H,W,C) or (N,H,W,C)
        # We are interested in H, W for resizing the PIL image
        if len(input_vstream_infos[0].shape) == 3: # H, W, C
            hef_expected_h_in = input_vstream_infos[0].shape[0]
            hef_expected_w_in = input_vstream_infos[0].shape[1]
        elif len(input_vstream_infos[0].shape) == 4: # N, H, W, C
            hef_expected_h_in = input_vstream_infos[0].shape[1]
            hef_expected_w_in = input_vstream_infos[0].shape[2]
        else:
            raise RuntimeError(f"Unexpected HEF input shape: {input_vstream_infos[0].shape}")
        
        print(f"HEF model expects input HxW: {hef_expected_h_in}x{hef_expected_w_in}")

        # Resize pil_image_for_comparison to HEF's expected dimensions
        pil_for_hef_resized = pil_image_for_comparison.resize(
            (hef_expected_w_in, hef_expected_h_in), Image.Resampling.BICUBIC
        )
        print(f"Resized PIL image for HEF to: {pil_for_hef_resized.size}")

        # Convert the resized PIL image to UINT8 NumPy array (H, W, C)
        hef_input_np_uint8_hwc = np.array(pil_for_hef_resized) # This is already uint8

        # Add batch dimension: (1, H, W, C)
        hef_input_final = np.expand_dims(hef_input_np_uint8_hwc, axis=0)
        print(f"HEF input tensor final shape (UINT8): {hef_input_final.shape}, dtype: {hef_input_final.dtype}")


        # 2. Perform HEF inference (hef_model_obj, input_vstream_infos, output_vstream_infos already defined)
        hef_input_name = input_vstream_infos[0].name
        hef_output_name = output_vstream_infos[0].name

        print(
            f"  HEF expected input name: {hef_input_name}, shape: {input_vstream_infos[0].shape}"
        )
        print(
            f"  HEF expected output name: {hef_output_name}, shape: {output_vstream_infos[0].shape}"
        )

        # Check if HEF input shape matches our prepared input shape (after batching)
        # input_vstream_infos[0].shape is (H, W, C) or (N, H, W, C).
        # Our hef_input_final is (N, H, W, C)
        expected_hef_shape_no_batch = tuple(
            input_vstream_infos[0].shape[-3:]
        )  # H, W, C
        actual_hef_input_shape_no_batch = tuple(hef_input_final.shape[1:])  # H, W, C (from our uint8 NHWC array)

        if expected_hef_shape_no_batch != actual_hef_input_shape_no_batch:
            print(
                f"Warning: HEF expected input HWC {expected_hef_shape_no_batch} but actual input HWC is {actual_hef_input_shape_no_batch} after preparation. This should not happen if resizing was correct."
            )
            # If this warning appears, there's a discrepancy between how input_vstream_infos[0].shape
            # was interpreted for resizing vs. for this check.
            # However, hef_input_final is already prepared with the target dimensions and uint8 type.

        print("Performing HEF inference...")
        # Create a VDevice and configure it with the HEF
        with hpf.VDevice() as target:
            # Configure the HEF on the device
            network_group = target.configure(hef_model_obj)[0]  # First network group from the HEF
            
            # Sanity check: Print what the HEF thinks it should receive
            print("HEF expected input stream properties:")
            # input_vstream_infos is already defined from hef_model_obj.get_input_vstream_infos() earlier in the script
            for vstream_info in input_vstream_infos:
                try:
                    order_str = str(vstream_info.format.order)
                except AttributeError:
                    order_str = "N/A"
                try:
                    format_type_str = str(vstream_info.format.type)
                except AttributeError:
                    format_type_str = "N/A"

                print(f"  Input '{vstream_info.name}': shape={vstream_info.shape}, order={order_str}, format_type={format_type_str}")
                # Note: Mean and Std for normalization are typically applied externally before HEF inference.
                # This script uses MEAN_255 and STD_255 defined globally.

            # Set up inference parameters
            in_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
            out_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
            
            # Resize the input tensor to match the expected input shape if necessary
            # THIS BLOCK IS NOW REDUNDANT as hef_input_final is already prepared with correct size and uint8 type.
            '''
            expected_h, expected_w = expected_hef_shape_no_batch[0], expected_hef_shape_no_batch[1]
            if actual_hef_input_shape_no_batch != expected_hef_shape_no_batch: # This check is mostly covered above
                print(f"Resizing HEF input from {actual_hef_input_shape_no_batch} to {expected_hef_shape_no_batch} (H, W, C)")
                # from PIL import Image # Already imported
                # hef_input_final is (N,H,W,C) and float32. Convert to uint8 for PIL.
                img_to_resize_uint8 = hef_input_final[0].astype(np.uint8)
                img_pil_to_resize = Image.fromarray(img_to_resize_uint8) # Now expects uint8
                img_resized_pil = img_pil_to_resize.resize((expected_w, expected_h), Image.Resampling.BICUBIC)
                # Convert back to float32 numpy array as was the original intention for hef_input_final
                resized_np_float32 = np.array(img_resized_pil).astype(np.float32)
                # No hailo_normalize here
                # Add batch dimension back
                hef_input_final = np.expand_dims(resized_np_float32, axis=0)
                print(f"Resized HEF input tensor shape: {hef_input_final.shape}")
            '''
            
            # Activate the network and perform inference
            with network_group.activate():
                with hpf.InferVStreams(network_group, in_params, out_params) as infer_pipeline:
                    hef_output_dict = infer_pipeline.infer({hef_input_name: hef_input_final})

        hef_features_raw_np = hef_output_dict[hef_output_name]  # Get the output from the output name
        print(
            f"HEF backbone raw output shape: {hef_features_raw_np.shape}"
        )  # Expected (1, H_feat, W_feat, C_feat)

        # 3. Process HEF output to match PyTorch/ONNX format (B, C, H_feat, W_feat)
        if (
            hef_features_raw_np.ndim == 4 and hef_features_raw_np.shape[0] == 1
        ):  # Assuming batch size 1
            # Input: (1, H_feat, W_feat, C_feat) -> Output: (1, C_feat, H_feat, W_feat)
            hef_features_processed_np = np.transpose(hef_features_raw_np, (0, 3, 1, 2))
        elif (
            hef_features_raw_np.ndim == 3
        ):  # Case: (H_feat, W_feat, C_feat) -> add batch dim and transpose
            print("Warning: HEF output was 3D, adding batch dimension and transposing.")
            hef_features_processed_np = np.transpose(
                np.expand_dims(hef_features_raw_np, axis=0), (0, 3, 1, 2)
            )
        else:
            print(
                f"Warning: HEF output is not 4D with batch 1, or 3D. Shape is {hef_features_raw_np.shape}. Attempting to use as is or permute if C is last."
            )
            if (
                hef_features_raw_np.shape[-1] == pytorch_features.shape[1]
            ):  # if last dim is channel
                hef_features_processed_np = np.transpose(
                    hef_features_raw_np, (0, 3, 1, 2)
                )  # B H W C -> B C H W
            else:  # Fallback, hoping it's already B C H W
                hef_features_processed_np = hef_features_raw_np

        hef_features = torch.from_numpy(hef_features_processed_np).to(device)
        print(f"HEF backbone processed output shape (B,C,H,W): {hef_features.shape}")

        # 4. Compare with PyTorch features
        rtol_val = 1e-02  # from ONNX comparison
        atol_val = 1e-04  # from ONNX comparison
        print(
            f"Comparing PyTorch features with HEF features (rtol={rtol_val}, atol={atol_val})..."
        )

        try:
            np.testing.assert_allclose(
                pytorch_features.cpu().numpy(),
                hef_features.cpu().numpy(),
                rtol=rtol_val,
                atol=atol_val,
            )
            abs_diff_hef = torch.abs(pytorch_features.cpu() - hef_features.cpu())
            print(
                f"  Max absolute difference (PyTorch vs HEF): {abs_diff_hef.max().item():.6e}"
            )
            print(
                f"  Mean absolute difference (PyTorch vs HEF): {abs_diff_hef.mean().item():.6e}"
            )
            print("SUCCESS: PyTorch backbone features and HEF features are close.")
        except AssertionError as e_hef:
            print("FAILURE: PyTorch backbone features and HEF features differ.")
            print(
                str(e_hef)
            )  # Print only the message part of assertion error for brevity
            abs_diff_hef = torch.abs(pytorch_features.cpu() - hef_features.cpu())
            print(
                f"  Max absolute difference (PyTorch vs HEF): {abs_diff_hef.max().item():.6e}"
            )
            print(
                f"  Mean absolute difference (PyTorch vs HEF): {abs_diff_hef.mean().item():.6e}"
            )

    except FileNotFoundError as e_fnf:
        print(f"HEF related file not found: {e_fnf}")
    except RuntimeError as e_rt:
        print(f"Runtime error during HEF comparison: {e_rt}")
        import traceback

        traceback.print_exc()
    except ImportError as e_imp:
        print(
            f"ImportError related to Hailo Platform: {e_imp}. Is hailo_platform installed and configured?"
        )
        print(
            "Please ensure 'hailo_platform' is installed and your environment is set up for Hailo SDK."
        )
    except Exception as e_gen:
        print(f"An unexpected error occurred during HEF comparison: {e_gen}")
        import traceback

        traceback.print_exc()

print("--- End HEF Comparison ---\\n")
from facenet_pytorch import MTCNN

# import numpy as np # 이미 위에서 임포트됨

# initialize once (on GPU if available)
mtcnn = MTCNN(keep_all=True, device="cpu")  # device 변수 사용

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

# --- GazeLLE 추론 with Pytorch Features (새로운 섹션) ---
print("\\n--- Running GazeLLE with Pre-extracted PyTorch Features ---")
if "pytorch_features" in locals() and pytorch_features is not None:
    if "pil_image_for_comparison" in locals() and pil_image_for_comparison is not None:
        try:
            # 1. pytorch_features의 정보 가져오기
            # pytorch_features는 (B, C, H_feat, W_feat) 형태
            _, feature_dim, feat_h, feat_w = pytorch_features.shape
            print(f"Using PyTorch features with shape: B={_}, C={feature_dim}, H_feat={feat_h}, W_feat={feat_w}")

            # 2. 새로운 GazeLLE 모델 인스턴스 생성 (backbone 없이)
            gazelle_from_features = GazeLLE(
                backbone=None, 
                inout=model.inout, 
                dim=model.dim, 
                num_layers=model.num_layers,
                featmap_h=feat_h,
                featmap_w=feat_w,
                feature_dim=feature_dim,
                out_size=model.out_size
            )
            
            if args.pth and os.path.exists(args.pth):
                print(f"Loading GazeLLE head weights for gazelle_from_features from: {args.pth}")
                checkpoint_feat = torch.load(args.pth, map_location=device)
                actual_state_dict_feat = None
                if isinstance(checkpoint_feat, dict):
                    if "model_state_dict" in checkpoint_feat: actual_state_dict_feat = checkpoint_feat["model_state_dict"]
                    elif "state_dict" in checkpoint_feat: actual_state_dict_feat = checkpoint_feat["state_dict"]
                    elif "model" in checkpoint_feat: actual_state_dict_feat = checkpoint_feat["model"]
                    else: actual_state_dict_feat = checkpoint_feat
                else: actual_state_dict_feat = checkpoint_feat
                
                if actual_state_dict_feat is None:
                     raise ValueError("Loaded checkpoint for features model is None.")

                gazelle_from_features.load_gazelle_state_dict(actual_state_dict_feat, include_backbone=False)
                print("Successfully loaded GazeLLE head weights into gazelle_from_features.")
            else:
                print("No .pth file provided or found for gazelle_from_features. Using uninitialized head.")

            gazelle_from_features.eval()
            gazelle_from_features.to(device)

            width_feat, height_feat = pil_image_for_comparison.size
            boxes_feat, _ = mtcnn.detect(pil_image_for_comparison)

            if boxes_feat is None:
                print("No faces detected in the image for GazeLLE with features. Skipping.")
            else:
                print(f"Detected face boxes for GazeLLE with features: {boxes_feat}")
                norm_bboxes_feat = [[np.array(bbox) / np.array([width_feat, height_feat, width_feat, height_feat]) for bbox in boxes_feat]]

                input_data_feat = {
                    "extracted_features": pytorch_features.to(device), 
                    "bboxes": norm_bboxes_feat,
                }
                
                hef_input_data_feat = {
                    "extracted_features": hef_features.to(device), 
                    "bboxes": norm_bboxes_feat,
                }

                if not norm_bboxes_feat or not norm_bboxes_feat[0]:
                    print("No bounding boxes to process for GazeLLE with features. Skipping inference.")
                else:
                    with torch.no_grad():
                        # output_feat = gazelle_from_features(input_data_feat)
                        output_feat = gazelle_from_features(hef_input_data_feat)
                    print("Visualizing results from GazeLLE with PyTorch features:")
                    for i in range(len(boxes_feat)):
                        plt.figure(figsize=(8,8))
                        plt.imshow(
                            visualize_heatmap(
                                pil_image_for_comparison, 
                                output_feat["heatmap"][0][i],
                                norm_bboxes_feat[0][i],
                                inout_score=output_feat["inout"][0][i] if output_feat["inout"] is not None else None,
                            )
                        )
                        plt.title(f"GazeLLE (PyTorch Feat) - Person {i+1}")
                        plt.axis("off")
                        #plt.show()
                        # 생성된 이미지를 파일로 저장합니다.
                        output_filename = f"gazelle_pytorch_feat_person_{i+1}.png"
                        plt.savefig(output_filename)
                        plt.close() # 다음 플롯을 위해 현재 플롯을 닫습니다.
                        print(f"Saved GazeLLE (PyTorch Feat) visualization to {output_filename}")
        except Exception as e_feat:
            print(f"Error during GazeLLE inference with PyTorch features: {e_feat}")
            import traceback
            traceback.print_exc()
    else:
        print("pil_image_for_comparison not available. Skipping GazeLLE with PyTorch features.")
else:
    print("PyTorch features (pytorch_features) not available. Skipping GazeLLE with PyTorch features.")
print("--- End GazeLLE with PyTorch Features ---\\n")

