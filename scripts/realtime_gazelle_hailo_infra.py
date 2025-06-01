#!/usr/bin/env python3
"""
Hailo-8 인프라와 GStreamer를 사용한 실시간 시선 추정.
camera capture와 processing pipeline을 위해 hailo_apps_infra를 사용합니다.
"""
import argparse  # 명령줄 인수 파싱을 위한 라이브러리
import sys  # 시스템 관련 매개변수와 함수에 접근
import time  # 시간 관련 함수 제공
from pathlib import Path  # 객체 지향적 파일시스템 경로 처리
import gi  # GObject Introspection 바인딩
gi.require_version('Gst', '1.0')  # GStreamer 1.0 버전 요구
from gi.repository import Gst, GLib  # GStreamer와 GLib 라이브러리 import
import os  # 운영체제 인터페이스
import numpy as np  # 수치 계산을 위한 라이브러리
import cv2  # OpenCV 컴퓨터 비전 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
from PIL import Image  # Python Imaging Library
import matplotlib  # 플롯팅 라이브러리
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt  # matplotlib의 pyplot 인터페이스
from facenet_pytorch import MTCNN  # 얼굴 감지를 위한 MTCNN 모델
import hailo  # Hailo AI 가속기 라이브러리
from hailo_platform import (  # Hailo 플랫폼 관련 클래스들
    VDevice, HailoStreamInterface, InferVStreams, 
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)

# gazelle 모듈을 import하기 위해 부모 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from gazelle.model import GazeLLE  # GazeLLE 시선 추정 모델
from gazelle.backbone import DinoV2Backbone  # DINOv2 backbone 네트워크

# Hailo 인프라 관련 모듈들 import
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,  # GStreamer pad에서 format 정보 추출
    get_numpy_from_buffer,  # GStreamer buffer를 numpy 배열로 변환
    app_callback_class,  # 애플리케이션 callback 클래스
    get_default_parser,  # 기본 argument parser 생성
    detect_hailo_arch,  # Hailo 아키텍처 자동 감지
)
from hailo_apps_infra.gstreamer_app import GStreamerApp  # GStreamer 애플리케이션 기본 클래스
from hailo_apps_infra.gstreamer_helper_pipelines import (  # GStreamer 파이프라인 헬퍼들
    QUEUE,  # Queue 요소 생성
    SOURCE_PIPELINE,  # 소스 파이프라인 생성
    DISPLAY_PIPELINE,  # 디스플레이 파이프라인 생성
)


# ============================================================================
# 설정 및 상수
# ============================================================================

# 기본 설정값들을 정의한 딕셔너리
DEFAULT_CONFIG = {
    'output_dir': './output_frames',  # 출력 프레임 저장 디렉토리
    'inference_output_dir': './inference_results',  # 추론 결과 저장 디렉토리
    'device': 'cpu',  # PyTorch 연산 디바이스 (cpu 또는 cuda)
    'save_interval': 1.0,  # 프레임 저장 간격 (초 단위)
    'max_frames': 10,  # 최대 저장할 프레임 수
    'skip_frames': 0,  # 건너뛸 프레임 수
    'save_mode': 'time',  # 저장 모드 ('time' 또는 'frame')
    'nominal_fps': 30,  # 명목상 FPS
}

# 디버그 출력 간격 설정
DEBUG_INTERVALS = {
    'frame_info': 10,      # N 프레임마다 프레임 정보 출력
    'heartbeat': 5,        # N초마다 heartbeat 메시지 출력
    'processing_avg': 10,  # 마지막 N 프레임의 평균 처리 시간 계산
}


# ============================================================================
# 유틸리티 함수들
# ============================================================================

def create_directories(paths):
    """디렉토리가 존재하지 않으면 생성합니다."""
    for path in paths:  # 각 경로에 대해 반복
        Path(path).mkdir(exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)


def normalize_bounding_boxes(boxes, width, height):
    """바운딩 박스를 [0,1] 범위로 정규화합니다."""
    # 각 바운딩 박스를 이미지 크기로 나누어 정규화
    return [[np.array(bbox) / np.array([width, height, width, height]) for bbox in boxes]]


def get_hef_input_dimensions(hef_model):
    """HEF 모델에서 입력 차원을 추출합니다."""
    input_vs = hef_model.get_input_vstream_infos()  # 입력 스트림 정보 가져오기
    if not input_vs:  # 입력 정보가 없으면 오류 발생
        raise ValueError("HEF 입력 정보를 가져오는데 실패했습니다")
    
    shape = input_vs[0].shape  # 첫 번째 입력의 shape 가져오기
    if len(shape) == 3:  # HWC 포맷인 경우
        return shape[0], shape[1]  # Height, Width 반환
    else:  # NHWC 포맷인 경우
        return shape[1], shape[2]  # Height, Width 반환


def process_dino_features(feat_raw):
    """DINOv2 raw features를 올바른 텐서 포맷으로 처리합니다."""
    if feat_raw.ndim == 3:  # 3차원 배열인 경우
        # [H, W, C] -> [1, C, H, W] 포맷으로 변환
        feat_processed = np.transpose(feat_raw, (2, 0, 1))  # 차원 순서 변경
        feat_processed = np.expand_dims(feat_processed, 0)  # 배치 차원 추가
    elif feat_raw.ndim == 4:  # 4차원 배열인 경우
        if feat_raw.shape[-1] == 384:  # [N, H, W, C] 포맷인 경우
            # [N, H, W, C] -> [N, C, H, W] 포맷으로 변환
            feat_processed = np.transpose(feat_raw, (0, 3, 1, 2))
        else:  # 이미 [N, C, H, W] 포맷인 경우
            feat_processed = feat_raw  # 그대로 사용
    else:  # 예상치 못한 차원인 경우
        raise ValueError(f"예상치 못한 feature shape: {feat_raw.shape}")
    
    return feat_processed  # 처리된 feature 반환


def find_gaze_point(heatmap):
    """히트맵에서 시선 지점을 찾습니다 (최대값 위치)."""
    if heatmap.ndim == 3:  # 3차원인 경우
        heatmap = heatmap.squeeze()  # 차원 축소
    gaze_y, gaze_x = np.unravel_index(heatmap.argmax(), heatmap.shape)  # 최대값의 인덱스를 좌표로 변환
    return gaze_x, gaze_y  # x, y 좌표 반환


def compute_gaze_targets(heatmaps, object_detections, img_width, img_height):
    """히트맵과 감지 결과를 기반으로 어떤 객체를 보고 있는지 결정합니다."""
    gaze_targets = []  # 시선 타겟 리스트 초기화
    
    for heatmap in heatmaps:  # 각 히트맵에 대해 반복
        if heatmap.ndim == 3:  # 3차원인 경우
            heatmap = heatmap.squeeze()  # 차원 축소
        
        # 필요시 히트맵을 이미지 크기에 맞게 조정
        if heatmap.shape != (img_height, img_width):
            heatmap = cv2.resize(heatmap, (img_width, img_height))
        
        # 가장 높은 시선 확률을 가진 객체 찾기
        best_object = None  # 최적 객체 초기화
        best_score = 0.0  # 최고 점수 초기화
        
        for detection in object_detections:  # 각 감지된 객체에 대해 반복
            bbox = detection['bbox']  # 바운딩 박스 가져오기
            x1, y1, x2, y2 = bbox  # 좌표 분해
            
            # 이 바운딩 박스에 해당하는 히트맵 영역 추출
            bbox_heatmap = heatmap[y1:y2, x1:x2]
            
            if bbox_heatmap.size > 0:  # 히트맵 영역이 존재하는 경우
                # 이 박스 내의 평균 시선 확률 계산
                avg_gaze_prob = np.mean(bbox_heatmap)
                max_gaze_prob = np.max(bbox_heatmap)  # 최대 시선 확률
                
                # 강건성을 위해 평균과 최대값의 조합 사용
                combined_score = 0.7 * max_gaze_prob + 0.3 * avg_gaze_prob
                
                if combined_score > best_score:  # 현재까지의 최고 점수보다 높으면
                    best_score = combined_score  # 최고 점수 업데이트
                    best_object = {  # 최적 객체 정보 저장
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'bbox': bbox,
                        'gaze_score': combined_score,
                        'max_gaze_prob': max_gaze_prob,
                        'avg_gaze_prob': avg_gaze_prob
                    }
        
        # 피크 시선 지점도 가져오기
        gaze_x, gaze_y = find_gaze_point(heatmap)
        
        # 시선 타겟 정보 추가
        gaze_targets.append({
            'gaze_point': (gaze_x, gaze_y),  # 시선 지점 좌표
            'gaze_object': best_object,  # 시선 대상 객체
            'heatmap_max': np.max(heatmap)  # 히트맵 최대값
        })
    
    return gaze_targets  # 시선 타겟 리스트 반환


# ============================================================================
# Hailo 추론 관리자
# ============================================================================

class HailoInferenceManager:
    """DINOv2 추론을 위한 Hailo AI 가속기를 관리합니다."""
    
    def __init__(self, hef_path, vdevice=None):
        """Hailo 추론 관리자 초기화"""
        self.hef_path = hef_path  # HEF 모델 파일 경로 저장
        self.vdevice = vdevice  # 가상 디바이스 객체 저장
        self._init_device()  # 디바이스 초기화 메서드 호출
    
    def _init_device(self):
        """Hailo 디바이스를 초기화하고 모델을 로드합니다."""
        import hailo_platform as hpf  # Hailo 플랫폼 라이브러리 import
        
        # 가상 디바이스가 제공되지 않았으면 생성
        if self.vdevice is None:
            self.vdevice = VDevice()  # 새로운 가상 디바이스 생성
        
        # HEF 모델 로드
        self.hef = hpf.HEF(self.hef_path)  # HEF 파일에서 모델 로드
        configure_params = ConfigureParams.create_from_hef(  # HEF에서 설정 파라미터 생성
            self.hef, interface=HailoStreamInterface.PCIe  # PCIe 인터페이스 사용
        )
        network_group = self.vdevice.configure(self.hef, configure_params)[0]  # 네트워크 그룹 설정
        
        # 스트림 설정
        self.input_vstreams_params = InputVStreamParams.make(  # 입력 스트림 파라미터 생성
            network_group, quantized=True, format_type=FormatType.UINT8  # 양자화된 UINT8 형식
        )
        self.output_vstreams_params = OutputVStreamParams.make(  # 출력 스트림 파라미터 생성
            network_group, format_type=FormatType.FLOAT32  # FLOAT32 형식
        )
        
        self.network_group = network_group  # 네트워크 그룹 저장
        self._print_model_info()  # 모델 정보 출력
    
    def _print_model_info(self):
        """모델 입력/출력 사양을 출력합니다."""
        pass  # 현재는 아무것도 하지 않음 (추후 구현 예정)
    
    def run_inference(self, frame):
        """입력 프레임에 대해 추론을 실행합니다."""
        # 입력 차원 가져오기
        input_info = self.hef.get_input_vstream_infos()[0]  # 첫 번째 입력 스트림 정보
        input_shape = input_info.shape  # 입력 형태 가져오기
        
        if len(input_shape) == 3:  # HWC 형식인 경우
            h, w = input_shape[0], input_shape[1]  # 높이, 너비 추출
        else:  # NHWC 형식인 경우
            h, w = input_shape[1], input_shape[2]  # 높이, 너비 추출
        
        # DETR용으로 원본 크기와 모델 입력 크기 저장
        if isinstance(self, DETRInferenceManager) and self.detr_input_size is None:
            self.detr_input_size = (h, w)  # DETR 입력 크기 저장
        
        # 프레임 크기 조정
        resized = cv2.resize(frame, (w, h))  # 모델 입력 크기에 맞게 리사이즈
        
        # 추론 실행
        with InferVStreams(self.network_group,   # 추론 스트림 생성
                          self.input_vstreams_params,   # 입력 스트림 파라미터
                          self.output_vstreams_params) as infer_pipeline:  # 출력 스트림 파라미터
            input_data = {  # 입력 데이터 준비
                list(self.input_vstreams_params.keys())[0]: np.expand_dims(resized, axis=0)  # 배치 차원 추가
            }
            
            with self.network_group.activate(None):  # 네트워크 그룹 활성화
                results = infer_pipeline.infer(input_data)  # 추론 실행
        
        return results  # 추론 결과 반환


class SCRFDInferenceManager(HailoInferenceManager):
    """SCRFD 객체 감지를 위한 전문 관리자입니다."""
    
    def __init__(self, hef_path, vdevice=None):
        """SCRFD 추론 관리자 초기화"""
        # 공유 vdevice로 부모 클래스 생성자 호출
        super().__init__(hef_path, vdevice)
    
    def process_detections(self, outputs, threshold=0.5):
        """SCRFD 출력을 처리하여 얼굴 감지 결과를 얻습니다."""
        # SCRFD는 일반적으로 다중 스케일 예측을 출력합니다
        # 이것은 단순화된 버전입니다 - 실제 모델 출력에 맞게 조정하세요
        detections = []  # 감지 결과 리스트 초기화
        
        for output_name, output_data in outputs.items():  # 각 출력에 대해 반복
            if 'bbox' in output_name or 'loc' in output_name:  # 바운딩 박스 출력인 경우
                # 바운딩 박스 예측 처리
                # 형태는 SCRFD 변형에 따라 다름
                continue
            elif 'conf' in output_name or 'cls' in output_name:  # 신뢰도 출력인 경우
                # 신뢰도 점수 처리
                continue
        
        # 현재는 빈 리스트 반환 - 실제 SCRFD 출력 형식에 맞게 구현 필요
        return detections


class DETRInferenceManager(HailoInferenceManager):
    """DETR 객체 감지를 위한 전문 관리자입니다."""
    
    def __init__(self, hef_path, vdevice=None, confidence_threshold=0.7, nms_threshold=0.3):
        """DETR 추론 관리자 초기화"""
        # 공유 vdevice로 부모 클래스 생성자 호출
        super().__init__(hef_path, vdevice)
        self.confidence_threshold = confidence_threshold  # 신뢰도 임계값 설정
        self.nms_threshold = nms_threshold  # NMS 임계값 설정
        # DETR용 COCO 클래스 이름 - 플레이스홀더를 포함한 92개 클래스
        # 모델은 92개의 로짓을 출력: 91개 객체 클래스 (0-90) + 1개 "no-object" 클래스 (91)
        self.COCO_CLASSES_92 = {  # COCO 클래스 라벨 딕셔너리
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
    91: "hair",}
        # Store original image size and DETR input size for proper scaling
        self.detr_input_size = None
    
    def process_detections(self, outputs, img_width, img_height, threshold=None):
        """Process DETR outputs to get object detections."""
        if threshold is None:
            threshold = self.confidence_threshold
        detections = []
        
        # DETR outputs: conv113 (class logits), conv116 (boxes)
        boxes_output = None
        scores_output = None
        
        for output_name, output_data in outputs.items():
            # Based on your output shapes:
            # conv116: (1, 1, 100, 4) - boxes
            # conv113: (1, 1, 100, 92) - class logits
            if 'conv116' in output_name or (output_data.shape[-1] == 4 and len(output_data.shape) >= 3):
                boxes_output = output_data
            elif 'conv113' in output_name or (output_data.shape[-1] > 4 and len(output_data.shape) >= 3):
                scores_output = output_data
        
        if boxes_output is not None and scores_output is not None:
            
            # Reshape to remove extra dimensions
            # From (1, 1, 100, 4) to (100, 4)
            if len(boxes_output.shape) == 4:
                boxes_output = boxes_output[0, 0]  # Remove batch and extra dim
            elif len(boxes_output.shape) == 3:
                boxes_output = boxes_output[0]  # Remove batch
                
            # From (1, 1, 100, 92) to (100, 92)
            if len(scores_output.shape) == 4:
                scores_output = scores_output[0, 0]  # Remove batch and extra dim
            elif len(scores_output.shape) == 3:
                scores_output = scores_output[0]  # Remove batch
            
            # Apply sigmoid to boxes if needed
            if boxes_output.min() < 0 or boxes_output.max() > 1:
                boxes_output = 1 / (1 + np.exp(-boxes_output))
            
            # Apply softmax to get probabilities from logits
            scores_probs = self._softmax(scores_output, axis=-1)
            
            
            # Process each detection
            num_queries = min(boxes_output.shape[0], 100)
            detection_count = 0
            
            for i in range(num_queries):
                box = boxes_output[i]
                class_probs = scores_probs[i]
                
                # Get best class including background
                best_class_idx = np.argmax(class_probs)
                best_score = class_probs[best_class_idx]
                
                # Skip if best class is background (index 91 in DETR output)
                if best_class_idx == 91:
                    continue
                
                # Filter: above threshold
                if best_score > threshold:
                    # DETR uses center_x, center_y, width, height format (normalized)
                    cx, cy, w, h = box
                    
                    # Get class name from the 92-class list, with bounds checking
                    if best_class_idx < len(self.COCO_CLASSES_92):
                        class_name = self.COCO_CLASSES_92[best_class_idx]
                    else:
                        # Unknown class index - skip this detection
                        if not hasattr(self, '_unknown_classes_logged'):
                            self._unknown_classes_logged = set()
                        if best_class_idx not in self._unknown_classes_logged:
                            print(f"[DEBUG] Warning: Unknown class index {best_class_idx} detected (confidence: {best_score:.3f})")
                            self._unknown_classes_logged.add(best_class_idx)
                        continue
                    
                    # Skip "N/A" placeholder classes or background
                    if class_name == "N/A" or class_name == "__background__":
                        continue
                    
                    
                    # Convert to corner format
                    # Box coordinates are normalized to [0,1] on the DETR input size (800x800)
                    # We need to scale them to the output image size (1280x720)
                    x1 = (cx - w/2) * img_width
                    y1 = (cy - h/2) * img_height
                    x2 = (cx + w/2) * img_width
                    y2 = (cy + h/2) * img_height
                    
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y2 = max(0, min(y2, img_height - 1))
                    
                    
                    # Convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Skip very small boxes (likely noise)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # Minimum absolute size
                    if box_width < 20 or box_height < 20:
                        continue
                    
                    # Minimum relative size (0.1% of image area)
                    min_relative_area = 0.001 * img_width * img_height
                    if box_area < min_relative_area:
                        continue
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': float(best_score)
                    })
                    detection_count += 1
        
        # Apply Non-Maximum Suppression (NMS) to remove duplicates
        if len(detections) > 0:
            num_before_nms = len(detections)
            detections = self._apply_nms(detections, iou_threshold=self.nms_threshold)
            num_after_nms = len(detections)
            
        
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return detections
        
        
        # Group detections by class for class-specific NMS
        detections_by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        
        # Apply NMS per class
        kept_detections = []
        nms_stats = {}  # Track removals per class
        
        for cls, class_detections in detections_by_class.items():
            # Sort by confidence (descending)
            class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS within this class
            kept_class_detections = []
            removed_count = 0
            
            for i, det in enumerate(class_detections):
                # Check if this detection overlaps too much with any kept detection of same class
                should_keep = True
                for j, kept in enumerate(kept_class_detections):
                    iou = self._compute_iou(det['bbox'], kept['bbox'])
                    
                    
                    if iou > iou_threshold:
                        should_keep = False
                        removed_count += 1
                        break
                
                if should_keep:
                    kept_class_detections.append(det)
            
            nms_stats[cls] = {'original': len(class_detections), 'kept': len(kept_class_detections), 'removed': removed_count}
            kept_detections.extend(kept_class_detections)
        
        
        # Sort all kept detections by confidence and limit to top N
        kept_detections = sorted(kept_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 20 detections (reduced from 25)
        return kept_detections[:20]
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        
        # Compute IoU
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        
        
        return iou
    
    def _softmax(self, x, axis=-1):
        """Compute softmax values for array x along specified axis."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ============================================================================
# Frame Processor
# ============================================================================

class FrameProcessor:
    """Handles frame processing logic."""
    
    def __init__(self, gazelle_model, hailo_manager, device='cpu', scrfd_manager=None, detr_manager=None):
        self.gazelle_model = gazelle_model
        self.hailo_manager = hailo_manager
        self.scrfd_manager = scrfd_manager
        self.detr_manager = detr_manager
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
    
    def process_frame(self, frame, width, height):
        """Process a single frame with unified gazelle and DETR processing."""
        processing_results = {
            'features': None,
            'boxes': [],
            'heatmaps': [],
            'object_detections': [],
            'gaze_targets': [],
            'highest_probability_target': None,
            'should_save': False
        }
        
        # Step 1: Run Hailo inference for DINOv2 features
        if not self.hailo_manager:
            return processing_results
            
        infer_results = self.hailo_manager.run_inference(frame)
        output_name = list(infer_results.keys())[0]
        feat_raw = infer_results[output_name]
        feat_processed = process_dino_features(feat_raw)
        feat_tensor = torch.from_numpy(feat_processed).to(self.device)
        processing_results['features'] = feat_tensor
        
        # Step 2: Face detection for gazelle processing
        face_detections = []
        if self.scrfd_manager:
            scrfd_results = self.scrfd_manager.run_inference(frame)
            scrfd_detections = self.scrfd_manager.process_detections(scrfd_results)
            face_detections.extend(scrfd_detections)
            face_boxes = [d['bbox'] for d in scrfd_detections if d.get('class') == 'face']
            if face_boxes:
                boxes = np.array(face_boxes)
            else:
                boxes, probs = self.mtcnn.detect(frame)
        else:
            boxes, probs = self.mtcnn.detect(frame)
        
        if boxes is None or len(boxes) == 0:
            boxes = np.array([[width*0.25, height*0.25, width*0.75, height*0.75]])
        
        processing_results['boxes'] = boxes
        
        # Step 3: DETR object detection
        object_detections = []
        if self.detr_manager:
            detr_results = self.detr_manager.run_inference(frame)
            detr_detections = self.detr_manager.process_detections(detr_results, width, height)
            object_detections.extend(detr_detections)
        
        processing_results['object_detections'] = face_detections + object_detections
        
        # Step 4: Only proceed with gazelle processing if we have both features and objects
        if len(processing_results['object_detections']) == 0:
            return processing_results
        
        # Step 5: Run GazeLLE inference
        norm_bboxes = normalize_bounding_boxes(boxes, width, height)
        with torch.no_grad():
            out = self.gazelle_model({
                "extracted_features": feat_tensor, 
                "bboxes": norm_bboxes
            })
        
        heatmaps = out["heatmap"][0].cpu().numpy()
        processing_results['heatmaps'] = heatmaps
        
        # Step 6: Compute gaze targets and find highest probability
        gaze_targets = compute_gaze_targets(heatmaps, processing_results['object_detections'], width, height)
        processing_results['gaze_targets'] = gaze_targets
        
        # Step 7: Determine highest probability target across all results
        highest_prob_target = None
        highest_prob_score = 0.0
        
        # Check gaze targets
        for target in gaze_targets:
            if target['gaze_object'] and target['gaze_object']['gaze_score'] > highest_prob_score:
                highest_prob_score = target['gaze_object']['gaze_score']
                highest_prob_target = {
                    'type': 'gaze_target',
                    'object': target['gaze_object'],
                    'probability': target['gaze_object']['gaze_score'],
                    'source': 'gazelle_gaze'
                }
        
        # Check object detection confidence scores
        for detection in processing_results['object_detections']:
            if detection['confidence'] > highest_prob_score:
                highest_prob_score = detection['confidence']
                highest_prob_target = {
                    'type': 'object_detection',
                    'object': detection,
                    'probability': detection['confidence'],
                    'source': 'detr_detection'
                }
        
        processing_results['highest_probability_target'] = highest_prob_target
        
        # Step 8: Only save if we have a high confidence result
        if highest_prob_target and highest_prob_score > 0.5:
            processing_results['should_save'] = True
        
        return processing_results


# ============================================================================
# Result Saver
# ============================================================================

class ResultSaver:
    """Handles saving of frames and inference results."""
    
    def __init__(self, output_dir, inference_output_dir=None):
        self.output_dir = Path(output_dir)
        self.inference_output_dir = Path(inference_output_dir) if inference_output_dir else None
        self.saved_frames = 0
        self.saved_inference = 0
        
        # Create directories
        dirs = [self.output_dir]
        if self.inference_output_dir:
            dirs.append(self.inference_output_dir)
        create_directories(dirs)
    
    def save_visualization(self, frame, boxes, heatmaps, object_detections=None, gaze_targets=None, highest_prob_target=None):
        """Save frame with gaze visualization and object detections."""
        try:
            frame_pil = Image.fromarray(frame)
            
            # Use first heatmap for visualization
            heatmap = heatmaps[0] if len(heatmaps) > 0 else np.zeros((frame.shape[0], frame.shape[1]))
            
            if heatmap.ndim == 3:
                heatmap = heatmap.squeeze()
            
            # Normalize and resize heatmap
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            if heatmap.shape != (frame.shape[0], frame.shape[1]):
                heatmap_norm = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]))
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(frame_pil)
            # Reduce heatmap opacity to ensure objects are visible
            plt.imshow(heatmap_norm, alpha=0.3, cmap='jet')
            
            # Get current axes for drawing
            ax = plt.gca()
            
            # Draw face bounding boxes
            for i, bbox in enumerate(boxes):
                xmin, ymin, xmax, ymax = bbox
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   fill=False, edgecolor='lime', linewidth=3)
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f'Face {i+1}',
                       color='lime', fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # Get gazed objects for highlighting
            gazed_objects = []
            if gaze_targets:
                for target in gaze_targets:
                    if target['gaze_object']:
                        gazed_objects.append(target['gaze_object'])
            
            # Draw object detections if available
            if object_detections:
                non_face_detections = [d for d in object_detections if d.get('class') != 'face']
                
                for detection in object_detections:
                    if detection.get('class') != 'face':  # Skip faces as they're already drawn
                        bbox = detection['bbox']
                        xmin, ymin, xmax, ymax = bbox
                        
                        # Check if this object is being gazed at
                        is_gazed = False
                        gaze_score = 0
                        for gazed_obj in gazed_objects:
                            if (gazed_obj['bbox'] == bbox and 
                                gazed_obj['class'] == detection['class']):
                                is_gazed = True
                                gaze_score = gazed_obj['gaze_score']
                                break
                        
                        # Use different colors and styles for gazed objects
                        if is_gazed:
                            linewidth = 5
                            linestyle = '-'
                            # Red color for gazed object
                            color = 'red'
                            label = f"[GAZE] {detection['class']} {detection.get('confidence', 0):.2f} (gaze: {gaze_score:.2f})"
                        else:
                            linewidth = 3
                            linestyle = '-'  # Changed from '--' to '-' for solid lines
                            # More visible color scheme
                            if 'person' in detection.get('class', '').lower():
                                color = 'cyan'
                            elif any(x in detection.get('class', '').lower() for x in ['chair', 'couch', 'bed']):
                                color = 'orange'
                            elif any(x in detection.get('class', '').lower() for x in ['mouse', 'keyboard', 'laptop', 'tv']):
                                color = 'magenta'
                            elif 'plant' in detection.get('class', '').lower():
                                color = 'green'
                            else:
                                color = 'yellow'
                            label = f"{detection['class']} {detection.get('confidence', 0):.2f}"
                        
                        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           fill=False, edgecolor=color, linewidth=linewidth,
                                           linestyle=linestyle)
                        ax.add_patch(rect)
                        
                        # Draw text label
                        text_y = ymin-5 if ymin > 20 else ymax+20  # Adjust text position if too close to top
                        ax.text(xmin, text_y, label,
                               color=color, fontsize=10 if not is_gazed else 12, 
                               weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='black' if not is_gazed else 'darkred', 
                                       alpha=0.7))
            
            # Draw gaze points
            if gaze_targets:
                for i, target in enumerate(gaze_targets):
                    gaze_x, gaze_y = target['gaze_point']
                    # Draw crosshair at gaze point
                    ax.plot(gaze_x, gaze_y, 'r+', markersize=20, markeredgewidth=3)
                    ax.add_patch(plt.Circle((gaze_x, gaze_y), 10, 
                                          fill=False, edgecolor='red', linewidth=2))
            
            # Update title with unified processing information
            title = f"Unified Processing - Frame {self.saved_frames + 1}"
            if highest_prob_target:
                target_obj = highest_prob_target['object']
                title += f" | Highest Prob: {target_obj['class']} ({highest_prob_target['probability']:.2f}) [{highest_prob_target['source']}]"
            elif gaze_targets and gaze_targets[0]['gaze_object']:
                gazed_class = gaze_targets[0]['gaze_object']['class']
                title += f" | Looking at: {gazed_class}"
            
            plt.title(title)
            plt.axis('off')
            
            # Force redraw to ensure all patches are rendered
            plt.draw()
            
            # Save
            output_path = self.output_dir / f"frame_{self.saved_frames + 1:04d}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            print(f"Saved frame {self.saved_frames + 1} to {output_path}")
            if gaze_targets and gaze_targets[0]['gaze_object']:
                print(f"  User is looking at: {gaze_targets[0]['gaze_object']['class']} "
                      f"(confidence: {gaze_targets[0]['gaze_object']['confidence']:.2f}, "
                      f"gaze score: {gaze_targets[0]['gaze_object']['gaze_score']:.2f})")
            self.saved_frames += 1
            
        except Exception as e:
            print(f"Error saving frame: {e}")
    
    def save_inference_data(self, features, boxes, heatmaps, frame_count, object_detections=None, gaze_targets=None, highest_prob_target=None):
        """Save raw inference results."""
        if not self.inference_output_dir:
            return
        
        try:
            result_dict = {
                'frame_number': frame_count,
                'timestamp': time.time(),
                'features': features.cpu().numpy(),
                'boxes': boxes,
                'heatmaps': heatmaps,
            }
            
            if object_detections:
                result_dict['object_detections'] = object_detections
            
            if gaze_targets:
                result_dict['gaze_targets'] = gaze_targets
            
            if highest_prob_target:
                result_dict['highest_probability_target'] = highest_prob_target
            
            output_path = self.inference_output_dir / f"inference_{self.saved_inference + 1:04d}.npz"
            np.savez_compressed(output_path, **result_dict)
            
            print(f"Saved inference results {self.saved_inference + 1} to {output_path}")
            self.saved_inference += 1
            
        except Exception as e:
            print(f"Error saving inference results: {e}")


# ============================================================================
# Timing Manager
# ============================================================================

class TimingManager:
    """Manages frame timing and save intervals."""
    
    def __init__(self, save_mode='time', save_interval=1.0, skip_frames=0):
        self.save_mode = save_mode
        self.save_interval = save_interval
        self.skip_frames = skip_frames
        
        # Timing tracking
        self.first_pts = None
        self.nominal_fps = DEFAULT_CONFIG['nominal_fps']
        self.last_save_ts = -1e9
        self.last_saved_orig_idx = -1e9
        self.last_inference_save_ts = -1e9
        self.last_inference_saved_orig_idx = -1e9
    
    def should_skip_frame(self, frame_count):
        """Check if frame should be skipped."""
        if self.skip_frames > 0:
            return frame_count % (self.skip_frames + 1) != 0
        return False
    
    def update_timing(self, pts_ms):
        """Update timing information from PTS."""
        if self.first_pts is None:
            self.first_pts = pts_ms
        return (pts_ms - self.first_pts) / 1000.0  # seconds since start
    
    def should_save_frame(self, rel_t):
        """Check if frame should be saved based on timing."""
        if self.save_mode == 'time':
            if rel_t - self.last_save_ts >= self.save_interval:
                self.last_save_ts = rel_t
                return True
        else:  # frame mode
            orig_idx = int(rel_t * self.nominal_fps)
            if orig_idx - self.last_saved_orig_idx >= self.save_interval:
                self.last_saved_orig_idx = orig_idx
                return True
        return False
    
    def should_save_inference(self, rel_t):
        """Check if inference results should be saved."""
        if self.save_mode == 'time':
            if rel_t - self.last_inference_save_ts >= self.save_interval:
                self.last_inference_save_ts = rel_t
                return True
        else:  # frame mode
            orig_idx = int(rel_t * self.nominal_fps)
            if orig_idx - self.last_inference_saved_orig_idx >= self.save_interval:
                self.last_inference_saved_orig_idx = orig_idx
                return True
        return False


# ============================================================================
# GazeLLE Callback Class
# ============================================================================

class GazeLLECallbackClass(app_callback_class):
    """Simplified callback class for GazeLLE processing."""
    
    def __init__(self, gazelle_model, device='cpu', output_dir='./output_frames',
                 save_interval=1.0, max_frames=10, hef_path=None, skip_frames=0,
                 save_inference_results=False, inference_output_dir='./inference_results',
                 save_mode='time', scrfd_hef_path=None, detr_hef_path=None,
                 detr_confidence=0.7, detr_nms=0.3):
        super().__init__()
        
        # Configuration
        self.device = device
        self.max_frames = max_frames
        self.save_inference_results = save_inference_results
        
        # Initialize components with shared VDevice
        shared_vdevice = None
        self.hailo_manager = HailoInferenceManager(hef_path, shared_vdevice) if hef_path else None
        if self.hailo_manager:
            shared_vdevice = self.hailo_manager.vdevice
        self.scrfd_manager = SCRFDInferenceManager(scrfd_hef_path, shared_vdevice) if scrfd_hef_path else None
        self.detr_manager = DETRInferenceManager(detr_hef_path, shared_vdevice, detr_confidence, detr_nms) if detr_hef_path else None
        self.frame_processor = FrameProcessor(gazelle_model, self.hailo_manager, device, self.scrfd_manager, self.detr_manager)
        self.result_saver = ResultSaver(output_dir, inference_output_dir if save_inference_results else None)
        self.timing_manager = TimingManager(save_mode, save_interval, skip_frames)
        
        # Tracking
        self.frame_count = 0
        self.processing_times = []
        
        # Print configuration
        self._print_config(output_dir, save_mode, save_interval, inference_output_dir)
    
    def _print_config(self, output_dir, save_mode, save_interval, inference_output_dir):
        """Print configuration summary."""
        pass
    
    def should_continue(self):
        """Check if processing should continue."""
        return True


# ============================================================================
# Main Callback Function
# ============================================================================

def gazelle_callback(pad, info, user_data):
    """Main GStreamer callback for processing frames."""
    user_data.frame_count += 1
    
    # Get buffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Get timing
    pts_ms = buffer.pts / Gst.SECOND * 1000
    rel_t = user_data.timing_manager.update_timing(pts_ms)
    
    # Check if should skip
    if user_data.timing_manager.should_skip_frame(user_data.frame_count):
        return Gst.PadProbeReturn.OK
    
    # Check if already saved max frames
    if user_data.result_saver.saved_frames >= user_data.max_frames:
        return Gst.PadProbeReturn.OK
    
    # Measure processing time
    start_time = time.time()
    
    
    # Get frame info
    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        return Gst.PadProbeReturn.OK
    
    # Get frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        return Gst.PadProbeReturn.OK
    
    
    try:
        # Process frame
        results = user_data.frame_processor.process_frame(frame, width, height)
        
        # Log model timing every 10 frames
        if user_data.frame_count % 10 == 1:
            if 'timing' in results:
                timing = results['timing']
                timing_str = ", ".join([f"{k}: {v:.1f}ms" for k, v in timing.items()])
                print(f"[TIMING] Frame {user_data.frame_count}: {timing_str}")
            else:
                print(f"[TIMING] Frame {user_data.frame_count}: No timing data available")
        
        # Add to GStreamer ROI (for pipeline integration)
        _add_roi_to_buffer(buffer, results['boxes'], results['heatmaps'], width, height)
        
        # Save only if unified processing indicates we should save
        if results.get('should_save', False) and user_data.timing_manager.should_save_frame(rel_t):
            # Include highest probability target info in save
            user_data.result_saver.save_visualization(
                frame, results['boxes'], results['heatmaps'], 
                results.get('object_detections'), results.get('gaze_targets'),
                highest_prob_target=results.get('highest_probability_target')
            )
            
            # Print unified processing result
            if results.get('highest_probability_target'):
                target = results['highest_probability_target']
                print(f"[UNIFIED] Frame {user_data.frame_count}: {target['source']} - "
                      f"{target['object']['class']} (prob: {target['probability']:.3f})")
        
        # Save inference results only when we have meaningful data
        if (user_data.save_inference_results and results.get('should_save', False) and 
            user_data.timing_manager.should_save_inference(rel_t)):
            user_data.result_saver.save_inference_data(
                results['features'], results['boxes'], results['heatmaps'], 
                user_data.frame_count, results.get('object_detections'), 
                results.get('gaze_targets'), 
                highest_prob_target=results.get('highest_probability_target')
            )
            
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
    
    # Record processing time
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    if len(user_data.processing_times) > 100:
        user_data.processing_times.pop(0)
    
    return Gst.PadProbeReturn.OK


def _add_roi_to_buffer(buffer, boxes, heatmaps, width, height):
    """Add ROI data to GStreamer buffer."""
    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        main_bbox = hailo.HailoBBox(0, 0, 1, 1)
        roi = hailo.HailoROI(main_bbox)
        hailo.add_roi_to_buffer(buffer, roi)
    
    for i, (bbox, heatmap) in enumerate(zip(boxes, heatmaps)):
        xmin, ymin, xmax, ymax = bbox
        norm_xmin = xmin / width
        norm_ymin = ymin / height
        norm_width = (xmax - xmin) / width
        norm_height = (ymax - ymin) / height
        
        face_bbox = hailo.HailoBBox(norm_xmin, norm_ymin, norm_width, norm_height)
        detection = hailo.HailoDetection(face_bbox, "face_with_gaze", 0.9)
        roi.add_object(detection)


# ============================================================================
# Command Line Parser
# ============================================================================

def get_gazelle_parser():
    """Create argument parser."""
    parser = get_default_parser()
    
    # Model arguments
    parser.add_argument("--hef", required=True, help="Path to compiled HEF backbone model file")
    parser.add_argument("--pth", required=True, help="Path to GazeLLE head checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="Torch device for GazeLLE head (cpu or cuda)")
    parser.add_argument("--scrfd-hef", help="Path to SCRFD HEF model for face detection")
    parser.add_argument("--detr-hef", help="Path to DETR HEF model for object detection")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./output_frames", help="Directory to save output frames")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to save")
    
    # Saving configuration
    parser.add_argument("--save-mode", choices=['time', 'frame'], default='time',
                       help="Save mode: 'time' for time-based intervals, 'frame' for frame count-based")
    parser.add_argument("--save-interval", type=float, default=1.0,
                       help="Save interval: seconds (if save-mode=time) or frames (if save-mode=frame)")
    
    # Processing options
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Skip N frames between processing (0=process all)")
    
    # Inference saving
    parser.add_argument("--save-inference", action="store_true",
                       help="Save raw inference results")
    parser.add_argument("--inference-output-dir", default="./inference_results",
                       help="Directory to save inference results")
    
    # Detection parameters
    parser.add_argument("--detr-confidence", type=float, default=0.7,
                       help="Confidence threshold for DETR object detection (default: 0.7)")
    parser.add_argument("--detr-nms", type=float, default=0.3,
                       help="NMS IoU threshold for DETR object detection (default: 0.3)")
    
    return parser


# ============================================================================
# GStreamer Application
# ============================================================================

class GStreamerGazeLLEApp(GStreamerApp):
    """GStreamer application for real-time gaze estimation."""
    
    def __init__(self, args, user_data):
        parser = get_gazelle_parser()
        super().__init__(parser, user_data)
        
        self.app_callback = gazelle_callback
        self._auto_detect_arch()
        self.hef_path = self.options_menu.hef
        self.create_pipeline()
    
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
        source = SOURCE_PIPELINE(
            self.video_source,
            self.video_width,
            self.video_height,
            self.video_format
        )
        
        callback = (
            f'queue name=hailo_pre_callback_q leaky=downstream '
            f'max-size-buffers=60 max-size-bytes=0 max-size-time=0 ! '
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
        
        pipeline = f'{source} ! {QUEUE("pre_callback_q")} ! {callback} ! {display}'
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
        if msg.src == self.pipeline and new_state == Gst.State.PLAYING:
            self.setup_callback()
        super().on_pipeline_state_changed(bus, msg)


# ============================================================================
# Model Loading
# ============================================================================

def load_gazelle_model(pth_path, hef_path, device='cpu'):
    """Load and configure GazeLLE model."""
    import hailo_platform as hpf
    
    # Get HEF dimensions
    hef_model = hpf.HEF(hef_path)
    hef_h, hef_w = get_hef_input_dimensions(hef_model)
    
    # Create model
    backbone = DinoV2Backbone("dinov2_vits14")
    gazelle_model = GazeLLE(backbone=backbone, in_size=(hef_h, hef_w), out_size=(hef_h, hef_w))
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights (backbone excluded - using Hailo)
    gazelle_model.load_gazelle_state_dict(state_dict, include_backbone=False)
    gazelle_model.to(device)
    gazelle_model.eval()
    
    return gazelle_model


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    parser = get_gazelle_parser()
    args = parser.parse_args()
    
    # Load model
    gazelle_model = load_gazelle_model(args.pth, args.hef, args.device)
    
    # Create callback handler
    user_data = GazeLLECallbackClass(
        gazelle_model=gazelle_model,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        max_frames=args.max_frames,
        hef_path=args.hef,
        skip_frames=args.skip_frames,
        save_inference_results=args.save_inference,
        inference_output_dir=args.inference_output_dir,
        save_mode=args.save_mode,
        scrfd_hef_path=args.scrfd_hef,
        detr_hef_path=args.detr_hef,
        detr_confidence=args.detr_confidence,
        detr_nms=args.detr_nms
    )
    
    # Create and run application
    app = GStreamerGazeLLEApp(args, user_data)
    
    print("Starting GStreamer app...")
    app.run()


if __name__ == "__main__":
    main()