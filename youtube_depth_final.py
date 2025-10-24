"""
Real-time YouTube Object Detection with Faster R-CNN
+ Depth Model Fusion for Distance Estimation

(실시간 유튜브 객체 탐지 + 뎁스 모델 퓨전을 통한 거리 추정)
모든 유틸리티 함수가 포함된 단일 파일 버전
"""

# --- 기본 라이브러리 ---
import torch
import cv2  # OpenCV (이미지/비디오 처리)
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Python Imaging Library
import argparse  # 커맨드 라인 인자 처리
import time  # 시간 측정 (FPS 계산, 실시간 동기화)
import hashlib  # 비디오 URL 해싱 (캐시 파일명 생성)
import sys
import os

# --- Object Detection (OD) 모델 관련 (PyTorch/TorchVision) ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms  # 이미지 전처리 (텐서 변환, 정규화)

# --- 비디오 다운로드 ---
import yt_dlp  # 유튜브 비디오 다운로더

##### 1. DEPTH 모델을 위한 라이브러리 임포트 #####
# Hugging Face Transformers 라이브러리 (Depth Anything V2 모델 로드용)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# --- [개선] BBox 스타일 상수 정의 ---
BOX_THICKNESS = 2                # BBox 윤곽선 두께
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5                 # 라벨 폰트 크기
FONT_THICKNESS = 1               # 라벨 폰트 두께
LABEL_BG_COLOR = (0, 0, 0)       # 라벨 배경은 항상 검은색
TEXT_OUTLINE_COLOR = (255, 255, 255)   # [수정] 가독성을 위한 검은색 테두리
TEXT_OUTLINE_THICKNESS = 4       # [수정] 테두리 두께 (본문 폰트보다 두껍게)
LABEL_BG_OPACITY = 0.7           # 라벨 배경 반투명도 (60%)
V_PADDING = 5                    # 라벨 상하 여백
H_PADDING = 5                    # 라벨 좌우 여백
# --- [개선] ---

# ============================================================================
# 유틸리티 함수 (utils.dataset2.py 대체)
# ============================================================================

def letterbox_resize(image, target_size=640):
    """
    PIL 이미지를 가로세로 비율을 유지하며 레터박스 리사이징합니다.
    (PyTorch 모델 입력용)

    Args:
        image (PIL.Image): 원본 PIL 이미지.
        target_size (int): 목표 크기 (정사각형).

    Returns:
        new_image (PIL.Image): 리사이즈되고 패딩이 추가된 이미지.
        scale (float): 적용된 리사이즈 비율.
        pad_left (int): 왼쪽에 추가된 패딩 크기.
        pad_top (int): 위쪽에 추가된 패딩 크기.
    """
    iw, ih = image.size  # 원본 크기
    w, h = target_size, target_size  # 목표 크기

    # 리사이즈 비율 (작은 쪽 기준)
    scale = min(w / iw, h / ih)
    
    # 새 크기 계산
    nw = int(iw * scale)
    nh = int(ih * scale)

    # PIL 이미지 리사이즈 (LANCZOS/Antialias 사용)
    image_resized = image.resize((nw, nh), Image.LANCZOS)

    # 새 캔버스(배경) 생성 (128, 128, 128 그레이)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))

    # 패딩 계산 (좌상단 기준)
    pad_left = (w - nw) // 2
    pad_top = (h - nh) // 2

    # 새 캔버스에 리사이즈된 이미지 붙여넣기
    new_image.paste(image_resized, (pad_left, pad_top))

    return new_image, scale, pad_left, pad_top

def inverse_transform_bbox(box, scale, pad_left, pad_top):
    """
    레터박스 리사이즈된 BBox 좌표를 원본 이미지 좌표로 역변환합니다.

    Args:
        box (list or np.array): [x1, y1, x2, y2] (모델 출력 좌표)
        scale (float): letterbox_resize에서 반환된 비율
        pad_left (int): letterbox_resize에서 반환된 왼쪽 패딩
        pad_top (int): letterbox_resize에서 반환된 위쪽 패딩

    Returns:
        list: [x1, y1, x2, y2] (원본 이미지 기준 좌표)
    """
    # 1. 패딩 제거
    x1 = box[0] - pad_left
    y1 = box[1] - pad_top
    x2 = box[2] - pad_left
    y2 = box[3] - pad_top
    
    # 2. 스케일 역적용 (나누기)
    orig_x1 = x1 / scale
    orig_y1 = y1 / scale
    orig_x2 = x2 / scale
    orig_y2 = y2 / scale
    
    return [orig_x1, orig_y1, orig_x2, orig_y2]

# ============================================================================
# CONFIGURATION - 학습된 모델에 맞게 수정하세요!
# ============================================================================

CONFIG = {
    # --- 모델 설정 ---
    'model_path': 'ori2_woong_structured_pruned_50.pth',  # 학습된 Faster R-CNN 모델 파일 경로
    'class_names': ['CAR', 'PERSON', 'TRAFFIC'],  # 클래스 이름 (학습 시 사용한 순서대로, '__background__' 제외)

    ##### 2. DEPTH 모델 설정 추가 #####
    'depth_model_enabled': True,  # 뎁스 모델 퓨전 활성화 여부 (True/False)
    # 사용할 뎁스 모델 ID (Hugging Face 모델 허브 기준)
    # (추천) 야외 주행 영상용 Metric(절대 거리) 모델
    'depth_model_id': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
    # (대안) 실내용 Metric 모델
    # 'depth_model_id': 'depth-anything/depth-anything-v2-metric-hypersim',
    # (대안) 일반적인 Small 모델 (상대 거리만 추정)
    # 'depth_model_id': 'depth-anything/depth-anything-v2-small-hf',
    ####################################

    # --- 탐지 파라미터 ---
    'conf_threshold': 0.8,   # 초기 Confidence threshold (이 값 이상만 탐지)
    'iou_threshold': 0.5,    # 초기 IOU threshold (NMS - Non-Maximum Suppression 용)
    'resize_size': 640,      # 모델 입력 이미지 크기 (Faster R-CNN)

    # --- 비디오 설정 ---
    'cache_dir': 'video_cache',  # 다운로드한 비디오를 저장할 캐시 폴더
}

# ============================================================================
# (이하 EXAMPLE CONFIGURATIONS 생략)
# ============================================================================

# --- 디바이스 설정 ---
# 사용 가능한 GPU(CUDA 또는 Intel XPU) 확인, 없으면 CPU 사용 (CPU는 매우 느림)
device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')
print(f"Using device: {device}")

def get_fasterrcnn_model(num_classes, backbone_freeze=False):
    """
    Get Faster R-CNN model
    [v5 수정] backbone_freeze 파라미터 추가
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # [v5] 백본 동결 (하이퍼 파라미터)
    if backbone_freeze:
        print("  Applying backbone freeze.")
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval() # BN 통계도 동결

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def generate_colors(num_classes):
    """
    각 클래스별로 고유한 색상을 생성합니다. (BGR 포맷)
    Returns: dict (클래스 ID -> BGR 튜플)
    """
    np.random.seed(42)  # 일관된 색상 생성을 위해 시드 고정
    colors = {}

    # 자주 사용되는 클래스를 위한 미리 정의된 색상 (BGR 순서)
    predefined_colors = [
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (255, 0, 0),      # Blue
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
    ]

    # 1-indexed (0번은 __background__)
    for i in range(1, num_classes + 1):
        if i - 1 < len(predefined_colors):
            colors[i] = predefined_colors[i - 1]  # 미리 정의된 색상 사용
        else:
            # 랜덤 색상 생성
            colors[i] = tuple(np.random.randint(50, 255, 3).tolist())

    return colors


def validate_model_classes(model, expected_num_classes):
    """
    로드된 모델의 클래스 수와 CONFIG의 클래스 수가 일치하는지 확인합니다.
    (배경 클래스 포함)
    """
    try:
        # 모델의 출력 레이어에서 클래스 수 확인
        model_num_classes = model.roi_heads.box_predictor.cls_score.out_features

        if model_num_classes != expected_num_classes:
            print(f"\n⚠️  WARNING: 모델 클래스 불일치!")
            print(f"  CONFIG 클래스 수: {expected_num_classes} (배경 포함)")
            print(f"  로드된 모델의 클래스 수: {model_num_classes}")
            print(f"  CONFIG['class_names'] 설정을 확인하세요.")
            return False

        return True
    except Exception as e:
        print(f"\n⚠️  Warning: 모델 클래스 검증 실패: {e}")
        return True  # 검증 실패 시에도 일단 진행


def get_model_dtype(model):
    """모델의 데이터 타입(dtype)을 감지합니다 (FP32 or FP16)."""
    # 모델의 첫 번째 파라미터 타입을 확인
    return next(model.parameters()).dtype


def download_youtube_video(youtube_url, cache_dir, force_download=False):
    """
    YouTube 비디오를 다운로드합니다 (캐시 기능 지원).
    Returns: 비디오 파일 경로
    """
    print(f"\nExtracting video ID...")
    try:
        # yt-dlp를 사용해 비디오 정보만 추출 (다운로드 X)
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id', None)
    except:
        # ID 추출 실패 시 URL을 해시하여 고유 ID 생성
        video_id = hashlib.md5(youtube_url.encode()).hexdigest()[:11]

    print(f"Video ID: {video_id}")

    # 캐시 폴더 확인 및 생성
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cached_path = os.path.join(cache_dir, f"{video_id}.mp4")

    # 캐시 파일이 존재하고, 강제 다운로드가 아니면 캐시 사용
    if os.path.exists(cached_path) and not force_download:
        file_size = os.path.getsize(cached_path) / (1024 * 1024)
        print(f"\n✓ 캐시된 비디오 발견!")
        print(f"  Path: {cached_path}")
        print(f"  Size: {file_size:.1f} MB")
        print(f"  다운로드를 생략합니다...")
        return cached_path

    # --- 비디오 다운로드 ---
    print(f"\nDownloading YouTube video...")
    print(f"비디오 길이와 인터넷 속도에 따라 1-2분 정도 소요될 수 있습니다.")

    ydl_opts = {
        # 다운로드 포맷 설정: 720p 이하의 mp4 (avc1 코덱 우선)
        'format': (
            'best[height<=720][ext=mp4][vcodec^=avc1]/best[height<=720][ext=mp4]/'
            'bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/'
            'best[height<=720]'
        ),
        'outtmpl': cached_path,  # 저장 경로 (캐시 경로)
        'quiet': False,  # 다운로드 진행률 표시
        'no_warnings': True,
        'noplaylist': True,  # 플레이리스트 다운로드 방지
        'prefer_free_formats': False,
        # 다운로드 진행률 콜백 함수
        'progress_hooks': [lambda d: print(f"\rDownload: {d.get('_percent_str', 'N/A')} ", end='')
                           if d['status'] == 'downloading' else None],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\nFetching video info...")
            info = ydl.extract_info(youtube_url, download=True)  # 다운로드 실행
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

        if not os.path.exists(cached_path):
            raise FileNotFoundError(f"다운로드된 파일을 찾을 수 없습니다: {cached_path}")

        file_size = os.path.getsize(cached_path) / (1024 * 1024)

        print(f"\n✓ 비디오 다운로드 성공")
        print(f"  Title: {video_title}")
        print(f"  Duration: {duration // 60}m {duration % 60}s")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Cached to: {cached_path}")
        return cached_path

    except Exception as e:
        print(f"\nError: 비디오 다운로드 실패")
        print(f"Error details: {e}")
        raise


def preprocess_frame(frame, resize_size=640, model_dtype=torch.float32):
    """
    모델 추론을 위해 프레임을 전처리합니다 (dtype 자동 변환 포함).
    (이 함수는 스크립트 상단에 정의된 `letterbox_resize`를 사용합니다.)
    """
    # 1. BGR (OpenCV) -> RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 2. Letterbox 리사이징 (비율 유지)
    img_letterbox, scale, pad_left, pad_top = letterbox_resize(
        pil_image, target_size=resize_size
    )

    # 3. PyTorch 텐서 변환 및 정규화
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 기본값
                             std=[0.229, 0.224, 0.225])
    ])

    # 4. 배치 차원 추가 (C, H, W) -> (1, C, H, W) 및 GPU로 이동
    img_tensor = normalize(img_letterbox).unsqueeze(0).to(device)

    # 5. 모델이 FP16(half)이면 입력 텐서도 FP16으로 변환
    if model_dtype == torch.float16:
        img_tensor = img_tensor.half()

    return img_tensor, scale, pad_left, pad_top


def postprocess_predictions(predictions, scale, pad_left, pad_top,
                            conf_threshold=0.5, frame_shape=None):
    """
    모델의 추론 결과를 후처리합니다.
    (리사이즈된 좌표 -> 원본 프레임 좌표로 역변환)
    (이 함수는 스크립트 상단에 정의된 `inverse_transform_bbox`를 사용합니다.)
    """
    # 모델 출력 (배치 중 첫 번째 결과 사용)
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    # 1. Confidence 점수 기준으로 필터링
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # 2. Bounding Box 좌표를 원본 프레임 기준으로 역변환
    transformed_boxes = []
    for box in boxes:
        # letterbox_resize의 역함수(inverse_transform_bbox) 사용
        orig_box = inverse_transform_bbox(box, scale, pad_left, pad_top)
        transformed_boxes.append(orig_box)

    # 3. (Optional) 박스가 프레임 경계를 넘어가지 않도록 클리핑
    if frame_shape is not None and len(transformed_boxes) > 0:
        h, w = frame_shape[:2]
        transformed_boxes = np.array(transformed_boxes)
        # x좌표(0, 2)는 너비(w) 기준으로, y좌표(1, 3)는 높이(h) 기준으로 클리핑
        transformed_boxes[:, [0, 2]] = np.clip(transformed_boxes[:, [0, 2]], 0, w)
        transformed_boxes[:, [1, 3]] = np.clip(transformed_boxes[:, [1, 3]], 0, h)

    return transformed_boxes, labels, scores

# ##### 3. draw_predictions 함수 (UI 개선 버전 - 배경 제거, 텍스트 색상) #####
# def draw_predictions(frame, boxes, labels, scores, class_names, class_colors, conf_threshold, distances=None):
#     """
#     [개선] 프레임에 Bounding Box와 라벨을 그립니다.
#     - [수정] 라벨 배경을 제거합니다.
#     - [수정] 라벨 텍스트 색상을 BBox 색상과 동일하게 설정합니다.
#     - [수정] 가독성을 위해 텍스트에 검은색 테두리를 추가합니다.
#     - 화면 상단을 벗어날 경우 자동으로 박스 '안'에 배치합니다.
#     """
    
#     # [수정] 오버레이(overlay) 및 반투명(addWeighted) 관련 코드 제거

#     if distances is None:
#         distances = [None] * len(boxes)

#     for box, label, score, dist in zip(boxes, labels, scores, distances):
#         x1, y1, x2, y2 = map(int, box)
#         color = class_colors.get(label, (0, 255, 0)) # BBox 색상 (및 텍스트 색상)
#         class_name = class_names[label]

#         # --- 1. Bounding Box 윤곽선 그리기 ---
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS, cv2.LINE_AA)

#         # --- 2. 라벨 텍스트 준비 ---
#         label_text = f'{class_name}: {score:.2f}'
#         if dist is not None:
#             label_text += f' | {dist:.1f}m'

#         # --- 3. 라벨 크기 및 위치 계산 ---
#         (text_w, text_h), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
        
#         # 라벨의 전체 높이 (텍스트 + 여백)
#         label_h = text_h + baseline + (2 * V_PADDING)
        
#         # 기본 위치: 박스 위 (outside)
#         text_x = x1 + H_PADDING
#         text_y = y1 - V_PADDING - baseline # 텍스트 기준선(baseline) y좌표
        
#         # [개선] 화면 상단을 벗어날 경우 (y1 - label_h < 0): 박스 안 (inside)으로 전환
#         if (y1 - label_h) < 0:
#             text_y = y1 + text_h + V_PADDING # 텍스트 기준선 y좌표

#         # --- 4. [수정] 라벨 배경 그리기 제거 ---
#         # (cv2.rectangle(overlay, ...) 부분 삭제됨)

#         # --- 5. [수정] 라벨 텍스트 그리기 (테두리 + 본문) ---

#         # BBox 내부에 반투명 채우기 (시인성 향상)
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
#         cv2.addWeighted(overlay, LABEL_BG_OPACITY, frame, 1-LABEL_BG_OPACITY, 0, frame)
        
#         # 텍스트 테두리 (검은색, 더 두껍게)
#         cv2.putText(frame, label_text, (text_x, text_y),
#                     FONT, FONT_SCALE, TEXT_OUTLINE_COLOR, TEXT_OUTLINE_THICKNESS, cv2.LINE_AA)
        
#         # 텍스트 본문 (BBox 색상)
#         cv2.putText(frame, label_text, (text_x, text_y),
#                     FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

#     # --- 6. [수정] 반투명 오버레이 적용 제거 ---
#     # (cv2.addWeighted(...) 부분 삭제됨)

#     return frame

##### 3. draw_predictions 함수 (그리기 순서 버그 수정) #####
def draw_predictions(frame, boxes, labels, scores, class_names, class_colors, conf_threshold, distances=None):
    """
    [개선] 프레임에 Bounding Box와 라벨을 그립니다.
    - [버그 수정] 그리기 순서를 변경하여, 반투명 배경이 텍스트를 가리는 문제를 해결.
    - 1. (루프 1) 배경만 overlay에 그리기
    - 2. (합성) overlay를 frame에 합성
    - 3. (루프 2) BBox와 텍스트를 합성된 frame에 그리기
    """
    
    overlay = frame.copy()
    h, w = frame.shape[:2]

    if distances is None:
        distances_list = [None] * len(boxes)
    else:
        distances_list = distances
        
    # [수정] 나중(루프 2)에 텍스트와 BBox를 그리기 위해 데이터 저장
    draw_data = []

    # --- [수정] 1. 첫 번째 루프: 배경만 'overlay'에 그리기 ---
    for box, label, score, dist in zip(boxes, labels, scores, distances_list):
        x1, y1, x2, y2 = map(int, box)
        
        label_text = f'{class_names[label]}: {score:.2f}'
        if dist is not None:
            label_text += f' | {dist:.1f}m'

        (text_w, text_h), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
        
        label_h = text_h + baseline + (2 * V_PADDING)
        label_w = text_w + (2 * H_PADDING)
        
        label_bg_x1 = x1
        label_bg_x2 = x1 + label_w
        label_bg_y1 = y1 - label_h
        label_bg_y2 = y1
        text_x = x1 + H_PADDING
        text_y = y1 - V_PADDING - baseline
        is_inside = False

        if label_bg_y1 < 0:
            label_bg_y1 = y1
            label_bg_y2 = y1 + label_h
            text_y = y1 + text_h + V_PADDING
            is_inside = True

        label_bg_x2 = min(label_bg_x2, w) 
        if is_inside:
            label_bg_x2 = min(label_bg_x2, x2)
            label_bg_y2 = min(label_bg_y2, y2)

        # 1-1. 배경을 'overlay'에 그리기
        if label_bg_x2 > label_bg_x1 and label_bg_y2 > label_bg_y1:
             cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2),
                           LABEL_BG_COLOR, -1)
        
        # 1-2. 두 번째 루프에서 사용할 데이터 저장
        draw_data.append({
            'box': (x1, y1, x2, y2),
            'color': class_colors.get(label, (0, 255, 0)),
            'label_text': label_text,
            'text_pos': (text_x, text_y),
            'text_clip': (label_bg_x2, label_bg_y2) # 텍스트가 잘리지 않게
        })
    
    # --- [수정] 2. 반투명 배경 합성 ---
    # 'overlay'(배경만 있음)를 'frame'(원본 비디오)에 합성
    cv2.addWeighted(overlay, LABEL_BG_OPACITY, frame, 1 - LABEL_BG_OPACITY, 0, frame)

    # --- [수정] 3. 두 번째 루프: BBox와 텍스트를 'frame'에 그리기 ---
    # 이제 'frame'은 반투명 배경이 적용된 상태임
    for data in draw_data:
        x1, y1, x2, y2 = data['box']
        color = data['color']
        text_x, text_y = data['text_pos']
        label_bg_x2, label_bg_y2 = data['text_clip']
        
        # 3-1. BBox 윤곽선 그리기 (100% 불투명)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS, cv2.LINE_AA)
        
        # 3-2. 텍스트 그리기 (100% 불투명)
        if text_y < label_bg_y2 and text_x < label_bg_x2:
            cv2.putText(frame, data['label_text'], (text_x, text_y),
                        FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    return frame

##### 4. draw_info_panel 함수 (depth_model_active 및 timings 파라미터 추가) #####
def draw_info_panel(frame, class_counts, class_names, inference_fps, conf_threshold,
                    iou_threshold, video_fps, processing_rate, dropped_frames,
                    realtime_capable, model_dtype, depth_model_active=False,
                    timings=None):  # <-- [수정] timings 딕셔너리 인자 추가
    """프레임에 정보 패널을 그립니다 (동적 클래스 표시 및 성능 타이밍 포함)."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # --- [수정] 패널 높이 계산 (타이밍 정보 표시를 위해 높이 증가) ---
    num_classes = len(class_names) - 1  # Exclude background
    base_height = 420  # 뎁스 모델 + 타이밍 정보 추가로 기본 높이 증가
    timing_height = 10 * 22  # 타이밍 정보 라인 수 (약 10줄)
    class_height = num_classes * 22
    panel_height = min(base_height + class_height + timing_height, h - 100) # 최대 높이 제한
    # --- [수정] ---

    # Main info panel
    cv2.rectangle(overlay, (10, 10), (380, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Real-time status indicator
    status_color = (0, 255, 0) if realtime_capable else (0, 0, 255)

    # Model dtype display
    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    dtype_color = (255, 165, 0) if model_dtype == torch.float16 else (255, 255, 255)

    # Build info texts with dynamic class counts
    info_texts = []

    # Display each class count
    total_count = 0
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id]
        info_texts.append(f"{class_name}: {count}")
        total_count += count

    info_texts.append(f"Total: {total_count}")
    info_texts.append("")
    info_texts.append(f"OD Model: {dtype_str} (Faster R-CNN)")

    # 뎁스 모델 정보 추가
    if depth_model_active:
        info_texts.append(f"Depth Model: Active ({dtype_str})")
    else:
        info_texts.append(f"Depth Model: Inactive")

    # --- [수정] Performance (ms) 섹션 추가 ---
    info_texts.append("")  # Separator
    info_texts.append("--- Performance (ms) ---")

    if timings:
        # 패널에 표시할 순서와 이름 정의
        timing_order = [
            ('Total Process', 'Total'),
            ('OD Inference', '  OD Infer'),
            ('OD Preprocess', '  OD Pre'),
            ('OD Postprocess', '  OD Post'),
            ('Depth Inference', '  Depth Infer'),
            ('Depth Preprocess', '  Depth Pre'),
            ('Depth Postprocess', '  Depth Post'),
            ('Depth Fusion', '  Depth Fusion'),
            ('Draw BBoxes', '  Draw BBoxes'),
            ('Draw Info Panel', '  Draw Panel')
        ]
        
        for key, name in timing_order:
            # timings 딕셔너리에서 값 가져오기 (없으면 0.0)
            time_ms = timings.get(key, 0.0)
            info_texts.append(f"{name}: {time_ms:.1f} ms")
    else:
        info_texts.append("Timings: N/A")

    info_texts.append("")  # Separator
    info_texts.append("--- Video & Thresholds ---") # 섹션 제목
    # --- [수정] ---
    
    info_texts.append(f"Video FPS: {video_fps:.1f}")
    info_texts.append(f"Inference FPS: {inference_fps:.1f}")
    info_texts.append(f"Processing Rate: {processing_rate:.1f}%")
    info_texts.append(f"Dropped Frames: {dropped_frames}")
    info_texts.append("")
    info_texts.append(f"Confidence: {conf_threshold:.2f}")
    info_texts.append(f"IOU: {iou_threshold:.2f}")

    y_offset = 35
    for i, text in enumerate(info_texts):
        if text:
            # --- [수정] 타이밍 정보 색상 코딩 추가 ---
            color = (255, 255, 255)  # Default white
            if "Model:" in text:
                color = dtype_color
            elif "---" in text:
                color = (255, 255, 100)  # Section header (Yellowish)
            elif "Total" in text and "ms" in text:
                color = (0, 255, 255)  # Total Time (Yellow)
            elif "Infer" in text:
                color = (0, 165, 255)  # Inference Times (Orange)
            elif "Video FPS:" in text:
                color = (100, 255, 255)
            elif "Inference FPS:" in text:
                color = (255, 255, 100)
            elif "Processing Rate:" in text:
                color = (0, 255, 0) if processing_rate >= 99 else (0, 165, 255)
            # --- [수정] ---

            cv2.putText(frame, text, (20, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Controls info at bottom
    controls = [
        "Controls: [+/-] Conf | [[/]] IOU | [SPACE] Pause | [ESC] Exit"
    ]

    for i, text in enumerate(controls):
        cv2.putText(frame, text, (10, h - 20 - i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
################################################################


class RealtimePerformanceTracker:
    """실시간 처리 성능을 추적하는 클래스"""
    def __init__(self, video_fps):
        self.video_fps = video_fps
        # 1프레임당 허용 시간 (예: 30fps -> 1/30 = 0.033초)
        self.frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033
        self.total_frames = 0  # 비디오의 총 경과 프레임
        self.processed_frames = 0  # 실제로 처리한 프레임
        self.dropped_frames = 0  # 처리가 늦어 스킵한 프레임
        self.inference_times = []  # 최근 추론 시간 기록 (이동 평균 계산용)
        self.max_history = 30  # FPS 계산 시 최근 30개 프레임의 평균 사용

    def update(self, processing_time, frames_to_skip=0):
        """성능 지표 업데이트"""
        self.total_frames += 1
        self.processed_frames += 1
        self.dropped_frames += frames_to_skip  # 스킵된 프레임 수 누적

        self.inference_times.append(processing_time)
        # 오래된 기록 삭제
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)

    def get_inference_fps(self):
        """평균 '추론' FPS 계산"""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_processing_rate(self):
        """'실시간 처리율' 계산 (전체 비디오 프레임 중 몇 %를 처리했는가)"""
        total_elapsed = self.processed_frames + self.dropped_frames
        if total_elapsed == 0:
            return 100.0
        return (self.processed_frames / total_elapsed) * 100

    def is_realtime_capable(self):
        """현재 실시간 처리가 가능한지 (추론 시간이 프레임 간격보다 짧은지) 확인"""
        if not self.inference_times:
            return True
        # 최근 10개 프레임의 평균 처리 시간
        avg_time = np.mean(self.inference_times[-10:])
        return avg_time < self.frame_interval  # 처리 시간 < 허용 시간


def main():
    # --- 커맨드 라인 인자 파서 ---
    parser = argparse.ArgumentParser(description='Real-time YouTube Object Detection + Depth Fusion')
    parser.add_argument('--url', type=str, required=False, help='YouTube URL (Optional)')
    parser.add_argument('--input', type=str, default='video_cache/fkzyY4TGybw.mp4',
                       help='Path to local video file (default: video_cache/sample_video.mp4)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download even if cached (if --url is used)')
    args = parser.parse_args()
    # --- (종료) ---

    print("=" * 70)
    print("Universal Real-time YouTube Object Detection + Depth Fusion")  # Title
    print("=" * 70)
    print(f"Device: {device}")

    # --- 설정값 출력 ---
    print("\nConfiguration:")
    print(f"  OD Model: {CONFIG['model_path']}")
    print(f"  Classes: {CONFIG['class_names']}")
    ##### 5. 뎁스 모델 설정 출력 #####
    if CONFIG['depth_model_enabled']:
        print(f"  Depth Model: {CONFIG['depth_model_id']} (Enabled)")
    else:
        print("  Depth Model: (Disabled)")
    ####################################
    print(f"  Confidence threshold: {CONFIG['conf_threshold']}")
    print(f"  IOU threshold: {CONFIG['iou_threshold']}")
    print(f"  Resize size: {CONFIG['resize_size']}")
    print(f"  Cache directory: {CONFIG['cache_dir']}")

    # --- 클래스 준비 ---
    class_names = ['__background__'] + CONFIG['class_names']
    num_classes = len(class_names)
    print(f"  Total classes (with background): {num_classes}")

    # 클래스별 BBox 색상 생성
    class_colors = generate_colors(len(CONFIG['class_names']))
    print(f"  Generated {len(class_colors)} class colors")

    # --- 1. Object Detection (OD) 모델 로드 ---
    print(f"\nLoading OD model (Faster R-CNN)...")
    if not os.path.exists(CONFIG['model_path']):
        print(f"✗ ERROR: Model not found at: {CONFIG['model_path']}")
        print("  Please check the 'model_path' in CONFIG.")
        return
    
    try:
        model = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
        model.to(device)  # 모델을 GPU로 이동
        model.eval()      # 추론 모드로 설정
    except Exception as e:
        print(f"✗ ERROR: Failed to load OD model: {e}")
        return

    # 로드된 모델의 dtype (FP32/FP16) 감지
    model_dtype = get_model_dtype(model)
    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    print(f"✓ OD Model loaded")
    print(f"  OD Model dtype: {dtype_str}")

    # 모델 클래스 수 검증
    if not validate_model_classes(model, num_classes):
        print("\n⚠️  경고: 클래스 수가 불일치하지만 강행합니다.")
        print("  CONFIG['class_names']가 학습된 모델과 일치하는지 확인하세요.")

    ##### 6. 뎁스 모델 로드 #####
    depth_model = None
    depth_processor = None
    if CONFIG['depth_model_enabled']:
        try:
            print(f"\nLoading Depth model ({CONFIG['depth_model_id']})...")
            # Transformers 라이브러리를 사용해 사전 학습된 뎁스 모델과 프로세서 로드
            depth_processor = AutoImageProcessor.from_pretrained(CONFIG['depth_model_id'])
            depth_model = AutoModelForDepthEstimation.from_pretrained(CONFIG['depth_model_id'])
            depth_model.to(device)  # GPU로 이동
            depth_model.half() # 메모리 절약을 위해 FP16으로 변환
            depth_model.eval()      # 추론 모드
            if model_dtype == torch.float16:
                depth_model.half()  # OD 모델이 FP16이면 뎁스 모델도 FP16으로 맞춰 성능 향상
            print("✓ Depth model loaded")
        except Exception as e:
            # 모델 로드 실패 시 뎁스 기능 비활성화
            print(f"\n⚠️ WARNING: 뎁스 모델 로드 실패. 뎁스 퓨전을 비활성화합니다.")
            print(f"  Error: {e}")
            CONFIG['depth_model_enabled'] = False
    #################################

    # --- 비디오 경로 결정 ---
    video_path = '/home/intel/teni/visionAI/sesac/video_cache/fkzyY4TGybw.mp4'
    if args.url:
        print(f"\n--url provided. Downloading video...")
        try:
            video_path = download_youtube_video(args.url, CONFIG['cache_dir'],
                                               force_download=args.force_download)
        except Exception as e:
            print(f"✗ ERROR: Failed to download YouTube video. Exiting.")
            return
    else:
        print(f"\nUsing local video: {video_path}")

    # --- 비디오 파일 열기 ---
    print("\nOpening video file...")
    if not os.path.exists(video_path):
        print(f"✗ ERROR: Video file not found: {video_path}")
        print("  Please check the path or use --url to download a video.")
        return
        
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"✗ Error: 비디오 파일을 열 수 없습니다: {video_path}")
        return

    # --- 비디오 속성 확인 ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("✓ Video file opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  Video FPS: {video_fps:.1f} ← 여기에 맞춰 재생 속도를 동기화합니다.")
    print(f"  Total frames: {total_frames}")
    if video_fps > 0 and total_frames > 0:
        duration = total_frames / video_fps
        print(f"  Duration: {int(duration // 60)}m {int(duration % 60)}s")
        print(f"  Frame interval: {1000/video_fps:.1f}ms per frame") # 1프레임당 시간

    # 첫 프레임 읽기 테스트
    print("\nTesting frame reading...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("✗ ERROR: 비디오 프레임을 읽을 수 없습니다!")
        cap.release()
        return
    else:
        print(f"✓ 첫 프레임 읽기 성공: {test_frame.shape}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오를 다시 처음으로 되돌림

    print("\nStarting real-time detection...")
    print("Press [+/-] to adjust confidence threshold")
    print("Press [[/]] to adjust IOU threshold")
    print("Press [SPACE] to pause/resume")
    print("Press [ESC] to exit")
    print("-" * 70)

    # --- 초기 변수 설정 ---
    conf_threshold = CONFIG['conf_threshold']  # 현재 Confidence
    iou_threshold = CONFIG['iou_threshold']    # 현재 IOU
    performance_tracker = RealtimePerformanceTracker(video_fps)  # 성능 추적기
    frame_count = 0  # 총 경과 프레임 카운터
    paused = False   # 일시정지 상태
    last_frame = None  # 마지막으로 처리된 프레임 (일시정지 시 표시용)

    # 실시간 동기화를 위한 변수
    frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033  # 1프레임당 시간 (초)
    next_frame_time = time.time()  # 다음 프레임을 처리해야 할 시간

    # --- [신규] 프레임별 타이밍 저장을 위한 딕셔너리 ---
    timings = {}

    try:
        # --- 메인 루프 ---
        while True:
            current_time = time.time()  # 현재 시간

            if not paused:  # 일시정지가 아닐 때
                # --- 실시간 동기화 ---
                if current_time >= next_frame_time:
                    ret, frame = cap.read()  # 비디오에서 프레임 읽기
                    if not ret:
                        # 비디오 끝에 도달
                        if frame_count == 0:
                            print("\n✗ Error: 첫 프레임을 읽지 못했습니다!")
                        else:
                            print(f"\n✓ 비디오 재생 완료")
                            print(f"  Total frames: {frame_count}")
                            print(f"  Processed: {performance_tracker.processed_frames}")
                            print(f"  Dropped: {performance_tracker.dropped_frames}")
                            print(f"  Processing rate: {performance_tracker.get_processing_rate():.1f}%")
                        break  # 루프 종료

                    if frame is None:
                        # 프레임이 손상되었거나 읽기 실패
                        frame_count += 1
                        next_frame_time += frame_interval
                        continue

                    frame_count += 1

                    # 6프레임마다 1번씩 추론, 나머지는 패스
                    continue_processing = (frame_count % 6 == 0)
                    frames_to_skip = 0
                    if not continue_processing:
                        frames_to_skip = 1  # 1프레임 스킵
                        performance_tracker.update(0.0, frames_to_skip=frames_to_skip)
                        next_frame_time += frame_interval
                        continue  # 다음 프레임으로

                    # --- [수정] 전체 처리 시간 측정 시작 ---
                    process_start = time.time()
                    
                    # (실시간 변경) 모델 NMS 임계값 업데이트
                    model.roi_heads.nms_thresh = iou_threshold

                    # --- 1. Object Detection (Faster R-CNN) ---
                    
                    # [타이밍 1: OD Preprocess]
                    t_start_pre_od = time.time()
                    img_tensor, scale, pad_left, pad_top = preprocess_frame(
                        frame, resize_size=CONFIG['resize_size'], model_dtype=model_dtype
                    )
                    timings['OD Preprocess'] = (time.time() - t_start_pre_od) * 1000 # ms

                    # [타이밍 2: OD Inference] (GPU 동기화 포함)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device) # 이전 작업 완료 대기
                    t_start_inf_od = time.time()
                    
                    with torch.no_grad():
                        predictions = model(img_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device) # 현재 작업 완료 대기
                    timings['OD Inference'] = (time.time() - t_start_inf_od) * 1000 # ms
                    
                    
                    # [타이밍 3: OD Postprocess]
                    t_start_post_od = time.time()
                    boxes, labels, scores = postprocess_predictions(
                        predictions, scale, pad_left, pad_top,
                        conf_threshold=conf_threshold,
                        frame_shape=frame.shape
                    )
                    timings['OD Postprocess'] = (time.time() - t_start_post_od) * 1000 # ms
                    
                    
                    ##### 7. 뎁스 추정 및 퓨전 로직 #####
                    distances = []
                    if CONFIG['depth_model_enabled'] and depth_model is not None and len(boxes) > 0:
                        
                        # [타이밍 4: Depth Preprocess]
                        t_start_pre_depth = time.time()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        inputs = depth_processor(images=pil_image, return_tensors="pt").to(device)
                        if model_dtype == torch.float16:
                            inputs['pixel_values'] = inputs['pixel_values'].half() # FP16
                        timings['Depth Preprocess'] = (time.time() - t_start_pre_depth) * 1000 # ms
                        
                        
                        # [타이밍 5: Depth Inference] (GPU 동기화 포함)
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        t_start_inf_depth = time.time()
                        
                        with torch.no_grad():
                            outputs = depth_model(**inputs)
                            predicted_depth = outputs.predicted_depth
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        timings['Depth Inference'] = (time.time() - t_start_inf_depth) * 1000 # ms
                        
                        
                        # [타이밍 6: Depth Postprocess (Interpolate)]
                        t_start_post_depth = time.time()
                        original_h, original_w = frame.shape[:2]
                        depth_map = torch.nn.functional.interpolate(
                            predicted_depth.unsqueeze(1),             # (1, 1, H_out, W_out)
                            size=(original_h, original_w),            # 목표 크기 (원본 프레임 크기)
                            mode='bicubic',                           # 'bicubic'이 'bilinear'보다 부드러운 결과
                            align_corners=False
                        )
                        depth_map = depth_map.squeeze().cpu().numpy() # (H_orig, W_orig)
                        timings['Depth Postprocess'] = (time.time() - t_start_post_depth) * 1000 # ms
                        
                        
                        # [타이밍 7: Depth Fusion (Median Extraction)]
                        t_start_fuse_depth = time.time()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            # BBox 영역에 해당하는 뎁스 값들만 추출
                            object_depth_region = depth_map[y1:y2, x1:x2]
                            
                            if object_depth_region.size > 0:
                                # 아웃라이어(이상치)에 강건한 중앙값(median)을 거리로 사용
                                median_depth = np.median(object_depth_region)
                                distances.append(median_depth)
                            else:
                                distances.append(0.0)  # BBox 영역이 0인 경우
                        timings['Depth Fusion'] = (time.time() - t_start_fuse_depth) * 1000 # ms
                        
                    else:
                        distances = [None] * len(boxes)
                        # 뎁스가 비활성화된 경우 0으로 채움 (패널 표시용)
                        timings['Depth Preprocess'] = 0.0
                        timings['Depth Inference'] = 0.0
                        timings['Depth Postprocess'] = 0.0
                        timings['Depth Fusion'] = 0.0
                    ################################################
                    
                    
                    # --- 디버그 출력 (30프레임마다) ---
                    if frame_count % 30 == 0:
                        print(f"\n[Frame {frame_count}] Detected: {len(boxes)} objects")
                        if len(boxes) > 0:
                            for i, (label, score, dist) in enumerate(zip(labels, scores, distances)):
                                class_name = class_names[label]
                                dist_str = f"| {dist:.1f}m" if dist is not None else ""
                                print(f"  {i+1}. {class_name}: {score:.3f} {dist_str}")
                    
                    # --- 정보 패널용 클래스 카운트 ---
                    class_counts = {}
                    for class_id in range(1, num_classes):  # 배경(0) 제외
                        count = np.sum(labels == class_id)
                        if count > 0:
                            class_counts[class_id] = count
                    
                    
                    # [타이밍 8: Draw BBoxes]
                    t_start_draw_pred = time.time()
                    frame = draw_predictions(frame, boxes, labels, scores, 
                                           class_names, class_colors, conf_threshold,
                                           distances=distances)
                    timings['Draw BBoxes'] = (time.time() - t_start_draw_pred) * 1000 # ms
                    
                    
                    # --- [수정] 전체 처리 시간 계산 ---
                    processing_time = time.time() - process_start
                    timings['Total Process'] = processing_time * 1000 # ms
                    
                    
                    # --- 성능 측정 및 프레임 스킵 로직 ---
                    frames_to_skip = 0
                    if processing_time > frame_interval:
                        frames_behind = int(processing_time / frame_interval)
                        frames_to_skip = frames_behind
                        for _ in range(frames_to_skip):
                            cap.read()
                            frame_count += 1
                    
                    # 성능 추적기 업데이트
                    performance_tracker.update(processing_time, frames_to_skip)
                    
                    
                    # [타이밍 9: Draw Info Panel]
                    t_start_draw_info = time.time()
                    frame = draw_info_panel(
                        frame, class_counts, class_names,
                        performance_tracker.get_inference_fps(),
                        conf_threshold, iou_threshold, video_fps,
                        performance_tracker.get_processing_rate(),
                        performance_tracker.dropped_frames,
                        performance_tracker.is_realtime_capable(),
                        model_dtype,
                        depth_model_active=CONFIG['depth_model_enabled'],
                        timings=timings  # <-- [수정] 타이밍 딕셔너리 전달
                    )
                    timings['Draw Info Panel'] = (time.time() - t_start_draw_info) * 1000 # ms
                    
                    last_frame = frame
                    
                    # --- 다음 프레임 시간 예약 ---
                    next_frame_time += frame_interval
                    if next_frame_time < current_time:
                        next_frame_time = current_time
            
            # --- 프레임 표시 ---
            if last_frame is not None:
                cv2.imshow('YouTube Real-time Detection + Depth', last_frame)

            # --- 키 입력 대기 ---
            time_until_next_frame = next_frame_time - time.time()
            wait_ms = max(1, int(time_until_next_frame * 1000))

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == 27:  # ESC 키
                print("\nExiting...")
                break
            elif key == ord(' '):  # 스페이스바 (일시정지/재개)
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
                if not paused:
                    next_frame_time = time.time()
            elif key == ord('+') or key == ord('='):  # Confidence 증가
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):  # Confidence 감소
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord(']'):  # IOU 증가
                iou_threshold = min(iou_threshold + 0.05, 0.95)
                print(f"\nIOU threshold: {iou_threshold:.2f}")
            elif key == ord('['):  # IOU 감소
                iou_threshold = max(iou_threshold - 0.05, 0.30)
                print(f"\nIOU threshold: {iou_threshold:.2f}")

    except KeyboardInterrupt:
        # Ctrl+C로 종료 시
        print("\n\nInterrupted by user")
    
    finally:
        # --- 리소스 정리 ---
        cap.release()
        cv2.destroyAllWindows()
        
        # --- 최종 성능 요약 출력 ---
        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)
        print(f"OD Model: {dtype_str} (Faster R-CNN)")
        if CONFIG['depth_model_enabled']:
             print(f"Depth Model: {dtype_str} (Active)")
        else:
             print("Depth Model: (Inactive)")
        print(f"Classes: {CONFIG['class_names']}")
        print(f"Video FPS: {video_fps:.1f}")
        print(f"Average Inference FPS: {performance_tracker.get_inference_fps():.1f}")
        print(f"Real-time Processing Rate: {performance_tracker.get_processing_rate():.1f}%")
        print(f"Total Frames: {frame_count}")
        print(f"Processed Frames: {performance_tracker.processed_frames}")
        print(f"Dropped Frames: {performance_tracker.dropped_frames}")
        if performance_tracker.is_realtime_capable():
            print("Status: ✓ REAL-TIME CAPABLE")
        else:
            print("Status: ✗ NOT REAL-TIME (Consider model optimization or smaller models)")
        print(f"\nVideo file at: {video_path}")
        print("="*70)


if __name__ == "__main__":
    main()  # 스크립트 실행