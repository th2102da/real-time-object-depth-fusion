"""
Real-time YouTube Object Detection with Faster R-CNN
+ Depth Model Fusion for Distance Estimation
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import time
import hashlib
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import yt_dlp

# Import preprocessing functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset2 import letterbox_resize, inverse_transform_bbox

##### 1. DEPTH 모델을 위한 라이브러리 임포트 #####
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
################################################

# ============================================================================
# CONFIGURATION - 학습된 모델에 맞게 수정하세요!
# ============================================================================

CONFIG = {
    # Model configuration
    'model_path': 'fasterrcnn_model.pth',
    'class_names': ['car','person'],  # Classes (WITHOUT '__background__')
    
    ##### 2. DEPTH 모델 설정 추가 #####
    'depth_model_enabled': True,  # 뎁스 모델 퓨전 활성화
    # (추천) 야외 주행 영상용 Metric(절대 거리) 모델
    'depth_model_id': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
    # (대안) 실내용 Metric 모델
    # 'depth_model_id': 'depth-anything/depth-anything-v2-metric-hypersim',
    # (대안) 일반적인 Small 모델 (상대 거리만 추정)
    # 'depth_model_id': 'depth-anything/depth-anything-v2-small-hf',
    ####################################
    
    # Detection parameters
    'conf_threshold': 0.5,   # Initial confidence threshold
    'iou_threshold': 0.5,    # Initial IOU threshold for NMS
    'resize_size': 640,      # Input size for model (Faster R-CNN)
    
    # Video settings
    'cache_dir': 'video_cache',  # Cache directory for downloaded videos
}

# ============================================================================
# (이하 EXAMPLE CONFIGURATIONS 생략)
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')
print(device)

def generate_colors(num_classes):
    """
    Generate distinct colors for each class
    Returns: dict mapping class_id (1-indexed) to BGR color
    """
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = {}
    
    # Predefined colors for common classes
    predefined_colors = [
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
    ]
    
    for i in range(1, num_classes + 1):
        if i - 1 < len(predefined_colors):
            colors[i] = predefined_colors[i - 1]
        else:
            # Generate random distinct color
            colors[i] = tuple(np.random.randint(50, 255, 3).tolist())
    
    return colors


def validate_model_classes(model, expected_num_classes):
    """
    Validate that model's number of classes matches expected
    
    Args:
        model: Faster R-CNN model
        expected_num_classes: Expected number of classes (including background)
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Get model's number of output classes
        model_num_classes = model.roi_heads.box_predictor.cls_score.out_features
        
        if model_num_classes != expected_num_classes:
            print(f"\n⚠️  WARNING: Model class mismatch!")
            print(f"  Expected classes: {expected_num_classes} (including background)")
            print(f"  Model classes: {model_num_classes}")
            print(f"  Please check your CONFIG['class_names']")
            return False
        
        return True
    except Exception as e:
        print(f"\n⚠️  Warning: Could not validate model classes: {e}")
        return True  # Continue anyway


def get_model_dtype(model):
    """Detect model's dtype (FP32 or FP16)"""
    return next(model.parameters()).dtype


def download_youtube_video(youtube_url, cache_dir, force_download=False):
    """
    Download YouTube video with caching support
    Returns: path to video file (cached or newly downloaded)
    """
    print(f"\nExtracting video ID...")
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id', None)
    except:
        video_id = hashlib.md5(youtube_url.encode()).hexdigest()[:11]
    
    print(f"Video ID: {video_id}")
    
    # Check cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cached_path = os.path.join(cache_dir, f"{video_id}.mp4")
    
    if os.path.exists(cached_path) and not force_download:
        file_size = os.path.getsize(cached_path) / (1024 * 1024)
        print(f"\n✓ Found cached video!")
        print(f"  Path: {cached_path}")
        print(f"  Size: {file_size:.1f} MB")
        print(f"  Skipping download...")
        return cached_path
    
    # Download video
    print(f"\nDownloading YouTube video...")
    print(f"This will take 1-2 minutes depending on video length and your internet speed")
    
    ydl_opts = {
        'format': (
            'best[height<=720][ext=mp4][vcodec^=avc1]/best[height<=720][ext=mp4]/'
            'bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/'
            'best[height<=720]'
        ),
        'outtmpl': cached_path,
        'quiet': False,
        'no_warnings': True,
        'noplaylist': True,
        'prefer_free_formats': False,
        'progress_hooks': [lambda d: print(f"\rDownload: {d.get('_percent_str', 'N/A')} ", end='') 
                          if d['status'] == 'downloading' else None],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\nFetching video info...")
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
        if not os.path.exists(cached_path):
            raise FileNotFoundError(f"Downloaded file not found: {cached_path}")
        
        file_size = os.path.getsize(cached_path) / (1024 * 1024)
        
        print(f"\n✓ Video downloaded successfully")
        print(f"  Title: {video_title}")
        print(f"  Duration: {duration // 60}m {duration % 60}s")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Cached to: {cached_path}")
        return cached_path
        
    except Exception as e:
        print(f"\nError: Failed to download video")
        print(f"Error details: {e}")
        raise


def preprocess_frame(frame, resize_size=640, model_dtype=torch.float32):
    """
    Preprocess frame for model inference with automatic dtype conversion
    
    Args:
        frame: OpenCV BGR frame
        resize_size: target size for letterbox resize
        model_dtype: dtype of the model (torch.float32 or torch.float16)
    
    Returns: 
        tensor, scale, pad_left, pad_top
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    img_letterbox, scale, pad_left, pad_top = letterbox_resize(
        pil_image, target_size=resize_size
    )
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = normalize(img_letterbox).unsqueeze(0).to(device)
    
    # Convert to FP16 if model is FP16
    if model_dtype == torch.float16:
        img_tensor = img_tensor.half()
    
    return img_tensor, scale, pad_left, pad_top


def postprocess_predictions(predictions, scale, pad_left, pad_top, 
                           conf_threshold=0.5, frame_shape=None):
    """
    Postprocess model predictions
    Returns: filtered boxes, labels, scores in original image coordinates
    """
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Transform boxes back to original coordinates
    transformed_boxes = []
    for box in boxes:
        orig_box = inverse_transform_bbox(box, scale, pad_left, pad_top)
        transformed_boxes.append(orig_box)
    
    # Clip boxes to frame boundaries if frame_shape provided
    if frame_shape is not None and len(transformed_boxes) > 0:
        h, w = frame_shape[:2]
        transformed_boxes = np.array(transformed_boxes)
        transformed_boxes[:, [0, 2]] = np.clip(transformed_boxes[:, [0, 2]], 0, w)
        transformed_boxes[:, [1, 3]] = np.clip(transformed_boxes[:, [1, 3]], 0, h)
    
    return transformed_boxes, labels, scores


##### 3. draw_predictions 함수 수정 (distances 파라미터 추가) #####
def draw_predictions(frame, boxes, labels, scores, class_names, class_colors, conf_threshold, distances=None):
    """Draw bounding boxes and labels on frame"""
    
    # 퓨전을 위해 distances가 None이 아니면 zip에 포함시킴
    if distances is None:
        distances = [None] * len(boxes)
        
    for box, label, score, dist in zip(boxes, labels, scores, distances):
        x1, y1, x2, y2 = map(int, box)
        color = class_colors.get(label, (0, 255, 0))
        class_name = class_names[label]
        
        # Draw thicker, brighter bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add semi-transparent fill for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # 레이블 텍스트에 거리 정보 추가
        label_text = f'{class_name}: {score:.2f}'
        if dist is not None:
            # Metric 모델은 'm' 단위로 표시
            label_text += f' | {dist:.1f}m'
            
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background with more visibility
        cv2.rectangle(frame, (x1, y1 - text_h - 12), (x1 + text_w + 4, y1), color, -1)
        
        # Draw label text with black outline for better readability
        cv2.putText(frame, label_text, (x1 + 2, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
################################################################


##### 4. draw_info_panel 함수 수정 (depth_model_active 파라미터 추가) #####
def draw_info_panel(frame, class_counts, class_names, inference_fps, conf_threshold, 
                   iou_threshold, video_fps, processing_rate, dropped_frames, 
                   realtime_capable, model_dtype, depth_model_active=False):
    """Draw information panel on frame with dynamic class display"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Calculate panel height based on number of classes
    num_classes = len(class_names) - 1  # Exclude background
    base_height = 300 # 뎁스 모델 정보 추가로 높이 증가
    class_height = num_classes * 22
    panel_height = min(base_height + class_height, h - 100)
    
    # Main info panel
    cv2.rectangle(overlay, (10, 10), (380, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Real-time status indicator
    status_color = (0, 255, 0) if realtime_capable else (0, 0, 255)
    status_text = "REAL-TIME ✓" if realtime_capable else "DROPPING FRAMES ✗"
    
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
            # Color coding
            if "Model:" in text:
                color = dtype_color
            elif "Video FPS:" in text:
                color = (100, 255, 255)
            elif "Inference FPS:" in text:
                color = (255, 255, 100)
            elif "Processing Rate:" in text:
                color = (0, 255, 0) if processing_rate >= 99 else (0, 165, 255)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, text, (20, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    # Real-time status
    cv2.putText(frame, status_text, (20, y_offset + len(info_texts) * 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
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
    """Track real-time processing performance"""
    def __init__(self, video_fps):
        self.video_fps = video_fps
        self.frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033
        self.total_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.inference_times = []
        self.max_history = 30
        
    def update(self, processing_time, frames_to_skip=0):
        """Update performance metrics"""
        self.total_frames += 1
        self.processed_frames += 1
        self.dropped_frames += frames_to_skip
        
        self.inference_times.append(processing_time)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
    
    def get_inference_fps(self):
        """Calculate average inference FPS"""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_processing_rate(self):
        """Calculate real-time processing rate"""
        if self.total_frames == 0:
            return 100.0
        return (self.processed_frames / (self.processed_frames + self.dropped_frames)) * 100
    
    def is_realtime_capable(self):
        """Check if currently processing in real-time"""
        if not self.inference_times:
            return True
        avg_time = np.mean(self.inference_times[-10:])
        return avg_time < self.frame_interval


def main():
    parser = argparse.ArgumentParser(description='Real-time YouTube Object Detection')
    parser.add_argument('--url', type=str, required=True, help='YouTube URL')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if cached')
    args = parser.parse_args()
    
    print("="*70)
    print("Universal Real-time YouTube Object Detection + Depth Fusion") # Title
    print("="*70)
    print(f"Device: {device}")
    
    # Print configuration
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
    
    # Prepare class names with background
    class_names = ['__background__'] + CONFIG['class_names']
    num_classes = len(class_names)
    
    print(f"  Total classes (with background): {num_classes}")
    
    # Generate colors for classes
    class_colors = generate_colors(len(CONFIG['class_names']))
    print(f"  Generated {len(class_colors)} class colors")
    
    # Load model
    print(f"\nLoading OD model (Faster R-CNN)...")
    if not os.path.exists(CONFIG['model_path']):
        raise FileNotFoundError(f"Model not found: {CONFIG['model_path']}")
    
    model = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Detect model dtype
    model_dtype = get_model_dtype(model)
    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    print(f"✓ OD Model loaded")
    print(f"  OD Model dtype: {dtype_str}")
    
    # Validate model classes
    if not validate_model_classes(model, num_classes):
        print("\n⚠️  Continuing anyway, but results may be incorrect!")
        print("  Make sure CONFIG['class_names'] matches your trained model")
        
    ##### 6. 뎁스 모델 로드 #####
    depth_model = None
    depth_processor = None
    if CONFIG['depth_model_enabled']:
        try:
            print(f"\nLoading Depth model ({CONFIG['depth_model_id']})...")
            depth_processor = AutoImageProcessor.from_pretrained(CONFIG['depth_model_id'])
            depth_model = AutoModelForDepthEstimation.from_pretrained(CONFIG['depth_model_id'])
            depth_model.to(device)
            depth_model.eval()
            if model_dtype == torch.float16:
                depth_model.half() # OD 모델과 동일한 타입 사용
            print("✓ Depth model loaded")
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to load depth model. Disabling depth fusion.")
            print(f"  Error: {e}")
            CONFIG['depth_model_enabled'] = False
    #################################

    # Download video (with caching)
    video_path = download_youtube_video(args.url, CONFIG['cache_dir'], 
                                       force_download=args.force_download)
    
    # Open video file
    print("\nOpening video file...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("✓ Video file opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  Video FPS: {video_fps:.1f} ← Playback synchronized to this")
    print(f"  Total frames: {total_frames}")
    if video_fps > 0 and total_frames > 0:
        duration = total_frames / video_fps
        print(f"  Duration: {int(duration // 60)}m {int(duration % 60)}s")
        print(f"  Frame interval: {1000/video_fps:.1f}ms per frame")
    
    # Test reading first frame
    print("\nTesting frame reading...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("✗ ERROR: Cannot read video frames!")
        cap.release()
        return
    else:
        print(f"✓ Successfully read first frame: {test_frame.shape}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nStarting real-time detection...")
    print("Press [+/-] to adjust confidence threshold")
    print("Press [[/]] to adjust IOU threshold")
    print("Press [SPACE] to pause/resume")
    print("Press [ESC] to exit")
    print("-"*70)
    
    # Initialize variables
    conf_threshold = CONFIG['conf_threshold']
    iou_threshold = CONFIG['iou_threshold']
    performance_tracker = RealtimePerformanceTracker(video_fps)
    frame_count = 0
    paused = False
    last_frame = None
    
    frame_interval = 1.0 / video_fps if video_fps > 0 else 0.033
    next_frame_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            if not paused:
                # Check if it's time for the next frame
                if current_time >= next_frame_time:
                    ret, frame = cap.read()
                    if not ret:
                        if frame_count == 0:
                            print("\n✗ Error: Could not read first frame!")
                        else:
                            print(f"\n✓ Video completed")
                            print(f"  Total frames: {frame_count}")
                            print(f"  Processed: {performance_tracker.processed_frames}")
                            print(f"  Dropped: {performance_tracker.dropped_frames}")
                            print(f"  Processing rate: {performance_tracker.get_processing_rate():.1f}%")
                        break
                    
                    if frame is None:
                        frame_count += 1
                        next_frame_time += frame_interval
                        continue
                    
                    frame_count += 1
                    process_start = time.time()
                    
                    # Update model's NMS threshold
                    model.roi_heads.nms_thresh = iou_threshold
                    
                    # --- 1. Object Detection (Faster R-CNN) ---
                    img_tensor, scale, pad_left, pad_top = preprocess_frame(
                        frame, resize_size=CONFIG['resize_size'], model_dtype=model_dtype
                    )
                    
                    with torch.no_grad():
                        predictions = model(img_tensor)
                    
                    boxes, labels, scores = postprocess_predictions(
                        predictions, scale, pad_left, pad_top,
                        conf_threshold=conf_threshold,
                        frame_shape=frame.shape
                    )
                    
                    ##### 7. 뎁스 추정 및 퓨전 로직 #####
                    distances = []
                    if CONFIG['depth_model_enabled'] and depth_model is not None and len(boxes) > 0:
                        # 뎁스 모델은 BGR -> RGB 변환된 PIL 이미지를 입력으로 받음
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # 뎁스 모델 전처리
                        inputs = depth_processor(images=pil_image, return_tensors="pt").to(device)
                        if model_dtype == torch.float16:
                            inputs['pixel_values'] = inputs['pixel_values'].half()
                        
                        # 뎁스 모델 추론
                        with torch.no_grad():
                            outputs = depth_model(**inputs)
                            predicted_depth = outputs.predicted_depth
                        
                        # 뎁스 맵을 원본 프레임 크기로 보간(Interpolate)
                        # BBox 좌표와 일치시키기 위해 필수
                        original_h, original_w = frame.shape[:2]
                        depth_map = torch.nn.functional.interpolate(
                            predicted_depth.unsqueeze(1),
                            size=(original_h, original_w),
                            mode='bicubic', # bicubic이 좀 더 부드러운 결과를 줌
                            align_corners=False
                        )
                        depth_map = depth_map.squeeze().cpu().numpy()
                        
                        # 각 BBox의 중앙값(median) 뎁스 계산
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            # BBox 영역의 뎁스 값들 추출
                            object_depth_region = depth_map[y1:y2, x1:x2]
                            
                            if object_depth_region.size > 0:
                                # 아웃라이어에 강건한 중앙값(median) 사용
                                median_depth = np.median(object_depth_region)
                                distances.append(median_depth)
                            else:
                                distances.append(0.0) # 영역이 없는 경우
                    else:
                        # 뎁스 모델이 비활성화되었거나 객체가 없는 경우
                        distances = [None] * len(boxes)
                    ################################################
                    
                    # Debug output - show what's detected
                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"\n[Frame {frame_count}] Detected: {len(boxes)} objects")
                        if len(boxes) > 0:
                            for i, (label, score, dist) in enumerate(zip(labels, scores, distances)):
                                class_name = class_names[label]
                                dist_str = f"| {dist:.1f}m" if dist is not None else ""
                                print(f"  {i+1}. {class_name}: {score:.3f} {dist_str}")
                        else:
                            print(f"  No objects detected (conf threshold: {conf_threshold:.2f})")
                    
                    # Count objects by class
                    class_counts = {}
                    for class_id in range(1, num_classes):  # Skip background
                        count = np.sum(labels == class_id)
                        if count > 0:
                            class_counts[class_id] = count
                    
                    ##### 8. 수정된 draw_predictions 호출 (distances 전달) #####
                    frame = draw_predictions(frame, boxes, labels, scores, 
                                           class_names, class_colors, conf_threshold,
                                           distances=distances) # distances 전달
                    
                    processing_time = time.time() - process_start
                    
                    # Calculate frames to skip if processing is too slow
                    frames_to_skip = 0
                    if processing_time > frame_interval:
                        frames_behind = int(processing_time / frame_interval)
                        frames_to_skip = frames_behind
                        # Skip frames to catch up
                        for _ in range(frames_to_skip):
                            cap.read()
                            frame_count += 1
                    
                    # Update performance tracker
                    performance_tracker.update(processing_time, frames_to_skip)
                    
                    ##### 9. 수정된 draw_info_panel 호출 (depth_model_active 전달) #####
                    frame = draw_info_panel(
                        frame, class_counts, class_names,
                        performance_tracker.get_inference_fps(),
                        conf_threshold, iou_threshold, video_fps,
                        performance_tracker.get_processing_rate(),
                        performance_tracker.dropped_frames,
                        performance_tracker.is_realtime_capable(),
                        model_dtype,
                        depth_model_active=CONFIG['depth_model_enabled'] # 뎁스 활성화 여부 전달
                    )
                    
                    last_frame = frame
                    
                    # Schedule next frame
                    next_frame_time += frame_interval
                    
                    # Adjust if we're too far behind
                    if next_frame_time < current_time:
                        next_frame_time = current_time
            
            # Display frame
            if last_frame is not None:
                cv2.imshow('YouTube Real-time Detection + Depth', last_frame) # Window title
            
            # Calculate wait time
            time_until_next_frame = next_frame_time - time.time()
            wait_ms = max(1, int(time_until_next_frame * 1000))
            
            # Handle keyboard input
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == 27:  # ESC
                print("\nExiting...")
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
                if not paused:
                    next_frame_time = time.time()
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                print(f"\nConfidence threshold: {conf_threshold:.2f}")
            elif key == ord(']'):
                iou_threshold = min(iou_threshold + 0.05, 0.95)
                print(f"\nIOU threshold: {iou_threshold:.2f}")
            elif key == ord('['):
                iou_threshold = max(iou_threshold - 0.05, 0.30)
                print(f"\nIOU threshold: {iou_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
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
        print(f"\nVideo cached at: {video_path}")
        print("="*70)


if __name__ == "__main__":
    main()