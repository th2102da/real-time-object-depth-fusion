# 🚀 Vision-Only 기반 실시간 ADAS 솔루션

> **단일 카메라(Vision-Only)**와 **딥러닝 퓨전(Deep Fusion)**을 활용한  
> **실시간 3D 객체 거리 인식 시스템**

> ![Model Fusion Demo](modelfusion.gif)

---

## 📚 목차 (Table of Contents)

- [프로젝트 개요 (Project Overview)](#-프로젝트-개요-project-overview)
- [핵심 기술 스택 (Tech Stack)](#-핵심-기술-스택-tech-stack)
- [시스템 아키텍처 및 퓨전 로직](#-시스템-아키텍처-및-퓨전-로직)
  - [Key Logic 1: 해상도 매칭 (Interpolation)](#-key-logic-1-해상도-매칭-interpolation)
  - [Key Logic 2: 이상치 제거 (Median-vs-Mean)](#-key-logic-2-이상치-제거-median-vs-mean)
- [실시간 성능 최적화 (Optimization)](#-실시간-성능-최적화-optimization)
  - [1️⃣ 정확도(Accuracy) 향상 전략](#1️⃣-정확도accuracy-향상-전략)
  - [2️⃣ 속도(Speed) 향상 전략](#2️⃣-속도speed-향상-전략)
- [실행 방법 (Installation & Usage)](#-실행-방법-installation--usage)
- [알려진 한계점 (Known Limitations)](#️-알려진-한계점-known-limitations)
- [향후 계획 (Future Work)](#-향후-계획-future-work)
- [팀원 (Contributors)](#-팀원-contributors)

---

## 🎯 프로젝트 개요 (Project Overview)

본 프로젝트의 시작은 **Faster R-CNN** 모델을 커스텀 데이터셋에 전이 학습시켜  
2D 객체를 탐지하는 과제였습니다.  

하지만 2D 객체 탐지는  
> “저것이 차다 (What)”는 정보만 제공할 뿐,  
> “그 차가 얼마나 멀리 있는가 (How Far)”를 알 수 없다는 명확한 한계가 있었습니다.  

이에 저희는 과제를 실제 **ADAS 솔루션**으로 확장했습니다.  

**🎯 최종 목표**  
> 고가의 LiDAR나 레이더 없이,  
> **단일 카메라(Vision-Only)**만으로  
> 2D 객체 탐지(Object Detection) + 3D 거리 추정(Depth Estimation)을 동시에 수행하는  
> **실시간 Deep Fusion 파이프라인 구축**

---

## 🛠️ 핵심 기술 스택 (Tech Stack)

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Deep Learning** | PyTorch | 모델 학습, 추론 및 FP16 최적화 |
| **Object Detection** | Faster R-CNN (Torchvision) | 👁️ The “Eyes” — 2D 객체 탐지 |
| **Depth Estimation** | Depth Anything V2 (Transformers) | 📏 The “Ruler” — 픽셀별 3D 거리 추정 |
| **Optimization** | Optuna (AutoML), Pruning | HPO 튜닝 및 모델 경량화 |
| **Data & Utility** | OpenCV, yt-dlp, NumPy | 비디오 I/O, 데이터 수집, Fusion 로직 |

---

## 📈 시스템 아키텍처 및 퓨전 로직

본 시스템은 입력된 비디오 프레임을  
두 개의 딥러닝 모델로 병렬 처리한 뒤, 결과를 융합(Fusion)합니다.

> *(⬆️ 아래 다이어그램 자리에 시스템 아키텍처 이미지를 삽입하세요)*

**Pipeline Flow:**
1. **Input:** 유튜브 스트림 또는 로컬 비디오 프레임
2. **Branch 1 (Object Detection):** Pruned Faster R-CNN이 객체의 BBox 탐지  
3. **Branch 2 (Depth Estimation):** Depth Anything V2가 각 픽셀의 Depth Map 생성  
4. **Fusion Module:** 두 결과를 융합하여 객체별 3D 거리 산출  

---

### 💡 Key Logic 1: 해상도 매칭 (Interpolation)

**문제:**  
OD의 BBox는 원본 해상도(예: 1920x1080) 기준,  
Depth Map은 저해상도(예: 512x512)로 불일치함.

**해결:**  
`torch.nn.functional.interpolate`로 Depth Map을 원본 크기로 업샘플링.  
- `mode='bicubic'` 옵션으로 부드러운 보간 처리  
- 픽셀 복사가 아닌 주변 픽셀 참조 기반 자연스러운 업샘플링 구현

---

### 💡 Key Logic 2: 이상치 제거 (Median vs. Mean)

**문제:**  
BBox 내부에 ‘하늘(500m)’ 같은 이상치 포함 → 평균 사용 시 왜곡 발생  

**해결:**  
- **평균(Mean)** 대신 **중앙값(Median)** 사용  
- `np.median()`으로 극단값 무시  
- 예: 실제 거리 20m → 평균 250m → 중앙값 20.5m (정확도 향상)

---

## ⚡ 실시간 성능 최적화 (Optimization)

두 개의 모델을 동시에 실행 시  
GPU 메모리 병목 및 **프레임 드랍(초기 5 FPS)** 발생  

이를 해결하기 위해 **정확도(Accuracy)**와 **속도(Speed)** 두 축에서 최적화를 진행했습니다.

---

### 1️⃣ 정확도(Accuracy) 향상 전략

> “속도를 희생하더라도 정확도를 먼저 확보하여 가지치기(Pruning)의 기반을 마련한다.”

- **커스텀 데이터셋 구축:**  
  서울 도로 주행 영상 1,000프레임 수집, COCO 포맷 어노테이션  
- **AutoML (Optuna):**  
  최적 하이퍼파라미터 탐색 (학습률, 배치 사이즈 등)
- **Backbone Fine-tuning:**  
  ResNet-50을 `Freeze=False`로 설정, 커스텀 데이터에 맞게 재학습  
  → 추론 0.005ms 증가, mAP 상승, 모델 압축 여유 확보

---

### 2️⃣ 속도(Speed) 향상 전략

> “확보된 정확도를 담보로 모델을 경량화하여 실시간 성능 달성.”

#### 🔹 Pruning (The "Eye")

- **대상:** ResNet-50의 Bottleneck conv1, conv2 레이어  
- **결과:**  
  - 속도: 🚀 2.09배 향상 (1.57 → 3.28 FPS)  
  - 모델 크기: 📉 32% 감소 (158MB → 107MB)  
  - 정확도: ✅ 97% 보존 (mIoU 0.960 → 0.932)

#### 🔹 FP16 변환 (The "Ruler")

- **방법:**  
  `depth_model.half()` 한 줄로 FP32 → FP16 변환  
- **결과:**  
  Depth 모델 추론 속도 **2~3배 향상**, 파이프라인 병목 해소  
- *(참고: OD 모델은 FP16 변환 시 정확도 저하로 FP32 유지)*

#### 🔹 Frame Skipping

- **문제:**  
  30 FPS 영상 vs. 모델 처리 20 FPS → 프레임 지연  
- **해결:**  
  2번 프레임을 스킵하고 최신 프레임(3번)만 처리  
- **결과:**  
  끊김 없는 실시간성 유지 (Processing Rate 100%)

---

## 🚀 실행 방법 (Installation & Usage)

### 1️⃣ Git Clone & 환경 설정

```bash
# 1. 레포지토리 클론 (URL을 실제 GitHub 주소로 변경)
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. 필수 라이브러리 설치
pip install torch torchvision transformers opencv-python-headless yt-dlp numpy
