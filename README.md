# AI Measurement

컴퓨터 비전과 딥러닝을 이용한 물리적 능력 측정 프로젝트입니다.

## 설치

### 1. 환경 설정
```bash
# Conda 환경 생성
conda create -n ai-measurement python=3.10 -y
conda activate ai-measurement

# 의존성 설치
pip install -r requirements.txt
```

### 2. 의존성
- OpenCV 4.11+ (opencv-contrib-python)
- NumPy 2.2+
- 기타 표준 라이브러리

## 기능

### 1. ChArUco 보드 생성
- 카메라 캘리브레이션용 ChArUco 보드 자동 생성
- A4 용지에 최적화된 PDF 형식 출력

### 2. 웹캠 캘리브레이션
- 실시간 웹캠을 이용한 자동 캘리브레이션
- 캘리브레이션 후 실시간 왜곡 보정 비교

## 사용법

### 1. ChArUco 보드 생성
```bash
cd scripts
python generate_charuco_board.py
```
생성된 `charuco_board_7x5.pdf` 파일을 A4 용지에 인쇄하여 사용하세요.

### 2. 웹캠 캘리브레이션
```bash
cd scripts
python webcam_charuco_calibration.py

# 특정 카메라 지정
python webcam_charuco_calibration.py 0
```

**컨트롤:**
- `SPACE`: 현재 프레임 캡처
- `R`: 캡처된 이미지 초기화
- `C`: 캘리브레이션 수행 (완료 후 실시간 비교 모드로 전환)
- `Q`: 종료

## 출력 파일

### 캘리브레이션 결과
- `calibration_YYYYMMDD_HHMMSS.json`: 캘리브레이션 데이터
- `comparison_*.jpg`: 보정 전/후 비교 이미지
- `frame_*.jpg`: 캡처된 개별 프레임 (동영상 캘리브레이션)

### JSON 형식
```json
{
  "timestamp": "20241227_143052",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "reprojection_error": 0.234,
  "image_size": [1920, 1080]
}
```

## 팁

### 웹캠 캘리브레이션
1. 조명이 좋은 환경에서 촬영
2. 다양한 각도와 거리에서 보드 촬영
3. 보드가 화면 전체를 차지하지 않도록 주의
4. 15개 이상의 코너가 검출될 때 캡처

### 동영상 캘리브레이션
1. 핸드폰을 천천히 움직이며 촬영
2. 보드의 다양한 각도가 포함된 동영상 사용
3. 초점이 맞고 흔들림이 적은 영상 사용
4. 보드가 명확하게 보이는 구간이 충분히 포함된 영상

## 프로젝트 구조
```
ai-measurement/
├── scripts/
│   ├── generate_charuco_board.py      # ChArUco 보드 생성
│   ├── webcam_charuco_calibration.py  # 웹캠 캘리브레이션
│   ├── video_charuco_calibration.py   # 동영상 캘리브레이션
│   └── output/                        # 출력 파일들
├── requirements.txt                   # 의존성 목록
└── README.md                         # 프로젝트 설명
```

## 다음 단계
- [ ] 물리적 측정 기능 구현
- [ ] 포즈 추정 기능 추가
- [ ] 스테레오 캘리브레이션 지원
- [ ] GUI 인터페이스 개발