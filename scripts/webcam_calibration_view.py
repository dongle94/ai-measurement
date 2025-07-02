"""
웹캠 실시간 캘리브레이션 적용 뷰어

기능:
1. 저장된 캘리브레이션 데이터로 실시간 왜곡 보정
2. 보정 전/후 화면 나란히 표시
3. 자동 캘리브레이션 파일 탐색 (카메라 번호 기반)

사용법:
    python webcam_undistort_view.py [camera_index] [calibration_file_path]
    
    camera_index (필수): 사용할 카메라 인덱스 (0, 1, 2, ...)
    calibration_file_path (선택사항): 캘리브레이션 JSON 파일 경로
        생략 시 output/webcam_calibration/camera_[camera_index]/ 폴더에서 자동 탐색

컨트롤:
    Q: 종료
"""

import cv2
import numpy as np
import os
import json
import glob
import sys
import argparse
from datetime import datetime

def parse_arguments():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='웹캠 실시간 보정 뷰어')
    parser.add_argument('camera_index', type=int, 
                        help='사용할 카메라 인덱스 (0, 1, 2, ...)')
    parser.add_argument('calibration_file', nargs='?', type=str,
                        help='캘리브레이션 JSON 파일 경로 (생략 시 자동 탐색)')
    return parser.parse_args()

def find_latest_calibration_file(camera_index):
    """지정된 카메라 인덱스에 대한 가장 최근 캘리브레이션 파일을 찾습니다."""
    device_str = str(camera_index)
    calibration_dir = os.path.join("output/webcam_calibration", f"camera_{device_str}")
    
    if not os.path.isdir(calibration_dir):
        return None
    
    # 해당 카메라의 모든 캘리브레이션 파일 찾기
    pattern = os.path.join(calibration_dir, f"calibration_{device_str}_*.json")
    calibration_files = glob.glob(pattern)
    
    if not calibration_files:
        return None
    
    # 가장 최신 파일 (파일명 기준)
    latest_file = max(calibration_files)
    return latest_file

def load_calibration_data(file_path):
    """캘리브레이션 파일을 로드하여 카메라 행렬과 왜곡 계수를 반환합니다."""
    try:
        with open(file_path, 'r') as f:
            calibration_data = json.load(f)
        
        camera_matrix = np.array(calibration_data['camera_matrix'])
        dist_coeffs = np.array(calibration_data['distortion_coefficients'])
        image_size = calibration_data.get('image_size')
        fps = calibration_data.get('fps', 30)  # fps 정보 읽기 (없으면 기본값 30)
        
        print(f"✅ 캘리브레이션 파일을 로드했습니다: {file_path}")
        print(f"📊 재투영 오차: {calibration_data.get('reprojection_error', 'N/A')} 픽셀")
        print(f"📐 이미지 크기: {image_size[0]}x{image_size[1]}")
        print(f"⏱️ 프레임 레이트: {fps} fps")
        
        return camera_matrix, dist_coeffs, image_size, fps
    
    except Exception as e:
        print(f"❌ 캘리브레이션 파일 로드 오류: {e}")
        return None, None, None, None

def run_undistorted_view(camera_index, camera_matrix, dist_coeffs, image_size=None, fps=30):
    """왜곡 보정된 실시간 웹캠 뷰를 실행합니다."""
    # 웹캠 초기화
    print(f"🎥 카메라 {camera_index} 초기화 중...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_index}를 열 수 없습니다.")
        print("💡 다른 카메라 인덱스를 시도해보세요.")
        return
    
    # 웹캠 해상도 설정 (캘리브레이션 시 사용된 해상도로 설정)
    if image_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS 설정 (캘리브레이션 시 사용된 FPS로 설정)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    
    # 실제 설정된 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ 카메라 {camera_index} 초기화 완료")
    print(f"   해상도: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    print("📋 실시간 왜곡 보정 뷰어 실행 중")
    print("   왼쪽: 원본 | 오른쪽: 보정됨")
    print("   [Q]키를 눌러 종료")
    
    # 두 창을 나란히 배치하기 위한 설정
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Undistorted', cv2.WINDOW_NORMAL)
    
    # 창 크기 및 위치 설정
    window_width = actual_width // 2
    window_height = actual_height // 2
    cv2.resizeWindow('Original', window_width, window_height)
    cv2.resizeWindow('Undistorted', window_width, window_height)
    
    # 창 위치 설정 (왼쪽, 오른쪽)
    cv2.moveWindow('Original', 50, 50)
    cv2.moveWindow('Undistorted', 50 + window_width + 20, 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break
        
        # 왜곡 보정 적용
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        # 화면에 정보 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Original", (30, 50), font, 1.2, (0, 0, 255), 2)
        cv2.putText(undistorted, "Undistorted", (30, 50), font, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'Q' to quit", (30, frame.shape[0] - 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(undistorted, "Press 'Q' to quit", (30, undistorted.shape[0] - 30), font, 0.7, (255, 255, 255), 2)
        
        # 각각 별도의 창에 표시
        cv2.imshow('Original', frame)
        cv2.imshow('Undistorted', undistorted)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 명령행 인자 파싱
    args = parse_arguments()
    camera_index = args.camera_index
    calibration_file = args.calibration_file
    
    # 캘리브레이션 파일 경로가 지정되지 않았으면 자동 탐색
    if calibration_file is None:
        calibration_file = find_latest_calibration_file(camera_index)
        if calibration_file is None:
            print(f"❌ 카메라 {camera_index}에 대한 캘리브레이션 파일을 찾을 수 없습니다.")
            print(f"   경로 확인: output/webcam_calibration/camera_{camera_index}/")
            sys.exit(1)
    
    # 캘리브레이션 데이터 로드
    camera_matrix, dist_coeffs, image_size, fps = load_calibration_data(calibration_file)
    if camera_matrix is None or dist_coeffs is None:
        print("❌ 유효한 캘리브레이션 데이터를 로드할 수 없습니다.")
        sys.exit(1)
    
    # 실시간 보정 뷰어 실행
    run_undistorted_view(camera_index, camera_matrix, dist_coeffs, image_size, fps)
