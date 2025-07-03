"""
웹캠용 ChArUco 보드 캘리브레이션 스크립트

기능:
1. 다중 웹캠 지원 및 자동 감지
2. 실시간 웹캠 미리보기 및 ChArUco 코너 검출
3. 사용자 가이드와 함께 캘리브레이션 이미지 수집
4. 수집된 이미지들로 카메라 캘리브레이션 수행
5. 캘리브레이션 결과 시각화 (왜곡 보정 전후 비교)

사용법:
    python webcam_calibration.py [camera_index]
    
    camera_index (선택사항): 사용할 카메라 인덱스 (0, 1, 2, ...)
    생략 시 사용 가능한 카메라 목록을 표시하고 선택 가능

컨트롤:
    SPACE: 현재 프레임 캡처
    R: 캡처된 이미지 모두 삭제하고 다시 시작
    C: 캘리브레이션 수행 (완료 후 실시간 비교 모드로 전환)
    Q: 종료
"""

import cv2
import numpy as np
import os
import json
import glob
import sys
import argparse
import subprocess
from datetime import datetime

def detect_available_cameras(max_index=10):
    """사용 가능한 카메라들을 감지합니다."""
    available_cameras = []
    
    print("🔍 사용 가능한 카메라 감지 중...")
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 카메라 정보 가져오기
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 실제로 프레임을 읽어서 동작하는지 확인
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append({
                    'index': i,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'name': f"Camera {i}"
                })
                print(f"  ✅ 카메라 {i}: {width}x{height} @ {fps}fps")
            
            cap.release()
        else:
            # 연속으로 3개 카메라가 없으면 검색 중단
            if i > 2 and len(available_cameras) == 0:
                break
    
    return available_cameras

def select_camera(available_cameras):
    """사용자가 카메라를 선택할 수 있게 합니다."""
    if not available_cameras:
        print("❌ 사용 가능한 카메라가 없습니다.")
        return None
    
    print(f"\n📷 사용 가능한 카메라: {len(available_cameras)}개")
    print("=" * 50)
    
    for i, camera in enumerate(available_cameras):
        print(f"  {i}: {camera['name']} - {camera['resolution']} @ {camera['fps']}fps")
    
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"사용할 카메라 번호를 선택하세요 (0-{len(available_cameras)-1}): ")
            selected_idx = int(choice)
            
            if 0 <= selected_idx < len(available_cameras):
                selected_camera = available_cameras[selected_idx]
                print(f"✅ 선택됨: {selected_camera['name']} (인덱스 {selected_camera['index']})")
                return selected_camera['index']
            else:
                print(f"❌ 잘못된 선택입니다. 0-{len(available_cameras)-1} 범위에서 선택하세요.")
        
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            return None

def parse_arguments():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='웹캠 ChArUco 캘리브레이션')
    parser.add_argument('camera_index', nargs='?', type=int, 
                       help='사용할 카메라 인덱스 (0, 1, 2, ...). 생략 시 자동 선택')
    return parser.parse_args()

class WebcamCalibration:
    def __init__(self, camera_index=0):
        # 카메라 설정
        self.camera_index = camera_index
        self.device_str = str(camera_index)
        # 캘리브레이션 세션 타임스탬프 (YYYYMMDD_HHMMSS)
        self.session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count = 0  # 캡처 인덱스
        
        # 웹캠 프로필 선택
        print(f"\n🎯 카메라 {camera_index} 프로필 설정")
        self.profiles = get_webcam_profiles(camera_index)
        self.selected_profile = None
        if self.profiles:
            self.selected_profile = select_webcam_profile(self.profiles)
        else:
            print("기본 해상도(1280x720)와 자동 FPS를 사용합니다.")

        # ChArUco 보드 설정 (generate_charuco_board.py와 동일)
        self.squares_x = 6
        self.squares_y = 9
        self.square_length = 0.030  # 30mm in meters
        self.marker_length = 0.0225  # 22.5mm in meters

        # ArUco 딕셔너리 및 보드 생성
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )

        # CharucoDetector 생성 (OpenCV 4.11+)
        detector_params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params, detector_params)

        # 캘리브레이션 데이터
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.captured_images = []
        self.image_size = None

        # 캘리브레이션 결과
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None

        # 디렉토리 설정 (카메라별 하위 폴더)
        self.output_dir = os.path.join("output/webcam_calibration", f"camera_{self.device_str}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 상태 표시를 위한 변수들
        self.last_detection_status = "Waiting..."
        self.target_images = 25  # 목표 이미지 수
        
    def detect_charuco_corners(self, image):
        """이미지에서 ChArUco 코너를 검출합니다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 새로운 CharucoDetector API 사용 (OpenCV 4.7+)
        charuco_corners, charuco_ids, aruco_corners, aruco_ids = self.charuco_detector.detectBoard(gray)
        
        # 검출된 코너 수 계산
        num_corners = len(charuco_corners) if charuco_corners is not None else 0
        
        return num_corners, charuco_corners, charuco_ids, aruco_corners, aruco_ids
    
    def draw_detection_status(self, image, num_corners, charuco_corners, charuco_ids, aruco_corners, aruco_ids):
        """검출 상태를 이미지에 표시합니다."""
        # 검출된 마커와 코너 그리기
        if len(aruco_corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, aruco_corners, aruco_ids)
            
        if num_corners > 0 and charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
        
        # 상태 정보 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 배경 사각형
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (600, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 텍스트 정보
        texts = [
            f"Captured: {len(self.captured_images)}/{self.target_images}",
            f"Corners: {num_corners}",
            f"Status: {self.get_detection_quality(num_corners)}",
            "",
            "Controls: [SPACE]Capture [R]Reset [C]Calibrate+Compare [Q]Quit"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 0) if num_corners > 15 else (0, 255, 255) if num_corners > 5 else (0, 0, 255)
            if i >= 3:  # 컨트롤 텍스트는 흰색
                color = (255, 255, 255)
            cv2.putText(image, text, (20, 35 + i * 25), font, 0.6, color, 2)
        
        return image
    
    def get_detection_quality(self, num_corners):
        """코너 검출 품질을 평가합니다."""
        if num_corners >= 20:
            return "Excellent - Ready to capture!"
        elif num_corners >= 15:
            return "Good - Can capture"
        elif num_corners >= 10:
            return "Fair - Adjust angle"
        elif num_corners >= 5:
            return "Poor - Make board clearer"
        else:
            return "Failed - Check board visibility"
    
    def capture_image(self, image, charuco_corners, charuco_ids):
        """현재 프레임을 캘리브레이션용으로 캡처합니다."""
        if charuco_corners is not None and len(charuco_corners) >= 10:
            # 이미지 저장
            idx_str = f"{self.capture_count:03d}"
            image_name = f"capture_{self.device_str}_{self.session_time}_{idx_str}.jpg"
            image_path = os.path.join(self.output_dir, image_name)
            cv2.imwrite(image_path, image)

            # 캘리브레이션 데이터 저장
            self.captured_images.append(image_path)
            self.all_charuco_corners.append(charuco_corners)
            self.all_charuco_ids.append(charuco_ids)

            if self.image_size is None:
                self.image_size = image.shape[:2][::-1]  # (width, height)

            self.capture_count += 1
            print(f"✅ 이미지 캡처됨: {len(self.captured_images)}/{self.target_images}")
            return True
        else:
            print("❌ 캡처 실패: 충분한 코너가 검출되지 않음")
            return False
    
    def reset_capture(self):
        """캡처된 모든 데이터를 리셋합니다."""
        # 저장된 이미지 파일들 삭제
        for image_path in self.captured_images:
            if os.path.exists(image_path):
                os.remove(image_path)

        # 데이터 초기화
        self.all_charuco_corners.clear()
        self.all_charuco_ids.clear()
        self.captured_images.clear()
        self.image_size = None
        self.capture_count = 0

        print("🔄 모든 캡처 데이터가 리셋되었습니다.")
    
    def perform_calibration(self):
        """수집된 데이터로 카메라 캘리브레이션을 수행합니다."""
        if len(self.all_charuco_corners) < 10:
            print("❌ 캘리브레이션 실패: 최소 10개의 이미지가 필요합니다.")
            return False
        
        print("🔄 캘리브레이션 수행 중...")
        
        try:
            # ChArUco 캘리브레이션 수행
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self.all_charuco_corners,
                self.all_charuco_ids,
                self.board,
                self.image_size,
                None,
                None
            )
            
            if ret:
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                self.calibration_error = ret
                
                print(f"✅ 캘리브레이션 성공!")
                print(f"   - 재투영 오차: {ret:.4f} 픽셀")
                print(f"   - 사용된 이미지: {len(self.all_charuco_corners)}개")
                
                # 결과 저장
                self.save_calibration_results()
                return True
            else:
                print("❌ 캘리브레이션 실패: 계산 오류")
                return False
                
        except Exception as e:
            print(f"❌ 캘리브레이션 오류: {e}")
            return False
    
    def save_calibration_results(self):
        """캘리브레이션 결과를 파일로 저장합니다."""
        # 카메라에서 현재 정보 가져오기
        cap = cv2.VideoCapture(self.camera_index)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1280
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720
        actual_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
        cap.release()
        
        # JSON 형태로 저장
        calibration_data = {
            "timestamp": self.session_time,
            "camera_index": self.device_str,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "reprojection_error": float(self.calibration_error),
            "image_size": [actual_width, actual_height],
            "fps": actual_fps,
            "num_images": len(self.captured_images),
            "board_info": {
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_mm": self.square_length * 1000,
                "marker_length_mm": self.marker_length * 1000
            }
        }
        
        # 선택된 웹캠 프로필 정보 추가
        if self.selected_profile:
            calibration_data["webcam_profile"] = {
                "codec": self.selected_profile["codec"],
                "codec_name": self.selected_profile["codec_name"],
                "width": self.selected_profile["width"],
                "height": self.selected_profile["height"],
                "fps": self.selected_profile["fps"]
            }
            # 호환성을 위해 codec 정보도 최상위 레벨에 추가
            calibration_data["codec"] = self.selected_profile["codec"]

        json_name = f"calibration_{self.device_str}_{self.session_time}.json"
        json_path = os.path.join(self.output_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"📁 캘리브레이션 결과 저장: {json_path}")
        if self.selected_profile:
            print(f"   프로필: {self.selected_profile['codec']} {self.selected_profile['width']}x{self.selected_profile['height']} @{self.selected_profile['fps']}fps")
    
    def create_comparison_visualization(self):
        """왜곡 보정 전후 비교 이미지를 생성합니다."""
        if self.camera_matrix is None:
            print("❌ 캘리브레이션이 완료되지 않았습니다.")
            return

        # 최근 캡처된 이미지 몇 개로 비교 이미지 생성
        num_compare = min(4, len(self.captured_images))

        for i in range(num_compare):
            image_path = self.captured_images[-(i+1)]  # 최근 이미지부터
            image = cv2.imread(image_path)

            if image is not None:
                # 왜곡 보정
                undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

                # 나란히 배치
                comparison = np.hstack([image, undistorted])

                # 제목 추가
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, "Original", (50, 50), font, 1, (0, 0, 255), 2)
                cv2.putText(comparison, "Undistorted", (image.shape[1] + 50, 50), font, 1, (0, 255, 0), 2)

                # 저장
                comp_name = f"comparison_{self.device_str}_{self.session_time}_{i:03d}.jpg"
                comp_path = os.path.join(self.output_dir, comp_name)
                cv2.imwrite(comp_path, comparison)

        print(f"📸 비교 이미지 {num_compare}개 생성 완료")
    
    def run_realtime_comparison(self, cap):
        """캘리브레이션 후 실시간 보정 전/후 비교를 실행합니다."""
        if self.camera_matrix is None:
            print("❌ 캘리브레이션이 완료되지 않았습니다.")
            return
        
        print("\n🔄 실시간 왜곡 보정 비교 모드")
        print("   왼쪽: 원본 (기존 창) | 오른쪽: 보정됨 (새 창)")
        print("   [Q]키를 눌러 종료")
        
        # 보정된 영상용 새 창 생성
        cv2.namedWindow('Undistorted - Calibration Result', cv2.WINDOW_NORMAL)
        
        # 기존 캘리브레이션 창의 위치와 크기 가져오기
        try:
            # 기존 창 크기 확인
            ret, frame = cap.read()
            if ret:
                original_height, original_width = frame.shape[:2]
                
                # 새 창 크기 설정 (기존 창과 동일)
                cv2.resizeWindow('Undistorted - Calibration Result', original_width, original_height)
                
                # 새 창을 기존 창 오른쪽에 배치
                cv2.moveWindow('Undistorted - Calibration Result', original_width + 50, 50)
        except:
            pass  # 창 위치 설정 실패해도 계속 진행
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break
            
            # 왜곡 보정 적용
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # 기존 캘리브레이션 창에는 원본 프레임 표시 (상태 정보 추가)
            display_frame = frame.copy()
            
            # 상태 정보 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 배경 사각형
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # 텍스트 정보
            texts = [
                "Calibration Complete!",
                "Left: Original | Right: Undistorted",
                "Press [Q] to quit comparison mode"
            ]
            
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(display_frame, text, (20, 35 + i * 25), font, 0.6, color, 2)
            
            # 보정된 프레임에 제목 추가
            undistorted_display = undistorted.copy()
            cv2.putText(undistorted_display, "Undistorted", (30, 50), font, 1.2, (0, 255, 0), 2)
            cv2.putText(undistorted_display, "Press [Q] to quit", (30, undistorted_display.shape[0] - 30), 
                       font, 0.7, (255, 255, 255), 2)
            
            # 두 창에 각각 표시
            cv2.imshow('ChArUco Calibration', display_frame)  # 기존 창 (원본)
            cv2.imshow('Undistorted - Calibration Result', undistorted_display)  # 새 창 (보정됨)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        # 보정된 영상 창만 닫기 (기존 캘리브레이션 창은 유지)
        cv2.destroyWindow('Undistorted - Calibration Result')
    
    def run(self):
        """메인 캘리브레이션 루프를 실행합니다."""
        # 웹캠 초기화
        print(f"\n🎥 카메라 {self.camera_index} 초기화 중...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {self.camera_index}를 열 수 없습니다.")
            print("💡 다른 카메라 인덱스를 시도해보세요.")
            return
        
        # 선택한 웹캠 프로필이 있으면 적용
        if self.selected_profile:
            success = set_webcam_profile(cap, self.camera_index, self.selected_profile)
            if not success:
                print("⚠️ 선택된 프로필을 적용하는데 문제가 발생했습니다. 기본 설정을 사용합니다.")
                # 기본 해상도 설정
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            # 웹캠 기본 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 실제 설정된 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ 카메라 {self.camera_index} 초기화 완료")
        print(f"   해상도: {actual_width}x{actual_height}")
        print(f"   FPS: {actual_fps}")
        print()
        print("🎯 웹캠 캘리브레이션 시작")
        print("📋 ChArUco 보드를 웹캠 앞에 놓고 다양한 각도로 촬영하세요")
        print("   컨트롤: [SPACE]캡처 [R]리셋 [C]캘리브레이션+실시간비교 [Q]종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break
            
            # ChArUco 코너 검출
            num_corners, charuco_corners, charuco_ids, aruco_corners, aruco_ids = self.detect_charuco_corners(frame)
            
            # 표시용 프레임 생성 (원본 복사)
            display_frame = frame.copy()
            
            # 검출 상태 표시 (표시용 프레임에만)
            display_frame = self.draw_detection_status(display_frame, num_corners, charuco_corners, charuco_ids, aruco_corners, aruco_ids)
            
            # 화면 표시 (오버레이가 있는 프레임)
            cv2.imshow('ChArUco Calibration', display_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # 스페이스바: 캡처
                # 원본 프레임(오버레이 없음)을 캡처
                self.capture_image(frame, charuco_corners, charuco_ids)
                
            elif key == ord('r') or key == ord('R'):  # R: 리셋
                self.reset_capture()
                
            elif key == ord('c') or key == ord('C'):  # C: 캘리브레이션
                if self.perform_calibration():
                    self.create_comparison_visualization()
                    # 캘리브레이션 완료 후 실시간 비교 모드로 전환
                    self.run_realtime_comparison(cap)
                    
            elif key == ord('q') or key == ord('Q'):  # Q: 종료
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 최종 결과 출력
        if self.camera_matrix is not None:
            print("\n🎉 캘리브레이션 완료!")
            print(f"📊 최종 재투영 오차: {self.calibration_error:.4f} 픽셀")
            print(f"📁 결과 저장 위치: {self.output_dir}")
        else:
            print("\n❌ 캘리브레이션이 완료되지 않았습니다.")

def check_v4l2_ctl_available():
    """v4l2-ctl 명령어가 사용 가능한지 확인합니다."""
    try:
        subprocess.run(["which", "v4l2-ctl"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_webcam_profiles(camera_index):
    """웹캠의 지원 프로필(코덱, 해상도, FPS)을 v4l2-ctl을 사용하여 조회합니다."""
    if not check_v4l2_ctl_available():
        print("⚠️ v4l2-ctl이 설치되어 있지 않습니다. 기본 설정을 사용합니다.")
        return None
    
    device_path = f"/dev/video{camera_index}"
    if not os.path.exists(device_path):
        print(f"⚠️ 카메라 디바이스 경로를 찾을 수 없습니다: {device_path}")
        return None
    
    try:
        print(f"🔍 카메라 {camera_index}의 지원 프로필 조회 중...")
        # v4l2-ctl을 사용하여 지원되는 포맷 목록 가져오기
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-formats-ext"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        output = result.stdout
        
        # 출력 파싱하여 포맷 정보 추출
        profiles = []
        current_codec = None
        current_codec_name = None
        current_resolution = None
        
        for line in output.splitlines():
            line = line.strip()
            
            # 픽셀 포맷(코덱) 정보 추출 - 실제 형식: [0]: 'MJPG' (Motion-JPEG, compressed)
            if "]: '" in line and "(" in line and ")" in line:
                # 'MJPG' 부분 추출
                start_quote = line.find("'") + 1
                end_quote = line.find("'", start_quote)
                if start_quote > 0 and end_quote > start_quote:
                    current_codec = line[start_quote:end_quote]
                    
                    # 괄호 안의 설명 추출
                    start_paren = line.find("(") + 1
                    end_paren = line.find(")", start_paren)
                    if start_paren > 0 and end_paren > start_paren:
                        current_codec_name = line[start_paren:end_paren]
                    else:
                        current_codec_name = current_codec
            
            # 해상도 정보 추출 - 실제 형식: Size: Discrete 1920x1080
            elif "Size: Discrete" in line and current_codec:
                size_part = line.split("Size: Discrete")[1].strip()
                if "x" in size_part:
                    try:
                        width, height = map(int, size_part.split("x"))
                        current_resolution = (width, height)
                    except ValueError:
                        continue
            
            # FPS 정보 추출 - 실제 형식: Interval: Discrete 0.017s (60.000 fps)
            elif "Interval: Discrete" in line and "fps)" in line and current_codec and current_resolution:
                # (60.000 fps) 부분에서 fps 값 추출
                fps_start = line.find("(") + 1
                fps_end = line.find(" fps)", fps_start)
                if fps_start > 0 and fps_end > fps_start:
                    try:
                        fps_str = line[fps_start:fps_end]
                        fps = float(fps_str)
                        
                        # 프로필 정보 저장
                        profile = {
                            'codec': current_codec,
                            'codec_name': current_codec_name,
                            'width': current_resolution[0],
                            'height': current_resolution[1],
                            'fps': fps
                        }
                        profiles.append(profile)
                        
                    except ValueError:
                        continue
        
        return profiles
    
    except subprocess.CalledProcessError as e:
        print(f"⚠️ v4l2-ctl 명령 실행 오류: {e}")
        return None
    except Exception as e:
        print(f"⚠️ 웹캠 프로필 조회 중 오류 발생: {e}")
        return None

def select_webcam_profile(profiles):
    """사용자가 웹캠 프로필(코덱, 해상도, FPS)을 선택할 수 있게 합니다."""
    if not profiles:
        print("⚠️ 사용 가능한 웹캠 프로필이 없습니다. 기본 설정을 사용합니다.")
        return None
    
    # 코덱별로 프로필 그룹화
    codec_profiles = {}
    for profile in profiles:
        codec = profile['codec']
        if codec not in codec_profiles:
            codec_profiles[codec] = []
        codec_profiles[codec].append(profile)
    
    # 프로필 선택 UI
    print("\n📹 사용 가능한 웹캠 프로필:")
    print("=" * 50)
    
    # 코덱 선택
    print("\n1️⃣ 코덱 선택:")
    codecs = list(codec_profiles.keys())
    for i, codec in enumerate(codecs):
        sample_profile = codec_profiles[codec][0]
        print(f"  {i}: {codec} ({sample_profile['codec_name']})")
    
    codec_idx = -1
    while codec_idx < 0 or codec_idx >= len(codecs):
        try:
            choice = input(f"코덱 번호를 선택하세요 (0-{len(codecs)-1}, 기본값=0): ")
            if choice.strip() == "":
                codec_idx = 0
                break
            
            codec_idx = int(choice)
            if codec_idx < 0 or codec_idx >= len(codecs):
                print(f"❌ 잘못된 선택입니다. 0-{len(codecs)-1} 범위에서 선택하세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로필 선택을 취소합니다.")
            return None
    
    selected_codec = codecs[codec_idx]
    print(f"✅ 선택됨: {selected_codec}")
    
    # 선택된 코덱의 해상도 목록 추출
    selected_profiles = codec_profiles[selected_codec]
    resolutions = set()
    for profile in selected_profiles:
        resolutions.add((profile['width'], profile['height']))
    
    resolutions = sorted(list(resolutions), reverse=True)  # 높은 해상도가 먼저 오도록 정렬
    
    # 해상도 선택
    print("\n2️⃣ 해상도 선택:")
    for i, res in enumerate(resolutions):
        print(f"  {i}: {res[0]}x{res[1]}")
    
    res_idx = -1
    while res_idx < 0 or res_idx >= len(resolutions):
        try:
            choice = input(f"해상도 번호를 선택하세요 (0-{len(resolutions)-1}, 기본값=0): ")
            if choice.strip() == "":
                res_idx = 0
                break
            
            res_idx = int(choice)
            if res_idx < 0 or res_idx >= len(resolutions):
                print(f"❌ 잘못된 선택입니다. 0-{len(resolutions)-1} 범위에서 선택하세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로필 선택을 취소합니다.")
            return None
    
    selected_resolution = resolutions[res_idx]
    print(f"✅ 선택됨: {selected_resolution[0]}x{selected_resolution[1]}")
    
    # 선택된 해상도의 FPS 목록 추출
    fps_options = []
    for profile in selected_profiles:
        if profile['width'] == selected_resolution[0] and profile['height'] == selected_resolution[1]:
            fps_options.append(profile['fps'])
    
    fps_options = sorted(list(set(fps_options)), reverse=True)  # 중복 제거 후 높은 FPS가 먼저 오도록 정렬
    
    # FPS 선택
    print("\n3️⃣ 프레임레이트(FPS) 선택:")
    for i, fps in enumerate(fps_options):
        print(f"  {i}: {fps} fps")
    
    fps_idx = -1
    while fps_idx < 0 or fps_idx >= len(fps_options):
        try:
            choice = input(f"FPS 번호를 선택하세요 (0-{len(fps_options)-1}, 기본값=0): ")
            if choice.strip() == "":
                fps_idx = 0
                break
            
            fps_idx = int(choice)
            if fps_idx < 0 or fps_idx >= len(fps_options):
                print(f"❌ 잘못된 선택입니다. 0-{len(fps_options)-1} 범위에서 선택하세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로필 선택을 취소합니다.")
            return None
    
    selected_fps = fps_options[fps_idx]
    print(f"✅ 선택됨: {selected_fps} fps")
    
    # 선택된 프로필 생성
    selected_profile = {
        'codec': selected_codec,
        'codec_name': next(p['codec_name'] for p in selected_profiles if p['codec'] == selected_codec),
        'width': selected_resolution[0],
        'height': selected_resolution[1],
        'fps': selected_fps
    }
    
    print("\n🎯 선택된 프로필:")
    print(f"  코덱: {selected_profile['codec']} ({selected_profile['codec_name']})")
    print(f"  해상도: {selected_profile['width']}x{selected_profile['height']}")
    print(f"  FPS: {selected_profile['fps']}")
    
    return selected_profile

def set_webcam_profile(cap, camera_index, profile):
    """웹캠 설정을 선택된 프로필에 맞게 설정합니다."""
    if not profile:
        return False
    
    try:
        # 코덱 설정 (OpenCV 내장 기능 사용)
        if 'codec' in profile and profile['codec']:
            codec = profile['codec']
            try:
                print(f"🎬 코덱 설정 시도: {codec}")
                # 4자리 문자열을 FOURCC로 변환
                if len(codec) == 4:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    print(f"✅ 코덱 설정 성공: {codec}")
                else:
                    print(f"⚠️ 유효하지 않은 코덱 형식: {codec} (4자리 문자열이어야 함)")
            except Exception as e:
                print(f"⚠️ 코덱 설정 실패: {e}, 기본 코덱 사용")
        
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile['height'])
        
        # FPS 설정
        cap.set(cv2.CAP_PROP_FPS, profile['fps'])
        
        # 실제 설정된 값 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 설정된 카메라 프로필:")
        print(f"  코덱: {profile['codec']} ({profile['codec_name']})")
        print(f"  해상도: {actual_width}x{actual_height} (요청: {profile['width']}x{profile['height']})")
        print(f"  FPS: {actual_fps} (요청: {profile['fps']})")
        
        return True
    except Exception as e:
        print(f"⚠️ 웹캠 프로필 설정 중 오류: {e}")
        return False

if __name__ == "__main__":
    # 명령행 인자 파싱
    args = parse_arguments()
    
    if args.camera_index is not None:
        # 명령행에서 카메라 인덱스 지정됨
        print(f"📷 지정된 카메라 인덱스: {args.camera_index}")
        calibrator = WebcamCalibration(camera_index=args.camera_index)
        calibrator.run()
    else:
        # 사용 가능한 카메라 자동 감지 및 선택
        available_cameras = detect_available_cameras()
        
        if not available_cameras:
            print("❌ 사용 가능한 카메라가 없습니다.")
            sys.exit(1)
        
        # 카메라가 하나만 있으면 자동 선택
        if len(available_cameras) == 1:
            camera_index = available_cameras[0]['index']
            print(f"📷 카메라 1개 발견, 자동 선택: 카메라 {camera_index}")
        else:
            # 여러 카메라 중 선택
            camera_index = select_camera(available_cameras)
            if camera_index is None:
                print("👋 사용자가 취소했습니다.")
                sys.exit(0)
        
        calibrator = WebcamCalibration(camera_index=camera_index)
        calibrator.run()
