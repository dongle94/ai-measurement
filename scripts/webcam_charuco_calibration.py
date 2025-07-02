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
        # 카메라에서 현재 FPS 정보 가져오기
        cap = cv2.VideoCapture(self.camera_index)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30
        cap.release()
        
        # JSON 형태로 저장
        calibration_data = {
            "timestamp": self.session_time,
            "camera_index": self.device_str,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "reprojection_error": float(self.calibration_error),
            "image_size": self.image_size,
            "fps": fps,
            "num_images": len(self.captured_images),
            "board_info": {
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_mm": self.square_length * 1000,
                "marker_length_mm": self.marker_length * 1000
            }
        }

        json_name = f"calibration_{self.device_str}_{self.session_time}.json"
        json_path = os.path.join(self.output_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"📁 캘리브레이션 결과 저장: {json_path}")
    
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
        print("   왼쪽: 원본 | 오른쪽: 보정 후")
        print("   [Q]키를 눌러 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break
            
            # 왜곡 보정 적용
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # 두 이미지를 나란히 배치
            comparison = np.hstack([frame, undistorted])
            
            # 제목과 구분선 추가
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # 상단에 제목 추가
            cv2.putText(comparison, "Original", (50, 40), font, font_scale, (0, 0, 255), thickness)
            cv2.putText(comparison, "Undistorted", (frame.shape[1] + 50, 40), font, font_scale, (0, 255, 0), thickness)
            
            # 중앙에 구분선 추가
            cv2.line(comparison, (frame.shape[1], 0), (frame.shape[1], comparison.shape[0]), (255, 255, 255), 2)
            
            # 하단에 안내 메시지 추가
            msg_y = comparison.shape[0] - 20
            cv2.putText(comparison, "Press [Q] to quit", (comparison.shape[1]//2 - 100, msg_y), 
                       font, 0.6, (255, 255, 255), 2)
            
            # 화면 표시
            cv2.imshow('Calibration Comparison - Before/After', comparison)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        cv2.destroyWindow('Calibration Comparison - Before/After')
    
    def run(self):
        """메인 캘리브레이션 루프를 실행합니다."""
        # 웹캠 초기화
        print(f"🎥 카메라 {self.camera_index} 초기화 중...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {self.camera_index}를 열 수 없습니다.")
            print("💡 다른 카메라 인덱스를 시도해보세요.")
            return
        
        # 웹캠 해상도 설정
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
        
        print("🎥 웹캠 캘리브레이션 시작")
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
