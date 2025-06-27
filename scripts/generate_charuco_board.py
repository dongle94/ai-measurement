"""
ChArUco 보드 생성 스크립트

ChArUco(Chessboard + ArUco) 보드는 카메라 캘리브레이션을 위한 패턴으로,
체스보드의 코너 검출 정확도와 ArUco 마커의 강건성을 결합한 것입니다.

사용법:
    python generate_charuco_board.py

출력:
    charuco_A4_6x9_30mm.png - A4 크기로 인쇄 가능한 ChArUco 보드
"""

import cv2
import numpy as np
import os

def generate_charuco_board():
    """ChArUco 보드를 생성하고 이미지로 저장합니다."""
    
    # ❶ 보드 파라미터 설정
    squares_x, squares_y = 6, 9        # 흰칸(=마커가 들어갈 칸) 개수
    square_len_mm = 30                 # 흰칸 한 변 길이 [mm]
    marker_len_mm = 22.5               # 마커 한 변 (흰칸의 75%)
    
    print(f"ChArUco 보드 파라미터:")
    print(f"  - 격자 크기: {squares_x} x {squares_y}")
    print(f"  - 사각형 크기: {square_len_mm}mm")
    print(f"  - 마커 크기: {marker_len_mm}mm")
    
    # ❷ ArUco 딕셔너리 선택
    # DICT_4X4_50: 4x4 비트, 50개 마커 (작은 보드에 적합)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # ❸ ChArUco 보드 객체 생성 (OpenCV ≥4.7)
    # 캘리브레이션에서는 실제 크기가 중요하므로 미터 단위로 변환
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_len_mm / 1000.0,        # mm → m 변환
        marker_len_mm / 1000.0,        # mm → m 변환
        aruco_dict
    )
    
    # ❹ A4 용지 크기로 이미지 생성
    # A4(210×297 mm)를 300 DPI로 변환 → 2480×3508 픽셀
    a4_px = (2480, 3508)               # (width, height)
    
    # 보드 이미지 생성
    # marginSize: 가장자리 여백 (픽셀)
    # borderBits: 마커 주변 검은색 테두리 두께
    img = board.generateImage(a4_px, marginSize=60, borderBits=1)
    
    # ❺ 이미지 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"charuco_A4_{squares_x}x{squares_y}_{square_len_mm}mm.png"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, img)
    print(f"\n✅ ChArUco 보드가 생성되었습니다: {filepath}")
    print(f"   크기: {a4_px[0]} x {a4_px[1]} 픽셀 (A4, 300 DPI)")
    
    # ❻ 보드 정보 저장 (캘리브레이션 시 필요)
    board_info = {
        'squares_x': squares_x,
        'squares_y': squares_y,
        'square_length_mm': square_len_mm,
        'marker_length_mm': marker_len_mm,
        'dictionary': 'DICT_4X4_50'
    }
    
    info_filepath = os.path.join(output_dir, f"charuco_board_info_{squares_x}x{squares_y}.txt")
    with open(info_filepath, 'w') as f:
        for key, value in board_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📋 보드 정보 저장됨: {info_filepath}")
    
    return board, filepath

def print_usage_instructions():
    """사용법 안내를 출력합니다."""
    print("\n" + "="*60)
    print("📖 ChArUco 보드 사용법")
    print("="*60)
    print("1. 생성된 PNG 파일을 A4 용지에 인쇄하세요")
    print("   - 실제 크기로 인쇄 (크기 조정 없음)")
    print("   - 고품질 프린터 사용 권장")
    print("   - 평평하고 단단한 표면에 부착")
    print()
    print("2. 캘리브레이션 촬영 시:")
    print("   - 다양한 각도에서 20-30장 촬영")
    print("   - 보드가 이미지 전체를 차지하도록")
    print("   - 조명이 균일하고 그림자가 없도록")
    print("   - 블러나 왜곡이 없도록 천천히 촬영")
    print()
    print("3. 캘리브레이션 시 board_info 파일의 정보를 사용하세요")

if __name__ == "__main__":
    print("🎯 ChArUco 보드 생성기")
    print("-" * 30)
    
    try:
        board, filepath = generate_charuco_board()
        print_usage_instructions()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("OpenCV 버전을 확인하세요 (≥4.7 권장)")
