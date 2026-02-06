import pickle
import os
import numpy as np
from tqdm import tqdm

# =========================================================
# 프로젝트 구조에 맞춘 절대 경로 설정
# =========================================================
# 현재 파일(tools/make_mini_info.py)의 상위 폴더 = 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NuScenes 데이터 경로 (extern/pointcept/data/nuscenes 라고 가정)
DATA_ROOT = os.path.join(PROJECT_ROOT, "extern", "pointcept", "data", "nuscenes")

SOURCE_PKL = os.path.join(DATA_ROOT, "info", "nuscenes_infos_10sweeps_val.pkl")
TARGET_PKL = os.path.join(DATA_ROOT, "info", "nuscenes_infos_10sweeps_mini.pkl")
# =========================================================

def main():
    if not os.path.exists(SOURCE_PKL):
        print(f"[오류] 원본 메타데이터 파일을 찾을 수 없습니다: {SOURCE_PKL}")
        print(f"      경로 확인: {SOURCE_PKL}")
        return

    print(f"[1/3] 원본 메타데이터 로드 중... ({SOURCE_PKL})")
    with open(SOURCE_PKL, "rb") as f:
        data_list = pickle.load(f)
    
    print(f"      총 {len(data_list)}개의 샘플을 확인했습니다.")

    print("[2/3] 파일 존재 여부 검사 및 필터링 중...")
    valid_data = []
    missing_count = 0
    
    for item in tqdm(data_list):
        # NuScenesDataset은 'raw' 폴더를 참조하므로 경로를 맞춰서 검사
        lidar_rel_path = item["lidar_path"] 
        full_path = os.path.join(DATA_ROOT, "raw", lidar_rel_path)
        
        if os.path.exists(full_path):
            valid_data.append(item)
        else:
            missing_count += 1

    print(f"[3/3] 새로운 메타데이터 저장 중... ({TARGET_PKL})")
    print(f"      - 원본: {len(data_list)}개")
    print(f"      - 유효: {len(valid_data)}개 (Mini 데이터셋)")
    print(f"      - 제외: {missing_count}개")

    if len(valid_data) == 0:
        print("\n[경고] 유효한 파일이 0개입니다! 경로 설정이나 데이터 존재 여부를 확인하세요.")
        return

    with open(TARGET_PKL, "wb") as f:
        pickle.dump(valid_data, f)
    
    print("\n[완료] 'mini'용 메타데이터 생성이 완료되었습니다!")

if __name__ == "__main__":
    main()