import argparse
import os
import numpy as np
import torch
from collections import OrderedDict

import _init_pointcept_path

# Pointcept 라이브러리 임포트
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.visualization import save_point_cloud

# S3DIS 클래스별 색상 맵
S3DIS_COLOR_MAP = np.array([
    [0,   255, 0],   # ceiling
    [255, 0,   0],   # floor
    [0,   0,   255], # wall
    [0,   255, 255], # beam
    [255, 255, 0],   # column
    [255, 0,   255], # window
    [100, 100, 255], # door
    [200, 200, 100], # table
    [170, 120, 200], # chair
    [255, 180, 0],   # sofa
    [10,  170, 10],  # bookcase
    [0,   0,   0],   # board
    [50,  50,  50],  # clutter
])

def main():
    # 1. 인자 파싱 (launch.json의 args를 받음)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to config file")
    # parser.add_argument("--weight", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()

    # 2. Config 파일 로드 (fromfile 사용)
    cfg = Config.fromfile(args.config_file)

    # 3. 데이터 설정 (Single-pass 추론용으로 덮어쓰기)
    data_config = dict(
        type="S3DISDataset",
        split="Area_5",
        data_root="data/s3dis",
        # test_area=5,  <-- 이 부분이 에러 원인이므로 제거합니다.
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.04,  # RTX 4050 메모리 최적화
                hash_type="fnv",
                mode="train",    # 핵심: Fragment 없이 한 번에 추론
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    )

    # 4. 모델 빌드 (Config 파일의 'model' 섹션 사용)
    model = build_model(cfg.model).cuda()

    # 5. 가중치 로드
    checkpoint = torch.load(args.weight, map_location="cuda")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # 키 매핑 수정
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # remove 'module.'
        new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 6. 데이터셋 로드 및 추론
    idx = 0
    dataset = build_dataset(data_config)
    data = dataset[idx]

    # GPU 전송
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda(non_blocking=True)

    # 7. 추론 및 결과 저장
    print(f"추론 시작: {data['name']} (Points: {data['coord'].shape[0]})")
    
    with torch.no_grad():
        output = model(data)
        seg_logits = seg_logits = output['seg_logits']
        pred_labels = seg_logits.argmax(-1).cpu().numpy()

    # 시각화 (색상 매핑)
    pred_colors = S3DIS_COLOR_MAP[pred_labels]
    
    os.makedirs("test/s3dis_inference", exist_ok=True)
    save_path = f"test/s3dis_inference/pred_{data['name']}.ply"
    
    save_point_cloud(
        coord=data['coord'].cpu().numpy(),
        color=pred_colors / 255.0,
        file_path=save_path,
    )
    
    print(f"추론 완료! 결과 저장됨: {save_path}")

if __name__ == "__main__":
    main()