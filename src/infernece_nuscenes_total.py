import argparse
import os
import numpy as np
import torch
import pickle
from collections import OrderedDict
from tqdm import tqdm

import _init_pointcept_path

from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.visualization import save_point_cloud
# 기존 Dataset 클래스 임포트
from pointcept.datasets.nuscenes import NuScenesDataset
from pointcept.datasets.builder import DATASETS

# -------------------------------------------------------------------------
# [핵심] Config 설정을 받아들이기 위한 커스텀 데이터셋
# split="mini"를 입력받으면, 방금 만든 mini.pkl을 로드하도록 수정
# -------------------------------------------------------------------------
@DATASETS.register_module()
class NuScenesMiniDataset(NuScenesDataset):
    def get_info_path(self, split):
        # "mini" 라는 split이 들어오면 우리가 만든 파일 경로 반환
        if split == "mini":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_mini.pkl"
            )
        # 나머지는 기존 로직(train/val/test) 그대로 사용
        return super().get_info_path(split)

# -------------------------------------------------------------------------

NUSCENES_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation"
]

NUSCENES_COLOR_MAP = np.array([
    [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [200, 180, 0], [255, 0, 0], [255, 240, 150],
    [135, 60, 0], [160, 32, 240], [255, 0, 255], [139, 137, 137],
    [75, 0, 75], [150, 240, 80], [230, 230, 250], [0, 175, 0],
])

def calculate_view_stats(pred, gt, num_classes=16, ignore_index=-1):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for i in range(num_classes):
        mask_pred = (pred == i)
        mask_gt = (gt == i)
        intersection[i] = np.sum(mask_pred & mask_gt & (gt != ignore_index))
        union[i] = np.sum((mask_pred | mask_gt) & (gt != ignore_index))
    return intersection, union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    
    # -------------------------------------------------------------------------
    # [핵심] 데이터 설정 변경
    # type: 커스텀 클래스 (NuScenesMiniDataset)
    # split: "mini" (새로 만든 pkl 파일을 가리킴)
    # -------------------------------------------------------------------------
    data_config = dict(
        type="NuScenesMiniDataset",  # 우리가 정의한 클래스 사용
        split="mini",                # 여기서 'mini'를 호출!
        data_root="data/nuscenes",
        transform=[
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=("coord", "grid_coord", "segment", "name"), 
                feat_keys=("strength",) 
            ),
        ],
        test_mode=False,
    )

    # 모델 빌드 및 로드
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(args.weight, map_location="cuda")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 데이터셋 로드
    print(">>> Mini 데이터셋 로드 시작...")
    dataset = build_dataset(data_config)
    num_samples = len(dataset)
    
    # 검증
    if num_samples == 0:
        print("[오류] 데이터셋이 비어 있습니다. make_mini_info.py를 먼저 실행했는지 확인하세요.")
        return
    
    print(f">>> 성공! 총 {num_samples}개의 유효한 샘플로 추론을 시작합니다.")

    total_intersection = np.zeros(16)
    total_union = np.zeros(16)
    save_dir = "test/nuscenes_inference"
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(range(num_samples), desc="추론 진행 중"):
        data = dataset[idx]
        gt_labels = data['segment'].numpy()

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data)
            pred_labels = output['seg_logits'].argmax(-1).cpu().numpy()

        inter, uni = calculate_view_stats(pred_labels, gt_labels, num_classes=16)
        total_intersection += inter
        total_union += uni

        pred_colors = NUSCENES_COLOR_MAP[pred_labels % 16]
        save_point_cloud(
            coord=data['coord'].cpu().numpy(),
            color=pred_colors / 255.0,
            file_path=f"{save_dir}/pred_{data['name']}.ply",
        )

    # 결과 출력
    iou_per_class = total_intersection / (total_union + 1e-6)
    miou = np.nanmean(iou_per_class)

    print("\n" + "="*40)
    print(f" nuScenes (Mini) 평가 결과 (mIoU) ")
    print("="*40)
    for i, class_name in enumerate(NUSCENES_NAMES):
        print(f"{class_name:<20}: {iou_per_class[i]*100:>6.2f}%")
    print("-" * 40)
    print(f"평균 mIoU: {miou * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()