import argparse
import os
import numpy as np
import torch
from collections import OrderedDict
from tqdm import tqdm

import _init_pointcept_path

# Pointcept 라이브러리 임포트
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.visualization import save_point_cloud
from pointcept.datasets.nuscenes import NuScenesDataset
from pointcept.datasets.builder import DATASETS

# nuScenes 설정
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

# -------------------------------------------------------------------------
# 정답 파일이 없어도 죽지 않는 데이터셋 클래스 (RobustDataset)
# -------------------------------------------------------------------------
@DATASETS.register_module()
class NuScenesRobustDataset(NuScenesDataset):
    def get_data(self, idx):
        # 1. 메타데이터 가져오기
        data = self.data_list[idx % len(self.data_list)]
        
        # 2. 포인트 클라우드 로드
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        if not os.path.exists(lidar_path):
             lidar_path_no_raw = os.path.join(self.data_root, data["lidar_path"])
             if os.path.exists(lidar_path_no_raw):
                 lidar_path = lidar_path_no_raw
        
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255

        # 3. 정답 라벨(GT) 로드 (없으면 Dummy)
        segment = None
        if "gt_segment_path" in data.keys():
            gt_path = os.path.join(self.data_root, "raw", data["gt_segment_path"])
            if os.path.exists(gt_path):
                segment = np.fromfile(str(gt_path), dtype=np.uint8, count=-1).reshape([-1])
                segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int64)
        
        if segment is None:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

# -------------------------------------------------------------------------

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
    print("--- [Step 1] Arguments Parsed ---") #### 추가
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    args = parser.parse_args()

    print("--- [Step 2] Config Loaded ---") #### 추가
    cfg = Config.fromfile(args.config_file)
    
    # -------------------------------------------------------------------------
    # [수정됨] 데이터 설정 (feat_keys에 "coord" 추가!)
    # -------------------------------------------------------------------------
    data_config = dict(
        type="NuScenesRobustDataset",
        split="val",
        data_root="data/nuscenes",
        transform=[
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(
                type="Collect", 
                keys=("coord", "grid_coord", "segment", "name"), 
                # [중요] 모델이 4채널(coord:3 + strength:1)을 기대하므로 coord를 추가해야 함
                feat_keys=("coord", "strength") 
            ),
        ],
        test_mode=False,
    )
    print("--- [Step 3] Building Model (This might take time) ---") #### 추가   
    model = build_model(cfg.model).cuda()
    print("--- [Step 4] Model Built. Loading Weights... ---") #### 추가
    checkpoint = torch.load(args.weight, map_location="cuda")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 데이터셋 로드
    print("--- [Step 5] Weights Loaded. Building Dataset... ---") #### 추가
    print(">>> 데이터셋 로드 및 필터링 준비...")
    dataset = build_dataset(data_config)
    
    # 필터링 로직 (파일이 존재하는 것만 남김)
    valid_data_list = []
    for item in tqdm(dataset.data_list, desc="포인트 클라우드 파일 검사"):
        path1 = os.path.join(dataset.data_root, "raw", item["lidar_path"])
        path2 = os.path.join(dataset.data_root, item["lidar_path"]) 
        if os.path.exists(path1) or os.path.exists(path2):
            valid_data_list.append(item)
    
    dataset.data_list = valid_data_list
    num_samples = len(dataset.data_list)
    print(f">>> 추론 대상: {num_samples}개")
    
    if num_samples == 0:
        print("[오류] 포인트 클라우드 파일을 찾을 수 없습니다.")
        return

    # 추론 시작
    total_intersection = np.zeros(16)
    total_union = np.zeros(16)
    save_dir = "test/nuscenes_inference"
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(range(num_samples), desc="추론 진행 중"):
        try:
            data = dataset[idx] 
        except Exception as e:
            print(f"\n[Error] {e}")
            continue

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
    print(f" nuScenes 평가 결과 (정답 파일 없으면 0%) ")
    print("="*40)
    for i, class_name in enumerate(NUSCENES_NAMES):
        print(f"{class_name:<20}: {iou_per_class[i]*100:>6.2f}%")
    print("-" * 40)
    print(f"평균 mIoU: {miou * 100:.2f}%")
    print(f"결과 저장 경로: {save_dir}")
    print("="*40)

if __name__ == "__main__":
    main()