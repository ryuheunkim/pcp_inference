import os
import numpy as np
import torch
import torch.nn.functional as F
import _init_pointcept_path

from collections import OrderedDict
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.utils.config import Config
from pointcept.utils.visualization import save_point_cloud
from pointcept.utils.comm import get_world_size
from pointcept.datasets.utils import point_collate_fn

from tensorboardX import SummaryWriter

POINTCEPT_PATH = os.path.join('extern', 'pointcept')

VERSION = "s3dis" 

if VERSION == "nuscenes": # regsitry 데이터베이스에서 못찾음 이슈 / 
    type = "NuScenesMiniDataset" # 되긴 되는데 부모가 없다고 뜸, 데이터베이스엔 있는듯 v1.0-mini 나 v1.0-trainvald이나...
    split = "mini"
    data_root = POINTCEPT_PATH+"/data/nuscenes"
    CONFIGFILE = POINTCEPT_PATH+"/exp/nuscenes/nuscenes-semseg-pt-v3m1-0-base/config.py"
    WEIGHTFILE = POINTCEPT_PATH+"/exp/nuscenes/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth"
elif VERSION == "s3dis": # 메모리 뻗음 이슈(RTX4050...)
    type = "S3DISDataset"
    split = "test" # 아마, 폴더 하나에 방 전체 포인트들이 들어있어서, 너무 커서 메모리가 부족한 걸 꺼...
    data_root = POINTCEPT_PATH+"/data/s3dis"
    CONFIGFILE = POINTCEPT_PATH+"/exp/s3dis/s3dis-semseg-pt-v3m1-1-ppt-extreme/config.py"
    WEIGHTFILE = POINTCEPT_PATH+"/exp/s3dis/s3dis-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth"
elif VERSION == "scannet":
    type="ScanNetDataset"
    split="train"
    data_root=POINTCEPT_PATH+"/data/scannet"
    CONFIGFILE = POINTCEPT_PATH+"/exp/scannet/semseg-pt-v3m1-0-base/config.py"
    WEIGHTFILE = POINTCEPT_PATH+"/exp/scannet/semseg-pt-v3m1-0-base/model/model_last.pth"

data_config = dict(
    type=type,
    split=split,
    data_root=data_root,
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
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

cfg = Config.fromfile(CONFIGFILE)

checkpoint = WEIGHTFILE
keywords = "backbone." # module로 바꿔야 하는 거 아닌지?
replacement = ""
idx = 300

if __name__ == "__main__":
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(
        checkpoint, map_location=lambda storage, loc: storage.cuda()
    )
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("module."):
            key = "module." + key  # xxx.xxx -> module.xxx.xxx
        if keywords in key:
            key = key.replace(keywords, replacement)
        if get_world_size() == 1:
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
        weight[key] = value
    load_state_info = model.load_state_dict(weight, strict=False)
    print(load_state_info)
    model.eval() # 평가 모델로 전환 (추가된 거임...)
    dataset = build_dataset(data_config)
    data = dataset[idx]
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda(non_blocking=True)
    # [수정 2] 모델 전체가 아닌 'backbone'만 통과시켜 Feature 추출
    # model(data) -> 분류 결과(Logits) 또는 Loss 반환
    # model.backbone(data) -> 특징(Feature) 반환
    with torch.no_grad():
        output = model.backbone(data)
    # [수정 3] output 타입에 따른 안전한 feat 접근
    if isinstance(output, dict):
        feat = output["feat"]  # 딕셔너리로 반환된 경우
    else:
        feat = output.feat     # 객체(Point 등)로 반환된 경우
    # output = model(data)기존 코드
    # feat = output.feat # 모델이 학습 상태였기에 오류가 발생하였음 ... s3dis 모델은 왜 작동된 거지?
    feat = feat - feat.mean(dim=-2, keepdim=True)
    u, s, v = torch.pca_lowrank(feat, center=False, q=3)
    projection = feat @ v
    min_val = projection.min(dim=0, keepdim=True)[0]
    max_val = projection.max(dim=0, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    pca_color = (projection - min_val) / div
    os.makedirs("test/vis_pca", exist_ok=True)
    save_point_cloud(
        coord=output.coord,
        color=pca_color,
        file_path=f"test/vis_pca/pca_color{idx}.ply",
    )
    vertices = output.coord.unsqueeze(0).clone().detach().cpu().numpy() / 3
    colors = pca_color.unsqueeze(0).clone().detach().cpu().numpy() * 255

    writer = SummaryWriter("test/vis_pca")
    writer.add_mesh(
        f"train/{data['name']}",
        vertices=vertices,
        colors=colors,
        global_step=1,
    )