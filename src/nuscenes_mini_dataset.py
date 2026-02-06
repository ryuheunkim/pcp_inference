import os
import sys

# 프로젝트 루트 경로 추가 (src/_init_pointcept_path.py를 부르기 위함)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

import _init_pointcept_path # Pointcept 경로 연결

from pointcept.datasets import NuScenesDataset
from pointcept.datasets.builder import DATASETS

@DATASETS.register_module()
class NuScenesMiniDataset(NuScenesDataset):
    def get_info_path(self, split):
        # "mini" split이 들어오면 우리가 만든 pkl 파일 경로 반환
        if split == "mini":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_mini.pkl"
            )
        return super().get_info_path(split)