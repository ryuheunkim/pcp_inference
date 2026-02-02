#!/bin/bash

# =========================================================
# 사용법:
# ./run_inference.sh [Config경로] [가중치경로]
#
# 인자를 입력하지 않으면 아래 설정된 기본값을 사용합니다.
# =========================================================

# 1. 기본값 설정 (자주 쓰는 경로를 여기에 적어두세요)
DEFAULT_CONFIG="extern/pointcept/configs/nuscenes/semseg-pt-v3m1-0-base.py"
DEFAULT_WEIGHT="extern/pointcept/exp/nuscenes/semseg-pt-v3m1-0-base/model/model_best.pth"

# 2. 인자 처리 (입력값이 있으면 쓰고, 없으면 기본값 사용)
# $1: 첫 번째 인자, $2: 두 번째 인자
CONFIG_PATH=${1:-$DEFAULT_CONFIG}
WEIGHT_PATH=${2:-$DEFAULT_WEIGHT}

echo "======================================================="
echo " [Run Inference Script] "
echo " - Config : $CONFIG_PATH"
echo " - Weight : $WEIGHT_PATH"
echo "======================================================="

# 3. Python 실행
# src 폴더의 inference.py를 실행하며 인자를 전달합니다.
python src/inference.py \
    --config-file "$CONFIG_PATH" \
    --weight "$WEIGHT_PATH"

echo "Done."