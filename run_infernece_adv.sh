#!/bin/sh

# 1. 스크립트 실행 위치를 '프로젝트 루트'로 이동
cd $(dirname $(dirname "$0")) || exit

# ==============================================================================
# 2. Pointcept 라이브러리 경로 설정 (핵심 변경 사항)
# ==============================================================================
# 루트 기준, Pointcept 외부 라이브러리가 위치한 경로
POINTCEPT_ROOT="extern/pointcept"

PYTHON=python
INF_CODE=inference.py  # 실행할 파일 이름 (경로는 아래에서 POINTCEPT_ROOT와 합쳐짐)

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

# ==============================================================================
# 3. 인자 파싱 (이전과 동일)
# ==============================================================================
while getopts "p:d:c:n:w:g:m:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_NAME=$OPTARG ;;
    w) WEIGHT=$OPTARG ;;
    g) NUM_GPU=$OPTARG ;;
    m) NUM_MACHINE=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" ;;
  esac
done

# ==============================================================================
# 4. 환경 설정 및 경로 재구성 (extern/pointcept 반영)
# ==============================================================================

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "GPU Num: $NUM_GPU"

# SLURM 설정 (기존 로직 유지)
if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

# ------------------------------------------------------------------
# [중요] 경로 수정: 모든 데이터/설정 경로는 extern/pointcept 내부를 가리킴
# ------------------------------------------------------------------

# 실험 결과(exp) 폴더가 extern/pointcept/exp/... 에 있다고 가정
EXP_DIR=${POINTCEPT_ROOT}/exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CONFIG_DIR=${EXP_DIR}/config.py

if [ "${CONFIG}" = "None" ]
then
    # Config 인자가 없으면 exp 폴더 내의 config.py 사용
    CONFIG_DIR=${EXP_DIR}/config.py
else
    # Config 인자가 있으면 extern/pointcept/configs/... 사용
    CONFIG_DIR=${POINTCEPT_ROOT}/configs/${DATASET}/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR

# ------------------------------------------------------------------
# [중요] PYTHONPATH 설정
# 파이썬이 'import pointcept'를 수행할 때 extern/pointcept 폴더를 참조하도록 설정
# ------------------------------------------------------------------
export PYTHONPATH=./${POINTCEPT_ROOT}
echo "Running code in: ${POINTCEPT_ROOT}/tools/$TEST_CODE"


# ==============================================================================
# 5. 추론(Inference) 실행
# ==============================================================================
echo " =========> RUN INFERENCE <========="
ulimit -n 65536

# 실행할 파이썬 스크립트 경로도 POINTCEPT_ROOT 아래로 지정
$PYTHON -u ${POINTCEPT_ROOT}/tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$NUM_GPU" \
  --num-machines "$NUM_MACHINE" \
  --machine-rank ${SLURM_NODEID:-0} \
  --dist-url ${DIST_URL} \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth