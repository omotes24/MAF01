#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${HOME_ROOT:=/home/omote}"
: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=${PROJECT_DIR}/.venv}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${TMUX_SESSION:=maf-ood-track1}"
: "${DATA_SRC:=${HOME_ROOT}/WILD_DATA/splits}"
: "${SAVE_ROOT:=${HOME_ROOT}/maf_ood_v51}"
: "${LIST_ROOT:=${SAVE_ROOT}/official_inputs}"
: "${VIT_TIMM_MODEL:=vit_base_patch16_224}"
: "${OODD_REPO:=${HOME_ROOT}/OODD}"
: "${OODD_CHECKPOINT:=${OODD_REPO}/checkpoints/resnet50-0676ba61.pth}"
: "${TORCH_VERSION:=2.6.0}"
: "${TORCHVISION_VERSION:=0.21.0}"
: "${TORCH_WHL_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${EXTRA_PIP_PACKAGES:=numpy<2 scipy scikit-learn pandas matplotlib seaborn timm open-clip-torch pillow faiss-cpu opencv-python-headless imgaug}"

RUN_CMD="cd '${PROJECT_DIR}' && \
${PYTHON_BIN} -m venv '${VENV_DIR}' && \
source '${VENV_DIR}/bin/activate' && \
python -m pip install --upgrade pip setuptools wheel && \
pip install --force-reinstall --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_WHL_INDEX} && \
pip install ${EXTRA_PIP_PACKAGES} && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python official_repro/prepare_wild_lists.py --data-src '${DATA_SRC}' --out-root '${LIST_ROOT}' && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python official_repro/run_vim_gen_track_i.py --data-src '${DATA_SRC}' --list-root '${LIST_ROOT}/vim_gen' --save-root '${SAVE_ROOT}/official_runs/vim_gen' --methods ViM GEN --model '${VIT_TIMM_MODEL}' && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python official_repro/run_knn_track_i_wild.py --data-src '${DATA_SRC}' --save-root '${SAVE_ROOT}/official_runs/knn' --model '${VIT_TIMM_MODEL}' && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' OODD_REPO='${OODD_REPO}' DATA_SRC='${DATA_SRC}' LIST_ROOT='${LIST_ROOT}/oodd' OUT_ROOT='${SAVE_ROOT}/official_runs/oodd' OODD_CHECKPOINT='${OODD_CHECKPOINT}' bash official_repro/run_oodd_track_i.sh && \
python official_repro/combine_official_track_i.py --inputs '${SAVE_ROOT}/official_runs/vim_gen/summary_import.csv' '${SAVE_ROOT}/official_runs/knn/summary_import.csv' '${SAVE_ROOT}/official_runs/oodd/summary_import.csv' --output '${SAVE_ROOT}/official_track_i_results.csv'"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session '${TMUX_SESSION}' already exists."
  echo "Attach with: tmux attach -t ${TMUX_SESSION}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash -lc \"${RUN_CMD}\""

echo "Started tmux session: ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "CUDA device : ${CUDA_VISIBLE_DEVICES}"
echo "Torch wheel : torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} @ ${TORCH_WHL_INDEX}"
echo "Data source : ${DATA_SRC}"
echo "Save root   : ${SAVE_ROOT}"
echo "TIMM_MODEL  : ${VIT_TIMM_MODEL}"
echo "OODD_REPO   : ${OODD_REPO}"
echo "OODD_CKPT   : ${OODD_CHECKPOINT}"
