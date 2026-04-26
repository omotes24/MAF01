#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${HOME_ROOT:=/home/omote}"
: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=${PROJECT_DIR}/.venv}"
: "${DATA_SRC:=${HOME_ROOT}/WILD_DATA/splits}"
: "${SAVE_ROOT:=${HOME_ROOT}/maf_ood_v51}"
: "${BACKBONES:=imagenet_vit bioclip}"
: "${TMUX_SESSION:=maf-ood}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${OFFICIAL_TRACK_I_CSV:=}"
: "${TORCH_VERSION:=2.6.0}"
: "${TORCHVISION_VERSION:=0.21.0}"
: "${TORCH_WHL_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${EXTRA_PIP_PACKAGES:=numpy scipy scikit-learn pandas matplotlib seaborn timm open-clip-torch pillow}"

RUN_CMD="cd '${PROJECT_DIR}' && \
${PYTHON_BIN} -m venv '${VENV_DIR}' && \
source '${VENV_DIR}/bin/activate' && \
python -m pip install --upgrade pip setuptools wheel && \
pip install --force-reinstall --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_WHL_INDEX} && \
pip install ${EXTRA_PIP_PACKAGES} && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python run_seed42_dualtrack.py --data-src '${DATA_SRC}' --save-root '${SAVE_ROOT}' --backbones ${BACKBONES}"

if [[ -n "${OFFICIAL_TRACK_I_CSV}" ]]; then
  RUN_CMD="${RUN_CMD} --official-track-i-csv '${OFFICIAL_TRACK_I_CSV}'"
fi

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
echo "Backbones   : ${BACKBONES}"
if [[ -n "${OFFICIAL_TRACK_I_CSV}" ]]; then
  echo "Track I CSV : ${OFFICIAL_TRACK_I_CSV}"
fi
