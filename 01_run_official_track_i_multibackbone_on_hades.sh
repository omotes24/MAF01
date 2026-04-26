#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${HOME_ROOT:=/home/omote}"
: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=/home/omote/.venv-maf-ood-v51}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${TMUX_SESSION:=maf-track1-multibackbone-gpu0}"
: "${DATA_SRC:=${HOME_ROOT}/WILD_DATA/splits}"
: "${SAVE_ROOT:=${HOME_ROOT}/260421_term1}"
: "${TRACK_I_PAIRS:=imagenet_vit=vit_base_patch16_224 resnet50=resnet50 swin_base=swin_base_patch4_window7_224}"
: "${OUTPUT_CSV:=${SAVE_ROOT}/official_track_i_results_multi_backbone.csv}"
: "${INSTALL_DEPS:=0}"
: "${REINSTALL_TORCH:=0}"
: "${TORCH_VERSION:=2.6.0}"
: "${TORCHVISION_VERSION:=0.21.0}"
: "${TORCH_WHL_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${EXTRA_PIP_PACKAGES:=numpy scipy scikit-learn pandas matplotlib seaborn timm open-clip-torch pillow faiss-cpu}"

mkdir -p "${SAVE_ROOT}" "${SAVE_ROOT}/logs"
: "${LOG_PATH:=${SAVE_ROOT}/logs/maf_track1_multibackbone_gpu${CUDA_VISIBLE_DEVICES}.log}"

RUN_CMD="cd '${PROJECT_DIR}' && \
if [[ ! -d '${VENV_DIR}' ]]; then ${PYTHON_BIN} -m venv '${VENV_DIR}'; fi && \
source '${VENV_DIR}/bin/activate'"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} && \
python -m pip install --upgrade pip setuptools wheel"
  if [[ "${REINSTALL_TORCH}" == "1" ]]; then
    RUN_CMD="${RUN_CMD} && \
pip install --force-reinstall --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_WHL_INDEX}"
  fi
  RUN_CMD="${RUN_CMD} && \
pip install ${EXTRA_PIP_PACKAGES}"
fi

RUN_CMD="${RUN_CMD} && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python official_repro/01_run_track_i_multibackbone.py \
--data-src '${DATA_SRC}' \
--save-root '${SAVE_ROOT}' \
--pairs ${TRACK_I_PAIRS} \
--output '${OUTPUT_CSV}' \
2>&1 | tee '${LOG_PATH}'"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session '${TMUX_SESSION}' already exists."
  echo "Attach with: tmux attach -t ${TMUX_SESSION}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash -lc \"${RUN_CMD}\""

echo "Started tmux session: ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "CUDA device : ${CUDA_VISIBLE_DEVICES}"
echo "Save root   : ${SAVE_ROOT}"
echo "Pairs       : ${TRACK_I_PAIRS}"
echo "Output CSV  : ${OUTPUT_CSV}"
echo "Log path    : ${LOG_PATH}"
