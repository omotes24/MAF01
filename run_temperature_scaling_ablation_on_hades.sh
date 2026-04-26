#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${HOME_ROOT:=/home/omote}"
: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=/home/omote/.venv-maf-ood-v51}"
: "${DATA_SRC:=${HOME_ROOT}/WILD_DATA/splits}"
: "${SAVE_ROOT:=${HOME_ROOT}/260422_temperature_ablation}"
: "${ARTIFACT_ROOT:=${HOME_ROOT}/maf_ood_v51}"
: "${BACKBONES:=imagenet_vit openai_clip_b16 bioclip}"
: "${SEEDS:=42 123 456}"
: "${CUDA_VISIBLE_DEVICES:=2}"
: "${TMUX_SESSION:=maf-temperature-ablation-gpu2}"
: "${OUTPUT_SUBDIR:=temperature_scaling_ablation}"
: "${TEMPERATURE_GRID:=0.25 0.5 1.0 2.0 4.0}"
: "${TEMPERATURE_SCHEMES:=raw sqrt_dim}"
: "${DISTANCE_MODE:=mah_t}"
: "${EVAL_ONLY:=1}"
: "${FORCE_REEXTRACT:=0}"
: "${SKIP_MISSING:=1}"
: "${INSTALL_DEPS:=0}"
: "${TORCH_VERSION:=2.6.0}"
: "${TORCHVISION_VERSION:=0.21.0}"
: "${TORCH_WHL_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${EXTRA_PIP_PACKAGES:=numpy scipy scikit-learn pandas matplotlib seaborn timm open-clip-torch pillow}"

mkdir -p "${SAVE_ROOT}" "${SAVE_ROOT}/logs"
: "${LOG_PATH:=${SAVE_ROOT}/logs/maf_temperature_ablation_gpu${CUDA_VISIBLE_DEVICES}.log}"

RUN_CMD="cd '${PROJECT_DIR}' && \
if [[ ! -d '${VENV_DIR}' ]]; then ${PYTHON_BIN} -m venv '${VENV_DIR}'; fi && \
source '${VENV_DIR}/bin/activate'"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} && \
python -m pip install --upgrade pip setuptools wheel && \
pip install --force-reinstall --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_WHL_INDEX} && \
pip install ${EXTRA_PIP_PACKAGES}"
fi

RUN_CMD="${RUN_CMD} && \
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' python run_temperature_scaling_ablation.py \
--data-src '${DATA_SRC}' \
--save-root '${SAVE_ROOT}' \
--artifact-root '${ARTIFACT_ROOT}' \
--backbones ${BACKBONES} \
--seeds ${SEEDS} \
--output-subdir '${OUTPUT_SUBDIR}' \
--temperature-grid ${TEMPERATURE_GRID} \
--temperature-schemes ${TEMPERATURE_SCHEMES} \
--distance-mode '${DISTANCE_MODE}'"

if [[ "${EVAL_ONLY}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} --eval-only"
fi

if [[ "${FORCE_REEXTRACT}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} --force-reextract"
fi

if [[ "${SKIP_MISSING}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} --skip-missing"
fi

RUN_CMD="${RUN_CMD} 2>&1 | tee '${LOG_PATH}'"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session '${TMUX_SESSION}' already exists."
  echo "Attach with: tmux attach -t ${TMUX_SESSION}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash -lc \"${RUN_CMD}\""

echo "Started tmux session: ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
echo "CUDA device : ${CUDA_VISIBLE_DEVICES}"
echo "Artifact dir: ${ARTIFACT_ROOT}"
echo "Save root   : ${SAVE_ROOT}"
echo "Backbones   : ${BACKBONES}"
echo "Seeds       : ${SEEDS}"
echo "Temp grid   : ${TEMPERATURE_GRID}"
echo "Temp scheme : ${TEMPERATURE_SCHEMES}"
echo "Distance    : ${DISTANCE_MODE}"
echo "Output dir  : ${SAVE_ROOT}/${OUTPUT_SUBDIR}"
echo "Log path    : ${LOG_PATH}"
