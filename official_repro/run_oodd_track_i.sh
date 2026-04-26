#!/usr/bin/env bash
set -euo pipefail

: "${OODD_REPO:?Set OODD_REPO to the cloned OODD repository path}"
: "${DATA_SRC:?Set DATA_SRC to ~/WILD_DATA/splits}"
: "${LIST_ROOT:?Set LIST_ROOT to the generated official_inputs/oodd path}"
: "${OUT_ROOT:?Set OUT_ROOT to an output directory}"
: "${OODD_CHECKPOINT:?Set OODD_CHECKPOINT to the official OODD/ImageNet checkpoint path}"
: "${NUM_WORKERS:=4}"
: "${CUDA_VISIBLE_DEVICES:=0}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_ROOT="${OUT_ROOT}/oodd_configs"
RESULT_ROOT="${OUT_ROOT}/oodd_run"

python "${PROJECT_DIR}/official_repro/make_oodd_wild_configs.py" \
  --data-src "${DATA_SRC}" \
  --list-root "${LIST_ROOT}" \
  --out-root "${CONFIG_ROOT}"

cd "${OODD_REPO}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python main.py \
  --config "${CONFIG_ROOT}/wild_id.yml" \
  "${CONFIG_ROOT}/wild_ood.yml" \
  configs/networks/resnet50.yml \
  configs/pipelines/test/test_ood.yml \
  configs/preprocessors/base_preprocessor.yml \
  configs/postprocessors/knn.yml \
  --output_dir "${RESULT_ROOT}" \
  --num_workers "${NUM_WORKERS}" \
  --ood_dataset.image_size 256 \
  --dataset.test.batch_size 256 \
  --dataset.val.batch_size 256 \
  --network.pretrained True \
  --network.checkpoint "${OODD_CHECKPOINT}" \
  --postprocessor.postprocessor_args.K1 100 \
  --postprocessor.postprocessor_args.K2 5 \
  --postprocessor.postprocessor_args.ALPHA 0.5 \
  --postprocessor.postprocessor_args.queue_size 2048 \
  --merge_option merge

OOD_CSV="$(find "${RESULT_ROOT}" -name ood.csv | sort | tail -n 1)"
if [[ -z "${OOD_CSV}" ]]; then
  echo "ood.csv not found under ${RESULT_ROOT}" >&2
  exit 1
fi

python "${PROJECT_DIR}/official_repro/collect_oodd_track_i.py" \
  --input-csv "${OOD_CSV}" \
  --scores-dir "$(dirname "${OOD_CSV}")/scores" \
  --output-csv "${OUT_ROOT}/summary_import.csv"

echo "OODD raw csv   : ${OOD_CSV}"
echo "OODD import csv: ${OUT_ROOT}/summary_import.csv"
