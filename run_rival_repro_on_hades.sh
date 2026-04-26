#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

cd /home/omote/MAF-OOD-v51
source /home/omote/.venv-maf-ood-v51/bin/activate

python run_rival_repro_comparison.py \
  --artifact-root /home/omote/maf_ood_v51 \
  --output-root /home/omote/maf_ood_v51/rival_repro_comparison \
  --backbones dinov2_vitb14 imagenet_vit openai_clip_b16 bioclip \
  --seeds 42 123 456
