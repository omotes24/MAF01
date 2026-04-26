#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash run_temperature_scaling_ablation_on_hades.sh "$@"
