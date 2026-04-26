#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash run_multiseed_adaptive_on_hades.sh "$@"
