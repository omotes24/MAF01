#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash run_rival_repro_on_hades.sh "$@"
