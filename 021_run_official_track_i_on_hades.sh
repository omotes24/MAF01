#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash run_official_track_i_on_hades.sh "$@"
