#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec bash 01_run_term1_foundation_on_hades.sh "$@"
