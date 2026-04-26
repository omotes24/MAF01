#!/usr/bin/env bash
set -euo pipefail

# Replace these before use, or pass them as env vars.
: "${HADES_HOST:?Set HADES_HOST, e.g. hades.example.ac.jp or your ssh alias}"
: "${HADES_USER:=$USER}"
: "${REMOTE_DIR:=/home/omote/MAF-OOD-v51}"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${HADES_USER}@${HADES_HOST}"

echo "Uploading ${LOCAL_DIR} -> ${TARGET}:${REMOTE_DIR}"
ssh "${TARGET}" "mkdir -p '${REMOTE_DIR}'"
rsync -av \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${LOCAL_DIR}/" "${TARGET}:${REMOTE_DIR}/"

echo "Done."
echo "Next:"
echo "  ssh ${TARGET}"
echo "  cd ${REMOTE_DIR}"
echo "  bash run_on_hades.sh"
