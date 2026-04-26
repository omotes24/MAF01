#!/usr/bin/env bash
set -euo pipefail

: "${HOME_ROOT:=/home/omote}"
: "${OODD_REPO:=${HOME_ROOT}/OODD}"
: "${OODD_COMMIT:=edbb1a32e5fe81e443942156f0a9cafb0297d95b}"
: "${OODD_CHECKPOINT:=${OODD_REPO}/checkpoints/resnet50-0676ba61.pth}"
: "${OODD_REMOTE:=https://github.com/zxk1212/OODD.git}"
: "${RESNET50_URL:=https://download.pytorch.org/models/resnet50-0676ba61.pth}"

if [[ ! -d "${OODD_REPO}/.git" ]]; then
  git clone --depth 1 "${OODD_REMOTE}" "${OODD_REPO}"
fi

cd "${OODD_REPO}"
git fetch --depth 1 origin "${OODD_COMMIT}"
git checkout "${OODD_COMMIT}"

mkdir -p "$(dirname "${OODD_CHECKPOINT}")"
if [[ ! -f "${OODD_CHECKPOINT}" ]]; then
  curl -L -s -o "${OODD_CHECKPOINT}" "${RESNET50_URL}"
fi

echo "OODD_REPO=${OODD_REPO}"
echo "OODD_CHECKPOINT=${OODD_CHECKPOINT}"
echo "OODD_COMMIT=${OODD_COMMIT}"
