#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SID_MLP_DIR="${SID_MLP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SID_MLP_DIR}/.." && pwd)}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <AmazonReviews2023 category> [gpu_id]" >&2
  exit 1
fi

CATEGORY="$1"
shift
GPU_ID="0"
if [[ $# -gt 0 && "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
export PROJECT_DIR
export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${SID_MLP_DIR}/main.py" \
  --model=TIGER \
  --dataset=AmazonReviews2023 \
  --category="${CATEGORY}" \
  --cache_dir="${CACHE_DIR:-cache}" \
  --log_dir="${LOG_DIR:-logs}" \
  --tensorboard_log_dir="${TENSORBOARD_DIR:-tensorboard}" \
  --ckpt_dir="${CKPT_DIR:-ckpt}" \
  "$@"
