#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash SID-MLP/scripts/build_valid_set.sh <dataset_tag>

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_tag>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SID_MLP_DIR="${SID_MLP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SID_MLP_DIR}/.." && pwd)}"
PY="${PY:-python}"
GRAPH_DIR="${GRAPH_DIR:-${SID_MLP_GRAPH_DIR:-${PROJECT_DIR}/Graph}}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SID_MLP_GRAPH_DIR="${GRAPH_DIR}"

"${PY}" -m sid_mlp.build_valid_set \
    --graph_dir "${GRAPH_DIR}" \
    --dataset_tag "$1"
