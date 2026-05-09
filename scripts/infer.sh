#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash SID-MLP/scripts/infer.sh sidmlp <category> <gpu> <lens_path> [split]
#   bash SID-MLP/scripts/infer.sh sidmlp-pp <category> <gpu> <lens_path> <encoder_ckpt> <tag_label> [split]

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sidmlp|sidmlp-pp> ..." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SID_MLP_DIR="${SID_MLP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SID_MLP_DIR}/.." && pwd)}"
PY="${PY:-python}"
DATASET="${DATASET:-AmazonReviews2023}"
GRAPH_DIR="${GRAPH_DIR:-${SID_MLP_GRAPH_DIR:-${PROJECT_DIR}/Graph}}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/downloaded_models}"
SEM_ID_DIR="${SEM_ID_DIR:-${MODEL_DIR}/semantic_ids}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${GRAPH_DIR}/SID-MLP/ckpt}"
MODE="$1"
shift

cd "${PROJECT_DIR}"
export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SID_MLP_GRAPH_DIR="${GRAPH_DIR}"

run_infer() {
    local category="$1"
    local gpu_id="$2"
    local lens_path="$3"
    local dataset_tag="$4"
    local split="$5"
    shift 5

    local tiger_ckpt="${TIGER_CKPT:-${MODEL_DIR}/TIGER-${DATASET}-category_${category}.pth}"
    local sem_ids="${SEM_IDS:-${SEM_ID_DIR}/${DATASET}-${category}_sentence-t5-base_256,256,256,256.sem_ids}"
    local valid_set="${VALID_SET:-${BASE_CKPT_DIR}/valid_item_set_${dataset_tag}.pt}"
    local config="${CONFIG:-${SID_MLP_DIR}/configs/infer.yaml}"
    local fp32_args=()

    if [[ "${FP32:-0}" == "1" ]]; then
        fp32_args+=(--fp32)
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -m sid_mlp.cli.infer \
        --config "${config}" \
        --checkpoint "${tiger_ckpt}" \
        --dataset "${DATASET}" \
        --category "${category}" \
        --dataset_tag "${dataset_tag}" \
        --lens_path "${lens_path}" \
        --valid_set "${valid_set}" \
        --sem_ids "${sem_ids}" \
        --graph_dir "${GRAPH_DIR}" \
        --batch_size "${BATCH_SIZE:-32}" \
        --split "${split}" \
        "${fp32_args[@]}" \
        "$@"
}

case "${MODE}" in
    sidmlp)
        if [[ $# -lt 3 ]]; then
            echo "Usage: $0 sidmlp <category> <gpu> <lens_path> [split]" >&2
            exit 1
        fi
        category="$1"
        gpu_id="$2"
        lens_path="$3"
        split="test"
        shift 3
        if [[ $# -gt 0 && "$1" != --* ]]; then
            split="$1"
            shift
        fi
        run_infer "${category}" "${gpu_id}" "${lens_path}" "${DATASET_TAG:-${DATASET}_${category}}" "${split}" "$@"
        ;;
    sidmlp-pp)
        if [[ $# -lt 5 ]]; then
            echo "Usage: $0 sidmlp-pp <category> <gpu> <lens_path> <encoder_ckpt> <tag_label> [split]" >&2
            exit 1
        fi
        category="$1"
        gpu_id="$2"
        lens_path="$3"
        encoder_ckpt="$4"
        tag_label="$5"
        split="test"
        shift 5
        if [[ $# -gt 0 && "$1" != --* ]]; then
            split="$1"
            shift
        fi
        run_infer "${category}" "${gpu_id}" "${lens_path}" "${DATASET_TAG:-${DATASET}_${category}_${tag_label}}" "${split}" \
            --encoder_ckpt "${encoder_ckpt}" "$@"
        ;;
    *)
        echo "Unknown infer mode: ${MODE}" >&2
        exit 1
        ;;
esac
