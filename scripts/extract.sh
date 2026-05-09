#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash SID-MLP/scripts/extract.sh teacher <category> <train|val|test> [gpu]
#   bash SID-MLP/scripts/extract.sh raw <category> <train|val|test> [gpu]
#   bash SID-MLP/scripts/extract.sh sidmlp-pp <category> <train|val|test> <gpu> <encoder_ckpt> <layers> <ffn_dim> <tag_label>

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <teacher|raw|sidmlp-pp> ..." >&2
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
MODE="$1"
shift
export SID_MLP_GRAPH_DIR="${GRAPH_DIR}"

split_suffix() {
    local split="$1"
    if [[ "${split}" == "val" ]]; then
        echo "_val"
    elif [[ "${split}" == "test" ]]; then
        echo "_test"
    else
        echo ""
    fi
}

run_teacher_or_raw() {
    local kind="$1"
    if [[ $# -lt 3 ]]; then
        echo "Usage: $0 ${kind} <category> <train|val|test> [gpu]" >&2
        exit 1
    fi
    local category="$2"
    local split="$3"
    local gpu_id="${4:-0}"
    local checkpoint="${CHECKPOINT:-${MODEL_DIR}/TIGER-${DATASET}-category_${category}.pth}"
    local sem_ids="${SEM_IDS:-${SEM_ID_DIR}/${DATASET}-${category}_sentence-t5-base_256,256,256,256.sem_ids}"

    cd "${PROJECT_DIR}"
    export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

    if [[ "${kind}" == "teacher" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -m sid_mlp.extract \
            --checkpoint "${checkpoint}" \
            --category "${category}" \
            --dataset "${DATASET}" \
            --split "${split}" \
            --mode all \
            --sem_ids "${sem_ids}" \
            --graph_dir "${GRAPH_DIR}"
    else
        local suffix
        suffix="$(split_suffix "${split}")"
        local output_dir="${OUTPUT_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_rawemb${suffix}}"
        mkdir -p "${output_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -m sid_mlp.extract \
            --checkpoint "${checkpoint}" \
            --category "${category}" \
            --dataset "${DATASET}" \
            --split "${split}" \
            --mode encoder_sequences \
            --raw_embeddings \
            --sem_ids "${sem_ids}" \
            --graph_dir "${GRAPH_DIR}" \
            --output_dir "${output_dir}"
    fi
}

run_sidmlp_pp() {
    if [[ $# -lt 7 ]]; then
        echo "Usage: $0 sidmlp-pp <category> <train|val|test> <gpu> <encoder_ckpt> <layers> <ffn_dim> <tag_label>" >&2
        exit 1
    fi
    local category="$1"
    local split="$2"
    local gpu_id="$3"
    local encoder_ckpt="$4"
    local layers="$5"
    local ffn_dim="$6"
    local tag_label="$7"
    local suffix
    suffix="$(split_suffix "${split}")"

    local raw_dir="${RAW_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_rawemb${suffix}}"
    local output_dir="${OUTPUT_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_${tag_label}${suffix}}"
    local logits_src="${GRAPH_DIR}/logits/${DATASET}_${category}${suffix}"
    local logits_dst="${GRAPH_DIR}/logits/${DATASET}_${category}_${tag_label}${suffix}"

    cd "${PROJECT_DIR}"
    export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p "${output_dir}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -m sid_mlp.cli.train sidmlp-pp-stage1 --extract \
        --raw_dir "${raw_dir}" \
        --d_model "${D_MODEL:-128}" \
        --ffn_dim "${ffn_dim}" \
        --num_layers "${layers}" \
        --dropout "${STAGE1_DROPOUT:-0.1}" \
        --encoder_ckpt "${encoder_ckpt}" \
        --extract_out "${output_dir}"

    if [[ -d "${logits_src}" && ! -e "${logits_dst}" ]]; then
        mkdir -p "$(dirname "${logits_dst}")"
        ln -sfn "${logits_src}" "${logits_dst}"
        echo "Linked logits: ${logits_dst} -> ${logits_src}"
    fi
}

case "${MODE}" in
    teacher)
        run_teacher_or_raw teacher "$@"
        ;;
    raw)
        run_teacher_or_raw raw "$@"
        ;;
    sidmlp-pp)
        run_sidmlp_pp "$@"
        ;;
    *)
        echo "Unknown extract mode: ${MODE}" >&2
        exit 1
        ;;
esac
