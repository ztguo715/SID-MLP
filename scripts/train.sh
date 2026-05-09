#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash SID-MLP/scripts/train.sh sidmlp <category> <gpu>
#   bash SID-MLP/scripts/train.sh sidmlp-pp-stage1 <category> <gpu> <layers> <ffn_dim> <tag_label>
#   bash SID-MLP/scripts/train.sh sidmlp-pp-stage2 <category> <gpu> <encoder_ckpt> <layers> <encoder_ffn_dim> <tag_label>

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sidmlp|sidmlp-pp-stage1|sidmlp-pp-stage2> ..." >&2
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
LOG_DIR="${LOG_DIR:-${GRAPH_DIR}/SID-MLP/logs}"
MODE="$1"
shift

cd "${PROJECT_DIR}"
export PYTHONPATH="${SID_MLP_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SID_MLP_GRAPH_DIR="${GRAPH_DIR}"

run_sidmlp() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: $0 sidmlp <category> <gpu>" >&2
        exit 1
    fi
    local category="$1"
    local gpu_id="$2"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    local tiger_ckpt="${TIGER_CKPT:-${MODEL_DIR}/TIGER-${DATASET}-category_${category}.pth}"
    local sem_ids="${SEM_IDS:-${SEM_ID_DIR}/${DATASET}-${category}_sentence-t5-base_256,256,256,256.sem_ids}"
    local dataset_tag="${DATASET_TAG:-${DATASET}_${category}}"
    local val_dataset_tag="${VAL_DATASET_TAG:-${DATASET}_${category}_val}"
    local ckpt_dir="${CKPT_DIR:-${BASE_CKPT_DIR}}"
    local log_dir="${LOG_DIR}"
    local config="${CONFIG:-${SID_MLP_DIR}/configs/sidmlp.yaml}"
    local storage_args=()
    if [[ "${STORAGE_FP16:-0}" == "1" ]]; then
        storage_args+=(--storage_fp16)
    fi
    mkdir -p "${ckpt_dir}" "${log_dir}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -u -m sid_mlp.cli.train sidmlp \
        --config "${config}" \
        --checkpoint "${tiger_ckpt}" \
        --dataset "${DATASET}" \
        --category "${category}" \
        --dataset_tag "${dataset_tag}" \
        --val_dataset_tag "${val_dataset_tag}" \
        --sem_ids "${sem_ids}" \
        --graph_dir "${GRAPH_DIR}" \
        --num_heads "${NUM_HEADS:-4}" \
        --attn_dim "${ATTN_DIM:-384}" \
        --ffn_dim "${LENS_FFN:-1024}" \
        --head_hidden "${HEAD_HIDDEN:-512}" \
        --head_layers "${HEAD_LAYERS:-1}" \
        --loss "${LOSS:-combined}" \
        --temperature "${TEMPERATURE:-1.0}" \
        --alpha "${ALPHA:-0.7}" \
        --d4_teacher "${D4_TEACHER:-combined}" \
        --lr "${LR:-5e-5}" \
        --dropout "${DROPOUT:-0.2}" \
        --weight_decay "${WEIGHT_DECAY:-1e-4}" \
        --epochs "${EPOCHS:-200}" \
        --batch_size "${BATCH_SIZE:-512}" \
        --val_batch_size "${VAL_BATCH_SIZE:-32}" \
        --beam_size "${BEAM_SIZE:-50}" \
        --final_top_n "${FINAL_TOP_N:-50}" \
        --ckpt_dir "${ckpt_dir}" \
        --log_dir "${log_dir}" \
        "${storage_args[@]}" \
        --seed "${SEED:-42}" \
        --run_tag "${RUN_TAG:-${category}_sidmlp_${ts}}" \
        2>&1 | tee "${log_dir}/${category}_sidmlp_${ts}.log"
}

run_sidmlp_pp_stage1() {
    if [[ $# -lt 5 ]]; then
        echo "Usage: $0 sidmlp-pp-stage1 <category> <gpu> <layers> <ffn_dim> <tag_label>" >&2
        exit 1
    fi
    local category="$1"
    local gpu_id="$2"
    local layers="$3"
    local ffn_dim="$4"
    local tag_label="$5"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    local raw_dir="${RAW_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_rawemb}"
    local teacher_dir="${TEACHER_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}}"
    local val_raw_dir="${VAL_RAW_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_rawemb_val}"
    local val_teacher_dir="${VAL_TEACHER_DIR:-${GRAPH_DIR}/hidden_states/${DATASET}_${category}_val}"
    local ckpt_dir="${CKPT_DIR:-${BASE_CKPT_DIR}/sidmlp_pp_stage1}"
    local log_dir="${LOG_DIR}"
    local output="${OUTPUT:-${ckpt_dir}/stage1_${tag_label}_${category}_${ts}.pth}"
    local config="${CONFIG:-${SID_MLP_DIR}/configs/sidmlp_pp_stage1.yaml}"
    local gpu_resident_args=()
    local storage_args=()
    if [[ "${GPU_RESIDENT:-1}" == "1" ]]; then
        gpu_resident_args+=(--gpu_resident)
    else
        gpu_resident_args+=(--no-gpu_resident)
    fi
    if [[ "${STORAGE_FP16:-0}" == "1" ]]; then
        storage_args+=(--storage_fp16)
    fi
    mkdir -p "${ckpt_dir}" "${log_dir}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -u -m sid_mlp.cli.train sidmlp-pp-stage1 \
        --config "${config}" \
        --raw_dir "${raw_dir}" \
        --teacher_dir "${teacher_dir}" \
        --val_raw_dir "${val_raw_dir}" \
        --val_teacher_dir "${val_teacher_dir}" \
        --d_model "${D_MODEL:-128}" \
        --ffn_dim "${ffn_dim}" \
        --num_layers "${layers}" \
        --dropout "${STAGE1_DROPOUT:-0.1}" \
        --batch_size "${BATCH_SIZE:-512}" \
        --lr "${LR:-1e-3}" \
        --weight_decay "${WEIGHT_DECAY:-1e-4}" \
        --epochs "${EPOCHS:-100}" \
        --grad_clip "${GRAD_CLIP:-1.0}" \
        "${gpu_resident_args[@]}" \
        "${storage_args[@]}" \
        --num_workers "${NUM_WORKERS:-0}" \
        --output "${output}" \
        2>&1 | tee "${log_dir}/${category}_sidmlp_pp_stage1_${ts}.log"
}

run_sidmlp_pp_stage2() {
    if [[ $# -lt 6 ]]; then
        echo "Usage: $0 sidmlp-pp-stage2 <category> <gpu> <encoder_ckpt> <layers> <encoder_ffn_dim> <tag_label>" >&2
        exit 1
    fi
    local category="$1"
    local gpu_id="$2"
    local encoder_ckpt="$3"
    local layers="$4"
    local enc_ffn="$5"
    local tag_label="$6"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    local tiger_ckpt="${TIGER_CKPT:-${MODEL_DIR}/TIGER-${DATASET}-category_${category}.pth}"
    local sem_ids="${SEM_IDS:-${SEM_ID_DIR}/${DATASET}-${category}_sentence-t5-base_256,256,256,256.sem_ids}"
    local dataset_tag="${DATASET_TAG:-${DATASET}_${category}_${tag_label}}"
    local val_dataset_tag="${VAL_DATASET_TAG:-${dataset_tag}_val}"
    local ckpt_dir="${CKPT_DIR:-${BASE_CKPT_DIR}}"
    local log_dir="${LOG_DIR}"
    local config="${CONFIG:-${SID_MLP_DIR}/configs/sidmlp_pp_stage2.yaml}"
    local storage_args=()
    if [[ "${STORAGE_FP16:-0}" == "1" ]]; then
        storage_args+=(--storage_fp16)
    fi
    mkdir -p "${ckpt_dir}" "${log_dir}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" -u -m sid_mlp.cli.train sidmlp-pp-stage2 \
        --config "${config}" \
        --checkpoint "${tiger_ckpt}" \
        --dataset "${DATASET}" \
        --category "${category}" \
        --dataset_tag "${dataset_tag}" \
        --val_dataset_tag "${val_dataset_tag}" \
        --sem_ids "${sem_ids}" \
        --graph_dir "${GRAPH_DIR}" \
        --encoder_ckpt "${encoder_ckpt}" \
        --num_heads "${NUM_HEADS:-4}" \
        --attn_dim "${ATTN_DIM:-384}" \
        --ffn_dim "${LENS_FFN:-1024}" \
        --head_hidden "${HEAD_HIDDEN:-512}" \
        --head_layers "${HEAD_LAYERS:-1}" \
        --loss "${LOSS:-combined}" \
        --temperature "${TEMPERATURE:-1.0}" \
        --alpha "${ALPHA:-0.7}" \
        --d4_teacher "${D4_TEACHER:-combined}" \
        --lr "${LR:-5e-5}" \
        --dropout "${DROPOUT:-0.2}" \
        --weight_decay "${WEIGHT_DECAY:-1e-4}" \
        --epochs "${EPOCHS:-200}" \
        --batch_size "${BATCH_SIZE:-512}" \
        --val_batch_size "${VAL_BATCH_SIZE:-32}" \
        --beam_size "${BEAM_SIZE:-50}" \
        --final_top_n "${FINAL_TOP_N:-50}" \
        --ckpt_dir "${ckpt_dir}" \
        --log_dir "${log_dir}" \
        "${storage_args[@]}" \
        --seed "${SEED:-42}" \
        --run_tag "${RUN_TAG:-${category}_${tag_label}_sidmlp_pp_L${layers}_ef${enc_ffn}_${ts}}" \
        2>&1 | tee "${log_dir}/${category}_sidmlp_pp_stage2_${ts}.log"
}

case "${MODE}" in
    sidmlp)
        run_sidmlp "$@"
        ;;
    sidmlp-pp-stage1)
        run_sidmlp_pp_stage1 "$@"
        ;;
    sidmlp-pp-stage2)
        run_sidmlp_pp_stage2 "$@"
        ;;
    *)
        echo "Unknown train mode: ${MODE}" >&2
        exit 1
        ;;
esac
