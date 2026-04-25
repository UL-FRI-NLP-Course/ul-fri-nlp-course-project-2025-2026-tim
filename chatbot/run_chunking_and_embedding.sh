#!/bin/bash

set -euo pipefail

SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img
LLM_MODELS_DIR=/d/hpc/projects/onj_fri/group-tim
INPUT_FILE=/workspace/data/register-predpisov.jsonl

#my hf auth token - bregar
HUGGINGFACE_HUB_TOKEN=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

echo "Choose chunking strategy:"
echo "  1) normal         whitelist + keyword relevance filtering"
echo "  2) whitelist      only explicitly whitelisted laws"
echo "  3) all-laws       every law record, keeping nonparsed fallback chunks"
read -r -p "Strategy [1]: " STRATEGY
STRATEGY=${STRATEGY:-1}

case "$STRATEGY" in
    1|normal)
        STRATEGY_NAME=normal
        OUT_DIR=/workspace/rag_store/register_predpisov_normal
        CHUNK_FLAGS=""
        ;;
    2|whitelist)
        STRATEGY_NAME=whitelist
        OUT_DIR=/workspace/rag_store/register_predpisov_whitelist
        CHUNK_FLAGS="--strict-whitelist"
        ;;
    3|all|all-laws)
        STRATEGY_NAME=all_laws
        OUT_DIR=/workspace/rag_store/register_predpisov_all_laws
        CHUNK_FLAGS="--all-laws --keep-nonparsed"
        ;;
    *)
        echo "Unknown strategy: $STRATEGY"
        exit 1
        ;;
esac

echo "Selected strategy: $STRATEGY_NAME"
echo "Output directory: $OUT_DIR"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --partition=gpu \
    --time=00:30:00 \
    --pty \
    singularity exec --nv \
        --overlay $OVERLAY_FILE:ro \
        -B $(pwd):/workspace \
        -B ${LLM_MODELS_DIR}:/models \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate

            export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
            export HF_HOME=/models/hf_cache
            export TRANSFORMERS_CACHE=/models/hf_cache
            export HUGGINGFACE_HUB_CACHE=/models/hf_cache
            export TORCH_HOME=/models/torch_cache
            export TRITON_CACHE_DIR=/models/triton_cache
            export VLLM_CACHE_ROOT=/models/vllm_cache
            
            python /workspace/src/run_chunking_and_embedding.py \
                --input $INPUT_FILE \
                --out-dir $OUT_DIR \
                $CHUNK_FLAGS
        "
