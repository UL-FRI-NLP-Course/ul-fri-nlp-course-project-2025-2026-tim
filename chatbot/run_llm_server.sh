#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

LLM_MODELS_DIR="${LLM_MODELS_DIR:-/d/hpc/projects/onj_fri/group-tim}"

SIF_FILE="${SCRIPT_DIR}/containers/chatbot_container.sif"
OVERLAY_FILE="${SCRIPT_DIR}/containers/chatbot_overlay.img"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --partition=gpu \
    --time=01:30:00 \
    --pty \
    singularity exec --nv \
        --overlay $OVERLAY_FILE:ro \
        -B "${SCRIPT_DIR}":/workspace \
        -B ${LLM_MODELS_DIR}:/models \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate

            export VLLM_USE_FLASHINFER_SAMPLER=0
            export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
            export HF_HOME=/models/hf_cache
            export TRANSFORMERS_CACHE=/models/hf_cache
            export HUGGINGFACE_HUB_CACHE=/models/hf_cache
            export TORCH_HOME=/models/torch_cache
            export TRITON_CACHE_DIR=/models/triton_cache
            export VLLM_CACHE_ROOT=/models/vllm_cache
            
            python /workspace/src/run_openai_llm_server.py
        "