#!/bin/bash

SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img
LLM_MODELS_DIR=/d/hpc/projects/onj_fri/group-tim

#my hf auth token - bregar
HUGGINGFACE_HUB_TOKEN=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

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

            export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
            export HF_HOME=/models/hf_cache
            export TRANSFORMERS_CACHE=/models/hf_cache
            export HUGGINGFACE_HUB_CACHE=/models/hf_cache
            export TORCH_HOME=/models/torch_cache
            export TRITON_CACHE_DIR=/models/triton_cache
            export VLLM_CACHE_ROOT=/models/vllm_cache
            
            python /workspace/src/run_openai_llm_server.py
        "