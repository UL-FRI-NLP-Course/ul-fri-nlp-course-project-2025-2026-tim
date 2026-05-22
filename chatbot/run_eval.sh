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

SIF_FILE="/d/hpc/projects/onj_fri/group-tim_shared_containers/chatbot_container.sif"
OVERLAY_FILE="/d/hpc/projects/onj_fri/group-tim_shared_containers/chatbot_overlay.img"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --partition=gpu \
    --time=00:45:00 \
    --pty \
    singularity exec --nv \
        --overlay $OVERLAY_FILE:ro \
        -B "${REPO_ROOT}":/workspace \
        -B ${LLM_MODELS_DIR}:/models \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate

            python /workspace/evaluation/scripts/evaluate_predictions.py \
                --input /workspace/evaluation/outputs/predictions/predictions3.jsonl \
                --output /workspace/evaluation/outputs/evaluations/judgmentsLLAMA2.jsonl \
                --summary /workspace/evaluation/outputs/evaluations/judgment_summaryLLAMA2.json \
                --judge-prompt /workspace/evaluation/prompts/judge_prompt.yaml
        "
