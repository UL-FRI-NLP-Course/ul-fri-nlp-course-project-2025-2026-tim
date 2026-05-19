#!/bin/bash

LLM_MODELS_DIR=/d/hpc/projects/onj_fri/group-tim

SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --partition=gpu \
    --time=00:10:00 \
    --pty \
    singularity exec --nv \
        --overlay $OVERLAY_FILE:ro \
        -B $(pwd):/workspace \
        -B ${LLM_MODELS_DIR}:/models \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate
            python /workspace/src/run_chatbot_client.py
        "