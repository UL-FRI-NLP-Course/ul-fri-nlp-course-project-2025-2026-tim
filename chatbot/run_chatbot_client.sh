#!/bin/bash

SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --partition=gpu \
    --time=00:10:00 \
    --pty \
    singularity exec \
        --overlay $OVERLAY_FILE:ro \
        -B $(pwd):/workspace \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate
            python /workspace/src/run_chatbot_client.py
        "