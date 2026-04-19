#!/bin/bash

SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --partition=gpu \
    --time=00:30:00 \
    --pty \
    singularity exec --nv \
        --overlay $OVERLAY_FILE \
        -B $(pwd):/workspace \
        $SIF_FILE \
        bash -c "
            echo 'Running on: ' \$(hostname)
            source /opt/venv/bin/activate
            python /workspace/run.py
        "