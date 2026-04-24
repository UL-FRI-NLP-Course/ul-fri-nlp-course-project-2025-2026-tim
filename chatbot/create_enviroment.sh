#!/bin/bash

CONTAINER=docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img
OVERLAY_SIZE_GB=16

set -e
mkdir -p ./containers

echo "Building Singularity container from Docker Hub..."
singularity build $SIF_FILE $CONTAINER

echo "Creating overlay..."
if [ ! -f "$OVERLAY_FILE" ]; then
    OVERLAY_SIZE_MB=$((OVERLAY_SIZE_GB * 1024))
    singularity overlay create \
        --size $OVERLAY_SIZE_MB \
        $OVERLAY_FILE
fi

echo "Setting up virtual environment inside overlay..."

singularity exec \
    --overlay $OVERLAY_FILE \
    -B $(pwd):/workspace \
    $SIF_FILE \
    bash -c "
        set -e

        VENV_PATH=/opt/venv

        if [ ! -d \$VENV_PATH ]; then
            echo 'Creating venv in overlay...'
            python -m venv --system-site-packages \$VENV_PATH
        fi

        echo 'Installing dependencies into venv...'
        source \$VENV_PATH/bin/activate

        pip install --no-cache-dir --upgrade pip
        pip install --no-cache-dir -r /workspace/requirements.txt
    "

echo "Container and overlay ready:"
echo "  SIF: $SIF_FILE"
echo "  Overlay: $OVERLAY_FILE"
echo "  Venv: /opt/venv (inside overlay)"