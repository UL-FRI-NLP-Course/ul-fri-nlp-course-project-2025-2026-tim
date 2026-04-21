#!/bin/bash

CONTAINER=docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
SIF_FILE=./containers/chatbot_container.sif
OVERLAY_FILE=./containers/chatbot_overlay.img
OVERLAY_SIZE_GB=16

set -e

mkdir -p ./containers

echo "Rebuilding overlay..."

if [ -f "$OVERLAY_FILE" ]; then
    echo "Removing old overlay..."
    rm $OVERLAY_FILE
fi

OVERLAY_SIZE_MB=$((OVERLAY_SIZE_GB * 1024))
echo "Creating new overlay (${OVERLAY_SIZE_GB}GB)..."
singularity overlay create \
    --size $OVERLAY_SIZE_MB \
    $OVERLAY_FILE

echo "Setting up virtual environment and reinstalling dependencies..."

singularity exec \
    --overlay $OVERLAY_FILE \
    -B $(pwd):/workspace \
    $SIF_FILE \
    bash -c "
        set -e

        VENV_PATH=/opt/venv

        echo 'Creating fresh venv...'
        python -m venv --system-site-packages \$VENV_PATH

        echo 'Activating venv...'
        source \$VENV_PATH/bin/activate

        echo 'Upgrading pip...'
        pip install --no-cache-dir --upgrade pip

        echo 'Installing updated requirements...'
        pip install --no-cache-dir -r /workspace/requirements.txt
    "

echo "Overlay successfully rebuilt and packages updated:"
echo "  Overlay: $OVERLAY_FILE"
echo "  Venv: /opt/venv (inside overlay)"