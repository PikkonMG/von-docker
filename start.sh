#!/bin/sh
# Set the Von device for this image, prepare /data for PUID:PGID, check the
# NVIDIA GPU is visible, then start the command as that user.
set -eu

DATA_DIR=/data

case "$VON_TARGET" in
    cpu) export VON_DEVICE=openvino:cpu ;;
    nvidia) export VON_DEVICE=cuda ;;
    *)
        echo "Unknown image target: '$VON_TARGET'" >&2
        exit 1
        ;;
esac

as_user=""
if [ "$(id -u)" = 0 ]; then
    mkdir -p "$HF_HOME"
    find "$DATA_DIR" \( ! -user "$PUID" -o ! -group "$PGID" \) -exec chown -h "$PUID:$PGID" {} +
    as_user="setpriv --reuid=$PUID --regid=$PGID --clear-groups --inh-caps=-all"
fi

if [ "$VON_DEVICE" = cuda ]; then
    gpu_name=$($as_user python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')") || gpu_name=""
    if [ -z "$gpu_name" ]; then
        echo "This is the nvidia image but PyTorch cannot see a GPU. Install the Unraid NVIDIA Driver plugin and add --runtime=nvidia to Extra Parameters." >&2
        exit 1
    fi
    echo "Device: cuda ($gpu_name)"
else
    echo "Device: cpu (OpenVINO)"
fi

exec $as_user "$@"
