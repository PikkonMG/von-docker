"""Start the Von HTTP server. With VON_PRELOAD=1 the model loads before the server listens."""
import os

import torch
import uvicorn
from von.engine import VON_VERSION, VonEngine
from von.server import app

PRELOAD_ON = "1"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"
VON_MODEL = f"von-{VON_VERSION}"
WARM_UP_STATE = "Von is starting."
WARM_UP_QUESTIONS = {"ready": {"type": "noul", "instructions": "Is the server starting?"}}
GPU_FULL_WARNING = (
    "WARNING: The GPU does not have enough free memory for Von. Von runs on the CPU instead, "
    "which is slower and uses CPU for every request. Free GPU memory and restart the container "
    "to run on the GPU again."
)


def warm_up():
    # One real request downloads the weights, loads them on the device and,
    # on the cpu image, builds the OpenVINO graph.
    VonEngine.get_instance().evaluate(state=WARM_UP_STATE, questions=WARM_UP_QUESTIONS)


def load_model():
    """Load the model, and move it to the CPU when the GPU is full instead of stopping."""
    try:
        warm_up()
    except torch.OutOfMemoryError:
        if os.environ["VON_DEVICE"] != CUDA_DEVICE:
            raise
        print(GPU_FULL_WARNING, flush=True)
        torch.cuda.empty_cache()
        os.environ["VON_DEVICE"] = CPU_DEVICE
        VonEngine.set_backend(VON_MODEL, device=CPU_DEVICE)
        warm_up()


def main():
    if os.environ["VON_PRELOAD"] == PRELOAD_ON:
        load_model()
    uvicorn.run(
        app,
        host=os.environ["VON_HOST"],
        port=int(os.environ["VON_PORT"]),
        log_level=os.environ["VON_LOG_LEVEL"],
    )


if __name__ == "__main__":
    main()
