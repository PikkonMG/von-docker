"""Start the Von HTTP server. With VON_PRELOAD=1 the model loads before the server listens."""
import os

import uvicorn
from von.engine import VonEngine
from von.server import app

PRELOAD_ON = "1"
WARM_UP_STATE = "Von is starting."
WARM_UP_QUESTIONS = {"ready": {"type": "noul", "instructions": "Is the server starting?"}}


def main():
    if os.environ["VON_PRELOAD"] == PRELOAD_ON:
        # One real request downloads the weights, loads them on the device and,
        # on the cpu image, builds the OpenVINO graph.
        VonEngine.get_instance().evaluate(state=WARM_UP_STATE, questions=WARM_UP_QUESTIONS)
    uvicorn.run(
        app,
        host=os.environ["VON_HOST"],
        port=int(os.environ["VON_PORT"]),
        log_level=os.environ["VON_LOG_LEVEL"],
    )


if __name__ == "__main__":
    main()
