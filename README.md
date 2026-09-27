# Von System One Docker image

Builds the Docker image for the Von Unraid template from the
[Von](https://github.com/wfzyx/von) release on PyPI (`von-sdk`). Von has no
official image.

| Tag | For | Extra Parameters on Unraid |
|---|---|---|
| `pikkonmg/von-system-one:cpu` | Any CPU, runs on OpenVINO | Empty |
| `pikkonmg/von-system-one:nvidia` | NVIDIA GPU, driver 560 or newer | `--runtime=nvidia` |

Each release also gets a fixed tag, for example `1.2.3-cpu`.

## How it updates

`.github/workflows/build.yml` runs every day. When PyPI has a `von-sdk`
release that Docker Hub does not have yet, it builds both tags, runs
`smoke_test.sh`, and pushes them. A push to `main` or a manual run rebuilds
the latest release.

The repository needs two secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`: a Docker Hub personal access token with write access

## Build on your own computer

```bash
docker build --build-arg TARGET=cpu --build-arg VON_VERSION=1.2.3 -t pikkonmg/von-system-one:cpu .
./smoke_test.sh pikkonmg/von-system-one:cpu cpu
```

Use `TARGET=nvidia` for the NVIDIA image.

## Files

| File | Job |
|---|---|
| `Dockerfile` | Python 3.12, PyTorch 2.14 for the target, Von from PyPI, OpenVINO on cpu |
| `start.sh` | Sets the device, fixes `/data` ownership, checks the GPU, drops to `PUID:PGID` |
| `serve.py` | Loads the model when `VON_PRELOAD=1`, then starts the server |
| `smoke_test.sh` | Checks the image without a GPU or model download |
