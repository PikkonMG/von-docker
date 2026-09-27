#!/bin/sh
# Smoke test for one image tag. Downloads no model weights and needs no GPU.
# Usage: ./smoke_test.sh IMAGE TARGET   (TARGET is cpu or nvidia)
set -eu

image="$1"
target="$2"
container=von-smoke
host_port=18000
api_key=smoke-test-key
unraid_uid=99
unraid_gid=100
health_timeout_seconds=120
http_unauthorized=401
# Von validates the body before it checks the key, so the request must be well formed.
valid_request='{"state": "Smoke test.", "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}}'

cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
    echo "FAIL: $1" >&2
    docker logs "$container" 2>&1 | tail -40 >&2 || true
    exit 1
}

in_image() {
    docker run --rm --entrypoint python "$image" -c "$1"
}

echo "Test: PyTorch build matches $target, and Triton is removed"
case "$target" in
    cpu)
        in_image "import torch; assert torch.version.cuda is None, torch.__version__" || fail "not a CPU PyTorch build"
        in_image "import openvino" || fail "OpenVINO is missing"
        ;;
    nvidia) in_image "import torch; assert torch.version.cuda, torch.__version__" || fail "not a CUDA PyTorch build" ;;
    *) fail "TARGET must be cpu or nvidia" ;;
esac
in_image "import importlib.util; assert importlib.util.find_spec('triton') is None" || fail "triton is still installed"

if [ "$target" = nvidia ]; then
    echo "Test: nvidia image without a GPU stops with a hint"
    if output=$(docker run --rm "$image" 2>&1); then
        fail "nvidia image started without a GPU"
    fi
    echo "$output" | grep -q "PyTorch cannot see a GPU" || fail "no GPU hint: $output"
fi

echo "Test: server answers /health on CPU"
cleanup
docker run -d --name "$container" -p "127.0.0.1:$host_port:8000" \
    -e VON_TARGET=cpu -e VON_PRELOAD=0 -e VON_API_KEY="$api_key" "$image" >/dev/null
elapsed=0
until curl -fsS "http://127.0.0.1:$host_port/health" >/dev/null 2>&1; do
    [ "$elapsed" -lt "$health_timeout_seconds" ] || fail "/health did not answer in $health_timeout_seconds seconds"
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "Test: server runs as $unraid_uid:$unraid_gid and owns /data"
server_ids=$(docker exec "$container" awk '/^Uid:/ { uid = $2 } /^Gid:/ { gid = $2 } END { print uid ":" gid }' /proc/1/status)
[ "$server_ids" = "$unraid_uid:$unraid_gid" ] || fail "server runs as $server_ids"
data_owner=$(docker exec "$container" stat -c %u:%g /data/huggingface)
[ "$data_owner" = "$unraid_uid:$unraid_gid" ] || fail "/data/huggingface owned by $data_owner"

echo "Test: API key is required"
status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$host_port/v1/systemone" \
    -H 'content-type: application/json' --data "$valid_request")
[ "$status" = "$http_unauthorized" ] || fail "request without key returned $status"

echo "PASS: $image"
