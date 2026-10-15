#!/bin/sh
set -eu
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RELEASE_ROOT=$(CDPATH= cd -- "${TASK_DIR}/.." && pwd)
PORT="${PORT:-8010}"
HOST="${HOST:-127.0.0.1}"
MODEL_DIR="${MODEL_DIR:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-reward_model}"
VLLM_BIN="${VLLM_BIN:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${RELEASE_ROOT}}"
LOG_PATH="${LOG_PATH:-${RELEASE_ROOT}/logs/qwen3omni_rm_vllm_${PORT}.log}"
if [ -z "$MODEL_DIR" ]; then
  echo "Set MODEL_DIR to the local reward-model checkpoint." >&2
  exit 2
fi
command -v "$VLLM_BIN" >/dev/null 2>&1 || {
  echo "vllm executable not found: $VLLM_BIN" >&2
  exit 2
}
mkdir -p "$(dirname "$LOG_PATH")"
test -e "$MODEL_DIR"
MODEL_DIR=$(CDPATH= cd -- "$MODEL_DIR" && pwd)
ALLOWED_LOCAL_MEDIA_PATH=$(CDPATH= cd -- "$ALLOWED_LOCAL_MEDIA_PATH" && pwd)
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  >"$LOG_PATH" 2>&1
