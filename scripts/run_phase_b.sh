#!/usr/bin/env bash
set -euo pipefail
# Phase B — launch SGLang for the NLA AV checkpoint, decode all activations, kill server.
CFG="${1:-configs/exp0_calibration.yaml}"
NLA_MODEL="kitft/nla-qwen2.5-7b-L20-av"
PORT="$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['nla']['sglang_port'])")"

echo "[phase_b] launching SGLang for $NLA_MODEL on port $PORT"
python -m sglang.launch_server \
  --model-path "$NLA_MODEL" \
  --port "$PORT" \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  >sglang_nla.log 2>&1 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT

# Wait for server to be reachable.
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python -m sandbag_bench.phase_b_client --config "$CFG" --results results/exp0

kill $SGLANG_PID 2>/dev/null || true
wait $SGLANG_PID 2>/dev/null || true
