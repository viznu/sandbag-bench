#!/usr/bin/env bash
set -euo pipefail
# Phase C — launch SGLang for Mistral extractor, run extractor, kill server.
CFG="${1:-configs/exp0_calibration.yaml}"
EXTR_MODEL="$(python -c "import yaml; print(yaml.safe_load(open('${CFG}'))['extractor']['model_id'])")"
PORT="$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['extractor']['sglang_port'])")"

echo "[phase_c] launching SGLang for $EXTR_MODEL on port $PORT"
python -m sglang.launch_server \
  --model-path "$EXTR_MODEL" \
  --port "$PORT" \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --disable-cuda-graph \
  --trust-remote-code \
  >sglang_extractor.log 2>&1 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python -m sandbag_bench.phase_c_client --config "$CFG" --results results/exp0
python -m sandbag_bench.run_exp0 --config "$CFG" --results results/exp0 || true

kill $SGLANG_PID 2>/dev/null || true
wait $SGLANG_PID 2>/dev/null || true
