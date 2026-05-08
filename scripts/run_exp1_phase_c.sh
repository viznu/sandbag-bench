#!/usr/bin/env bash
# Phase 1a — Phase C (blinded extractor) for any condition's records.
# Usage: run_exp1_phase_c.sh <config> <results_dir>
set -euo pipefail
CFG="${1:?config path required}"
OUT="${2:?results dir required}"

EXTR_MODEL="$(python -c "import yaml; print(yaml.safe_load(open('${CFG}'))['extractor']['model_id'])")"
PORT="$(python -c "import yaml; print(yaml.safe_load(open('${CFG}'))['extractor']['sglang_port'])")"

echo "[exp1_phase_c] launching SGLang for $EXTR_MODEL on port $PORT"
setsid nohup python -m sglang.launch_server \
  --model-path "$EXTR_MODEL" \
  --port "$PORT" \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --disable-cuda-graph \
  --trust-remote-code \
  >sglang_extractor.log 2>&1 </dev/null &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python -m sandbag_bench.phase_c_client --config "$CFG" --results "$OUT"

kill $SGLANG_PID 2>/dev/null || true
wait $SGLANG_PID 2>/dev/null || true
