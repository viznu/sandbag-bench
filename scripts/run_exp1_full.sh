#!/usr/bin/env bash
# Phase 1a — full sequence on a fresh Vast box.
#
# Sequence:
#   1. Phase A on capability_emulation condition (keep wrong only) -> N positives
#   2. Low-elicitation 8-style sweep -> N controls
#   3. Phase B (NLA decode) on each
#   4. Phase C (blinded extractor) on each
#   5. Compute Phase 1a metrics: TPR @ 5%% FPR, AUROC, recovery rates
set -euo pipefail
N="${1:-300}"

CFG_S="configs/exp1_capability_emulation.yaml"
OUT_S="results/exp1_capability_emulation"
CFG_L="configs/exp1_low_elicitation.yaml"
OUT_L="results/exp1_low_elicitation"

mkdir -p "$OUT_S/activations" "$OUT_L/activations"

echo "=== 1/5 Phase A — capability_emulation sandbagger (target N=$N wrong) ==="
bash scripts/run_exp1_phase_a.sh "$N" "$CFG_S" "$OUT_S"

echo "=== 2/5 Low-elicitation 8-style sweep (target N=$N controls) ==="
bash scripts/run_exp1_low_elicitation.sh "$N" "$CFG_L" "$OUT_L"

echo "=== 3a/5 Phase B (NLA decode) on sandbagger condition ==="
bash scripts/run_exp1_phase_b.sh "$CFG_S" "$OUT_S"

echo "=== 3b/5 Phase B (NLA decode) on low-elicitation controls ==="
bash scripts/run_exp1_phase_b.sh "$CFG_L" "$OUT_L"

echo "=== 4a/5 Phase C (extractor) on sandbagger condition ==="
bash scripts/run_exp1_phase_c.sh "$CFG_S" "$OUT_S"

echo "=== 4b/5 Phase C (extractor) on low-elicitation controls ==="
bash scripts/run_exp1_phase_c.sh "$CFG_L" "$OUT_L"

echo "=== 5/5 Phase 1a metrics ==="
python -m sandbag_bench.metrics \
    --sandbagger "$OUT_S/records.jsonl" \
    --low-elic "$OUT_L/records.jsonl" \
    --out results/exp1_metrics.json
