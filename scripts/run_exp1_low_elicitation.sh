#!/usr/bin/env bash
# Phase 1a — low-elicitation wrong-answer control sweep.
# For each MMLU candidate, runs an 8-style elicitation suite. Items wrong
# on >=7/8 attempts (with parsing >=7/8) are kept as controls. Records a
# matching neutral L20 capture per kept item.
set -euo pipefail
N="${1:-300}"
CFG="${2:-configs/exp1_low_elicitation.yaml}"
OUT="${3:-results/exp1_low_elicitation}"
python -m sandbag_bench.elicitation \
    --config "$CFG" \
    --n "$N" \
    --out "$OUT"
