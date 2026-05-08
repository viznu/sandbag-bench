#!/usr/bin/env bash
# Phase 1a — Phase A for the prompted-sandbagger condition.
# Captures L20 activations and the (likely wrong) model answers for items
# where Qwen, prompted as a weak-student, emits a non-gold letter.
set -euo pipefail
N="${1:-300}"
CFG="${2:-configs/exp1_capability_emulation.yaml}"
OUT="${3:-results/exp1_capability_emulation}"
python -m sandbag_bench.phase_a \
    --config "$CFG" \
    --n "$N" \
    --out "$OUT" \
    --condition capability_emulation \
    --keep-wrong-only
