#!/usr/bin/env bash
set -euo pipefail
# Phase A — capture L20 activations and model answers from Qwen2.5-7B-Instruct.
# Owns the GPU exclusively.
N="${1:-200}"
CFG="${2:-configs/exp0_calibration.yaml}"
python -m sandbag_bench.phase_a --config "$CFG" --n "$N" --out results/exp0
