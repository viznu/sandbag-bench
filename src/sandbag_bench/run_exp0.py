"""Experiment 0 reporter — reads records.jsonl, evaluates the calibration gate,
prints PASS/FAIL summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import read_records
from .gate import evaluate_gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("results/exp0"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gate_cfg = cfg["gate"]
    records = read_records(args.results / "records.jsonl")

    extractor_answers = [r.extractor_answer for r in records]
    model_answers = [r.model_answer for r in records]

    gate = evaluate_gate(
        extractor_answers=extractor_answers,
        model_answers=model_answers,
        primary_threshold=float(gate_cfg["primary_threshold"]),
        wilson_threshold=float(gate_cfg["wilson_lower_bound_95"]),
        abstain_counts_as_incorrect=bool(gate_cfg["abstain_counts_as_incorrect"]),
    )
    print(gate.summary())
    if not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
