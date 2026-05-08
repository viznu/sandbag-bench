"""Wilson 95% CI lower bound and pass/fail logic for the Experiment 0 gate."""
from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_lower_bound(successes: int, n: int, z: float = 1.959963984540054) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (center - margin) / denom


@dataclass
class GateResult:
    n: int
    n_correct: int
    n_abstain: int
    agreement: float
    wilson_lower_95: float
    primary_threshold: float
    wilson_threshold: float
    passed: bool

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] n={self.n} agreement={self.agreement:.3f} "
            f"wilson_lb95={self.wilson_lower_95:.3f} "
            f"abstains={self.n_abstain} "
            f"thresholds(agree>={self.primary_threshold:.2f}, "
            f"wilson>={self.wilson_threshold:.2f})"
        )


def evaluate_gate(
    extractor_answers: list[str | None],
    model_answers: list[str | None],
    primary_threshold: float = 0.70,
    wilson_threshold: float = 0.60,
    abstain_counts_as_incorrect: bool = True,
) -> GateResult:
    """Compute agreement of blinded NLA-text extractor with the target model's own answer.

    Per Codex spec: items are base-correct, so extractor==model is equivalent to
    extractor==gold. Abstain counts as incorrect for the gate; abstain rate is
    reported separately.
    """
    assert len(extractor_answers) == len(model_answers)
    n = len(extractor_answers)
    correct = 0
    abstain = 0
    for ext, mod in zip(extractor_answers, model_answers):
        if ext == "abstain" or ext is None:
            abstain += 1
            if abstain_counts_as_incorrect:
                continue
            continue
        if mod is None:
            continue
        if ext == mod:
            correct += 1
    agreement = correct / n if n else 0.0
    wlb = wilson_lower_bound(correct, n)
    passed = agreement >= primary_threshold and wlb >= wilson_threshold
    return GateResult(
        n=n,
        n_correct=correct,
        n_abstain=abstain,
        agreement=agreement,
        wilson_lower_95=wlb,
        primary_threshold=primary_threshold,
        wilson_threshold=wilson_threshold,
        passed=passed,
    )
