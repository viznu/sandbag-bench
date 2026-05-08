from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import orjson

LETTERS = ("A", "B", "C", "D")
Letter = Literal["A", "B", "C", "D"]
ExtractorAnswer = Literal["A", "B", "C", "D", "abstain"]


@dataclass
class MMLUItem:
    item_id: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    gold: Letter

    def format_prompt(self) -> str:
        body = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, self.choices))
        return (
            f"The following is a multiple-choice question. Reply with the single letter "
            f"of the correct answer.\n\n"
            f"Question: {self.question}\n{body}\n\nAnswer:"
        )


@dataclass
class Phase0Record:
    item_id: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    gold: Letter
    model_answer: Letter | None = None
    model_logprob_top: float | None = None
    activation_layer: int = 20
    activation_position: str = "final_prompt_token"
    activation_token_idx: int | None = None
    activation_path: str | None = None
    nla_text: str | None = None
    extractor_answer: ExtractorAnswer | None = None
    extractor_confidence: float | None = None
    extractor_raw: str | None = None
    notes: dict = field(default_factory=dict)

    @property
    def agreement_with_model(self) -> bool | None:
        if self.extractor_answer is None or self.model_answer is None:
            return None
        return self.extractor_answer == self.model_answer

    @property
    def agreement_with_gold(self) -> bool | None:
        if self.extractor_answer is None:
            return None
        return self.extractor_answer == self.gold


def load_mmlu(
    subjects_exclude: Iterable[str] = (),
    split: str = "test",
    seed: int = 17,
) -> list[MMLUItem]:
    """Load MMLU as a flat list of MMLUItem; deterministic order via seeded shuffle."""
    from datasets import load_dataset

    excluded = set(subjects_exclude)
    ds = load_dataset("cais/mmlu", "all", split=split)
    items: list[MMLUItem] = []
    for i, row in enumerate(ds):
        subject = row["subject"]
        if subject in excluded:
            continue
        choices = tuple(row["choices"])
        if len(choices) != 4:
            continue
        gold_idx = int(row["answer"])
        items.append(
            MMLUItem(
                item_id=f"mmlu-{split}-{i:06d}",
                subject=subject,
                question=row["question"],
                choices=choices,  # type: ignore[arg-type]
                gold=LETTERS[gold_idx],
            )
        )
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def write_records(records: list[Phase0Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(asdict(r)))
            f.write(b"\n")


def read_records(path: Path) -> list[Phase0Record]:
    out: list[Phase0Record] = []
    with path.open("rb") as f:
        for line in f:
            d = orjson.loads(line)
            d["choices"] = tuple(d["choices"])
            out.append(Phase0Record(**d))
    return out
