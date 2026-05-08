"""Phase C — blinded extractor: read NLA text + question + choices, return A/B/C/D/abstain.

Talks to a separately-launched SGLang server hosting the extractor model.
The extractor never sees the target model's own answer or the gold answer.

We apply the chat template client-side and post to /generate as raw text,
because /generate bypasses chat templating by default and instruct models
fall into freeform reasoning mode without it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import LETTERS, read_records, write_records

SYSTEM_MSG = (
    "You are a strict, blinded evaluator. You will receive a multiple-choice "
    "question, four choices, and a paragraph that purports to describe what "
    "is in a target language model's internal state at the moment it was "
    "about to answer. Your job is NOT to answer the question yourself. Your "
    "job is to read the description and infer which answer letter (A, B, C, "
    "or D) the target model's internal state most strongly implies. If the "
    "description does not provide enough evidence to choose, output abstain. "
    "Output EXACTLY one line, no extra text, in this format:\n"
    "ANSWER=<A|B|C|D|abstain> CONF=<0.00-1.00>"
)

USER_TEMPLATE = """Question: {question}

Choices:
A. {a}
B. {b}
C. {c}
D. {d}

Description of the target model's internal state:
{nla_text}

Output ONLY one line: ANSWER=<letter or abstain> CONF=<0.00-1.00>"""

# Strict primary regex (compliant output).
_STRICT = re.compile(
    r"ANSWER\s*=\s*(A|B|C|D|abstain)\s+CONF\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
# Fallback for free-form prose like "the answer is D" / "most strongly implies B".
_PROSE = re.compile(
    r"\b(?:answer\s+is|most\s+strongly\s+implies|implies\s+(?:the\s+answer\s+is\s+)?)"
    r"\s*['\"]?([A-D])\b",
    re.IGNORECASE,
)


def _parse(raw: str) -> tuple[str | None, float | None, str]:
    """Returns (answer, confidence, parse_method)."""
    m = _STRICT.search(raw)
    if m:
        ans = m.group(1)
        conf = float(m.group(2))
        if ans.lower() == "abstain":
            return "abstain", conf, "strict"
        return ans.upper(), conf, "strict"
    m = _PROSE.search(raw)
    if m:
        return m.group(1).upper(), 0.5, "prose_fallback"
    return None, None, "unparsed"


def run_phase_c(config_path: Path, results_root: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    extr = cfg["extractor"]
    extr_id = extr["model_id"]
    port = int(extr["sglang_port"])
    max_new = int(extr["max_new_tokens"])
    temperature = float(extr["temperature"])
    base_url = f"http://127.0.0.1:{port}"

    print(f"[phase_c] loading tokenizer for {extr_id}")
    tok = AutoTokenizer.from_pretrained(extr_id, trust_remote_code=True)

    records_path = results_root / "records.jsonl"
    records = read_records(records_path)
    print(f"[phase_c] {len(records)} records to extract")

    n_strict = n_prose = n_unparsed = 0
    with httpx.Client(timeout=120.0) as client:
        for rec in tqdm(records, desc="phase_c"):
            if rec.nla_text is None:
                rec.extractor_answer = None
                rec.extractor_confidence = None
                rec.extractor_raw = "<no_nla_text>"
                continue
            user_msg = USER_TEMPLATE.format(
                question=rec.question,
                a=rec.choices[0],
                b=rec.choices[1],
                c=rec.choices[2],
                d=rec.choices[3],
                nla_text=rec.nla_text,
            )
            prompt_text = tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            payload = {
                "text": prompt_text,
                "sampling_params": {
                    "max_new_tokens": max_new,
                    "temperature": temperature,
                    "stop": ["\n\n"],
                },
            }
            r = client.post(f"{base_url}/generate", json=payload)
            r.raise_for_status()
            body = r.json()
            raw = _extract_text(body)
            ans, conf, method = _parse(raw)
            rec.extractor_raw = raw
            rec.extractor_answer = ans  # type: ignore[assignment]
            rec.extractor_confidence = conf
            rec.notes["parse_method"] = method
            if method == "strict":
                n_strict += 1
            elif method == "prose_fallback":
                n_prose += 1
            else:
                n_unparsed += 1

    print(
        f"[phase_c] parse: strict={n_strict} prose_fallback={n_prose} "
        f"unparsed={n_unparsed}"
    )
    write_records(records, records_path)
    print(f"[phase_c] updated {records_path}")
    return records_path


def _extract_text(body: dict[str, Any]) -> str:
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        if "text" in body and isinstance(body["text"], str):
            return body["text"]
        if "outputs" in body and body["outputs"]:
            first = body["outputs"][0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            if isinstance(first, str):
                return first
    import orjson
    return orjson.dumps(body).decode("utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("results/exp0"))
    args = p.parse_args()
    run_phase_c(args.config, args.results)


if __name__ == "__main__":
    main()
