"""Phase A — load Qwen, format prompts per condition, capture L20 hidden state
at the final prompt token, log model answer.

Single forward pass per item with output_hidden_states=True. No generation.
The L20 activation is taken at the last token of the formatted prompt
(immediately before the model would emit its letter answer). The model's
answer is read off the final-layer logits restricted to the four choice
letter token IDs.

Conditions (see conditions.py):
  neutral              raw text prompt (Experiment 0).
  capability_emulation chat-templated system + user prompt (van der Weij).
  target_accuracy      chat-templated system (with p=0 by default) + user.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .conditions import Condition, PromptSpec, build_prompt
from .data import LETTERS, MMLUItem, Phase0Record, load_mmlu, write_records


def _letter_token_ids(tokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for L in LETTERS:
        for cand in (f" {L}", L):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                out[L] = ids[0]
                break
        else:
            raise RuntimeError(f"Could not find single-token id for letter {L!r}")
    return out


def _materialize_prompt(spec: PromptSpec, tokenizer) -> str:
    """Return the final prompt string ready for tokenization.

    For neutral: spec.text is used verbatim.
    For chat-templated: applies the target's chat template with
    add_generation_prompt=True so the activation we capture is the
    final prompt-side token immediately before the model would emit.
    """
    if spec.text is not None:
        return spec.text
    assert spec.messages is not None
    return tokenizer.apply_chat_template(
        spec.messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_phase_a(
    config_path: Path,
    n: int,
    out_root: Path,
    *,
    condition: Condition = "neutral",
    target_accuracy_p: int = 0,
    base_correct_only_override: bool | None = None,
) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    target_id = cfg["target"]["model_id"]
    layer_idx = int(cfg["target"]["layer"])
    dtype = getattr(torch, cfg["target"]["dtype"])
    seed = int(cfg["sampling"]["seed"])
    subjects_exclude = cfg["dataset"].get("subjects_exclude") or []
    base_correct_only = (
        base_correct_only_override
        if base_correct_only_override is not None
        else bool(cfg["sampling"].get("base_correct_only", True))
    )

    activations_dir = out_root / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_root / "records.jsonl"

    print(f"[phase_a] condition={condition} target={target_id}")
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    letter_ids = _letter_token_ids(tokenizer)
    letter_id_tensor = torch.tensor(
        [letter_ids[L] for L in LETTERS], device=model.device
    )

    print(f"[phase_a] loading MMLU; excluding {len(subjects_exclude)} subjects")
    all_items = load_mmlu(subjects_exclude=subjects_exclude, seed=seed)
    print(f"[phase_a] {len(all_items)} candidate items; target N={n}")

    kept: list[Phase0Record] = []
    n_seen = 0

    for item in tqdm(all_items, desc=f"phase_a:{condition}"):
        if len(kept) >= n:
            break
        n_seen += 1
        rec = _process_item(
            item=item,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            letter_ids=letter_ids,
            letter_id_tensor=letter_id_tensor,
            activations_dir=activations_dir,
            condition=condition,
            target_accuracy_p=target_accuracy_p,
        )
        if base_correct_only:
            if rec.model_answer == rec.gold:
                kept.append(rec)
        else:
            kept.append(rec)

    print(
        f"[phase_a:{condition}] scanned {n_seen} items, kept {len(kept)} "
        f"(base_correct_only={base_correct_only})"
    )
    write_records(kept, records_path)
    print(f"[phase_a:{condition}] wrote {records_path}")
    return records_path


@torch.inference_mode()
def _process_item(
    *,
    item: MMLUItem,
    model,
    tokenizer,
    layer_idx: int,
    letter_ids: dict[str, int],
    letter_id_tensor: torch.Tensor,
    activations_dir: Path,
    condition: Condition,
    target_accuracy_p: int,
) -> Phase0Record:
    spec = build_prompt(item, condition, target_accuracy_p=target_accuracy_p)
    prompt = _materialize_prompt(spec, tokenizer)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    h = out.hidden_states[layer_idx]
    last_idx = enc["input_ids"].shape[1] - 1
    activation = h[0, last_idx].detach().to(torch.float32).cpu()

    logits_last = out.logits[0, last_idx]
    letter_logits = logits_last[letter_id_tensor]
    letter_probs = letter_logits.softmax(dim=-1)
    pred_idx = int(letter_probs.argmax().item())
    model_answer = LETTERS[pred_idx]
    top_logprob = float(torch.log(letter_probs[pred_idx] + 1e-12).item())

    act_path = activations_dir / f"{item.item_id}.safetensors"
    save_file({"activation": activation}, str(act_path))

    return Phase0Record(
        item_id=item.item_id,
        subject=item.subject,
        question=item.question,
        choices=item.choices,
        gold=item.gold,
        model_answer=model_answer,
        model_logprob_top=top_logprob,
        activation_layer=layer_idx,
        activation_position="final_prompt_token",
        activation_token_idx=last_idx,
        activation_path=str(act_path.relative_to(activations_dir.parent)),
        notes={"condition": condition, "target_accuracy_p": target_accuracy_p},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("results/exp0"))
    p.add_argument(
        "--condition",
        choices=["neutral", "capability_emulation", "target_accuracy"],
        default="neutral",
    )
    p.add_argument("--target-accuracy-p", type=int, default=0)
    p.add_argument(
        "--keep-wrong-only",
        action="store_true",
        help="For sandbagger conditions: keep only items where the sandbagger "
             "model's answer is wrong (i.e. !=gold). Implies "
             "base_correct_only=False.",
    )
    args = p.parse_args()
    base_correct_only_override = False if args.keep_wrong_only else None
    records_path = run_phase_a(
        args.config,
        args.n,
        args.out,
        condition=args.condition,
        target_accuracy_p=args.target_accuracy_p,
        base_correct_only_override=base_correct_only_override,
    )
    if args.keep_wrong_only:
        from .data import read_records
        recs = read_records(records_path)
        wrong = [r for r in recs if r.model_answer != r.gold]
        write_records(wrong, records_path)
        print(f"[phase_a:{args.condition}] kept-wrong-only: {len(wrong)} of {len(recs)}")


if __name__ == "__main__":
    main()
