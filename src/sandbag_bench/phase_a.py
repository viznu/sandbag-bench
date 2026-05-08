"""Phase A — load Qwen, capture L20 hidden state at final prompt token, log model answer.

Single forward pass per item with output_hidden_states=True. No generation.
The L20 activation is taken at the last token of the formatted prompt
(immediately before the model would emit its letter answer). The model's
answer is read off the final-layer logits restricted to the four choice
letter token IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import LETTERS, MMLUItem, Phase0Record, load_mmlu, write_records


def _letter_token_ids(tokenizer) -> dict[str, int]:
    """Token IDs for ' A', ' B', ' C', ' D' (and bare 'A' fallback)."""
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


def run_phase_a(config_path: Path, n: int, out_root: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    target_id = cfg["target"]["model_id"]
    layer_idx = int(cfg["target"]["layer"])
    dtype = getattr(torch, cfg["target"]["dtype"])
    seed = int(cfg["sampling"]["seed"])
    subjects_exclude = cfg["dataset"].get("subjects_exclude") or []

    activations_dir = out_root / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_root / "records.jsonl"

    print(f"[phase_a] loading {target_id} dtype={dtype}")
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
    print(f"[phase_a] {len(all_items)} candidate items; running first {n}")

    records: list[Phase0Record] = []
    base_correct: list[Phase0Record] = []

    for item in tqdm(all_items, desc="phase_a", total=min(n, len(all_items))):
        if len(base_correct) >= n:
            break
        rec = _process_item(
            item=item,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            letter_ids=letter_ids,
            letter_id_tensor=letter_id_tensor,
            activations_dir=activations_dir,
        )
        records.append(rec)
        if cfg["sampling"].get("base_correct_only", True):
            if rec.model_answer == rec.gold:
                base_correct.append(rec)
        else:
            base_correct.append(rec)

    print(f"[phase_a] kept {len(base_correct)} base-correct items")
    write_records(base_correct, records_path)
    print(f"[phase_a] wrote {records_path}")
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
) -> Phase0Record:
    prompt = item.format_prompt()
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # hidden_states is a tuple of length n_layers+1; index 0 is embeddings.
    # Layer 20 output corresponds to hidden_states[20] under the convention used by
    # kitft NLA training (decoder layer index, post-block residual).
    h = out.hidden_states[layer_idx]  # [1, T, D]
    last_idx = enc["input_ids"].shape[1] - 1
    activation = h[0, last_idx].detach().to(torch.float32).cpu()

    logits_last = out.logits[0, last_idx]  # [vocab]
    letter_logits = logits_last[letter_id_tensor]  # [4]
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
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("results/exp0"))
    args = p.parse_args()
    run_phase_a(args.config, args.n, args.out)


if __name__ == "__main__":
    main()
