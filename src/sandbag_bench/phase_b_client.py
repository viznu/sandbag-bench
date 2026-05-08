"""Phase B — decode saved L20 activations to natural-language NLA text.

Uses kitft/nla-inference's official NLAClient. The client expects a *local*
checkpoint directory containing nla_meta.yaml, so we snapshot_download the
HF repo first if needed.

The SGLang server hosting the AV checkpoint must be launched separately
(see scripts/run_phase_b.sh).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tqdm import tqdm

from .data import read_records, write_records


def _local_checkpoint(model_id: str) -> str:
    """If model_id is already a local dir, use it. Otherwise snapshot_download."""
    p = Path(model_id)
    if p.exists() and (p / "nla_meta.yaml").exists():
        return str(p)
    return snapshot_download(repo_id=model_id)


def run_phase_b(config_path: Path, results_root: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    nla_cfg = cfg["nla"]
    actor_id = nla_cfg["model_id"]
    port = int(nla_cfg["sglang_port"])
    max_new = int(nla_cfg["max_new_tokens"])
    temperature = float(nla_cfg["temperature"])
    sglang_url = f"http://127.0.0.1:{port}"

    from nla_inference import NLAClient  # type: ignore

    ckpt_dir = _local_checkpoint(actor_id)
    print(f"[phase_b] NLA checkpoint dir: {ckpt_dir}")
    client = NLAClient(checkpoint_dir=ckpt_dir, sglang_url=sglang_url)

    records_path = results_root / "records.jsonl"
    records = read_records(records_path)
    print(f"[phase_b] {len(records)} records to decode via NLA")

    for rec in tqdm(records, desc="phase_b"):
        assert rec.activation_path is not None
        act_path = results_root / rec.activation_path
        vec = load_file(str(act_path))["activation"].to(torch.float32)
        text = client.generate(
            vec,
            temperature=temperature,
            max_new_tokens=max_new,
        )
        rec.nla_text = text

    write_records(records, records_path)
    print(f"[phase_b] updated {records_path}")
    return records_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("results/exp0"))
    args = p.parse_args()
    run_phase_b(args.config, args.results)


if __name__ == "__main__":
    main()
