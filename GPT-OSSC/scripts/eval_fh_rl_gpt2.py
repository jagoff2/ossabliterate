#!/usr/bin/env python3
"""Evaluate baseline vs FH-RL GPT-2 on an OOD corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from meta_transformer.models.hf_gpt2_fh_rl import FHRLGPT2Config, load_fh_rl_gpt2


class JsonlSequenceDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples: List[Dict[str, List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        return torch.tensor(record["input_ids"], dtype=torch.long), torch.tensor(
            record["labels"], dtype=torch.long
        )


def collate(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, labels=labels)
            losses.append(outputs.loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def load_baseline(cfg: dict) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(cfg["model"]["base_model_name"])


def load_fh_model(cfg: dict) -> torch.nn.Module:
    fh_cfg = FHRLGPT2Config(
        base_model_name=cfg["model"]["base_model_name"],
        fh_rank=cfg["model"]["fh_rank"],
        fh_alpha=cfg["model"]["fh_alpha"],
        fh_beta=cfg["model"]["fh_beta"],
        fh_gamma=cfg["model"]["fh_gamma"],
        noise_std=cfg["model"]["noise_std"],
        detach_feedback=cfg["model"]["detach_feedback"],
    )
    return load_fh_rl_gpt2(fh_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path")
    parser.add_argument("--fh_rl", required=True, help="FH-RL checkpoint path")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    dataset = JsonlSequenceDataset(args.corpus)
    loader = DataLoader(dataset, batch_size=cfg["ood_eval"].get("batch_size", 4), collate_fn=collate)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    baseline_model = load_baseline(cfg).to(device)
    baseline_state = torch.load(args.baseline, map_location=device)
    baseline_model.load_state_dict(baseline_state)

    fh_model = load_fh_model(cfg).to(device)
    fh_state = torch.load(args.fh_rl, map_location=device)
    fh_model.load_state_dict(fh_state)

    baseline_loss = evaluate(baseline_model, loader, device)
    fh_loss = evaluate(fh_model, loader, device)

    def loss_to_ppl(loss):
        return float(torch.exp(torch.tensor(loss)).item())

    summary = {
        "baseline_loss": baseline_loss,
        "baseline_ppl": loss_to_ppl(baseline_loss),
        "fh_rl_loss": fh_loss,
        "fh_rl_ppl": loss_to_ppl(fh_loss),
    }
    print(json.dumps(summary, indent=2))

    out_path = Path(cfg["training"]["output_dir"]) / "ood_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
