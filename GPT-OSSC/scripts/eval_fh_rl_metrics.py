#!/usr/bin/env python3
"""Offline metric computation for FH-RL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from training.train_fh_rl import (
    ByteSequenceDataset,
    TinyFHRLConfig,
    TinyFHRLModel,
    compute_esri,
    compute_rdp,
)


def load_config(config_path: str) -> TinyFHRLConfig:
    cfg = json.loads(Path(config_path).read_text())
    return TinyFHRLConfig(
        vocab_size=cfg["model"]["vocab_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        mlp_hidden_size=cfg["model"]["mlp_hidden_size"],
        dropout=cfg["model"]["dropout"],
        fh_rank=cfg["model"]["fh_rank"],
        fh_alpha=cfg["model"]["fh_alpha"],
        fh_beta=cfg["model"]["fh_beta"],
        fh_gamma=cfg["model"]["fh_gamma"],
        noise_std=cfg["model"]["noise_std"],
        detach_feedback=cfg["model"]["detach_feedback"],
        seq_len=cfg["training"]["seq_len"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fh_rl_tiny.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=64, help="Number of sequences to evaluate")
    args = parser.parse_args()

    cfg_json = json.loads(Path(args.config).read_text())
    cfg = load_config(args.config)
    dataset = ByteSequenceDataset(cfg_json["training"]["corpus_path"], cfg.seq_len)
    indices = torch.randperm(len(dataset))[: args.limit]
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=8)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TinyFHRLModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.set_gamma(ckpt.get("gamma", cfg.fh_gamma))
    model.eval()

    irr_vals = []
    esri_vals = []
    rdp_vals = []
    with torch.no_grad():
        for input_ids, _ in loader:
            input_ids = input_ids.to(device)
            _, _, metrics, trace = model(input_ids, labels=None, track_metrics=True)
            irr_vals.append(metrics["irr"])
            if trace is not None:
                esri_vals.append(compute_esri(trace.cpu()))
                rdp_vals.append(compute_rdp(trace.cpu()))

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "irr": avg(irr_vals),
                "esri": avg(esri_vals),
                "rdp": avg(rdp_vals),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
