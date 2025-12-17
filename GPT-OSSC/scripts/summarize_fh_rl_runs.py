#!/usr/bin/env python3
"""Summarize FH-RL checkpoints (baseline vs reflective)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from training.train_fh_rl import (  # noqa: E402
    ByteSequenceDataset,
    TinyFHRLConfig,
    TinyFHRLModel,
    compute_esri,
    compute_rdp,
)


def load_configs(config_path: str) -> tuple[dict, TinyFHRLConfig]:
    cfg_json = json.loads(Path(config_path).read_text())
    cfg = TinyFHRLConfig(
        vocab_size=cfg_json["model"]["vocab_size"],
        hidden_size=cfg_json["model"]["hidden_size"],
        num_layers=cfg_json["model"]["num_layers"],
        num_heads=cfg_json["model"]["num_heads"],
        mlp_hidden_size=cfg_json["model"]["mlp_hidden_size"],
        dropout=cfg_json["model"]["dropout"],
        fh_rank=cfg_json["model"]["fh_rank"],
        fh_alpha=cfg_json["model"]["fh_alpha"],
        fh_beta=cfg_json["model"]["fh_beta"],
        fh_gamma=cfg_json["model"]["fh_gamma"],
        noise_std=cfg_json["model"]["noise_std"],
        detach_feedback=cfg_json["model"]["detach_feedback"],
        seq_len=cfg_json["training"]["seq_len"],
    )
    return cfg_json, cfg


def evaluate_checkpoint(
    ckpt_path: Path,
    cfg_json: dict,
    cfg: TinyFHRLConfig,
    device: torch.device,
    limit: int,
) -> Dict[str, float]:
    dataset = ByteSequenceDataset(cfg_json["training"]["corpus_path"], cfg.seq_len)
    if limit > 0 and limit < len(dataset):
        indices = torch.randperm(len(dataset))[:limit]
        subset = Subset(dataset, indices.tolist())
    else:
        subset = dataset
    loader = DataLoader(subset, batch_size=min(32, len(subset)))

    model = TinyFHRLModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.set_gamma(state.get("gamma", cfg.fh_gamma))
    model.eval()

    irr_vals: List[float] = []
    esri_vals: List[float] = []
    rdp_vals: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            _, loss, metrics, trace = model(input_ids, labels=labels, track_metrics=True)
            if loss is not None:
                losses.append(loss.item())
            irr_vals.append(metrics["irr"])
            if trace is not None:
                esri_vals.append(compute_esri(trace.cpu()))
                rdp_vals.append(compute_rdp(trace.cpu()))

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "checkpoint": ckpt_path.name,
        "gamma": float(state.get("gamma", cfg.fh_gamma)),
        "loss": avg(losses),
        "irr": avg(irr_vals),
        "esri": avg(esri_vals),
        "rdp": avg(rdp_vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    cfg_json, cfg = load_configs(args.config)
    run_dir = Path(args.runs)
    checkpoints = sorted(run_dir.glob("fh_rl_gamma_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {run_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for ckpt in checkpoints:
        rows.append(evaluate_checkpoint(ckpt, cfg_json, cfg, device, args.limit))

    header = f"{'checkpoint':>25}  {'gamma':>6}  {'loss':>8}  {'irr':>8}  {'esri':>8}  {'rdp':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['checkpoint']:>25}  {row['gamma']:6.2f}  {row['loss']:8.4f}  {row['irr']:8.4f}  {row['esri']:8.4f}  {row['rdp']:10.4f}"
        )

    summary_path = run_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
