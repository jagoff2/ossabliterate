#!/usr/bin/env python3
"""Training entry point for GPT-2 + FH-RL tiny model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

from meta_transformer.models.gpt2_fh_rl_block import GPT2FHRLBlock


@dataclass
class TinyFHRLConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float
    fh_rank: int
    fh_alpha: float
    fh_beta: float
    fh_gamma: float
    noise_std: float
    detach_feedback: bool
    seq_len: int


class ByteSequenceDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int) -> None:
        self.records: List[Dict[str, List[int]]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        input_ids = torch.tensor(record["input_ids"], dtype=torch.long)
        labels = torch.tensor(record["labels"], dtype=torch.long)
        return input_ids, labels


class TinyFHRLModel(nn.Module):
    def __init__(self, cfg: TinyFHRLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                GPT2FHRLBlock(
                    hidden_size=cfg.hidden_size,
                    mlp_hidden_size=cfg.mlp_hidden_size,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    fh_rank=cfg.fh_rank,
                    fh_alpha=cfg.fh_alpha,
                    fh_beta=cfg.fh_beta,
                    fh_gamma=cfg.fh_gamma,
                    noise_std=cfg.noise_std,
                    detach_feedback=cfg.detach_feedback,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        track_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float], Optional[torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        metrics_accum: Dict[str, float] = {"irr": 0.0}
        trace = x.detach() if track_metrics else None

        states: List = [None] * len(self.blocks)
        for idx, block in enumerate(self.blocks):
            x, block_state, fh_metrics = block(x, attention_mask=None, state=states[idx])
            states[idx] = block_state
            metrics_accum["irr"] += fh_metrics.get("avg_irr", 0.0)
            if track_metrics:
                trace = x.detach()

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(bsz * (seq_len - 1), -1),
                labels[:, 1:].reshape(-1),
            )
        metrics_accum["irr"] /= max(len(self.blocks), 1)
        return logits, loss, metrics_accum, trace

    def set_gamma(self, gamma: float) -> None:
        for block in self.blocks:
            block.fh_rl.gamma = gamma


def compute_esri(trace: torch.Tensor) -> float:
    """Approximate ESRI via cosine distance between successive covariance spectra."""
    bsz, seq_len, hidden = trace.shape
    covs = []
    for t in range(seq_len):
        ht = trace[:, t, :]
        ht = ht - ht.mean(dim=0, keepdim=True)
        cov = (ht.t() @ ht) / max(bsz - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).real
        covs.append(eigvals)
    distances = []
    for t in range(seq_len - 1):
        a = covs[t]
        b = covs[t + 1]
        denom = a.norm() * b.norm() + 1e-6
        distances.append(1.0 - (a * b).sum() / denom)
    return float(torch.stack(distances).mean().item()) if distances else 0.0


def compute_rdp(trace: torch.Tensor) -> float:
    sims = []
    for t in range(trace.size(1) - 1):
        a = trace[:, t, :].reshape(-1)
        b = trace[:, t + 1, :].reshape(-1)
        denom = a.norm() * b.norm() + 1e-6
        sims.append((a * b).sum() / denom)
    if not sims:
        return 0.0
    sim_tensor = torch.stack(sims)
    freq = torch.fft.rfft(sim_tensor)
    magnitudes = freq.abs()
    return float(magnitudes.max().item())


def cycle(data_loader: Iterable):
    while True:
        for batch in data_loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fh_rl_tiny.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg_data = json.loads(Path(args.config).read_text())
    model_cfg = TinyFHRLConfig(
        vocab_size=cfg_data["model"]["vocab_size"],
        hidden_size=cfg_data["model"]["hidden_size"],
        num_layers=cfg_data["model"]["num_layers"],
        num_heads=cfg_data["model"]["num_heads"],
        mlp_hidden_size=cfg_data["model"]["mlp_hidden_size"],
        dropout=cfg_data["model"]["dropout"],
        fh_rank=cfg_data["model"]["fh_rank"],
        fh_alpha=cfg_data["model"]["fh_alpha"],
        fh_beta=cfg_data["model"]["fh_beta"],
        fh_gamma=cfg_data["model"]["fh_gamma"],
        noise_std=cfg_data["model"]["noise_std"],
        detach_feedback=cfg_data["model"]["detach_feedback"],
        seq_len=cfg_data["training"]["seq_len"],
    )

    dataset = ByteSequenceDataset(cfg_data["training"]["corpus_path"], model_cfg.seq_len)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg_data["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_data["training"]["batch_size"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TinyFHRLModel(model_cfg).to(device)

    output_dir = Path(cfg_data["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for gamma in cfg_data["training"].get("gamma_sweep", [model_cfg.fh_gamma]):
            model.set_gamma(gamma)
            optimizer = AdamW(
                model.parameters(),
                lr=cfg_data["training"]["lr"],
                betas=(cfg_data["training"]["beta1"], cfg_data["training"]["beta2"]),
                weight_decay=cfg_data["training"]["weight_decay"],
            )
            train_iter = cycle(train_loader)
            max_steps = cfg_data["training"]["max_steps"]
            log_interval = cfg_data["training"].get("log_interval", 10)
            eval_interval = cfg_data["training"].get("eval_interval", 100)
            metrics_path = output_dir / f"metrics_gamma_{gamma:.2f}.jsonl"

            with metrics_path.open("w", encoding="utf-8") as metrics_fp:
                for step in range(1, max_steps + 1):
                    optimizer.zero_grad()
                    batch = next(train_iter)
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device)
                    logits, loss, metrics, trace = model(input_ids, labels=labels, track_metrics=True)
                    assert loss is not None
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    esri = compute_esri(trace.cpu()) if trace is not None else 0.0
                    rdp = compute_rdp(trace.cpu()) if trace is not None else 0.0

                    entry = {
                        "phase": "train",
                        "step": step,
                        "gamma": gamma,
                        "loss": float(loss.item()),
                        "irr": float(metrics["irr"]),
                        "esri": float(esri),
                        "rdp": float(rdp),
                    }
                    metrics_fp.write(json.dumps(entry) + "\n")
                    metrics_fp.flush()

                    if step % log_interval == 0:
                        print(
                            f"gamma={gamma:.2f} step={step} loss={loss.item():.4f} irr={metrics['irr']:.3f} esri={esri:.3f} rdp={rdp:.3f}",
                            flush=True,
                        )

                    if step % eval_interval == 0:
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for batch in val_loader:
                                input_ids = batch[0].to(device)
                                labels = batch[1].to(device)
                                _, val_loss, val_metrics, trace = model(
                                    input_ids, labels=labels, track_metrics=True
                                )
                                assert val_loss is not None
                                val_losses.append(val_loss.item())
                            avg_val = sum(val_losses) / len(val_losses)
                            print(f"[eval] gamma={gamma:.2f} val_loss={avg_val:.4f}")
                            eval_entry = {
                                "phase": "eval",
                                "step": step,
                                "gamma": gamma,
                                "val_loss": float(avg_val),
                            }
                            metrics_fp.write(json.dumps(eval_entry) + "\n")
                            metrics_fp.flush()
                        model.train()

            ckpt_path = output_dir / f"fh_rl_gamma_{gamma:.2f}.pt"
            torch.save({"model_state": model.state_dict(), "gamma": gamma}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
