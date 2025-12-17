from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

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
    input_ids = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True)
    return input_ids, labels


def cycle(data_loader: Iterable):
    while True:
        for batch in data_loader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["baseline", "fh_rl"], required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_fh_rl_gpt2(FHRLGPT2Config(**cfg["model"])).to(device)

    train_ds = JsonlSequenceDataset(cfg["training"]["train_path"])
    val_ds = JsonlSequenceDataset(cfg["training"]["val_path"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], collate_fn=collate)
    train_iter = cycle(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler()

    grad_accum = cfg["training"].get("grad_accum", 1)

    for step in range(1, cfg["training"]["max_steps"] + 1):
        optimizer.zero_grad()
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            with autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["training"].get("log_interval", 50) == 0:
            print(f"step={step} loss={loss.item():.4f}")

        if step % cfg["training"].get("eval_interval", 200) == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids = val_batch[0].to(device)
                    labels = val_batch[1].to(device)
                    outputs = model(input_ids, labels=labels)
                    val_losses.append(outputs.loss.item())
            avg_val = sum(val_losses) / len(val_losses)
            print(f"[eval] step={step} val_loss={avg_val:.4f}")
            model.train()

if __name__ == "__main__":
    main()
