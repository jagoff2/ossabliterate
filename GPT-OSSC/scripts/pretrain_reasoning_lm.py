from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_entries(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            completion = obj.get("gold_completion", "").strip()
            if not completion:
                continue
            entries.append(obj)
    if not entries:
        raise RuntimeError(f"no usable entries with gold_completion in {path}")
    return entries


class ReasoningTextDataset(Dataset):
    def __init__(self, entries: List[dict], tokenizer, max_length: int) -> None:
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        example = self.entries[idx]
        prompt = example["prompt"].strip()
        completion = example["gold_completion"].strip()
        prompt_ids = self.tokenizer.encode(prompt + "\n\n", add_special_tokens=False)
        eos = self.tokenizer.eos_token or ""
        completion_ids = self.tokenizer.encode(completion + eos, add_special_tokens=False)
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: List[dict], pad_id: int) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_needed = max_len - item["input_ids"].size(0)
        if pad_needed > 0:
            pad_tensor = torch.full((pad_needed,), pad_id, dtype=torch.long)
            input_ids.append(torch.cat([item["input_ids"], pad_tensor], dim=0))
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_needed, dtype=torch.long)], dim=0)
            )
            labels_pad = torch.full((pad_needed,), -100, dtype=torch.long)
            labels.append(torch.cat([item["labels"], labels_pad], dim=0))
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def train(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    entries = _load_entries(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device(args.device)
    model.to(device)
    torch.manual_seed(args.seed)
    dataset = ReasoningTextDataset(entries, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id or 0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.epochs * len(dataloader)
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation
            if (global_step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.log_every and (global_step % args.log_every == 0):
                print(
                    f"[epoch {epoch} step {global_step}/{total_steps}] loss={loss.item() * args.gradient_accumulation:.4f}"
                )
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} avg loss {avg_loss:.4f}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretraining for reasoning LM using gold completions.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="gpt2")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = train(args)
    print(f"Saved pretrained model to {output_dir}")


if __name__ == "__main__":
    main()
