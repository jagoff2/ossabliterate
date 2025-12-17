from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_transformer.config import MetaControllerConfig, MetaIntrospectorConfig, MetaWorkspaceConfig
from meta_transformer.models.gpt2_meta_transformer import Gpt2MetaConfig, Gpt2MetaTransformerLM


class MetaWarmupSample:
    def __init__(self, input_ids: List[int], labels: List[int], attention_mask: List[int], report_ids: List[int]) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.report_ids = report_ids


class MetaWarmupDataset(Dataset):
    def __init__(self, entries: List[dict], tokenizer, max_length: int, report_length: int) -> None:
        self.samples: List[MetaWarmupSample] = []
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        for entry in entries:
            prompt = entry.get("prompt", "").strip()
            completion = entry.get("gold_completion", "").strip()
            if not prompt or not completion:
                continue
            prompt_ids = tokenizer.encode(prompt + "\n\n", add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            if not completion_ids:
                continue
            seq = prompt_ids + completion_ids
            if len(seq) >= max_length:
                seq = seq[: max_length - 1]
            seq.append(eos_id)
            if len(seq) > max_length:
                seq = seq[:max_length]
            labels = list(seq)
            for idx in range(min(len(prompt_ids), len(labels))):
                labels[idx] = -100
            attention_mask = [1] * len(seq)
            report_text = entry.get("gold_report", "").strip() or completion
            report_ids = tokenizer.encode(report_text, add_special_tokens=False)
            if not report_ids:
                report_ids = [pad_id]
            if len(report_ids) >= report_length:
                report_ids = report_ids[:report_length]
            else:
                report_ids = report_ids + [pad_id] * (report_length - len(report_ids))
            self.samples.append(MetaWarmupSample(seq, labels, attention_mask, report_ids))
        if not self.samples:
            raise RuntimeError("no usable samples for warmup")
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MetaWarmupSample:
        return self.samples[idx]


def _collate(batch: List[MetaWarmupSample], pad_id: int) -> dict:
    max_len = max(len(sample.input_ids) for sample in batch)
    input_ids = []
    labels = []
    attention_mask = []
    report_ids = []
    for sample in batch:
        pad_amount = max_len - len(sample.input_ids)
        seq_ids = sample.input_ids + [pad_id] * pad_amount
        seq_labels = sample.labels + [-100] * pad_amount
        seq_mask = sample.attention_mask + [0] * pad_amount
        input_ids.append(torch.tensor(seq_ids, dtype=torch.long))
        labels.append(torch.tensor(seq_labels, dtype=torch.long))
        attention_mask.append(torch.tensor(seq_mask, dtype=torch.long))
        report_ids.append(torch.tensor(sample.report_ids, dtype=torch.long))
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "report_ids": torch.stack(report_ids, dim=0),
    }


def _load_entries(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)
    if not entries:
        raise RuntimeError(f"no entries found in {path}")
    return entries


def build_model(base_model: str, device: str) -> Gpt2MetaTransformerLM:
    controller_cfg = MetaControllerConfig(descriptor_dim=4, hidden_size=128, num_layers=2, num_heads=4, dropout=0.1)
    workspace_cfg = MetaWorkspaceConfig(descriptor_dim=4, num_slots=4, slot_dim=128, summary_dim=128, track_trace=True)
    introspector_cfg = MetaIntrospectorConfig(pool_size=8, hidden_size=256, num_layers=2, num_heads=4, dropout=0.1, report_length=32)
    config = Gpt2MetaConfig(
        base_model_name=base_model,
        descriptor_dim=4,
        controller=controller_cfg,
        workspace=workspace_cfg,
        introspector=introspector_cfg,
        device=device,
    )
    model = Gpt2MetaTransformerLM(config)
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def train(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    entries = _load_entries(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    dataset = MetaWarmupDataset(entries, tokenizer, args.max_length, args.report_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, dataset.pad_id),
    )
    model = build_model(args.base_model, args.device)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(dataloader)
    step = 0
    pad_id = tokenizer.pad_token_id or 0
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits, _ = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sample_gates=False,
                return_gate_details=False,
                force_open_gates=True,
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            summary, _, _, _ = model.get_introspection_state(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            report_logits, _ = model.report_head(summary)
            if report_logits.dim() == 2:
                report_logits = report_logits.unsqueeze(1)
            else:
                report_logits = report_logits.permute(1, 0, 2)
            report_loss = F.cross_entropy(
                report_logits.reshape(-1, report_logits.size(-1)),
                batch["report_ids"].reshape(-1),
                ignore_index=pad_id,
            )
            loss = lm_loss + args.report_weight * report_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_loss += loss.item()
            if args.log_every and (step % args.log_every == 0):
                print(f"[epoch {epoch} step {step}/{total_steps}] loss={loss.item():.4f} lm={lm_loss.item():.4f} report={report_loss.item():.4f}")
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} avg loss {avg_loss:.4f}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "meta_stack_warmup.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved warmup checkpoint to {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised warmup for GPT-2 meta transformer stack")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--report-length", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--report-weight", type=float, default=0.3)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
