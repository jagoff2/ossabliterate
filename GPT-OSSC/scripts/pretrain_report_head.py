from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from meta_transformer.config import MetaControllerConfig
from meta_transformer.models.gpt2_meta_transformer import Gpt2MetaConfig, Gpt2MetaTransformerLM
from meta_transformer.training.reasoning_tasks import load_reasoning_tasks


def _build_model(base_model: str, device: str) -> Gpt2MetaTransformerLM:
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    model_cfg = Gpt2MetaConfig(
        base_model_name=base_model,
        descriptor_dim=4,
        controller=controller_cfg,
        device=device,
    )
    return Gpt2MetaTransformerLM(model_cfg)


def _prepare_targets(tokenizer, text: str, length: int, device: torch.device) -> torch.Tensor:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=length,
    ).input_ids.to(device)
    if encoded.size(1) < length:
        pad = torch.full((encoded.size(0), length - encoded.size(1)), pad_id, dtype=encoded.dtype, device=device)
        encoded = torch.cat([encoded, pad], dim=1)
    else:
        encoded = encoded[:, :length]
    return encoded.view(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain report head/controller on gold reports")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to reasoning tasks JSON/JSONL with gold reports")
    parser.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save-path", type=Path, default=Path("report_pretrain.pt"))
    args = parser.parse_args()

    tasks = load_reasoning_tasks(str(args.tasks))
    gold_tasks = [task for task in tasks if task.gold_report]
    if not gold_tasks:
        raise RuntimeError("No tasks contain gold_report entries; cannot pretrain.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = _build_model(args.base_model, args.device).to(device)
    model.train()
    params = list(model.controller.parameters()) + list(model.report_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    report_len = model.config.introspector.report_length

    for epoch in range(args.epochs):
        total_loss = 0.0
        for task in gold_tasks:
            inputs = tokenizer(task.prompt, return_tensors="pt", truncation=True).to(device)
            summary_vec, _, _, _ = model.get_introspection_state(inputs.input_ids, inputs.attention_mask)
            logits, _ = model.report_head(summary_vec)
            target = _prepare_targets(tokenizer, task.gold_report, report_len, device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss={total_loss/len(gold_tasks):.4f}")

    payload = {
        "controller": model.controller.state_dict(),
        "report_head": model.report_head.state_dict(),
    }
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save_path)
    print(f"Saved pretraining weights to {args.save_path}")


if __name__ == "__main__":
    main()
