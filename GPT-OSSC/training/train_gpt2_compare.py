from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_transformer.config import MetaControllerConfig
from meta_transformer.models.gpt2_baseline import BaselineGPT2LM, Gpt2BaselineConfig
from meta_transformer.models.gpt2_meta_transformer import Gpt2MetaConfig, Gpt2MetaTransformerLM
from meta_transformer.training.gpt2_text_dataset import create_gpt2_text_dataloaders
from meta_transformer.training.training_utils import clip_gradients, set_seed


@dataclass
class Gpt2CompareTrainingConfig:
    batch_size: int = 4
    seq_len: int = 128
    num_epochs: int = 2
    steps_per_epoch: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 50
    tokenizer_name: str = "distilgpt2"
    train_paths: Tuple[str, ...] = ()
    val_paths: Tuple[str, ...] = ()


def _prepare_dataloaders(cfg: Gpt2CompareTrainingConfig) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
]:
    if not cfg.train_paths or not cfg.val_paths:
        default = ["README.md", "docs/architecture.md", "docs/workspace_training.md"]
        train_paths = [str((ROOT / p).resolve()) for p in default]
        val_paths = train_paths
    else:
        train_paths = list(cfg.train_paths)
        val_paths = list(cfg.val_paths)
    return create_gpt2_text_dataloaders(
        tokenizer_name=cfg.tokenizer_name,
        train_paths=train_paths,
        val_paths=val_paths,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        stride=cfg.seq_len,
    )


def _train_lm(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Gpt2CompareTrainingConfig,
    device: torch.device,
) -> List[float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    data_iter = iter(dataloader)
    for epoch in range(cfg.num_epochs):
        for step in range(cfg.steps_per_epoch):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Invalid loss during GPT-2 training")
            loss.backward()
            clip_gradients(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            if cfg.log_every > 0 and (len(losses) % cfg.log_every) == 0:
                print(f"[epoch {epoch} step {step}] step={len(losses)} loss={loss.item():.4f}", flush=True)
    return losses


def _evaluate_lm(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Invalid loss during GPT-2 evaluation")
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def _analyze_meta_gates(
    model: Gpt2MetaTransformerLM,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    import math

    model.eval()
    inputs, _ = next(iter(dataloader))
    descriptors, gates = model.analyze_gates(inputs)
    desc = descriptors.detach().cpu()
    g = gates.detach().cpu()
    num_layers, num_heads, d_dim = desc.shape
    print(f"[analysis] descriptors shape={desc.shape}, gates shape={g.shape}", flush=True)
    # Correlation between gate and entropy across heads/layers.
    entropy = desc[..., 0].reshape(-1)
    gate_vals = g.reshape(-1)
    entropy_mean = entropy.mean().item()
    gate_mean = gate_vals.mean().item()
    num = ((entropy - entropy_mean) * (gate_vals - gate_mean)).sum().item()
    den = math.sqrt(((entropy - entropy_mean) ** 2).sum().item() * ((gate_vals - gate_mean) ** 2).sum().item() + 1e-8)
    corr_entropy = num / den if den > 0 else 0.0
    print(f"[analysis] gate mean={gate_mean:.4f}, entropy-gate corr={corr_entropy:.4f}", flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline GPT-2 vs meta-attention GPT-2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--tokenizer-name", type=str, default="distilgpt2")
    return parser


def main(cli_args: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(cli_args)
    cfg = Gpt2CompareTrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        tokenizer_name=args.tokenizer_name,
    )
    set_seed(cfg.seed)
    train_loader, val_loader = _prepare_dataloaders(cfg)
    device = torch.device("cuda")

    # Baseline GPT-2 (LM head fine-tune).
    print("=== Training baseline GPT-2 ===", flush=True)
    baseline_cfg = Gpt2BaselineConfig(base_model_name=cfg.tokenizer_name, device="cuda")
    baseline_model = BaselineGPT2LM(baseline_cfg)
    baseline_opt = torch.optim.Adam(
        baseline_model.model.lm_head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    _train_lm(baseline_model, baseline_opt, train_loader, cfg, device)
    baseline_val_loss = _evaluate_lm(baseline_model, val_loader, device)
    baseline_ppl = float(torch.exp(torch.tensor(baseline_val_loss)))
    print(f"Baseline GPT-2 val loss={baseline_val_loss:.4f}, ppl={baseline_ppl:.2f}", flush=True)

    # Meta-attention GPT-2.
    print("=== Training meta-attention GPT-2 ===", flush=True)
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    meta_cfg = Gpt2MetaConfig(
        base_model_name=cfg.tokenizer_name,
        descriptor_dim=4,
        controller=controller_cfg,
        device="cuda",
    )
    meta_model = Gpt2MetaTransformerLM(meta_cfg)
    meta_params = list(meta_model.controller.parameters())
    # Also train LM head so it can adapt to gated attention.
    if hasattr(meta_model.model, "lm_head"):
        meta_params += list(meta_model.model.lm_head.parameters())
    meta_opt = torch.optim.Adam(
        meta_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    _train_lm(meta_model, meta_opt, train_loader, cfg, device)
    meta_val_loss = _evaluate_lm(meta_model, val_loader, device)
    meta_ppl = float(torch.exp(torch.tensor(meta_val_loss)))
    print(f"Meta-attention GPT-2 val loss={meta_val_loss:.4f}, ppl={meta_ppl:.2f}", flush=True)
    improvement = baseline_val_loss - meta_val_loss
    print(f"Meta-attention improvement (positive better)={improvement:.4f}", flush=True)

    _analyze_meta_gates(meta_model, val_loader)


if __name__ == "__main__":
    main()
