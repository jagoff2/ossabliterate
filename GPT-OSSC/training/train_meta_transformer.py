from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_transformer.config import MetaControllerConfig, MetaTransformerConfig, TransformerConfig
from meta_transformer.models.baseline_transformer import BaselineTransformerLM
from meta_transformer.models.meta_transformer import MetaTransformerLM
from meta_transformer.training.dummy_dataset import (
    create_dummy_dataloader,
    create_real_data_dataloader,
)
from meta_transformer.training.training_utils import clip_gradients, set_seed


@dataclass
class TrainingConfig:
    """Hyperparameters governing the training routine."""

    batch_size: int = 4
    num_samples: int = 256
    seq_len: int = 32
    vocab_size: int = 256
    num_epochs: int = 1
    steps_per_epoch: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 5
    corpus_paths: Tuple[str, ...] = ()


def build_model_and_optimizer(
    model_config: MetaTransformerConfig, training_config: TrainingConfig
) -> Tuple[MetaTransformerLM, Optimizer]:
    """Instantiate the MetaTransformerLM and its optimizer."""

    model = MetaTransformerLM(model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    return model, optimizer


def train(
    model: MetaTransformerLM,
    optimizer: Optimizer,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    training_config: TrainingConfig,
) -> List[float]:
    """Run full-training epochs and return tracked losses."""

    model.train()
    losses: List[float] = []
    criterion = nn.CrossEntropyLoss()
    data_iter = iter(dataloader)
    for epoch in range(training_config.num_epochs):
        for step in range(training_config.steps_per_epoch):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            optimizer.zero_grad(set_to_none=True)
            logits, gates = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Encountered invalid loss value during training")
            loss.backward()
            clip_gradients(model.parameters(), training_config.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            if training_config.log_every > 0 and (len(losses) % training_config.log_every) == 0:
                gate_mean = gates.mean().item()
                gate_std = gates.std().item()
                print(
                    f"[epoch {epoch} step {step}] step={len(losses)} "
                    f"loss={loss.item():.4f} gate_mean={gate_mean:.3f} gate_std={gate_std:.3f}",
                    flush=True,
                )
    return losses


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MetaTransformerLM on dummy data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--corpus-path",
        action="append",
        default=[],
        help="Path(s) to text files for real-data training (can be passed multiple times)",
    )
    return parser


def _build_default_model_config(seq_len: int, vocab_size: int) -> MetaTransformerConfig:
    base_cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max(seq_len, 32),
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        mlp_hidden_size=256,
        dropout=0.1,
    )
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    return MetaTransformerConfig(base=base_cfg, controller=controller_cfg)


def _prepare_dataloader(cfg: TrainingConfig) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    if cfg.corpus_paths:
        return create_real_data_dataloader(
            corpus_paths=cfg.corpus_paths,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
        )
    return create_dummy_dataloader(
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
    )


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    max_steps: int | None = None,
) -> float:
    """Compute mean cross-entropy loss over a validation iterator."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Invalid loss during evaluation")
            losses.append(loss.item())
    if not losses:
        raise RuntimeError("No batches were evaluated")
    return float(sum(losses) / len(losses))


def run_baseline_vs_meta_comparison(
    corpus_paths: Tuple[str, ...],
    base_cfg: TransformerConfig,
    controller_cfg: MetaControllerConfig,
    training_cfg: TrainingConfig,
) -> None:
    """Train baseline and meta models on real data and report validation losses."""

    # Shared train/validation loaders from the same corpus.
    train_loader = create_real_data_dataloader(
        corpus_paths=corpus_paths,
        batch_size=training_cfg.batch_size,
        seq_len=training_cfg.seq_len,
        vocab_size=training_cfg.vocab_size,
    )
    val_loader = create_real_data_dataloader(
        corpus_paths=corpus_paths,
        batch_size=training_cfg.batch_size,
        seq_len=training_cfg.seq_len,
        vocab_size=training_cfg.vocab_size,
    )

    # Baseline model.
    print("=== Training baseline TransformerLM ===", flush=True)
    baseline_cfg = MetaTransformerConfig(
        base=base_cfg,
        controller=controller_cfg,
        final_layer_norm=True,
    )
    set_seed(training_cfg.seed)
    baseline_model = BaselineTransformerLM(baseline_cfg)
    baseline_opt = torch.optim.Adam(
        baseline_model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    _ = train(baseline_model, baseline_opt, train_loader, training_cfg)
    baseline_val_loss = evaluate(baseline_model, val_loader, max_steps=50)
    print(f"Baseline validation loss: {baseline_val_loss:.4f}", flush=True)

    # Meta-attention model.
    print("=== Training MetaTransformerLM (two-pass) ===", flush=True)
    set_seed(training_cfg.seed)
    meta_cfg = MetaTransformerConfig(
        base=base_cfg,
        controller=controller_cfg,
        final_layer_norm=True,
    )
    meta_model, meta_opt = build_model_and_optimizer(meta_cfg, training_cfg)
    _ = train(meta_model, meta_opt, train_loader, training_cfg)
    meta_val_loss = evaluate(meta_model, val_loader, max_steps=50)
    print(f"MetaTransformer validation loss: {meta_val_loss:.4f}", flush=True)

    improvement = baseline_val_loss - meta_val_loss
    print(f"MetaTransformer improvement over baseline (positive is better): {improvement:.4f}", flush=True)


def main(cli_args: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(cli_args)
    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        corpus_paths=tuple(args.corpus_path),
    )
    set_seed(training_cfg.seed)
    base_cfg = TransformerConfig(
        vocab_size=training_cfg.vocab_size,
        max_seq_len=max(training_cfg.seq_len, 32),
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        mlp_hidden_size=256,
        dropout=0.1,
    )
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    if training_cfg.corpus_paths:
        run_baseline_vs_meta_comparison(
            corpus_paths=training_cfg.corpus_paths,
            base_cfg=base_cfg,
            controller_cfg=controller_cfg,
            training_cfg=training_cfg,
        )
    else:
        dataloader = _prepare_dataloader(training_cfg)
        model_config = MetaTransformerConfig(base=base_cfg, controller=controller_cfg)
        model, optimizer = build_model_and_optimizer(model_config, training_cfg)
        losses = train(model, optimizer, dataloader, training_cfg)
        print(f"Completed training with final loss {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
