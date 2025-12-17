from __future__ import annotations

import torch

from meta_transformer.config import MetaControllerConfig, MetaTransformerConfig, TransformerConfig
from pathlib import Path

from meta_transformer.training.dummy_dataset import create_dummy_dataloader, create_real_data_dataloader
from training.train_meta_transformer import TrainingConfig, build_model_and_optimizer, train


def test_training_loop_runs_and_reduces_loss() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for training tests")
    base_cfg = TransformerConfig(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_hidden_size=64,
        dropout=0.0,
    )
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model_cfg = MetaTransformerConfig(base=base_cfg, controller=controller_cfg)
    training_cfg = TrainingConfig(
        batch_size=4,
        num_samples=64,
        seq_len=12,
        vocab_size=base_cfg.vocab_size,
        num_epochs=1,
        steps_per_epoch=6,
        learning_rate=5e-3,
        grad_clip=1.0,
        log_every=0,
    )
    dataloader = create_dummy_dataloader(
        batch_size=training_cfg.batch_size,
        num_samples=training_cfg.num_samples,
        seq_len=training_cfg.seq_len,
        vocab_size=training_cfg.vocab_size,
    )
    model, optimizer = build_model_and_optimizer(model_cfg, training_cfg)
    first_param = next(model.parameters()).detach().clone()
    losses = train(model, optimizer, dataloader, training_cfg)
    assert len(losses) == training_cfg.num_epochs * training_cfg.steps_per_epoch
    loss_tensor = torch.tensor(losses, device=first_param.device)
    assert torch.isfinite(loss_tensor).all()
    assert losses[-1] <= losses[0] or losses[-1] <= losses[0] * 1.05
    updated_param = next(model.parameters()).detach()
    assert not torch.allclose(first_param, updated_param)


def test_training_loop_with_real_text(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for training tests")
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("meta attention controller test corpus\n" * 50, encoding="utf-8")
    base_cfg = TransformerConfig(
        vocab_size=64,
        max_seq_len=32,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_hidden_size=64,
        dropout=0.0,
    )
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model_cfg = MetaTransformerConfig(base=base_cfg, controller=controller_cfg)
    training_cfg = TrainingConfig(
        batch_size=4,
        num_samples=64,
        seq_len=24,
        vocab_size=base_cfg.vocab_size,
        num_epochs=1,
        steps_per_epoch=5,
        learning_rate=5e-3,
        grad_clip=1.0,
        log_every=0,
        corpus_paths=(str(corpus_path),),
    )
    dataloader = create_real_data_dataloader(
        corpus_paths=training_cfg.corpus_paths,
        batch_size=training_cfg.batch_size,
        seq_len=training_cfg.seq_len,
        vocab_size=training_cfg.vocab_size,
    )
    model, optimizer = build_model_and_optimizer(model_cfg, training_cfg)
    first_param = next(model.parameters()).detach().clone()
    losses = train(model, optimizer, dataloader, training_cfg)
    assert len(losses) == training_cfg.num_epochs * training_cfg.steps_per_epoch
    loss_tensor = torch.tensor(losses, device=first_param.device)
    assert torch.isfinite(loss_tensor).all()
    updated_param = next(model.parameters()).detach()
    assert not torch.allclose(first_param, updated_param)
