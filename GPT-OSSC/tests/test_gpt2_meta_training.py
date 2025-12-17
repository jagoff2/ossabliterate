from __future__ import annotations

from pathlib import Path

import torch

from transformers import AutoTokenizer

from meta_transformer.config import MetaControllerConfig
from meta_transformer.models.gpt2_meta_transformer import Gpt2MetaConfig, Gpt2MetaTransformerLM
from meta_transformer.training.reasoning_tasks import ReasoningTask
from training.train_gpt2_meta import Gpt2MetaTrainingConfig, train_gpt2_meta


def test_gpt2_meta_forward_and_training(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for GPT-2 meta tests")
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model_cfg = Gpt2MetaConfig(
        base_model_name="sshleifer/tiny-gpt2",
        descriptor_dim=4,
        controller=controller_cfg,
        device="cuda",
    )
    model = Gpt2MetaTransformerLM(model_cfg)
    training_cfg = Gpt2MetaTrainingConfig(
        num_epochs=1,
        episodes_per_epoch=2,
        learning_rate=1e-4,
        grad_clip=1.0,
        log_every=0,
        entropy_coef=0.0,
        max_new_tokens=8,
        reward_scale=1.0,
        run_dir=str(tmp_path / "run"),
    )
    batch = torch.randint(0, 32, (2, 16), dtype=torch.long)
    logits, gates = model(batch.to(model.device))
    assert logits.shape[0] == batch.shape[0]
    assert logits.shape[1] == batch.shape[1]
    assert gates.shape[0] == model.num_layers
    assert gates.shape[1] == model.num_heads
    assert torch.all((gates >= 0.0) & (gates <= 1.0))
    report_logits, report_tokens = model.generate_introspection_report(batch[:1].to(model.device))
    assert report_logits.shape[0] == model.config.introspector.report_length
    assert report_tokens.shape[-1] == model.config.introspector.report_length
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use a trivial reward structure so that REINFORCE updates emit a gradient deterministically.
    tasks = [
        ReasoningTask(
            prompt="Provide any explicit reasoning trace.",
            expected_answer="",
            reasoning_keywords=("",),
            max_new_tokens=4,
        )
    ]
    optimizer = torch.optim.Adam(model.controller.parameters(), lr=training_cfg.learning_rate)
    first_param = next(model.controller.parameters()).detach().clone()
    rewards = train_gpt2_meta(model, optimizer, tokenizer, tasks, training_cfg)
    reward_tensor = torch.tensor(rewards, device=model.device)
    assert torch.isfinite(reward_tensor).all()
    assert len(rewards) == training_cfg.num_epochs * training_cfg.episodes_per_epoch
    updated_param = next(model.controller.parameters()).detach()
    assert not torch.allclose(first_param, updated_param)
    memory_path = Path(training_cfg.run_dir) / "episode_memory.jsonl"
    assert memory_path.exists()
    report_dir = Path(training_cfg.run_dir) / "reports"
    assert report_dir.exists()
