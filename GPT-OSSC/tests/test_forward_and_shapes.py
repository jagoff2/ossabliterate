from __future__ import annotations

import torch

from meta_transformer.config import MetaControllerConfig, MetaTransformerConfig, TransformerConfig
from meta_transformer.models.meta_transformer import MetaTransformerLM


def test_meta_transformer_forward_shapes() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this test")
    base_cfg = TransformerConfig(
        vocab_size=128,
        max_seq_len=16,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        mlp_hidden_size=128,
        dropout=0.0,
        descriptor_top_k=2,
    )
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    config = MetaTransformerConfig(base=base_cfg, controller=controller_cfg)
    model = MetaTransformerLM(config)
    batch_size, seq_len = 2, 8
    dummy_input = torch.randint(0, base_cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    logits, gates = model(dummy_input)
    assert logits.shape == (batch_size, seq_len, base_cfg.vocab_size)
    assert gates.shape == (base_cfg.num_layers, base_cfg.num_heads)
    assert torch.all(gates > 0.0) and torch.all(gates < 1.0)
