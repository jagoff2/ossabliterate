from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from meta_transformer.config import MetaTransformerConfig
from meta_transformer.models.base_transformer import (
    TokenPositionalEmbedding,
    TransformerBlock,
)


class BaselineTransformerLM(nn.Module):
    """Single-pass decoder-only Transformer LM without meta-attention controller.

    This model shares the same base stack as MetaTransformerLM but does not
    perform a second gated pass or global meta-attention. It serves as a
    baseline for comparison in training and evaluation.
    """

    def __init__(self, config: MetaTransformerConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base
        if base.device != "cuda":
            raise RuntimeError("BaselineTransformerLM requires CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but required")
        self.device = torch.device(base.device)
        self.embedding = TokenPositionalEmbedding(
            vocab_size=base.vocab_size,
            hidden_size=base.hidden_size,
            max_seq_len=base.max_seq_len,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=base.hidden_size,
                    mlp_hidden_size=base.mlp_hidden_size,
                    num_heads=base.num_heads,
                    dropout=base.dropout,
                    descriptor_top_k=base.descriptor_top_k,
                )
                for _ in range(base.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(base.hidden_size)
        self.lm_head = nn.Linear(base.hidden_size, base.vocab_size, bias=False)
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        attn_mask = self._build_attention_mask(input_ids, attention_mask)
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden, _ = block(
                hidden,
                attn_mask,
                head_gates=None,
                collect_descriptors=False,
            )
        if self.config.final_layer_norm:
            hidden = self.final_ln(hidden)
        logits = self.lm_head(hidden)
        base = self.config.base
        dummy_gates = torch.ones(
            base.num_layers, base.num_heads, device=self.device, dtype=logits.dtype
        )
        return logits, dummy_gates

    def _build_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            mask = attention_mask.bool()
        return mask.to(self.device)

