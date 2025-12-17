from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from meta_transformer.config import MetaTransformerConfig
from meta_transformer.models.base_transformer import (
    TokenPositionalEmbedding,
    TransformerBlock,
)
from meta_transformer.models.meta_controller import MetaController


class MetaTransformerLM(nn.Module):
    """Two-pass transformer LM with a global meta-attention controller."""

    def __init__(self, config: MetaTransformerConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base
        if base.device != "cuda":
            raise RuntimeError("MetaTransformerLM requires CUDA device")
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
        controller_cfg = config.controller
        self.meta_controller = MetaController(
            num_layers=base.num_layers,
            num_heads=base.num_heads,
            config=controller_cfg,
        )
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        attn_mask = self._build_attention_mask(input_ids, attention_mask)
        embeddings = self.embedding(input_ids)
        first_pass_hidden = embeddings
        layer_descriptors = []
        for block in self.blocks:
            first_pass_hidden, descriptors = block(
                first_pass_hidden, attn_mask, head_gates=None, collect_descriptors=True
            )
            if descriptors is None:
                raise RuntimeError("Descriptor collection failed in first pass")
            layer_descriptors.append(descriptors)
        descriptor_tensor = torch.stack(layer_descriptors, dim=0)
        head_gates = self.meta_controller(descriptor_tensor)
        second_pass_hidden = embeddings
        for layer_idx, block in enumerate(self.blocks):
            second_pass_hidden, _ = block(
                second_pass_hidden,
                attn_mask,
                head_gates=head_gates[layer_idx],
                collect_descriptors=False,
            )
        if self.config.final_layer_norm:
            second_pass_hidden = self.final_ln(second_pass_hidden)
        logits = self.lm_head(second_pass_hidden)
        return logits, head_gates

    def _build_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            mask = attention_mask.bool()
        return mask.to(self.device)
