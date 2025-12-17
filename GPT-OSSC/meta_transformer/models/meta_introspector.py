from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from meta_transformer.config import MetaIntrospectorConfig


class MetaAttentionIntrospector(nn.Module):
    """Consumes attention tensors and produces guidance plus summary."""

    def __init__(
        self,
        config: MetaIntrospectorConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool = nn.AdaptiveAvgPool2d(config.pool_size)
        token_dim = num_heads * config.pool_size * config.pool_size
        self.token_proj = nn.Linear(token_dim, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.summary_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.bias_proj = nn.Linear(config.hidden_size, num_heads * head_dim)

    def forward(self, attentions: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens: List[torch.Tensor] = []
        for layer_attn in attentions:
            attn = layer_attn.mean(dim=0)  # [heads, seq, seq]
            if attn.dtype != torch.float32:
                attn = attn.to(torch.float32)
            pooled = self.pool(attn)
            token = pooled.reshape(-1)
            tokens.append(self.token_proj(token))
        token_seq = torch.stack(tokens, dim=0).unsqueeze(0)
        encoded = self.encoder(token_seq).squeeze(0)
        summary = self.summary_proj(encoded.mean(dim=0))
        bias = self.bias_proj(encoded).view(self.num_layers, self.num_heads, self.head_dim)
        return encoded, summary, bias
