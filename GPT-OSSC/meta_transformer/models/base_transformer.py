from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class TokenPositionalEmbedding(nn.Module):
    """Token + positional embedding module for decoder inputs."""

    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max_seq_len, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token(input_ids) + self.position(positions)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention with descriptor extraction."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        descriptor_top_k: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.descriptor_top_k = descriptor_top_k
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self._distance_cache: dict[Tuple[int, torch.device], torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        head_gates: Optional[torch.Tensor],
        collect_descriptors: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape_projection(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape_projection(q)
        k = shape_projection(k)
        v = shape_projection(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores + self._causal_mask(seq_len, x.device)

        if attention_mask is not None:
            # attention_mask assumed shape [batch, seq]
            padding_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)

        descriptors = None
        if collect_descriptors:
            descriptors = self._compute_descriptors(attn_probs, seq_len)

        if head_gates is not None:
            gates = head_gates.view(1, self.num_heads, 1, 1)
            context = context * gates

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        return output, descriptors

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def _compute_descriptors(self, attn_probs: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch, heads, _, _ = attn_probs.shape
        eps = 1e-9
        entropy = -(attn_probs * (attn_probs.clamp_min(eps).log())).sum(dim=-1)
        entropy = entropy.mean(dim=(0, 2))
        distances = self._distance_tensor(seq_len, attn_probs.device)
        mean_distance = (attn_probs * distances).sum(dim=-1).mean(dim=(0, 2))
        squared_distances = distances ** 2
        mean_square = (attn_probs * squared_distances).sum(dim=-1).mean(dim=(0, 2))
        std_distance = (mean_square - mean_distance ** 2).clamp_min(0.0).sqrt()
        top_k = min(self.descriptor_top_k, seq_len)
        topk_mass = attn_probs.topk(top_k, dim=-1).values.sum(dim=-1).mean(dim=(0, 2))
        descriptors = torch.stack(
            [entropy, mean_distance, std_distance, topk_mass], dim=-1
        )
        return descriptors

    def _distance_tensor(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        cached = self._distance_cache.get(key)
        if cached is None:
            positions = torch.arange(seq_len, device=device)
            dist = (positions[None, :] - positions[:, None]).abs().float()
            dist = dist.unsqueeze(0).unsqueeze(0)
            self._distance_cache[key] = dist
            cached = dist
        return cached


class TransformerBlock(nn.Module):
    """Single decoder block with attention descriptors."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        num_heads: int,
        dropout: float,
        descriptor_top_k: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            descriptor_top_k=descriptor_top_k,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        head_gates: Optional[torch.Tensor],
        collect_descriptors: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        attn_out, descriptors = self.attn(
            self.ln1(x), attention_mask, head_gates, collect_descriptors
        )
        x = residual + attn_out
        residual = x
        mlp_out = self.mlp(self.ln2(x))
        x = residual + mlp_out
        return x, descriptors
