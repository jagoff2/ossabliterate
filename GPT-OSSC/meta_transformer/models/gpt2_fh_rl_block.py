from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fh_rl_layer import FastWeightsHomeostaticReentryLayer, FHRLState


class CausalSelfAttentionWithQKV(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = tensor.shape
        return tensor.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        attn_scores = attn_scores + causal
        if attention_mask is not None:
            mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.out_proj(context)
        output = self.resid_dropout(output)
        return output, q, k, v


@dataclass
class GPT2FHRLBlockState:
    fh_state: Optional[FHRLState]


class GPT2FHRLBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        num_heads: int,
        dropout: float,
        fh_rank: int,
        fh_alpha: float,
        fh_beta: float,
        fh_gamma: float,
        noise_std: float = 1e-4,
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttentionWithQKV(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_dropout=dropout,
            resid_dropout=dropout,
        )
        self.fh_rl = FastWeightsHomeostaticReentryLayer(
            hidden_size=hidden_size,
            rank=fh_rank,
            alpha=fh_alpha,
            beta=fh_beta,
            gamma=fh_gamma,
            noise_std=noise_std,
            detach_feedback=detach_feedback,
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
        state: Optional[GPT2FHRLBlockState] = None,
    ) -> Tuple[torch.Tensor, GPT2FHRLBlockState, dict]:
        ln1 = self.ln1(x)
        attn_out, q, k, v = self.attn(ln1, attention_mask)
        x = x + attn_out

        q_seq = q.transpose(1, 2).contiguous()
        k_seq = k.transpose(1, 2).contiguous()
        v_seq = v.transpose(1, 2).contiguous()
        v_merge = v_seq.view(v_seq.size(0), v_seq.size(1), -1)
        fh_state = state.fh_state if state else None
        fh_out, new_state, fh_metrics = self.fh_rl(x, q_seq, k_seq, v_merge, fh_state)
        x = fh_out

        residual = x
        mlp_out = self.mlp(self.ln2(x))
        x = residual + mlp_out
        return x, GPT2FHRLBlockState(fh_state=new_state), fh_metrics

    def init_state(self, batch_size: int, device: torch.device) -> GPT2FHRLBlockState:
        return GPT2FHRLBlockState(fh_state=self.fh_rl.reset_state(batch_size, device))
