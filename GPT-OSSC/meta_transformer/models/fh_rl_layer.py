import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FHRLState:
    U: torch.Tensor  # (batch, rank, dim)
    V: torch.Tensor  # (batch, rank, dim)

    @staticmethod
    def init(batch_size: int, rank: int, dim: int, device: torch.device) -> "FHRLState":
        zeros = torch.zeros(batch_size, rank, dim, device=device)
        return FHRLState(U=zeros.clone(), V=zeros.clone())


class FastWeightsHomeostaticReentryLayer(nn.Module):
    """Implements FH-RL as described in Chae (2025)."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 32,
        alpha: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.1,
        noise_std: float = 1e-4,
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_std = noise_std
        self.detach_feedback = detach_feedback

        self.reentry_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.reentry_proj.weight, gain=math.sqrt(0.5))

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-6)

    def _homeostasis(self, y: torch.Tensor) -> torch.Tensor:
        norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = 1.0 / (1.0 + self.beta * (norm - 1.0))
        return y * scale

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v_value: torch.Tensor,
        state: Optional[FHRLState] = None,
    ) -> Tuple[torch.Tensor, FHRLState, dict]:
        """
        Args:
            x: (batch, seq, dim) hidden input after attention residual.
            q, k: (batch, seq, heads, dim_head) projections.
            v_value: (batch, seq, dim) value embeddings (post attention value proj).
            state: Optional fast-weight state (per batch).
        Returns:
            updated hidden, new state, metrics dict.
        """

        bsz, seq_len, dim = x.shape
        device = x.device
        if state is None:
            state = FHRLState.init(bsz, self.rank, dim, device)

        U = state.U
        V = state.V
        metrics = {
            "avg_irr": 0.0,
            "avg_in_norm": 0.0,
            "avg_feedback_norm": 0.0,
        }

        q_flat = q.reshape(bsz, seq_len, dim)
        k_flat = k.reshape(bsz, seq_len, dim)

        outputs = []
        total_feedback = 0.0
        total_input = 0.0

        for t in range(seq_len):
            q_t = q_flat[:, t, :]
            k_t = k_flat[:, t, :]
            v_t = v_value[:, t, :]
            x_t = x[:, t, :]

            if self.noise_std > 0:
                noise_u = torch.randn_like(q_t) * self.noise_std
                noise_v = torch.randn_like(k_t) * self.noise_std
            else:
                noise_u = torch.zeros_like(q_t)
                noise_v = torch.zeros_like(k_t)

            U = (1 - self.alpha) * U + self.alpha * self._normalize(q_t.unsqueeze(1) + noise_u.unsqueeze(1))
            V = (1 - self.alpha) * V + self.alpha * self._normalize(k_t.unsqueeze(1) + noise_v.unsqueeze(1))

            w_eff = torch.matmul(U.transpose(1, 2), V)  # (batch, dim, dim)
            y_t = torch.matmul(w_eff, v_t.unsqueeze(-1)).squeeze(-1)
            y_t = self._homeostasis(y_t)

            feedback = self.gamma * self.reentry_proj(y_t)
            if self.detach_feedback:
                feedback = feedback.detach()

            total_feedback += feedback.norm(dim=-1).mean().item()
            total_input += x_t.norm(dim=-1).mean().item()

            outputs.append(x_t + feedback)

        out = torch.stack(outputs, dim=1)

        seq_count = seq_len if seq_len > 0 else 1
        metrics["avg_feedback_norm"] = total_feedback / seq_count
        metrics["avg_in_norm"] = total_input / seq_count
        metrics["avg_irr"] = metrics["avg_feedback_norm"] / (metrics["avg_in_norm"] + 1e-6)

        new_state = FHRLState(U=U.detach(), V=V.detach())
        return out, new_state, metrics

    def reset_state(self, batch_size: int, device: torch.device) -> FHRLState:
        return FHRLState.init(batch_size, self.rank, self.hidden_size, device)
