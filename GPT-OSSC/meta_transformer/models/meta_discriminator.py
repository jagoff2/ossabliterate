from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConsistencyDiscriminator(nn.Module):
    """Scores consistency between current report and historical state."""

    def __init__(self, summary_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.summary_proj = nn.Linear(summary_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.report_encoder = nn.GRU(
            input_size=summary_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        summary_vec: torch.Tensor,
        prior_summary: Optional[torch.Tensor],
        prior_score: Optional[torch.Tensor],
        report_embedding: torch.Tensor,
    ) -> torch.Tensor:
        current = self.summary_proj(summary_vec)
        prior = self.summary_proj(prior_summary) if prior_summary is not None else torch.zeros_like(current)
        score_vec = prior_score if prior_score is not None else torch.zeros_like(current)
        context = torch.cat([current, prior, score_vec], dim=-1)
        context = self.context_proj(context)
        report_hidden, _ = self.report_encoder(report_embedding.unsqueeze(0))
        report_latent = report_hidden[:, -1, :]
        joint = torch.tanh(context + report_latent)
        return self.classifier(joint).squeeze(-1)
