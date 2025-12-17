from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MetaReportHead(nn.Module):
    """Generates textual summaries from workspace and introspection states."""

    def __init__(
        self,
        summary_dim: int,
        hidden_dim: int,
        vocab_size: int,
        report_length: int,
    ) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.report_length = report_length
        self.init_proj = nn.Linear(summary_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, summary: torch.Tensor, *, temperature: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.init_proj(summary)
        logits_seq: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        for _ in range(self.report_length):
            hidden = self.cell(hidden)
            logits = self.decoder(hidden)
            logits_seq.append(logits)
            if temperature and temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens.append(token)
        logits_tensor = torch.stack(logits_seq, dim=0)
        tokens_tensor = torch.cat(tokens, dim=-1)
        return logits_tensor, tokens_tensor
