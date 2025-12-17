import torch
import torch.nn as nn

class PlanReportHead(nn.Module):
    def __init__(self, hidden_size: int, plan_dim: int, report_dim: int) -> None:
        super().__init__()
        self.plan_classifier = nn.Linear(hidden_size, plan_dim)
        self.report_decoder = nn.Linear(hidden_size, report_dim)

    def forward(self, hidden_states: torch.Tensor):
        plan_logits = self.plan_classifier(hidden_states)
        report_logits = self.report_decoder(hidden_states)
        return plan_logits, report_logits
