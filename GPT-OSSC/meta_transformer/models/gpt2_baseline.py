from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class Gpt2BaselineConfig:
    base_model_name: str = "distilgpt2"
    device: str = "cuda"


class BaselineGPT2LM(nn.Module):
    """Baseline GPT-2 LM for comparison with meta-attention variant.

    The GPT-2 backbone is loaded and only the LM head is trained to keep
    parameter counts comparable to the meta-controller variant.
    """

    def __init__(self, config: Gpt2BaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        base_cfg = AutoConfig.from_pretrained(config.base_model_name)
        base_cfg.output_attentions = False
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name, config=base_cfg)
        self.model.to(self.device)
        # Freeze transformer, train only LM head.
        if hasattr(self.model, "transformer"):
            for param in self.model.transformer.parameters():
                param.requires_grad_(False)
        for param in self.model.lm_head.parameters():
            param.requires_grad_(True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits, None

