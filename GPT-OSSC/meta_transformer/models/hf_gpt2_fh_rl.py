from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from .fh_rl_layer import FastWeightsHomeostaticReentryLayer


@dataclass
class FHRLGPT2Config:
    base_model_name: str = "gpt2"
    fh_rank: int = 32
    fh_alpha: float = 0.2
    fh_beta: float = 0.1
    fh_gamma: float = 0.0
    noise_std: float = 1e-4
    detach_feedback: bool = True


class GPT2BlockWithFHRL(GPT2Block):
    def __init__(self, config: GPT2Config, fh_cfg: FHRLGPT2Config) -> None:
        super().__init__(config)
        self.last_fh_metrics: dict[str, float] = {}
        self.fh_rl = FastWeightsHomeostaticReentryLayer(
            hidden_size=config.hidden_size,
            rank=fh_cfg.fh_rank,
            alpha=fh_cfg.fh_alpha,
            beta=fh_cfg.fh_beta,
            gamma=fh_cfg.fh_gamma,
            noise_std=fh_cfg.noise_std,
            detach_feedback=fh_cfg.detach_feedback,
        )

    def _merge_qkv(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).contiguous().view(tensor.size(0), tensor.size(2), -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states_norm = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states_norm,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        # Compute Q/K/V from the same normalized input
        qkv = self.attn.c_attn(hidden_states_norm)
        query, key, value = qkv.split(self.attn.split_size, dim=2)
        query = self.attn._split_heads(query, self.attn.num_heads, self.attn.head_dim)
        key = self.attn._split_heads(key, self.attn.num_heads, self.attn.head_dim)
        value = self.attn._split_heads(value, self.attn.num_heads, self.attn.head_dim)
        q_seq = self._merge_qkv(query)
        k_seq = self._merge_qkv(key)
        v_seq = self._merge_qkv(value)

        fh_out, _, fh_metrics = self.fh_rl(hidden_states, q_seq, k_seq, v_seq, state=None)
        self.last_fh_metrics = fh_metrics
        hidden_states = fh_out

        if encoder_hidden_states is not None and hasattr(self, "crossattention"):
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


def load_fh_rl_gpt2(fh_cfg: FHRLGPT2Config) -> GPT2LMHeadModel:
    base = GPT2LMHeadModel.from_pretrained(fh_cfg.base_model_name)
    config = base.config
    for idx, block in enumerate(base.transformer.h):
        new_block = GPT2BlockWithFHRL(config, fh_cfg)
        missing, unexpected = new_block.load_state_dict(block.state_dict(), strict=False)
        if missing:
            # only FH-RL weights should be missing; ignore
            pass
        if unexpected:
            raise ValueError(f"Unexpected keys when loading block {idx}: {unexpected}")
        base.transformer.h[idx] = new_block
    return base
