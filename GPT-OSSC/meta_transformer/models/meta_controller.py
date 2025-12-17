from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from meta_transformer.config import MetaControllerConfig


@dataclass
class MetaControllerOutput:
    """Bundle of gate activations and policy statistics."""

    gates: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor | None = None
    state: Optional["MetaControllerState"] = None


@dataclass
class MetaControllerState:
    """Tracks recurrent controller context across reasoning steps."""

    hidden: torch.Tensor
    step: int = 0


class MetaTokenEmbedding(nn.Module):
    """Embeds descriptors with learned layer/head positional signals."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        descriptor_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = nn.Linear(descriptor_dim, hidden_size)
        self.layer_embedding = nn.Embedding(num_layers, hidden_size)
        self.head_embedding = nn.Embedding(num_heads, hidden_size)
        layer_indices = torch.arange(num_layers).unsqueeze(1).repeat(1, num_heads)
        head_indices = torch.arange(num_heads).unsqueeze(0).repeat(num_layers, 1)
        self.register_buffer("layer_indices", layer_indices.view(-1), persistent=False)
        self.register_buffer("head_indices", head_indices.view(-1), persistent=False)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        seq = descriptors.view(-1, descriptors.size(-1))
        tokens = self.input_proj(seq)
        layer_emb = self.layer_embedding(self.layer_indices.to(descriptors.device))
        head_emb = self.head_embedding(self.head_indices.to(descriptors.device))
        return tokens + layer_emb + head_emb


class MetaController(nn.Module):
    """Transformer encoder over head descriptors producing gates."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        config: MetaControllerConfig,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.config = config
        self.embedding = MetaTokenEmbedding(
            num_layers=num_layers,
            num_heads=num_heads,
            descriptor_dim=config.descriptor_dim,
            hidden_size=config.hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.gate_proj = nn.Linear(config.hidden_size, 1)
        self.gate_logit_scale = config.gate_logit_scale
        self.initial_gate_value = float(max(1e-3, min(1.0 - 1e-3, config.initial_gate_value)))
        self._init_gate_projection()
        self.workspace_mixer = nn.Linear(config.hidden_size, config.hidden_size)
        self.introspection_mixer = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_process_state = config.use_process_state
        if self.use_process_state:
            self.process_gru = nn.GRUCell(config.hidden_size, config.process_hidden_size)
            self.process_bias = nn.Linear(config.process_hidden_size, 1)
        else:
            self.process_gru = None
            self.process_bias = None

    def forward(
        self,
        descriptors: torch.Tensor,
        *,
        sample: bool = False,
        return_details: bool = False,
        state: Optional[MetaControllerState] = None,
        return_state: bool = False,
        workspace_summary: Optional[torch.Tensor] = None,
        introspection_summary: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, MetaControllerOutput]:
        if descriptors.dim() != 3:
            raise ValueError("descriptors must have shape [num_layers, num_heads, D]")
        meta_tokens = self.embedding(descriptors)
        meta_tokens = meta_tokens.unsqueeze(1)
        tokens = meta_tokens
        prepend_workspace = False
        if workspace_summary is not None:
            if workspace_summary.shape[-1] != self.config.hidden_size:
                raise ValueError("workspace_summary must have dimension equal to controller hidden size")
            workspace_token = self.workspace_mixer(workspace_summary).unsqueeze(0).unsqueeze(1)
            tokens = torch.cat([workspace_token, tokens], dim=0)
            prepend_workspace = True
        prepend_intro = False
        if introspection_summary is not None and self.config.use_introspection_token:
            intro_token = self.introspection_mixer(introspection_summary).unsqueeze(0).unsqueeze(1)
            tokens = torch.cat([intro_token, tokens], dim=0)
            prepend_intro = True
        encoded = self.encoder(tokens)
        encoded_tokens = encoded.squeeze(1)
        if prepend_workspace or prepend_intro:
            offset = int(prepend_workspace) + int(prepend_intro)
            encoded_tokens = encoded[offset:].squeeze(1)
        else:
            encoded_tokens = encoded.squeeze(1)
        logits = self.gate_proj(encoded_tokens)
        process_state: Optional[MetaControllerState] = None
        if self.use_process_state:
            summary = encoded_tokens.mean(dim=0, keepdim=False)
            hidden = summary.new_zeros(self.config.process_hidden_size)
            step = 0
            if state is not None:
                hidden = state.hidden
                step = state.step
            updated_hidden = self.process_gru(summary.unsqueeze(0), hidden.unsqueeze(0)).squeeze(0)
            step += 1
            process_state = MetaControllerState(hidden=updated_hidden, step=step)
            bias = self.process_bias(updated_hidden)
            logits = logits + bias
        gate_logits = self.gate_logit_scale * logits
        gate_probs = torch.sigmoid(gate_logits).view(self.num_layers, self.num_heads)
        log_probs: torch.Tensor | None = None
        if sample:
            bernoulli = torch.distributions.Bernoulli(probs=gate_probs.clamp(1e-6, 1 - 1e-6))
            gates = bernoulli.sample()
            if return_details:
                log_probs = bernoulli.log_prob(gates)
        else:
            gates = gate_probs
        if return_details or return_state:
            return MetaControllerOutput(gates=gates, probs=gate_probs, log_probs=log_probs, state=process_state)
        return gates

    def init_state(self, device: torch.device) -> MetaControllerState:
        if not self.use_process_state:
            raise RuntimeError("process state requested but controller was not configured with use_process_state=True")
        hidden = torch.zeros(self.config.process_hidden_size, device=device)
        return MetaControllerState(hidden=hidden, step=0)

    def _init_gate_projection(self) -> None:
        nn.init.zeros_(self.gate_proj.weight)
        if self.gate_proj.bias is not None:
            bias = self._bias_from_gate_value(self.initial_gate_value)
            nn.init.constant_(self.gate_proj.bias, bias)

    def _bias_from_gate_value(self, value: float) -> float:
        value = float(max(1e-5, min(1.0 - 1e-5, value)))
        logit = math.log(value / (1.0 - value))
        if self.gate_logit_scale == 0:
            return 0.0
        return logit / max(1e-6, self.gate_logit_scale)
