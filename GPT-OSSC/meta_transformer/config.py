from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    """Configuration for the base decoder-only transformer."""

    vocab_size: int = 512
    max_seq_len: int = 64
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_hidden_size: int = 256
    dropout: float = 0.1
    device: str = "cuda"
    pad_token_id: int = 0
    descriptor_top_k: int = 4


@dataclass
class MetaControllerConfig:
    """Configuration for the meta-attention controller."""

    descriptor_dim: int = 4
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_process_state: bool = False
    process_hidden_size: int = 128
    use_introspection_token: bool = True
    gate_logit_scale: float = 0.1
    initial_gate_value: float = 0.9


@dataclass
class MetaWorkspaceConfig:
    """Configuration for the global meta-attention workspace."""

    descriptor_dim: int = 4
    num_slots: int = 4
    slot_dim: int = 128
    summary_dim: int = 128
    track_trace: bool = True
    track_graph: bool = True


@dataclass
class MetaIntrospectorConfig:
    """Configuration for the attention introspector module."""

    pool_size: int = 8
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    report_length: int = 16


@dataclass
class MetaTransformerConfig:
    """Bundle configuration for the meta-transformer LM."""

    base: TransformerConfig = field(default_factory=TransformerConfig)
    controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    introspector: MetaIntrospectorConfig = field(default_factory=MetaIntrospectorConfig)
    final_layer_norm: bool = True
