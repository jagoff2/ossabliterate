from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2SdpaAttention, logger

from meta_transformer.config import MetaControllerConfig, MetaIntrospectorConfig, MetaWorkspaceConfig
from meta_transformer.models.meta_controller import MetaController, MetaControllerOutput, MetaControllerState
from meta_transformer.models.meta_introspector import MetaAttentionIntrospector
from meta_transformer.models.meta_report_head import MetaReportHead
from meta_transformer.models.meta_workspace import MetaWorkspace, MetaWorkspaceState

TRACE_CLASSIFIER_DIM = 128


_ORIGINAL_GPT2_ATTENTION_FORWARD = GPT2Attention.forward


def _apply_value_gates(
    value_states: torch.Tensor,
    meta_gates: Optional[torch.Tensor],
    meta_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if meta_gates is None:
        gated = value_states
    else:
        gates = meta_gates.to(device=value_states.device, dtype=value_states.dtype)
        gates = gates.view(1, -1, 1, 1)
        gated = value_states * gates
    if meta_bias is not None:
        bias = meta_bias.to(device=value_states.device, dtype=value_states.dtype)
        bias = bias.view(1, -1, 1, bias.size(-1))
        gated = gated + bias
    return gated
    gates = meta_gates.to(device=value_states.device, dtype=value_states.dtype)
    gates = gates.view(1, -1, 1, 1)
    return value_states * gates


def _patch_gpt2_attention_for_meta() -> None:
    """Monkey-patch GPT2Attention to support per-head gating via `meta_head_gates`."""

    if not getattr(GPT2Attention, "_meta_patched", False):

        signature = inspect.signature(GPT2Attention.forward)

        if "layer_past" in signature.parameters:

            def forward_with_gates(
                self,
                hidden_states: Optional[Tuple[torch.FloatTensor]],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
            ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
                if encoder_hidden_states is not None:
                    if not hasattr(self, "q_attn"):
                        raise ValueError(
                            "If class is used as cross attention, the weights `q_attn` have to be defined. "
                            "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                        )

                    query = self.q_attn(hidden_states)
                    key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                    attention_mask = encoder_attention_mask
                else:
                    query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

                query = self._split_heads(query, self.num_heads, self.head_dim)
                key = self._split_heads(key, self.num_heads, self.head_dim)
                value = self._split_heads(value, self.num_heads, self.head_dim)
                value = _apply_value_gates(
                    value,
                    getattr(self, "meta_head_gates", None),
                    getattr(self, "meta_value_bias", None),
                )

                if layer_past is not None:
                    past_key, past_value = layer_past
                    key = torch.cat((past_key, key), dim=-2)
                    value = torch.cat((past_value, value), dim=-2)

                if use_cache is True:
                    present = (key, value)
                else:
                    present = None

                if self.reorder_and_upcast_attn:
                    attn_output, attn_weights = self._upcast_and_reordered_attn(
                        query, key, value, attention_mask, head_mask
                    )
                else:
                    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

                attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
                attn_output = self.c_proj(attn_output)
                attn_output = self.resid_dropout(attn_output)

                outputs = (attn_output, present)
                if output_attentions:
                    outputs += (attn_weights,)

                return outputs

        else:

            def forward_with_gates(
                self,
                hidden_states: torch.Tensor,
                past_key_values: Optional[torch.Tensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                from transformers.cache_utils import Cache, EncoderDecoderCache  # local import
                from transformers.models.gpt2.modeling_gpt2 import (
                    eager_attention_forward,
                )
                from transformers.utils.deprecation import deprecate_kwarg  # noqa: F401

                try:
                    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS
                except ImportError:
                    ALL_ATTENTION_FUNCTIONS = {"eager": eager_attention_forward}

                is_cross_attention = encoder_hidden_states is not None
                if past_key_values is not None:
                    if isinstance(past_key_values, EncoderDecoderCache):
                        is_updated = past_key_values.is_updated.get(self.layer_idx)
                        if is_cross_attention:
                            curr_past_key_value = past_key_values.cross_attention_cache
                        else:
                            curr_past_key_value = past_key_values.self_attention_cache
                    else:
                        curr_past_key_value = past_key_values

                if is_cross_attention:
                    if not hasattr(self, "q_attn"):
                        raise ValueError(
                            "If class is used as cross attention, the weights `q_attn` have to be defined. "
                            "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                        )
                    query_states = self.q_attn(hidden_states)
                    attention_mask = encoder_attention_mask

                    if past_key_values is not None and is_updated:
                        key_states = curr_past_key_value.layers[self.layer_idx].keys
                        value_states = curr_past_key_value.layers[self.layer_idx].values
                    else:
                        key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                        key_states = key_states.view(shape_kv).transpose(1, 2)
                        value_states = value_states.view(shape_kv).transpose(1, 2)
                else:
                    query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
                    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                    key_states = key_states.view(shape_kv).transpose(1, 2)
                    value_states = value_states.view(shape_kv).transpose(1, 2)

                shape_q = (*query_states.shape[:-1], -1, self.head_dim)
                query_states = query_states.view(shape_q).transpose(1, 2)

                if (past_key_values is not None and not is_cross_attention) or (
                    past_key_values is not None and is_cross_attention and not is_updated
                ):
                    cache_kwargs = {"cache_position": cache_position} if not is_cross_attention else None
                    key_states, value_states = curr_past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                    if is_cross_attention:
                        past_key_values.is_updated[self.layer_idx] = True

                value_states = _apply_value_gates(
                    value_states,
                    getattr(self, "meta_head_gates", None),
                    getattr(self, "meta_value_bias", None),
                )

                is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

                attention_interface = eager_attention_forward
                if self.config._attn_implementation != "eager" and ALL_ATTENTION_FUNCTIONS is not None:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
                attn_output = self.c_proj(attn_output)
                attn_output = self.resid_dropout(attn_output)
                return attn_output, attn_weights

        GPT2Attention.forward = forward_with_gates  # type: ignore[assignment]
        GPT2Attention._meta_patched = True  # type: ignore[attr-defined]

    if not getattr(GPT2SdpaAttention, "_meta_patched", False):

        def sdpa_forward_with_gates(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
            if output_attentions or head_mask is not None:
                logger.warning_once(
                    "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` "
                    "does not support `output_attentions=True` or `head_mask`. Falling back to the manual attention "
                    "implementation, but specifying the manual implementation will be required from Transformers "
                    "version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation=\"eager\"` "
                    "when loading the model."
                )
                return super(GPT2SdpaAttention, self).forward(  # type: ignore[misc]
                    hidden_states=hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            bsz, q_len, _ = hidden_states.size()
            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention:
                if not hasattr(self, "q_attn"):
                    raise ValueError(
                        "If class is used as cross attention, the weights `q_attn` have to be defined. "
                        "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                    )
                query = self.q_attn(hidden_states)
                key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                attention_mask = encoder_attention_mask
            else:
                query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)
            value = _apply_value_gates(
                value,
                getattr(self, "meta_head_gates", None),
                getattr(self, "meta_value_bias", None),
            )

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            present = None
            if use_cache is True:
                present = (key, value)

            if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

            is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.embed_dim)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, present, None

        GPT2SdpaAttention.forward = sdpa_forward_with_gates  # type: ignore[assignment]
        GPT2SdpaAttention._meta_patched = True  # type: ignore[attr-defined]


@dataclass
class Gpt2MetaConfig:
    base_model_name: str = "sshleifer/tiny-gpt2"
    descriptor_dim: int = 4
    controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    workspace: MetaWorkspaceConfig = field(default_factory=MetaWorkspaceConfig)
    introspector: MetaIntrospectorConfig = field(default_factory=MetaIntrospectorConfig)
    device: str = "cuda"
    gradient_checkpointing: bool = False


class Gpt2MetaTransformerLM(nn.Module):
    """Two-pass GPT-2 LM with global meta-attention controller over head descriptors."""

    def __init__(self, config: Gpt2MetaConfig) -> None:
        super().__init__()
        _patch_gpt2_attention_for_meta()
        self.config = config
        self.device = torch.device(config.device)
        base_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.num_layers = base_cfg.n_layer
        self.num_heads = base_cfg.n_head
        self.hidden_size = getattr(base_cfg, "hidden_size", base_cfg.n_embd)
        self.head_dim = self.hidden_size // self.num_heads
        base_cfg.output_attentions = True
        # Force eager attention so patched GPT2Attention is always used.
        setattr(base_cfg, "attn_implementation", "eager")
        base_cfg._attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name, config=base_cfg)
        self.model.to(self.device)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            setattr(self.model.config, "gradient_checkpointing", True)
            self.model.config.use_cache = False
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.controller = MetaController(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            config=config.controller,
        ).to(self.device)
        workspace_cfg = MetaWorkspaceConfig(
            descriptor_dim=config.descriptor_dim,
            num_slots=config.workspace.num_slots,
            slot_dim=config.workspace.slot_dim,
            summary_dim=config.controller.hidden_size,
            track_trace=config.workspace.track_trace,
        )
        self.workspace = MetaWorkspace(workspace_cfg).to(self.device)
        introspector_cfg = config.introspector
        self.introspector = MetaAttentionIntrospector(
            introspector_cfg,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).to(self.device)
        self.introspection_bridge = nn.Linear(
            introspector_cfg.hidden_size,
            config.controller.hidden_size,
        ).to(self.device)
        trace_hidden = max(64, introspector_cfg.hidden_size // 2)
        self.trace_classifier_dim = TRACE_CLASSIFIER_DIM
        self.trace_classifier = nn.Sequential(
            nn.Linear(introspector_cfg.hidden_size, trace_hidden),
            nn.GELU(),
            nn.Linear(trace_hidden, self.trace_classifier_dim),
        ).to(self.device)
        vocab_size = self.model.config.vocab_size
        self.report_head = MetaReportHead(
            summary_dim=introspector_cfg.hidden_size,
            hidden_dim=self.hidden_size,
            vocab_size=vocab_size,
            report_length=introspector_cfg.report_length,
        ).to(self.device)
        self.diagnosis_head = nn.Linear(introspector_cfg.hidden_size, 1).to(self.device)
        judge_hidden = max(64, introspector_cfg.hidden_size // 2)
        self.judge_head = nn.Sequential(
            nn.Linear(introspector_cfg.hidden_size, judge_hidden),
            nn.GELU(),
            nn.Linear(judge_hidden, 1),
        ).to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        sample_gates: bool = False,
        return_gate_details: bool = False,
        controller_state: Optional[MetaControllerState] = None,
        return_controller_state: bool = False,
        workspace_state: Optional[MetaWorkspaceState] = None,
        return_workspace_state: bool = False,
        record_workspace_trace: bool = False,
        force_open_gates: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor | MetaControllerOutput],
        Tuple[torch.Tensor, torch.Tensor | MetaControllerOutput, Optional[MetaControllerState]],
        Tuple[
            torch.Tensor,
            torch.Tensor | MetaControllerOutput,
            Optional[MetaControllerState],
            Optional[MetaWorkspaceState],
        ],
    ]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        self._set_head_gates(None, None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("GPT-2 model did not return attentions; ensure output_attentions=True")
        descriptors = self._compute_descriptors(attentions)
        introspection_tokens, introspection_summary, value_bias = self.introspector(attentions)
        trace_logits = self.trace_classifier(introspection_summary)
        workspace_out = self.workspace(
            descriptors,
            state=workspace_state,
            record_trace=return_workspace_state or record_workspace_trace,
        )
        workspace_summary = workspace_out.summary
        need_details = return_gate_details or sample_gates or return_controller_state
        controller_out = self.controller(
            descriptors,
            sample=sample_gates,
            return_details=need_details,
            state=controller_state,
            return_state=return_controller_state,
            workspace_summary=workspace_summary,
            introspection_summary=self.introspection_bridge(introspection_summary),
        )
        if isinstance(controller_out, MetaControllerOutput):
            gates = controller_out.gates
        else:
            gates = controller_out
            controller_out = None
        if force_open_gates:
            logits = outputs.logits
        else:
            self._set_head_gates(gates, value_bias)
            outputs_gated = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs_gated.logits
            self._set_head_gates(None, None)
        output_gates: torch.Tensor | MetaControllerOutput
        if return_gate_details and controller_out is not None:
            output_gates = controller_out
        else:
            output_gates = gates
        controller_state_next = controller_out.state if (controller_out is not None and return_controller_state) else None
        workspace_state_next = workspace_out.state if return_workspace_state else None
        outputs: Tuple = (logits, output_gates)
        extras: List = []
        if return_controller_state:
            extras.append(controller_state_next)
        if return_workspace_state:
            extras.append(workspace_state_next)
        if extras:
            return outputs + tuple(extras)
        return outputs

    def _set_head_gates(self, gates: Optional[torch.Tensor], value_bias: Optional[torch.Tensor]) -> None:
        transformer = getattr(self.model, "transformer", None)
        if transformer is None or not hasattr(transformer, "h"):
            return
        layers = transformer.h
        for idx, block in enumerate(layers):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            if gates is None:
                attn.meta_head_gates = None
            else:
                attn.meta_head_gates = gates[idx]
            if value_bias is None:
                attn.meta_value_bias = None
            else:
                attn.meta_value_bias = value_bias[idx]

    def _compute_descriptors(self, attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute per-head descriptors from GPT-2 layer attentions."""

        num_layers = len(attentions)
        descs = []
        for layer_idx in range(num_layers):
            attn = attentions[layer_idx]  # [batch, heads, seq, seq]
            if attn.dtype != torch.float32:
                attn = attn.to(torch.float32)
            bsz, num_heads, q_len, k_len = attn.shape
            eps = 1e-9
            entropy = -(attn * (attn.clamp_min(eps).log())).sum(dim=-1)
            entropy = entropy.mean(dim=(0, 2))
            positions = torch.arange(k_len, device=attn.device, dtype=torch.float32)
            dist = (positions[None, :] - positions[:, None]).abs()
            dist = dist.view(1, 1, k_len, k_len)
            mean_distance = (attn * dist).sum(dim=-1).mean(dim=(0, 2))
            sq_dist = dist ** 2
            mean_square = (attn * sq_dist).sum(dim=-1).mean(dim=(0, 2))
            std_distance = (mean_square - mean_distance**2).clamp_min(0.0).sqrt()
            top_k = min(4, k_len)
            topk_mass = attn.topk(top_k, dim=-1).values.sum(dim=-1).mean(dim=(0, 2))
            layer_desc = torch.stack(
                [entropy, mean_distance, std_distance, topk_mass], dim=-1
            )  # [heads, 4]
            descs.append(layer_desc)
        descriptors = torch.stack(descs, dim=0)
        return descriptors.to(device=self.device)

    @torch.no_grad()
    def analyze_gates(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return descriptors and gates for analysis without a second pass."""

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        self._set_head_gates(None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("GPT-2 model did not return attentions")
        descriptors = self._compute_descriptors(attentions)
        gates = self.controller(descriptors)
        return descriptors, gates

    @torch.no_grad()
    def generate_introspection_report(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("GPT-2 model did not return attentions")
        _, summary, _ = self.introspector(attentions)
        logits, tokens = self.report_head(summary, temperature=temperature)
        return logits, tokens

    def get_introspection_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("GPT-2 model did not return attentions")
        tokens, summary, bias = self.introspector(attentions)
        return summary, tokens, attentions, bias

    def get_introspection_summary(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        summary, _, _, _ = self.get_introspection_state(input_ids, attention_mask)
        return summary

    @torch.no_grad()
    def extract_focus_terms(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        tokenizer,
        top_k: int = 4,
    ) -> List[str]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            return []
        scores = None
        for layer_attn in attentions:
            layer_score = layer_attn.mean(dim=(0, 1, 2))
            scores = layer_score if scores is None else scores + layer_score
        if scores is None:
            return []
        scores = scores / len(attentions)
        seq_len = int(scores.shape[0])
        top_k = min(top_k, seq_len)
        values, indices = torch.topk(scores, k=top_k)
        tokens: List[str] = []
        seen = set()
        for idx in indices.tolist():
            token_id = int(input_ids[0, idx].item())
            token = tokenizer.convert_ids_to_tokens(token_id)
            token = token.strip()
            if not token or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
        return tokens

    def diagnosis_prediction(self, summary_vec: torch.Tensor) -> torch.Tensor:
        return self.diagnosis_head(summary_vec)

    def judge_prediction(self, summary_vec: torch.Tensor) -> torch.Tensor:
        return self.judge_head(summary_vec)
