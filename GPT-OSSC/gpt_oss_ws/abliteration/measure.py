from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoProcessor,
  AutoTokenizer,
  BitsAndBytesConfig,
  PreTrainedModel,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast,
)

try:  # Vision-language models are optional in some transformer builds.
  from transformers import AutoModelForImageTextToText  # type: ignore
except ImportError:  # pragma: no cover - fallback for older transformers
  AutoModelForImageTextToText = AutoModelForCausalLM  # type: ignore

from .data import PromptSourceConfig, load_prompt_source
from .models import has_tied_weights
from .ops import magnitude_clip


def _maybe_empty_cache() -> None:
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


@dataclass
class MeasurementConfig:
  model_name: str
  output_path: str
  batch_size: int = 32
  clip_fraction: float = 1.0
  quant_mode: Optional[str] = None  # "4bit" | "8bit" | None
  flash_attention: bool = False
  add_deccp: bool = False
  projected: bool = False
  chat_template_path: Optional[str] = None
  sample_last_user_token: bool = False
  harmful_source: PromptSourceConfig = field(default_factory=lambda: PromptSourceConfig(path="data/harmful.parquet"))
  harmless_source: PromptSourceConfig = field(default_factory=lambda: PromptSourceConfig(path="data/harmless.parquet"))


def run_measurement(config: MeasurementConfig) -> Dict[str, torch.Tensor]:
  """Compute harmful/harmless means and refusal directions for every decoder layer."""

  harmful = list(load_prompt_source(config.harmful_source))
  harmless = list(load_prompt_source(config.harmless_source))
  if config.add_deccp:
    deccp = load_dataset("augmxnt/deccp", split="censored")
    harmful.extend(str(text) for text in deccp["text"])

  model_cfg = AutoConfig.from_pretrained(config.model_name)
  dtype = _resolve_dtype(model_cfg)
  has_vision = hasattr(model_cfg, "vision_config")

  quant_config, dtype = _resolve_quant_config(model_cfg, config.quant_mode, dtype)
  loader_cls = AutoModelForImageTextToText if has_vision else AutoModelForCausalLM
  loader_kwargs = dict(
    device_map="auto",
    torch_dtype=dtype if quant_config is None else None,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2" if config.flash_attention else None,
    trust_remote_code=True,
  )
  if quant_config is not None:
    loader_kwargs["quantization_config"] = quant_config

  model = loader_cls.from_pretrained(
    config.model_name,
    **loader_kwargs,
  )
  model.requires_grad_(False)
  model.eval()
  model_config_type = getattr(model_cfg, "model_type", "")
  if has_tied_weights(model_config_type):
    model.tie_weights()

  processor = None
  tokenizer = None
  if has_vision:
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
  if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"

  if config.chat_template_path:
    template_text = Path(config.chat_template_path).read_text(encoding="utf-8")
    tokenizer.chat_template = template_text
    if processor is not None and getattr(processor, "tokenizer", None) is not None:
      processor.tokenizer.chat_template = template_text

  harmful_formatted = _format_chats(tokenizer, harmful, processor)
  harmless_formatted = _format_chats(tokenizer, harmless, processor)

  layer_base = getattr(model, "model", model)
  if hasattr(layer_base, "language_model"):
    layer_base = layer_base.language_model
  num_layers = len(layer_base.layers)
  focus_layers = list(range(num_layers))

  harmful_stats = _welford_layers(
    harmful_formatted,
    "harmful",
    model,
    tokenizer,
    focus_layers,
    batch_size=config.batch_size,
    clip=config.clip_fraction,
    processor=processor,
    sample_last_user=config.sample_last_user_token,
  )
  _maybe_empty_cache()
  harmless_stats = _welford_layers(
    harmless_formatted,
    "harmless",
    model,
    tokenizer,
    focus_layers,
    batch_size=config.batch_size,
    clip=config.clip_fraction,
    processor=processor,
    sample_last_user=config.sample_last_user_token,
  )

  results: Dict[str, torch.Tensor | int] = {"layers": num_layers}
  for layer in tqdm(focus_layers, desc="Compiling layer measurements"):
    harmful_mean = harmful_stats[layer]
    harmless_mean = harmless_stats[layer]
    refusal_dir = harmful_mean - harmless_mean
    if config.projected:
      harmless_norm = torch.nn.functional.normalize(harmless_mean.float(), dim=0)
      projection = refusal_dir @ harmless_norm
      refusal_dir = refusal_dir - projection * harmless_norm
    results[f"harmful_{layer}"] = harmful_mean
    results[f"harmless_{layer}"] = harmless_mean
    results[f"refuse_{layer}"] = refusal_dir

  _maybe_empty_cache()
  gc.collect()

  output_path = Path(config.output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(results, output_path)
  return results


def _resolve_dtype(model_config) -> torch.dtype:
  precision = getattr(model_config, "torch_dtype", None) or getattr(model_config, "dtype", None)
  if precision is None:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  return _coerce_dtype(precision, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def _resolve_quant_config(model_config, quant_mode: Optional[str], dtype: torch.dtype):
  qmode = quant_mode
  dtype_override = dtype
  config_quant = getattr(model_config, "quantization_config", None)
  if config_quant:
    if config_quant.get("load_in_4bit"):
      qmode = "4bit"
      compute_dtype = config_quant.get("bnb_4bit_compute_dtype")
      if compute_dtype is not None:
        dtype_override = _coerce_dtype(compute_dtype, dtype_override)
    elif config_quant.get("load_in_8bit"):
      qmode = "8bit"
  if qmode == "4bit":
    return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=dtype_override,
      bnb_4bit_use_double_quant=True,
    ), dtype_override
  if qmode == "8bit":
    return BitsAndBytesConfig(load_in_8bit=True), dtype_override
  return None, dtype_override


def _coerce_dtype(value, fallback: torch.dtype) -> torch.dtype:
  if isinstance(value, torch.dtype):
    return value
  if isinstance(value, str):
    mapping = {
      "bfloat16": torch.bfloat16,
      "bf16": torch.bfloat16,
      "float16": torch.float16,
      "fp16": torch.float16,
      "float32": torch.float32,
      "fp32": torch.float32,
    }
    return mapping.get(value.lower(), fallback)
  return fallback


def _format_chats(
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  prompts: Iterable[str],
  processor=None,
) -> List[str]:
  actual_tokenizer = getattr(processor, "tokenizer", None) if processor is not None else tokenizer
  formatted = []
  for inst in prompts:
    formatted.append(
      actual_tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": inst}],
        add_generation_prompt=True,
        add_special_tokens=False,
        tokenize=False,
      )
    )
  return formatted


def _welford_layers(
  formatted_prompts: Sequence[str],
  desc: str,
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  layer_indices: Sequence[int],
  pos: int = -1,
  batch_size: int = 1,
  clip: float = 1.0,
  processor=None,
  sample_last_user: bool = False,
) -> Dict[int, torch.Tensor]:

  means: Dict[int, Optional[torch.Tensor]] = {layer_idx: None for layer_idx in layer_indices}
  counts: Dict[int, int] = {layer_idx: 0 for layer_idx in layer_indices}
  device = next(model.parameters()).device
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  assistant_id = tokenizer.convert_tokens_to_ids("assistant") if sample_last_user else None
  end_id = tokenizer.convert_tokens_to_ids("<|end|>") if sample_last_user else None

  for start in tqdm(range(0, len(formatted_prompts), batch_size), desc=f"{desc} batches"):
    batch_prompts = formatted_prompts[start:start + batch_size]
    if processor is not None:
      batch_encoding = processor(
        text=batch_prompts,
        return_tensors="pt",
        padding=True,
      )
      batch_input = batch_encoding["input_ids"].to(device)
      batch_mask = batch_encoding["attention_mask"].to(device)
    else:
      encoded = tokenizer(
        batch_prompts,
        padding=True,
        return_tensors="pt",
      )
      batch_input = encoded["input_ids"].to(device)
      batch_mask = encoded["attention_mask"].to(device)

    raw_output = model.generate(
      batch_input,
      attention_mask=batch_mask,
      max_new_tokens=1,
      return_dict_in_generate=True,
      output_hidden_states=True,
      pad_token_id=tokenizer.eos_token_id,
    )
    hidden_states = raw_output.hidden_states[0]
    del raw_output

    if sample_last_user:
      positions = _find_last_user_positions_tensor(
        batch_input,
        pad_id=pad_id,
        assistant_id=assistant_id,
        end_id=end_id,
      )
      position_tensor = torch.tensor(positions, device=hidden_states[0].device, dtype=torch.long)
    else:
      position_tensor = None

    for layer_idx in layer_indices:
      layer_hidden = hidden_states[layer_idx]
      if sample_last_user and position_tensor is not None:
        batch_indices = torch.arange(layer_hidden.size(0), device=layer_hidden.device)
        activations = layer_hidden[batch_indices, position_tensor, :].float()
      else:
        activations = layer_hidden[:, pos, :].float()
      if clip < 1.0:
        activations = magnitude_clip(activations, clip)
      batch_len = activations.size(0)
      total = counts[layer_idx] + batch_len
      if means[layer_idx] is None:
        means[layer_idx] = activations.mean(dim=0)
        counts[layer_idx] = batch_len
        continue
      delta = activations - means[layer_idx]
      means[layer_idx] = means[layer_idx] + delta.sum(dim=0) / total
      counts[layer_idx] = total

    del hidden_states, batch_input, batch_mask
    _maybe_empty_cache()

  result: Dict[int, torch.Tensor] = {}
  for layer_idx, mean in means.items():
    if mean is None:
      raise RuntimeError(f"Layer {layer_idx} produced no activations")
    result[layer_idx] = mean.to(device="cpu")
  return result


def _find_last_user_positions_tensor(
  input_ids: torch.Tensor,
  pad_id: int,
  assistant_id: Optional[int],
  end_id: Optional[int],
) -> List[int]:
  """Return per-row indices of the final user token before the assistant stub."""

  sequences = input_ids.tolist()
  positions: List[int] = []
  for seq in sequences:
    seq_len = len(seq)
    assistant_idx = None
    if assistant_id is not None:
      for idx in range(seq_len - 1, -1, -1):
        if seq[idx] == assistant_id:
          assistant_idx = idx
          break
    target_idx = None
    if assistant_idx is not None and end_id is not None:
      end_idx = None
      for j in range(assistant_idx - 1, -1, -1):
        if seq[j] == end_id:
          end_idx = j
          break
      if end_idx is not None:
        target_idx = max(end_idx - 1, 0)
      else:
        target_idx = max(assistant_idx - 1, 0)
    if target_idx is None:
      for j in range(seq_len - 1, -1, -1):
        if seq[j] != pad_id:
          target_idx = j
          break
    if target_idx is None:
      target_idx = max(seq_len - 1, 0)
    positions.append(target_idx)
  return positions
