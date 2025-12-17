from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers.utils import cached_file

from .mxfp4 import dequantize_mxfp4, quantize_mxfp4


@dataclass
class AblationOrder:
  layer: int
  measurement_layer: str  # int or "composite"
  scale: float = 1.0
  sparsity: float = 0.0


@dataclass
class AblationConfig:
  model: str
  measurements: str
  output_dir: str
  orders: List[AblationOrder]
  norm_preserve: bool = False
  projected: bool = False
  targets: List[str] | None = None


def load_ablation_config(yaml_path: Path, norm_preserve: bool, projected: bool) -> AblationConfig:
  data = yaml.safe_load(Path(yaml_path).read_text())
  orders = [
    AblationOrder(
      layer=int(item["layer"]),
      measurement_layer=str(item["measurement"]),
      scale=float(item.get("scale", 1.0)),
      sparsity=float(item.get("sparsity", 0.0)),
    )
    for item in data.get("ablate", [])
  ]
  if not orders:
    raise ValueError("Ablation config must contain at least one 'ablate' entry")
  return AblationConfig(
    model=str(data["model"]),
    measurements=str(data["measurements"]),
    output_dir=str(data["output"]),
    orders=orders,
    norm_preserve=norm_preserve,
    projected=projected,
    targets=list(data.get("targets") or []) or None,
  )


def run_ablation(config: AblationConfig) -> None:
  measures = torch.load(config.measurements, map_location="cpu")
  num_layers = int(measures.get("layers", 0))
  if num_layers == 0:
    raise ValueError("Measurements missing 'layers' metadata")
  composite_refusal = torch.nn.functional.normalize(
    torch.stack([measures[f"refuse_{i}"] for i in range(num_layers)], dim=0).mean(dim=0).float(),
    dim=0,
  )
  for order in config.orders:
    if order.layer >= num_layers:
      raise ValueError(f"Destination layer {order.layer} out of range (max {num_layers - 1})")
    if order.measurement_layer != "composite" and int(order.measurement_layer) >= num_layers:
      raise ValueError(f"Measurement layer {order.measurement_layer} out of range")
  index_path, model_dir, weight_map = _load_weight_index(config.model)
  layer_prefix = _detect_layer_prefix(weight_map)
  output_dir = Path(config.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  shard_to_edits: Dict[str, List[tuple[str, AblationOrder]]] = {}
  target_suffixes = config.targets or [
    "self_attn.o_proj.weight",
    "mlp.router.weight",
    "mlp.router.bias",
    "mlp.experts.gate_up_proj_blocks",
    "mlp.experts.down_proj_blocks",
    "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj_bias",
    "mlp.experts.gate_proj.weight",
    "mlp.experts.up_proj.weight",
    "mlp.experts.down_proj.weight",
  ]
  for order in config.orders:
    for suffix in target_suffixes:
      key = f"{layer_prefix}.layers.{order.layer}.{suffix}"
      shard = weight_map.get(key)
      if shard is None:
        continue
      shard_to_edits.setdefault(shard, []).append((key, order))
  if not shard_to_edits:
    raise ValueError("No matching parameters found for requested ablations")
  shard_files = sorted(set(weight_map.values()))
  # MXFP4 dequant/quant is memory heavy; keep on CPU to avoid GPU OOM.
  device = torch.device("cpu")
  for shard_name in tqdm(shard_files, desc="Processing shards"):
    shard_path = model_dir / shard_name
    edits = shard_to_edits.get(shard_name)
    if not edits:
      shutil.copy2(shard_path, output_dir / shard_name)
      continue
    state_dict = load_file(str(shard_path))
    for key, order in edits:
      if key not in state_dict:
        continue
      weight_tensor = state_dict[key]
      if order.measurement_layer == "composite":
        refusal_dir = composite_refusal.clone()
      else:
        refusal_dir = measures[f"refuse_{int(order.measurement_layer)}"].float()
      harmless_dir = measures.get(f"harmless_{order.layer}")
      if config.projected and harmless_dir is not None:
        harmless_unit = torch.nn.functional.normalize(harmless_dir.float(), dim=0)
        projection = refusal_dir @ harmless_unit
        refusal_dir = refusal_dir - projection * harmless_unit
      if order.sparsity > 0.0:
        refusal_dir = _magnitude_sparsify(refusal_dir, order.sparsity)
      refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
      if weight_tensor.dtype == torch.uint8 and key.endswith("_blocks"):
        scale_key = key.replace("_blocks", "_scales")
        if scale_key not in state_dict:
          continue
        scales = state_dict[scale_key]
        if scales.dtype != torch.uint8:
          continue
        dense = dequantize_mxfp4(weight_tensor, scales, device=device, dtype=torch.float32)
        # dense shape: (experts, out, nblocks, 32). Merge block dims into input dim.
        dense = dense.view(*dense.shape[:-2], -1)  # (..., out, input)
        dense_shape = dense.shape  # (experts, out, input)
        dense_2d = dense.view(-1, dense_shape[-1])
        updated = _apply_modification(
          dense_2d,
          refusal_dir,
          order.scale,
          config.norm_preserve,
        )
        updated = updated.view(dense_shape)
        updated_blocks = updated.view(*updated.shape[:-1], updated.shape[-1] // 32, 32)
        new_blocks, new_scales = quantize_mxfp4(updated_blocks, device=device)
        state_dict[key] = new_blocks.contiguous()
        state_dict[scale_key] = new_scales.contiguous()
      elif weight_tensor.is_floating_point():
        updated = _apply_modification(
          weight_tensor,
          refusal_dir,
          order.scale,
          config.norm_preserve,
        )
        state_dict[key] = updated.contiguous()
      else:
        continue
      del refusal_dir
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_file(state_dict, str(output_dir / shard_name))
    del state_dict
    gc.collect()
  shutil.copy2(index_path, output_dir / "model.safetensors.index.json")
  _copy_configs(config.model, model_dir, output_dir)


def _apply_modification(weight: torch.Tensor, direction: torch.Tensor, scale: float, norm_preserve: bool) -> torch.Tensor:
  original_dtype = weight.dtype
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  W = weight.to(device=device, dtype=torch.float32)
  refusal = direction.to(device=device, dtype=torch.float32).reshape(-1)
  refusal = torch.nn.functional.normalize(refusal, dim=0)
  # flatten all but last dim into rows
  if W.shape[-1] != refusal.shape[0]:
    return weight
  rows = W.view(-1, W.shape[-1])
  if norm_preserve:
    row_norm = torch.norm(rows, dim=1, keepdim=True)
    row_dir = torch.nn.functional.normalize(rows, dim=1)
    projection = torch.matmul(row_dir, refusal)
    new_dir = row_dir - scale * projection.unsqueeze(1) * refusal.unsqueeze(0)
    new_dir = torch.nn.functional.normalize(new_dir, dim=1)
    modified_rows = row_norm * new_dir
  else:
    projection = torch.matmul(rows, refusal)
    modified_rows = rows - scale * projection.unsqueeze(1) * refusal.unsqueeze(0)
  modified = modified_rows.view_as(W).to(dtype=original_dtype, device="cpu")
  return modified.detach().clone()


def _magnitude_sparsify(vector: torch.Tensor, fraction: float) -> torch.Tensor:
  if fraction >= 1.0:
    return vector
  if fraction <= 0.0:
    return torch.zeros_like(vector)
  flat = vector.flatten()
  k = max(1, int(flat.numel() * fraction))
  threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
  mask = vector.abs() >= threshold
  return vector * mask


def _load_weight_index(model: str):
  potential = Path(model)
  if potential.exists():
    index_path = potential / "model.safetensors.index.json"
    if not index_path.exists():
      raise FileNotFoundError(f"model.safetensors.index.json missing under {potential}")
  else:
    index_path = Path(cached_file(model, "model.safetensors.index.json"))
  with open(index_path, "r", encoding="utf-8") as handle:
    index = json.load(handle)
  weight_map = index["weight_map"]
  model_dir = index_path.parent
  return index_path, model_dir, weight_map


def _detect_layer_prefix(weight_map: Dict[str, str]) -> str:
  for key in weight_map:
    if ".layers." in key and ".self_attn." in key:
      return key.split(".layers.")[0]
  raise ValueError("Could not infer decoder layer prefix from weight map")


def _copy_configs(model: str, model_dir: Path, output_dir: Path) -> None:
  config_files = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
  ]
  for filename in config_files:
    src = model_dir / filename
    if src.exists():
      shutil.copy2(src, output_dir / filename)
      continue
    try:
      cached = Path(cached_file(model, filename))
    except Exception:
      continue
    if cached.exists():
      shutil.copy2(cached, output_dir / filename)
  template_dir = "additional_chat_templates"
  src_dir = model_dir / template_dir
  if src_dir.exists() and src_dir.is_dir():
    shutil.copytree(src_dir, output_dir / template_dir, dirs_exist_ok=True)
