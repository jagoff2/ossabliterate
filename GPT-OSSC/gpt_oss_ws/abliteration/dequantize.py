from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DequantizeConfig:
  model: str
  output_dir: str
  dtype: str = "float16"  # "float16" or "bfloat16"
  device_map: str = "cpu"


def run_dequantize(cfg: DequantizeConfig) -> None:
  dtype_map = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16"}
  torch_dtype = dtype_map.get(cfg.dtype.lower())
  if torch_dtype is None:
    raise ValueError(f"Unsupported dtype {cfg.dtype}; choose float16 or bfloat16")

  model = AutoModelForCausalLM.from_pretrained(
    cfg.model,
    torch_dtype=torch_dtype,
    device_map=cfg.device_map,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
  )
  model.save_pretrained(cfg.output_dir, safe_serialization=True)

  tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
  tokenizer.save_pretrained(cfg.output_dir)

  # Copy the config fields if trust_remote_code added extras; the save_pretrained call handles config already.
  Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

