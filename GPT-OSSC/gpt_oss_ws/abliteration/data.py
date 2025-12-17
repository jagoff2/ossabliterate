from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd
from datasets import load_dataset


@dataclass
class PromptDataset:
  """In-memory prompt collection with lightweight helpers."""

  prompts: List[str]
  source: str

  def __post_init__(self) -> None:
    # Normalize whitespace and drop empties.
    cleaned: List[str] = []
    for item in self.prompts:
      if not isinstance(item, str):
        continue
      text = item.strip()
      if text:
        cleaned.append(text)
    self.prompts = cleaned

  def __len__(self) -> int:  # pragma: no cover - small helper
    return len(self.prompts)

  def __iter__(self) -> Iterator[str]:  # pragma: no cover - small helper
    return iter(self.prompts)


@dataclass
class PromptSourceConfig:
  """Declarative description of where to fetch prompts."""

  path: Optional[str] = None
  hf_name: Optional[str] = None
  hf_subset: Optional[str] = None
  split: str = "train"
  text_field: str = "text"
  limit: Optional[int] = None
  shuffle: bool = False
  seed: int = 0

  def describe(self) -> str:
    if self.path:
      return f"file:{self.path}"
    hf_bits = self.hf_name or ""
    if self.hf_subset:
      hf_bits = f"{hf_bits}/{self.hf_subset}"
    return f"hf:{hf_bits}:{self.split}"


def load_prompt_file(path: str, text_field: str = "text") -> List[str]:
  """Load prompts from a local file (txt/json/jsonl/parquet)."""

  resolved = Path(path)
  if not resolved.exists():
    raise FileNotFoundError(f"Prompt file not found: {resolved}")
  suffix = resolved.suffix.lower()
  if suffix == ".txt":
    return [line.rstrip("\n") for line in resolved.read_text(encoding="utf-8").splitlines()]
  if suffix == ".json":
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return _coerce_json_payload(data, text_field)
  if suffix == ".jsonl":
    entries: List[str] = []
    with resolved.open("r", encoding="utf-8") as handle:
      for raw in handle:
        stripped = raw.strip()
        if not stripped:
          continue
        payload = json.loads(stripped)
        entries.append(_extract_text(payload, text_field))
    return entries
  if suffix == ".parquet":
    frame = pd.read_parquet(resolved)
    if text_field not in frame.columns:
      raise ValueError(f"Column '{text_field}' missing from {resolved}")
    return frame[text_field].astype(str).tolist()
  raise ValueError(f"Unsupported prompt file extension: {resolved.suffix}")


def load_prompt_source(config: PromptSourceConfig) -> PromptDataset:
  """Load prompts from either a local file or Hugging Face dataset."""

  prompts: List[str]
  if config.path:
    prompts = load_prompt_file(config.path, text_field=config.text_field)
  elif config.hf_name:
    dataset = load_dataset(config.hf_name, config.hf_subset, split=config.split)
    column = config.text_field
    if column not in dataset.features:
      raise ValueError(
        f"Column '{column}' missing from dataset {config.hf_name} ({config.hf_subset or 'default'})"
      )
    prompts = [str(entry) for entry in dataset[column]]
  else:
    raise ValueError("PromptSourceConfig must define either 'path' or 'hf_name'.")

  if config.shuffle and len(prompts) > 1:
    rng = random.Random(config.seed)
    rng.shuffle(prompts)
  if config.limit is not None:
    prompts = prompts[: max(0, config.limit)]
  return PromptDataset(prompts=prompts, source=config.describe())


def _coerce_json_payload(data: object, text_field: str) -> List[str]:
  if isinstance(data, list):
    return [_extract_text(item, text_field) for item in data]
  raise ValueError("JSON prompt files must contain a list of entries")


def _extract_text(item: object, text_field: str) -> str:
  if isinstance(item, str):
    return item
  if isinstance(item, dict):
    if text_field not in item:
      raise ValueError(f"JSON object missing '{text_field}' field: {item}")
    value = item[text_field]
    if not isinstance(value, str):
      return str(value)
    return value
  raise ValueError(f"Unsupported prompt entry type: {type(item).__name__}")

