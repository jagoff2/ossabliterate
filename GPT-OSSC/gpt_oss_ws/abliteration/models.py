from __future__ import annotations


def has_tied_weights(model_type: str) -> bool:
  """Return True when the HF config indicates shared input/output embeddings."""

  if not isinstance(model_type, str):
    return False
  gemma_family = {"gemma", "gemma2", "gemma3", "paligemma"}
  return model_type.lower() in gemma_family

