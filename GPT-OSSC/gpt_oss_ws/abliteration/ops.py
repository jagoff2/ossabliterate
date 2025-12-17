from __future__ import annotations

import torch


def magnitude_clip(vector: torch.Tensor, percentile: float = 1.0) -> torch.Tensor:
  """Clip tensor values by magnitude percentile (defaults to no clip)."""

  if percentile >= 1.0:
    return vector
  if percentile <= 0.0:
    return torch.zeros_like(vector)
  original_dtype = vector.dtype
  vector_float = vector.float()
  abs_vector = vector_float.abs()
  threshold = torch.quantile(abs_vector, percentile)
  clipped = vector_float.clamp(min=-threshold, max=threshold)
  return clipped.to(dtype=original_dtype)

