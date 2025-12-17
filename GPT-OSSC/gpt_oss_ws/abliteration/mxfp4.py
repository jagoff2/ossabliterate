from __future__ import annotations

from typing import Tuple

import torch

# MXFP4 lookup table from ggml-common.h (e2m1 values doubled)
_KVALUES = torch.tensor(
  [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
  dtype=torch.int8,
)


def _nibble_swap(x: torch.Tensor) -> torch.Tensor:
  return ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)


def transform_nibble_layout(blocks: torch.Tensor) -> torch.Tensor:
  """
  Convert Hugging Face MXFP4 layout to ggml layout.
  Port of llama.cpp convert_hf_to_gguf.py::transform_nibble_layout.
  blocks: uint8 tensor with last dim == 16.
  """
  t = _nibble_swap(blocks)
  blk_a, blk_b = torch.chunk(t, 2, dim=-1)
  blk_a0 = (blk_a & 0xF0).view(-1, 1)
  blk_a1 = (blk_a << 4).view(-1, 1)
  blk_a = torch.stack((blk_a0, blk_a1), dim=2).view(t.shape)
  blk_b0 = (blk_b >> 4).view(-1, 1)
  blk_b1 = (blk_b & 0x0F).view(-1, 1)
  blk_b = torch.stack((blk_b0, blk_b1), dim=2).view(t.shape)
  out = blk_a | blk_b
  out = _nibble_swap(out)
  return out


def inverse_transform_nibble_layout(blocks: torch.Tensor) -> torch.Tensor:
  """
  Inverse of transform_nibble_layout (ggml -> Hugging Face layout).
  """
  out_back = _nibble_swap(blocks)
  even = out_back[..., ::2]
  odd = out_back[..., 1::2]
  blk_a_high = even & 0xF0
  blk_a_low = (odd & 0xF0) >> 4
  blk_a = blk_a_high | blk_a_low
  blk_b_high = (even & 0x0F) << 4
  blk_b_low = odd & 0x0F
  blk_b = blk_b_high | blk_b_low
  t_swapped = torch.cat((blk_a, blk_b), dim=-1)
  return _nibble_swap(t_swapped)


def e8m0_to_fp32_half(e: torch.Tensor) -> torch.Tensor:
  """
  GGML_E8M0_TO_FP32_HALF: 0.5 * 2^(e-127) == 2^(e-128).
  """
  return torch.pow(2.0, e.to(torch.int16) - 128).float()


def dequantize_mxfp4(
  blocks: torch.Tensor,
  scales: torch.Tensor,
  device: torch.device | None = None,
  dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """
  Dequantize MXFP4 blocks+scales to dense weights.
  Inputs:
    blocks: (..., 16) uint8
    scales: (...,)   uint8
  Returns:
    dense tensor (..., 32) in requested dtype on CPU.
  """
  dev = device or torch.device("cpu")
  t = transform_nibble_layout(blocks.to(dev))
  low = (t & 0x0F).long()
  high = (t >> 4).long()
  k = _KVALUES.to(dev).float()
  vals = torch.cat((k[low], k[high]), dim=-1)
  d = e8m0_to_fp32_half(scales.to(dev))[..., None]
  out = vals * d
  return out.to(dtype=dtype, device="cpu")


def quantize_mxfp4(
  weight: torch.Tensor, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Quantize dense weights to Hugging Face MXFP4 blocks+scales.
  weight: (..., 32) where last dim is block size.
  Returns:
    blocks_hf: (..., 16) uint8 (HF layout)
    scales:    (...,)    uint8
  """
  dev = device or torch.device("cpu")
  w = weight.to(dev)
  amax = w.abs().amax(dim=-1)
  e = torch.zeros_like(amax, dtype=torch.uint8, device=dev)
  mask = amax > 0
  if mask.any():
    e_float = torch.floor(torch.log2(amax[mask])) - 2 + 127
    e[mask] = e_float.clamp(0, 255).to(torch.uint8)
  d = e8m0_to_fp32_half(e).unsqueeze(-1)
  table = _KVALUES.to(dev).float()
  scaled = torch.where(d == 0, torch.zeros_like(w), w / d)
  diff = (scaled.unsqueeze(-1) - table).abs()
  best = diff.argmin(dim=-1).to(torch.uint8)
  lo = best[..., :16]
  hi = best[..., 16:]
  qs = lo | (hi << 4)
  qs_hf = inverse_transform_nibble_layout(qs)
  return qs_hf.to(dtype=torch.uint8, device="cpu"), e.to(device="cpu")

