from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class AnalyzerConfig:
  input_path: str
  emit_chart: bool = False
  chart_path: Optional[str] = None


@dataclass
class AnalyzerResult:
  layers: int
  cosine_harmful_harmless: List[float]
  cosine_harmful_refusal: List[float]
  cosine_harmless_refusal: List[float]
  harmful_norms: List[float]
  harmless_norms: List[float]
  refusal_norms: List[float]
  signal_to_noise: List[float]
  purity_ratios: List[float]
  signal_quality: List[float]


def analyze_measurements(config: AnalyzerConfig) -> AnalyzerResult:
  blob = torch.load(config.input_path, map_location="cpu")
  layers = int(blob.get("layers", 0))
  if layers <= 0:
    raise ValueError("Measurement file missing 'layers' metadata")

  cos_hh: List[float] = []
  cos_hr: List[float] = []
  cos_ra: List[float] = []
  harmful_norms: List[float] = []
  harmless_norms: List[float] = []
  refusal_norms: List[float] = []
  snr_vals: List[float] = []
  purity_vals: List[float] = []
  quality_vals: List[float] = []

  for layer in range(layers):
    harmful = blob[f"harmful_{layer}"].float()
    harmless = blob[f"harmless_{layer}"].float()
    refusal = blob[f"refuse_{layer}"].float()
    cos_hh.append(_cosine(harmful, harmless))
    cos_hr.append(_cosine(harmful, refusal))
    cos_ra.append(_cosine(harmless, refusal))
    harmful_norm = harmful.norm().item()
    harmless_norm = harmless.norm().item()
    refusal_norm = refusal.norm().item()
    harmful_norms.append(harmful_norm)
    harmless_norms.append(harmless_norm)
    refusal_norms.append(refusal_norm)
    snr = refusal_norm / max(harmful_norm, harmless_norm, 1e-6)
    snr_vals.append(snr)
    purity = _purity_ratio(refusal, harmless)
    purity_vals.append(purity)
    quality_vals.append(snr * (1 - cos_hh[-1]) * purity)

  result = AnalyzerResult(
    layers=layers,
    cosine_harmful_harmless=cos_hh,
    cosine_harmful_refusal=cos_hr,
    cosine_harmless_refusal=cos_ra,
    harmful_norms=harmful_norms,
    harmless_norms=harmless_norms,
    refusal_norms=refusal_norms,
    signal_to_noise=snr_vals,
    purity_ratios=purity_vals,
    signal_quality=quality_vals,
  )
  if config.emit_chart:
    _render_chart(result, config.input_path, Path(config.chart_path) if config.chart_path else None)
  return result


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
  return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _purity_ratio(refusal: torch.Tensor, harmless: torch.Tensor) -> float:
  harmless_norm = torch.nn.functional.normalize(harmless, dim=0)
  projection = (refusal @ harmless_norm) * harmless_norm
  orth = refusal - projection
  return orth.norm().item() / max(refusal.norm().item(), 1e-6)


def _render_chart(result: AnalyzerResult, input_path: str, explicit_path: Optional[Path]) -> None:
  import matplotlib.pyplot as plt

  layers = list(range(result.layers))
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  axes[0, 0].plot(layers, result.harmful_norms, "r-o", label="Harmful")
  axes[0, 0].plot(layers, result.harmless_norms, "g-s", label="Harmless")
  axes[0, 0].plot(layers, result.refusal_norms, "b-^", label="Refusal")
  axes[0, 0].set_title("Mean Norms vs Layer")
  axes[0, 0].legend()
  axes[0, 0].grid(True, alpha=0.3)

  axes[0, 1].plot(layers, result.cosine_harmful_harmless, label="harmful↔harmless")
  axes[0, 1].plot(layers, result.cosine_harmful_refusal, label="harmful↔refusal")
  axes[0, 1].plot(layers, result.cosine_harmless_refusal, label="harmless↔refusal")
  axes[0, 1].set_title("Cosine Similarity")
  axes[0, 1].legend()
  axes[0, 1].grid(True, alpha=0.3)

  axes[1, 0].plot(layers, result.signal_to_noise, label="SNR")
  axes[1, 0].plot(layers, result.purity_ratios, label="Purity")
  axes[1, 0].set_title("Signal-to-Noise & Purity")
  axes[1, 0].legend()
  axes[1, 0].grid(True, alpha=0.3)

  axes[1, 1].plot(layers, result.signal_quality, label="Quality")
  axes[1, 1].set_title("Signal Quality")
  axes[1, 1].legend()
  axes[1, 1].grid(True, alpha=0.3)

  fig.suptitle("Refusal Direction Analysis", fontsize=16)
  fig.tight_layout()
  chart_path = explicit_path or Path(input_path).with_suffix(".png")
  fig.savefig(chart_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
