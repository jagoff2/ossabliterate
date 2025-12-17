from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import uvicorn

from gpt_oss_ws.abliteration import (
  AnalyzerConfig,
  DequantizeConfig,
  GgufExportConfig,
  MeasurementConfig,
  PromptSourceConfig,
  analyze_measurements,
  load_ablation_config,
  run_measurement,
  run_ablation,
  run_gguf_export,
  run_dequantize,
)
from gpt_oss_ws.api_server import create_app
from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.logging_utils import init_logger

app = typer.Typer(help="CLI entrypoints for GPT-OSS latent workspace package")
abliteration_app = typer.Typer(help="Refusal-direction measurement, analysis, ablation, and exports")


def _resolve_config(path: Optional[Path]) -> Path:
  if path is not None:
    return path
  default = Path("configs/server.yaml")
  if default.exists():
    return default
  raise typer.BadParameter("Config path must be provided if configs/server.yaml does not exist")


@app.command()
def serve(
  config: Optional[Path] = typer.Option(None, "--config", help="Path to server YAML config"),
  host: Optional[str] = typer.Option(None, "--host"),
  port: Optional[int] = typer.Option(None, "--port"),
  workspace_state: Optional[Path] = typer.Option(None, "--workspace-state", help="Path to trained workspace state checkpoint"),
  seed: int = typer.Option(99, "--seed", help="Base seed for deterministic generation"),
  temperature: float = typer.Option(0.0, "--temperature", help="Default sampling temperature"),
  top_p: float = typer.Option(1.0, "--top-p", help="Default nucleus sampling top-p (1.0 disables filtering)"),
  retro_generation: bool = typer.Option(False, "--retro-generation", "--retro", help="Enable retro refinement generation pipeline"),
  retro_margin: Optional[float] = typer.Option(None, "--retro-margin", help="Retro logit margin needed to roll back a chunk"),
  retro_window: Optional[int] = typer.Option(None, "--retro-window", help="Retro lookback window in tokens"),
  retro_max_retracts: Optional[int] = typer.Option(None, "--retro-max-retracts", help="Maximum retro retract attempts per chunk"),
  retro_iters: Optional[int] = typer.Option(None, "--retro-iters", help="Retro bidirectional smoothing passes"),
  retro_damping: Optional[float] = typer.Option(None, "--retro-damping", help="Retro damping factor (0-1)"),
  retro_chunk_size: Optional[int] = typer.Option(None, "--retro-chunk-size", help="Chunk size override for retro refinement"),
  retro_edit_budget: Optional[int] = typer.Option(None, "--retro-edit-budget", help="Maximum retro edits per position"),
  retro_max_tokens: Optional[int] = typer.Option(None, "--retro-max-tokens", help="Cap total new tokens for retro mode"),
  retro_blend: Optional[float] = typer.Option(None, "--retro-blend", help="Blend weight for retro logits vs forward logits (0-1)"),
  retro_diffusion_temperature: Optional[float] = typer.Option(None, "--retro-diffusion-temperature", help="Sampling temperature for diffusion-style retro updates"),
) -> None:
  cfg_path = _resolve_config(config)
  cfg = load_config(str(cfg_path))
  if host:
    cfg.api_host = host
  if port:
    cfg.api_port = port
  if workspace_state:
    cfg.workspace_state_path = str(workspace_state)
  logger = init_logger("cli.serve", cfg.log_level)
  logger.info("Starting API server", extra={"host": cfg.api_host, "port": cfg.api_port})
  overrides = {}
  if workspace_state:
    overrides["workspace_state_path"] = str(workspace_state)
  retro_overrides: Dict[str, Any] = {}
  if retro_generation:
    retro_overrides["enabled"] = True
  if retro_margin is not None:
    retro_overrides["margin"] = retro_margin
  if retro_window is not None:
    retro_overrides["window"] = retro_window
  if retro_max_retracts is not None:
    retro_overrides["max_retracts"] = retro_max_retracts
  if retro_iters is not None:
    retro_overrides["retro_iters"] = retro_iters
  if retro_damping is not None:
    retro_overrides["damping"] = retro_damping
  if retro_chunk_size is not None:
    retro_overrides["chunk_size"] = retro_chunk_size
  if retro_edit_budget is not None:
    retro_overrides["edit_budget"] = retro_edit_budget
  if retro_max_tokens is not None:
    retro_overrides["max_tokens"] = retro_max_tokens
  if retro_blend is not None:
    retro_overrides["diffusion_blend"] = retro_blend
  if retro_diffusion_temperature is not None:
    retro_overrides["diffusion_temperature"] = retro_diffusion_temperature
  if retro_overrides:
    existing_retro = overrides.get("retro", {})
    existing_retro.update(retro_overrides)
    overrides["retro"] = existing_retro
  app_instance = create_app(
    str(cfg_path),
    overrides=overrides or None,
    base_seed=seed,
    default_temperature=temperature,
    default_top_p=top_p,
  )
  uvicorn.run(
    app_instance,
    host=cfg.api_host,
    port=cfg.api_port,
    log_level=cfg.log_level.lower(),
  )


@app.command()
def eval(
  config: Optional[Path] = typer.Option(None, "--config", help="Path to eval config"),
  task: str = typer.Option("fluency", "--task", help="Which eval task to run"),
) -> None:
  cfg_path = _resolve_config(config)
  cfg = load_config(str(cfg_path))
  from evals import report

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None
  if loop is None:
    asyncio.run(report.run(task, cfg))
  else:
    loop.create_task(report.run(task, cfg))


@app.command(name="fluency-guard")
def fluency_guard(
  baseline_config: Optional[Path] = typer.Option(None, "--baseline", help="Baseline config path"),
  workspace_config: Optional[Path] = typer.Option(None, "--workspace", help="Workspace config path"),
  samples: int = typer.Option(64, "--samples", help="Number of samples to compare"),
) -> None:
  baseline = load_config(str(_resolve_config(baseline_config)))
  workspace = load_config(str(_resolve_config(workspace_config)))
  from evals import fluency_guard as fg

  asyncio.run(fg.compare(baseline, workspace, samples))


def _build_prompt_source(
  default_path: str,
  file_path: Optional[Path],
  hf_name: Optional[str],
  hf_subset: Optional[str],
  split: str,
  column: str,
  limit: Optional[int],
  shuffle: bool,
  seed: int,
) -> PromptSourceConfig:
  if file_path is not None:
    return PromptSourceConfig(
      path=str(file_path),
      text_field=column,
      limit=limit,
      shuffle=shuffle,
      seed=seed,
    )
  if hf_name is not None:
    return PromptSourceConfig(
      hf_name=hf_name,
      hf_subset=hf_subset,
      split=split,
      text_field=column,
      limit=limit,
      shuffle=shuffle,
      seed=seed,
    )
  return PromptSourceConfig(
    path=default_path,
    text_field=column,
    limit=limit,
    shuffle=shuffle,
    seed=seed,
  )


@abliteration_app.command("dequantize")
def abliteration_dequantize(
  model: Path = typer.Option(..., "--model", "-m", help="Source HF model dir or ID (e.g., MXFP4 quant)"),
  output: Path = typer.Option(..., "--output", "-o", help="Destination directory for dequantized weights"),
  dtype: str = typer.Option("float16", "--dtype", help="Target dtype: float16 or bfloat16"),
  device_map: str = typer.Option("cpu", "--device-map", help="Device map for loading (cpu recommended for full dequant)"),
) -> None:
  cfg = DequantizeConfig(
    model=str(model),
    output_dir=str(output),
    dtype=dtype,
    device_map=device_map,
  )
  typer.echo("Dequantizing model to dense weights; this may take time and RAM...")
  run_dequantize(cfg)
  typer.echo(f"Dequantized checkpoint written to {output}")


@abliteration_app.command("measure")
def abliteration_measure(
  model: str = typer.Option(..., "--model", "-m", help="Local checkpoint path or HF model ID"),
  output: Path = typer.Option(..., "--output", "-o", help="Destination .pt file for measurements"),
  chat_template: Optional[Path] = typer.Option(None, "--chat-template", help="Jinja chat template file to apply before measurement"),
  sample_last_user: bool = typer.Option(False, "--sample-last-user", help="Sample hidden states at final user token instead of assistant start"),
  data_harmful: Optional[Path] = typer.Option(None, "--data-harmful", help="Local harmful prompts file"),
  data_harmless: Optional[Path] = typer.Option(None, "--data-harmless", help="Local harmless prompts file"),
  hf_harmful: Optional[str] = typer.Option(None, "--hf-harmful", help="Hugging Face dataset ID for harmful prompts"),
  hf_harmful_subset: Optional[str] = typer.Option(None, "--hf-harmful-subset", help="Optional subset name for harmful dataset"),
  hf_harmful_split: str = typer.Option("train", "--harmful-split", help="Dataset split for harmful prompts"),
  hf_harmless: Optional[str] = typer.Option(None, "--hf-harmless", help="Hugging Face dataset ID for harmless prompts"),
  hf_harmless_subset: Optional[str] = typer.Option(None, "--hf-harmless-subset", help="Optional subset name for harmless dataset"),
  hf_harmless_split: str = typer.Option("train", "--harmless-split", help="Dataset split for harmless prompts"),
  harmful_column: str = typer.Option("text", "--harmful-column", help="Column name for harmful prompts"),
  harmless_column: str = typer.Option("text", "--harmless-column", help="Column name for harmless prompts"),
  harmful_limit: Optional[int] = typer.Option(None, "--harmful-limit", help="Optional max harmful prompts"),
  harmless_limit: Optional[int] = typer.Option(None, "--harmless-limit", help="Optional max harmless prompts"),
  shuffle_prompts: bool = typer.Option(False, "--shuffle-prompts/--no-shuffle-prompts", help="Shuffle prompts before limiting"),
  prompt_seed: int = typer.Option(0, "--prompt-seed", help="RNG seed when shuffling prompts"),
  batch_size: int = typer.Option(32, "--batch-size", help="Batch size for measurement inference"),
  clip: float = typer.Option(1.0, "--clip", help="Percentile (0-1] for magnitude clipping"),
  quant_measure: Optional[str] = typer.Option(None, "--quant-measure", case_sensitive=False, help="Force 4bit/8bit quantization during measurement"),
  flash_attn: bool = typer.Option(False, "--flash-attn", help="Enable FlashAttention 2"),
  deccp: bool = typer.Option(False, "--deccp", help="Append AUGMXNT/deccp topics to harmful prompts"),
  projected: bool = typer.Option(False, "--projected", help="Gram-Schmidt refusal vs harmless direction"),
) -> None:
  quant = quant_measure.lower() if quant_measure else None
  if quant not in {None, "4bit", "8bit"}:
    raise typer.BadParameter("--quant-measure must be '4bit' or '8bit'")
  if clip <= 0.0 or clip > 1.0:
    raise typer.BadParameter("--clip must be within (0, 1]")
  if batch_size <= 0:
    raise typer.BadParameter("--batch-size must be positive")
  harmful_source = _build_prompt_source(
    default_path="data/harmful.parquet",
    file_path=data_harmful,
    hf_name=hf_harmful,
    hf_subset=hf_harmful_subset,
    split=hf_harmful_split,
    column=harmful_column,
    limit=harmful_limit,
    shuffle=shuffle_prompts,
    seed=prompt_seed,
  )
  harmless_source = _build_prompt_source(
    default_path="data/harmless.parquet",
    file_path=data_harmless,
    hf_name=hf_harmless,
    hf_subset=hf_harmless_subset,
    split=hf_harmless_split,
    column=harmless_column,
    limit=harmless_limit,
    shuffle=shuffle_prompts,
    seed=prompt_seed,
  )
  measurement_cfg = MeasurementConfig(
    model_name=model,
    output_path=str(output),
    batch_size=batch_size,
    clip_fraction=clip,
    quant_mode=quant,
    flash_attention=flash_attn,
    add_deccp=deccp,
    projected=projected,
    chat_template_path=str(chat_template) if chat_template else None,
    sample_last_user_token=sample_last_user,
    harmful_source=harmful_source,
    harmless_source=harmless_source,
  )
  typer.echo("Starting refusal measurement run...")
  run_measurement(measurement_cfg)
  typer.echo(f"Saved measurements to {output}")


@abliteration_app.command("analyze")
def abliteration_analyze(
  measurements: Path = typer.Option(..., "--input", "-i", help="Measurement .pt file"),
  chart: bool = typer.Option(False, "--chart/--no-chart", help="Render matplotlib chart"),
  chart_path: Optional[Path] = typer.Option(None, "--chart-path", help="Optional chart output path"),
) -> None:
  cfg = AnalyzerConfig(
    input_path=str(measurements),
    emit_chart=chart,
    chart_path=str(chart_path) if chart_path else None,
  )
  result = analyze_measurements(cfg)
  for layer in range(result.layers):
    typer.echo(
      f"Layer {layer:02d} | cos(harmful,harmless)={result.cosine_harmful_harmless[layer]:.4f} "
      f"cos(harmful,refusal)={result.cosine_harmful_refusal[layer]:.4f} "
      f"cos(harmless,refusal)={result.cosine_harmless_refusal[layer]:.4f} "
      f"SNR={result.signal_to_noise[layer]:.4f} purity={result.purity_ratios[layer]:.4f} "
      f"quality={result.signal_quality[layer]:.4f}"
    )
  if chart:
    target = chart_path or measurements.with_suffix(".png")
    typer.echo(f"Chart saved to {target}")


@abliteration_app.command("ablate")
def abliteration_ablate(
  config_path: Path = typer.Option(..., "--config", "-c", help="YAML config describing ablation orders"),
  norm_preserve: bool = typer.Option(False, "--norm-preserve", help="Preserve row norms when ablating"),
  projected: bool = typer.Option(False, "--projected", help="Orthogonalize refusal vs harmless direction"),
) -> None:
  cfg = load_ablation_config(config_path, norm_preserve=norm_preserve, projected=projected)
  typer.echo("Starting sharded ablation run...")
  run_ablation(cfg)
  typer.echo(f"Ablated model written to {cfg.output_dir}")


@abliteration_app.command("export-gguf")
def abliteration_export(
  model_dir: Path = typer.Option(..., "--model-dir", "-m", help="Ablated Hugging Face checkpoint directory"),
  llama_cpp_path: Path = typer.Option(..., "--llama-cpp-path", help="Path to a local llama.cpp checkout"),
  outfile: Path = typer.Option(..., "--outfile", "-o", help="Destination GGUF file"),
  quantize: Optional[str] = typer.Option(None, "--quantize", help="Optional llama.cpp quantization preset (e.g., q4_0)"),
  outtype: str = typer.Option("f16", "--outtype", help="Base GGUF tensor type"),
  extra_arg: Optional[List[str]] = typer.Option(None, "--extra-arg", help="Additional convert.py arguments"),
) -> None:
  cfg = GgufExportConfig(
    model_dir=str(model_dir),
    llama_cpp_repo=str(llama_cpp_path),
    outfile=str(outfile),
    quantize=quantize,
    outtype=outtype,
    extra_args=list(extra_arg) if extra_arg else [],
  )
  typer.echo("Invoking llama.cpp converter...")
  run_gguf_export(cfg)
  typer.echo(f"GGUF artifact saved to {outfile}")


app.add_typer(abliteration_app, name="abliteration")


if __name__ == "__main__":
  app()
