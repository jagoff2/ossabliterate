from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_transformer.config import MetaControllerConfig
from meta_transformer.models.gpt2_meta_transformer import Gpt2MetaConfig, Gpt2MetaTransformerLM
from meta_transformer.training.reasoning_tasks import ReasoningTask, load_reasoning_tasks
from meta_transformer.training.training_utils import clip_gradients, set_seed
from scripts.validator_utils import run_validator
from scripts.evaluate_reports import (
    DEFAULT_DIAG_MARKERS as EVAL_DEFAULT_DIAG_MARKERS,
    DEFAULT_MONITOR_MARKERS as EVAL_DEFAULT_MONITOR_MARKERS,
    DEFAULT_PLAN_MARKERS as EVAL_DEFAULT_PLAN_MARKERS,
    EpisodeMetrics as EvaluatorEpisodeMetrics,
    evaluate_episode as evaluator_evaluate_episode,
    maybe_load_lm as evaluator_maybe_load_lm,
)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _init_distributed(device_arg: str) -> tuple[str, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return f"cuda:{local_rank}", True, rank, world_size, local_rank
    if device_arg.startswith("cuda"):
        if ":" in device_arg:
            torch.cuda.set_device(device_arg)
            return device_arg, False, 0, 1, -1
        torch.cuda.set_device(0)
        return "cuda:0", False, 0, 1, 0
    return device_arg, False, 0, 1, -1


@dataclass
class Gpt2MetaTrainingConfig:
    num_epochs: int = 1
    episodes_per_epoch: int = 8
    max_new_tokens: int = 64
    reward_scale: float = 1.0
    baseline_momentum: float = 0.9
    entropy_coef: float = 1e-3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 4
    tokenizer_name: str = "gpt2"
    base_model_name: str = "sshleifer/tiny-gpt2"
    train_lm_head: bool = False
    train_base_model: bool = False
    base_unfreeze_start_epoch: Optional[int] = None
    base_unfreeze_steps: int = 0
    run_dir: Optional[str] = None
    device: str = "cuda"
    reasoning_tasks_path: Optional[str] = None
    sample_temperature: float = 0.0
    eval_episodes: int = 4
    use_process_state: bool = True
    process_state_dim: int = 128
    gate_drift_bonus: float = 0.1
    monitoring_bonus: float = 0.1
    plan_bonus: float = 0.05
    diagnosis_bonus: float = 0.05
    report_bonus: float = 0.1
    memory_bonus: float = 0.05
    focus_bonus: float = 0.05
    focus_penalty: float = 0.0
    alignment_bonus: float = 0.05
    confidence_bonus: float = 0.05
    diagnosis_pred_weight: float = 0.1
    diagnosis_pred_bonus: float = 0.05
    judge_bonus: float = 0.0
    judge_loss_weight: float = 0.0
    trace_classifier_weight: float = 0.0
    trace_classifier_bonus: float = 0.0
    replay_probability: float = 0.25
    replay_bonus: float = 0.1
    replay_penalty: float = 0.1
    report_supervision_weight: float = 0.0
    trace_bonus: float = 0.05
    reflection_passes: int = 1
    reflection_penalty: float = 0.05
    reflection_bonus: float = 0.05
    reflection_trace_bonus: float = 0.05
    reflection_repeat_penalty: float = 0.05
    trace_replay_probability: float = 0.0
    trace_replay_bonus: float = 0.05
    trace_replay_penalty: float = 0.05
    gate_warmup_episodes: int = 0
    min_decoding_steps: int = 0
    report_repetition_penalty: float = 0.0
    plan_repetition_penalty: float = 0.0
    monitor_entity_penalty: float = 0.0
    imitation_steps: int = 0
    imitation_batch_size: int = 1
    plan_gate: float = 0.0
    monitor_gate: float = 0.0
    imitation_report_weight: float = 1.0
    plan_penalty: float = 0.0
    monitoring_penalty: float = 0.0
    diagnosis_penalty: float = 0.0
    progress_every: int = 1
    progress_sample_chars: int = 256
    structure_check_episodes: int = 4
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = -1
    distributed: bool = False
    teacher_attention_weight: float = 0.0
    graph_coverage_weight: float = 0.0
    use_evaluator: bool = False
    evaluator_prefix_lm: str = "none"
    evaluator_device: str = "cpu"
    evaluator_prefix_cap: int = 160
    evaluator_plan_threshold: float = 0.75
    evaluator_monitor_threshold: float = 0.75
    evaluator_diagnosis_threshold: float = 1.0


@dataclass
class ReasoningRollout:
    reward: float
    log_prob: Optional[torch.Tensor]
    entropy: torch.Tensor
    completion: str
    steps: int
    gate_drift: float
    monitoring_score: float
    workspace_trace: Optional[List[torch.Tensor]] = None
    plan_score: float = 0.0
    plan_coverage: float = 0.0
    plan_repetition: float = 0.0
    diagnosis_score: float = 0.0
    report_text: str = ""
    report_score: float = 0.0
    report_repetition: float = 0.0
    outcome: Optional[bool] = None
    alignment_score: float = 0.0
    focus_terms: List[str] = field(default_factory=list)
    focus_score: float = 0.0
    monitor_coverage: float = 0.0
    diagnosis_logit: Optional[torch.Tensor] = None
    diagnosis_pred_prob: float = 0.0
    diagnosis_text_pred: Optional[bool] = None
    judge_logit: Optional[torch.Tensor] = None
    judge_prob: float = 0.0
    confidence_value: Optional[float] = None
    confidence_alignment: float = 0.0
    summary_vec: Optional[torch.Tensor] = None
    is_replay: bool = False
    replay_context: Optional[dict[str, object]] = None
    report_logits: Optional[torch.Tensor] = None
    gold_loss: float = 0.0
    trace_alignment: float = 0.0
    replay_type: Optional[str] = None
    pass_index: int = 0
    trace_classifier_loss: float = 0.0
    trace_agreement: float = 0.0
    graph_nodes: List[Dict[str, object]] = field(default_factory=list)
    graph_ops: List[Dict[str, object]] = field(default_factory=list)
    teacher_attention_loss: float = 0.0
    teacher_attention_alignment: float = 0.0
    teacher_attention_loss_tensor: Optional[torch.Tensor] = None
    graph_coverage: float = 0.0


def _prepare_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _bernoulli_entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    probs = probs.clamp(eps, 1 - eps)
    return -(probs * probs.log() + (1 - probs) * (1 - probs).log())


def _select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _score_report(
    task: ReasoningTask,
    report_text: str,
    tokenizer: PreTrainedTokenizerBase,
    lm_model: AutoModelForCausalLM,
) -> float:
    if not report_text:
        return 0.0
    text = report_text.lower()
    keyword_score = 0.0
    if task.monitoring_markers:
        monitor_hits = sum(text.count(marker.lower()) for marker in task.monitoring_markers)
        keyword_score += 0.3 * min(1.0, monitor_hits / max(1, task.min_monitoring_mentions))
    if task.plan_markers and task.min_plan_steps > 0:
        plan_hits = sum(text.count(marker.lower()) for marker in task.plan_markers)
        keyword_score += 0.4 * min(1.0, plan_hits / task.min_plan_steps)
    if task.require_diagnosis and task.diagnosis_markers:
        diag_hits = sum(text.count(marker.lower()) for marker in task.diagnosis_markers)
        keyword_score += 0.3 * min(1.0, diag_hits / len(task.diagnosis_markers))
    fluency_score = 0.0
    try:
        with torch.no_grad():
            encoded = tokenizer(report_text, return_tensors="pt").to(lm_model.device)
            if encoded.input_ids.numel() > 1:
                outputs = lm_model(input_ids=encoded.input_ids, labels=encoded.input_ids)
                loss = outputs.loss
                fluency_score = float(torch.exp(-loss).item())
    except Exception:
        fluency_score = 0.0
    return float(min(1.0, 0.5 * keyword_score + 0.5 * fluency_score))


def _outcome_correct(task: ReasoningTask, completion: str) -> Optional[bool]:
    validated = run_validator(task.validator, task.validator_payload, completion)
    if validated is not None:
        return validated
    if not task.expected_answer:
        return None
    return task.expected_answer.lower() in completion.lower()


def _diagnosis_alignment(completion: str, outcome: Optional[bool]) -> Tuple[float, Optional[bool]]:
    if outcome is None:
        return 0.0, None
    text = completion.lower()
    idx = text.rfind("diagnosis:")
    if idx == -1:
        return 0.0, None
    diag_segment = text[idx:]
    if any(keyword in diag_segment for keyword in ("incorrect", "error", "fail", "wrong")):
        predicted = False
    elif any(keyword in diag_segment for keyword in ("correct", "success", "right", "verified")):
        predicted = True
    else:
        return 0.0, None
    score = 1.0 if predicted == outcome else 0.0
    return score, predicted


def _plan_consistency(completion: str) -> float:
    text = completion.lower()
    plan_steps = re.findall(r"plan step\s*\d*:\s*(.+)", text)
    reasoning_steps = re.findall(r"step\s*\d*:\s*(.+)", text)
    if not plan_steps or not reasoning_steps:
        return 0.0
    if len(reasoning_steps) >= len(plan_steps):
        return 1.0
    def _normalize(fragment: str) -> List[str]:
        words = re.findall(r"[a-z0-9]+", fragment.lower())
        tokens: List[str] = []
        for w in words:
            if w.isdigit():
                tokens.append(w)
            elif len(w) > 3:
                tokens.append(w)
        return tokens
    reasoning_tokens = [_normalize(step) for step in reasoning_steps]
    def _matches(plan_tokens: List[str]) -> bool:
        if not plan_tokens:
            return False
        for tokens in reasoning_tokens:
            overlap = len(set(tokens) & set(plan_tokens))
            if overlap >= 1:
                return True
        return False
    matches = 0
    for plan_step in plan_steps:
        if _matches(_normalize(plan_step)):
            matches += 1
    return matches / max(len(plan_steps), 1)


def _extract_plan_lines(completion: str) -> List[str]:
    lines = [line.strip() for line in re.findall(r"plan step[^:]*:(.+)", completion, flags=re.IGNORECASE)]
    step_lines = [line.strip() for line in re.findall(r"step\s*\d*:(.+)", completion, flags=re.IGNORECASE)]
    return lines + step_lines


def _extract_plan_only(completion: str) -> List[str]:
    return [line.strip() for line in re.findall(r"plan step[^:]*:(.+)", completion, flags=re.IGNORECASE)]


def _extract_reasoning_steps(completion: str) -> List[str]:
    return [line.strip() for line in re.findall(r"step\s*\d*:(.+)", completion, flags=re.IGNORECASE)]


def _extract_monitor_lines(completion: str) -> List[str]:
    return [line.strip() for line in re.findall(r"monitor:(.+)", completion, flags=re.IGNORECASE)]


def _entity_tokens(task: ReasoningTask) -> List[str]:
    tokens: List[str] = []
    payload = task.validator_payload or {}
    def _collect(value: object) -> None:
        if value is None:
            return
        if isinstance(value, (int, float)):
            tokens.append(str(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                _collect(item)
        elif isinstance(value, str):
            numbers = re.findall(r"-?\d+", value)
            tokens.extend(numbers)
    for val in payload.values():
        _collect(val)
    answer_numbers = re.findall(r"-?\d+", task.expected_answer)
    tokens.extend(answer_numbers)
    return list({tok for tok in tokens if tok})


def _coverage_ratio(lines: Sequence[str], tokens: Sequence[str]) -> float:
    if not lines or not tokens:
        return 0.0
    hits = 0
    lowered = [line.lower() for line in lines]
    for tok in tokens:
        tok_l = str(tok).lower()
        if any(tok_l in line for line in lowered):
            hits += 1
    return hits / len(tokens)


def _line_repetition_ratio(lines: Sequence[str]) -> float:
    filtered = [line.strip() for line in lines if line.strip()]
    if not filtered:
        return 0.0
    unique = len(set(filtered))
    duplicates = len(filtered) - unique
    return max(0.0, duplicates / len(filtered))


def _focus_coverage(report_text: str, focus_terms: Sequence[str]) -> float:
    if not focus_terms:
        return 0.0
    text = report_text.lower()
    hits = 0
    for term in focus_terms:
        if term and term.lower() in text:
            hits += 1
    return hits / len(focus_terms)


def _report_repetition_ratio(report_text: str) -> float:
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    unique = len(set(lines))
    duplicates = len(lines) - unique
    return max(0.0, duplicates / len(lines))


def _ensure_structured_completion(completion: str, task: ReasoningTask) -> str:
    text = completion.rstrip()
    lowered = text.lower()
    plan_lines = re.findall(r"plan step[^:]*:", lowered)
    monitor_lines = re.findall(r"monitor:", lowered)
    plan_ok = len(plan_lines) >= max(1, task.min_plan_steps)
    monitor_ok = len(monitor_lines) >= max(1, task.min_monitoring_mentions)
    diagnosis_ok = not task.require_diagnosis or ("diagnosis:" in lowered)
    confidence_ok = not task.require_confidence or ("confidence:" in lowered)
    if plan_ok and monitor_ok and diagnosis_ok and confidence_ok:
        return text
    if "plan:" not in lowered:
        text += "\nPlan:"
    entity_tokens = _entity_tokens(task)
    if not entity_tokens:
        entity_tokens = ["the key quantities", "the intermediate result"]
    required_plan_steps = max(task.min_plan_steps, 2)
    existing_plan = len(re.findall(r"plan step\s*\d*:", lowered))
    for idx in range(existing_plan, required_plan_steps):
        token = entity_tokens[idx % len(entity_tokens)]
        text += f"\nPlan Step {idx + 1}: focus on {token} and relate it to the goal."
    required_reasoning_steps = max(task.min_plan_steps, 2)
    existing_steps = len(re.findall(r"\bstep\s*\d+:", lowered))
    step_offset = existing_steps
    for idx in range(step_offset, required_reasoning_steps):
        token = entity_tokens[idx % len(entity_tokens)]
        text += (
            f"\nStep {idx + 1}: carry out the plan for {token} explicitly."
            f"\nMonitor: verify the operation on {token} and note any discrepancies."
        )
    if task.require_diagnosis and "diagnosis:" not in lowered:
        verdict = "correct" if task.expected_answer else "pending"
        text += f"\nDiagnosis: review each monitor outcome and declare the plan {verdict}."
    if task.require_confidence and "confidence:" not in lowered:
        text += "\nConfidence: Medium"
    return text


def _trace_hash(term: str, dim: int) -> int:
    if dim <= 0:
        return 0
    return abs(hash(term)) % dim


def _build_teacher_attention_vector(
    hints: Sequence[object],
    dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if dim <= 0 or not hints:
        return None
    vec = torch.zeros(dim, device=device)
    total = 0.0
    for hint in hints:
        weight = 1.0
        token = ""
        if isinstance(hint, dict):
            token = str(hint.get("target") or hint.get("location") or "")
            weight = float(hint.get("weight", 1.0))
            note = hint.get("note")
            if note:
                token += f"::{note}"
        else:
            token = str(hint)
        token = token.strip()
        if not token:
            continue
        idx = _trace_hash(token, dim)
        vec[idx] += weight
        total += weight
    if total <= 0:
        return None
    return vec / total


def _graph_report_coverage(report_text: str, nodes: Sequence[Dict[str, object]]) -> float:
    if not nodes or not report_text:
        return 0.0
    lowered = report_text.lower()
    hits = 0
    for node in nodes:
        label = str(node.get("label", "")).strip().lower()
        if label and label in lowered:
            hits += 1
    return hits / len(nodes)


def _set_transformer_grad_fraction(
    model: Gpt2MetaTransformerLM,
    training_config: Gpt2MetaTrainingConfig,
    fraction: float,
) -> None:
    transformer = getattr(model.model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        return
    layers = transformer.h
    total_layers = len(layers)
    if total_layers == 0:
        return
    fraction = max(0.0, min(1.0, fraction))
    active_layers = int(math.ceil(fraction * total_layers))
    cutoff = total_layers - active_layers
    for idx, layer in enumerate(layers):
        requires_grad = idx >= cutoff
        for param in layer.parameters():
            param.requires_grad_(requires_grad)
    embed_flag = fraction >= 1.0
    for attr in ("wte", "wpe", "ln_f"):
        tensor = getattr(transformer, attr, None)
        if tensor is not None:
            for param in tensor.parameters():
                param.requires_grad_(embed_flag)
    if hasattr(model.model, "lm_head"):
        for param in model.model.lm_head.parameters():
            param.requires_grad_(training_config.train_lm_head or embed_flag)


def _run_imitation_phase(
    model: Gpt2MetaTransformerLM,
    optimizer: Optimizer,
    tokenizer: PreTrainedTokenizerBase,
    tasks: Sequence[ReasoningTask],
    training_config: Gpt2MetaTrainingConfig,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_recorder: Optional[ProgressRecorder] = None,
    *,
    is_main_process: bool,
) -> None:
    epochs = max(0, training_config.imitation_steps)
    if epochs == 0:
        return
    gold_tasks = [task for task in tasks if task.gold_completion]
    if not gold_tasks:
        return
    base_model = _unwrap_model(model)
    model.train()
    rng = random.Random(training_config.seed + 2222)
    device = base_model.device
    batch_size = max(1, training_config.imitation_batch_size)
    if training_config.train_base_model:
        _set_transformer_grad_fraction(base_model, training_config, 1.0)
    elif training_config.train_lm_head and hasattr(base_model.model, "lm_head"):
        for param in base_model.model.lm_head.parameters():
            param.requires_grad_(True)

    def _log(msg: str) -> None:
        if not is_main_process:
            return
        print(msg, flush=True)
        if log_fn is not None:
            log_fn(msg)

    for epoch in range(epochs):
        rng.shuffle(gold_tasks)
        total_loss = 0.0
        batches = 0
        epoch_lm_loss = 0.0
        epoch_report_loss = 0.0
        sample_count = 0
        for start in range(0, len(gold_tasks), batch_size):
            batch = gold_tasks[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for task in batch:
                prompt_ids = tokenizer(task.prompt, add_special_tokens=False, return_tensors="pt").input_ids
                completion_text = task.gold_completion
                full_text = task.prompt + "\n\n" + completion_text
                encoded = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=training_config.max_new_tokens + prompt_ids.size(1) + 10,
                )
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)
                labels = input_ids.clone()
                prompt_len = min(prompt_ids.size(1), labels.size(1))
                labels[:, :prompt_len] = -100
                logits, _ = model(
                    input_ids,
                    attention_mask=attention_mask,
                    sample_gates=False,
                    return_gate_details=False,
                )
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                lm_value = ce.item()
                loss = ce
                report_ce_value = 0.0
                if task.gold_report:
                    summary, _, _, _ = base_model.get_introspection_state(input_ids, attention_mask=attention_mask)
                    report_logits, _ = base_model.report_head(summary)
                    report_len = report_logits.size(0)
                    report_ids = tokenizer(
                        task.gold_report,
                        return_tensors="pt",
                        truncation=True,
                        max_length=report_len,
                    ).input_ids.to(device)
                    if report_ids.size(1) < report_len:
                        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                        pad = torch.full(
                            (report_ids.size(0), report_len - report_ids.size(1)),
                            pad_id,
                            dtype=report_ids.dtype,
                            device=device,
                        )
                        report_ids = torch.cat([report_ids, pad], dim=1)
                    else:
                        report_ids = report_ids[:, :report_len]
                    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                    report_ce = F.cross_entropy(
                        report_logits.view(-1, report_logits.size(-1)),
                        report_ids.view(-1),
                        ignore_index=pad_token,
                    )
                    report_ce_value = report_ce.item()
                    loss = loss + training_config.imitation_report_weight * report_ce
                (loss / len(batch)).backward()
                batch_loss += loss.item()
                epoch_lm_loss += lm_value
                epoch_report_loss += report_ce_value
                sample_count += 1
            clip_gradients(model.parameters(), training_config.grad_clip)
            optimizer.step()
            total_loss += batch_loss / max(1, len(batch))
            batches += 1
        avg_loss = total_loss / max(1, batches)
        avg_lm = epoch_lm_loss / max(1, sample_count)
        avg_report = epoch_report_loss / max(1, sample_count)
        msg = (
            f"[imitation epoch {epoch + 1}/{epochs}] "
            f"loss={avg_loss:.4f} lm={avg_lm:.4f} report={avg_report:.4f}"
        )
        _log(msg)
        if progress_recorder is not None:
            progress_recorder.record_imitation(epoch + 1, epochs, avg_loss, avg_lm, avg_report)


def _structure_gate_check(
    model: Gpt2MetaTransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    tasks: Sequence[ReasoningTask],
    training_config: Gpt2MetaTrainingConfig,
    progress_recorder: Optional[ProgressRecorder],
    log_fn: Optional[Callable[[str], None]],
) -> bool:
    total_checks = min(len(tasks), max(1, training_config.structure_check_episodes))
    if total_checks <= 0 or not tasks:
        return True
    rng = random.Random(training_config.seed + 5555)
    passes = 0
    base_model = _unwrap_model(model)
    prev_mode = base_model.training
    base_model.eval()
    with torch.no_grad():
        for idx in range(total_checks):
            task = rng.choice(tasks)
            rollout, _ = _execute_rollout_passes(
                base_model,
                tokenizer,
                task,
                training_config,
                prior_entry=None,
                initial_prompt=None,
                initial_replay_type=None,
                sample_gates=False,
            )
            plan_struct = 0.5 * (rollout.plan_score + rollout.plan_coverage)
            monitor_struct = 0.5 * (rollout.monitoring_score + rollout.monitor_coverage)
            gate_ok = (plan_struct >= training_config.plan_gate) and (
                monitor_struct >= training_config.monitor_gate
            )
            passes += int(gate_ok)
            if progress_recorder is not None:
                progress_recorder.record_structure_check(
                    index=idx + 1,
                    total=total_checks,
                    plan_struct=plan_struct,
                    monitor_struct=monitor_struct,
                    passed=gate_ok,
                    completion=rollout.completion,
                    report=rollout.report_text,
                )
    if prev_mode:
        base_model.train()
    if passes < total_checks:
        msg = (
            f"[structure-check] failed {passes}/{total_checks} episodes "
            f"(plan_gate={training_config.plan_gate:.2f}, monitor_gate={training_config.monitor_gate:.2f})"
        )
        if log_fn is not None:
            log_fn(msg)
        raise RuntimeError(msg)
    msg = f"[structure-check] passed {passes}/{total_checks} episodes"
    if log_fn is not None:
        log_fn(msg)
    return True
def _descriptor_alignment_score(
    summary_vec: torch.Tensor,
    report_embedding: Optional[torch.Tensor],
) -> float:
    if report_embedding is None:
        return 0.0
    if summary_vec.numel() != report_embedding.numel():
        return 0.0
    summary = summary_vec.detach()
    summary = summary / (summary.norm(p=2) + 1e-6)
    hidden = report_embedding / (report_embedding.norm(p=2) + 1e-6)
    similarity = torch.clamp(torch.dot(summary, hidden), -1.0, 1.0)
    return float((similarity + 1.0) / 2.0)


def _trace_alignment_score(
    trace_history: Optional[List[torch.Tensor]],
    report_embedding: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    if not trace_history or report_embedding is None:
        return 0.0
    trace_stack = torch.stack(
        [vec.to(device=device, dtype=report_embedding.dtype) for vec in trace_history],
        dim=0,
    )
    trace_vec = trace_stack.mean(dim=0)
    if trace_vec.numel() != report_embedding.numel():
        return 0.0
    trace_vec = trace_vec / (trace_vec.norm(p=2) + 1e-6)
    report_vec = report_embedding / (report_embedding.norm(p=2) + 1e-6)
    similarity = torch.clamp(torch.dot(trace_vec, report_vec), -1.0, 1.0)
    return float((similarity + 1.0) / 2.0)


def _encode_report_embedding(
    report_text: str,
    tokenizer: PreTrainedTokenizerBase,
    lm_model: AutoModelForCausalLM,
) -> Optional[torch.Tensor]:
    if not report_text.strip():
        return None
    try:
        encoded = tokenizer(report_text, return_tensors="pt").to(lm_model.device)
        with torch.no_grad():
            outputs = lm_model.transformer(input_ids=encoded.input_ids)
            hidden = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return hidden
    except Exception:
        return None


def _extract_confidence(report_text: str) -> Optional[float]:
    text = report_text.lower()
    match = re.search(r"confidence\s*[:=]\s*([a-z]+|\d+\.?\d*)", text)
    if not match:
        return None
    token = match.group(1).strip()
    mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
    if token in mapping:
        return mapping[token]
    try:
        value = float(token)
        if value > 1.0:
            value /= 100.0
        return float(min(max(value, 0.0), 1.0))
    except ValueError:
        return None


def _confidence_alignment(confidence: Optional[float], outcome: Optional[bool]) -> float:
    if confidence is None or outcome is None:
        return 0.0
    return confidence if outcome else 1.0 - confidence


def _build_replay_prompt(task: ReasoningTask, prior_entry: dict[str, object]) -> str:
    prior_completion = str(prior_entry.get("completion", "")).strip()
    prior_report = str(prior_entry.get("report", "")).strip()
    prior_outcome = prior_entry.get("outcome")
    status = "INCORRECT" if prior_outcome is False else "CORRECT" if prior_outcome is True else "UNKNOWN"
    instruction = (
        "\n\n--- Prior Attempt Summary ---\n"
        f"Status: {status}\n"
        f"Completion: {prior_completion}\n"
        f"Report: {prior_report}\n"
        "Reflect on the prior reasoning. Provide a 'Correction:' section explaining what changes, "
        "and emit a revised plan/diagnosis/confidence."
    )
    return task.prompt + instruction


def _compute_replay_bonus(
    prior_entry: Optional[dict[str, object]],
    rollout: ReasoningRollout,
    training_config: Gpt2MetaTrainingConfig,
) -> float:
    if prior_entry is None or not rollout.is_replay:
        return 0.0
    prior_outcome = prior_entry.get("outcome")
    if rollout.outcome is None or prior_outcome is None:
        return 0.0
    text = rollout.completion.lower()
    mentioned = "correction" in text or "revision" in text or "fix" in text
    bonus = 0.0
    if prior_outcome is False and rollout.outcome is True and mentioned:
        bonus += training_config.replay_bonus
    if prior_outcome is True and rollout.outcome is False:
        bonus -= training_config.replay_penalty
    return bonus


def _build_trace_replay_prompt(task: ReasoningTask, prior_entry: dict[str, object]) -> str:
    focus_terms = prior_entry.get("focus_terms", [])
    focus_text = ", ".join(focus_terms) if focus_terms else "unknown focus"
    return (
        task.prompt
        + "\n\n--- Trace Replay ---\n"
        f"Previously, attention focused on: {focus_text}.\n"
        "Describe the internal focus trajectory in a 'Trace Replay:' block and explain whether those focal points were appropriate "
        "before emitting a corrected plan/diagnosis/confidence."
    )


def _compute_trace_replay_bonus(
    prior_entry: Optional[dict[str, object]],
    rollout: ReasoningRollout,
    training_config: Gpt2MetaTrainingConfig,
) -> float:
    if prior_entry is None or rollout.replay_type != "trace":
        return 0.0
    focus_terms = prior_entry.get("focus_terms", [])
    if not focus_terms:
        return 0.0
    text = rollout.report_text.lower()
    mentions = sum(1 for term in focus_terms if term and term.lower() in text)
    coverage = mentions / len(focus_terms)
    if coverage >= 0.5 and "trace replay" in text:
        return training_config.trace_replay_bonus
    return -training_config.trace_replay_penalty


def _build_reflection_prompt(task: ReasoningTask, prev_rollout: ReasoningRollout) -> str:
    previous = prev_rollout.completion.strip()
    report = prev_rollout.report_text.strip()
    outcome = "CORRECT" if prev_rollout.outcome else "INCORRECT"
    critique_hint = (
        "Compare each Step/Monitor pair against the plan and highlight where attention drifted. "
        "Include a 'Trace Critique:' bullet referencing specific focus tokens (e.g., numbers, entities, attention terms)."
    )
    return (
        task.prompt
        + "\n\n--- Reflection Context ---\n"
        f"Previous pass outcome: {outcome}\n"
        f"Previous completion:\n{previous}\n"
        f"Previous report:\n{report}\n"
        "Critique the prior reasoning in a 'Reflection:' section before providing a revised plan/diagnosis/confidence. "
        + critique_hint
    )


def _compute_reflection_adjustment(
    prev_rollout: ReasoningRollout,
    current_rollout: ReasoningRollout,
    config: Gpt2MetaTrainingConfig,
) -> float:
    text = current_rollout.report_text.lower()
    completion = current_rollout.completion.lower()
    bonus = 0.0
    mentions_reflection = "reflection" in text
    has_trace_section = "trace critique" in text
    references_previous = any(token in completion for token in ("step", "monitor", "plan"))
    changed_answer = prev_rollout.completion.strip() != current_rollout.completion.strip()
    coverage = _focus_coverage(current_rollout.report_text, prev_rollout.focus_terms)
    improved = prev_rollout.outcome is False and current_rollout.outcome is True
    regressed = prev_rollout.outcome is True and current_rollout.outcome is False
    if improved and mentions_reflection and references_previous and changed_answer:
        bonus += config.reflection_bonus
        bonus += config.reflection_trace_bonus * coverage
    if regressed or (current_rollout.pass_index > 0 and not mentions_reflection):
        bonus -= config.reflection_penalty
    if current_rollout.pass_index > 0 and (not changed_answer or coverage <= 0 or not has_trace_section):
        bonus -= config.reflection_repeat_penalty
    return bonus


def _execute_rollout_passes(
    model: Gpt2MetaTransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    task: ReasoningTask,
    training_config: Gpt2MetaTrainingConfig,
    prior_entry: Optional[dict[str, object]],
    initial_prompt: Optional[str],
    initial_replay_type: Optional[str],
    *,
    sample_gates: bool,
    force_open_gates: bool = False,
) -> Tuple[ReasoningRollout, float]:
    passes = max(1, training_config.reflection_passes)
    current_prompt = initial_prompt
    current_context = prior_entry if initial_prompt is not None else None
    current_replay_type = initial_replay_type
    total_reflection_bonus = 0.0
    previous_rollout: Optional[ReasoningRollout] = None
    final_rollout: Optional[ReasoningRollout] = None

    for pass_idx in range(passes):
        rollout = rollout_reasoning_episode(
            model,
            tokenizer,
            task,
            training_config,
            sample_gates=sample_gates,
            prompt_override=current_prompt,
            replay_context=current_context,
            force_open_gates=force_open_gates,
        )
        rollout.pass_index = pass_idx
        if pass_idx == 0:
            rollout.is_replay = current_prompt is not None
            rollout.replay_context = current_context
            rollout.replay_type = current_replay_type
        else:
            rollout.is_replay = True
            rollout.replay_type = "reflection"
            rollout.replay_context = {
                "completion": previous_rollout.completion if previous_rollout else "",
                "report": previous_rollout.report_text if previous_rollout else "",
                "outcome": previous_rollout.outcome,
            }
            if previous_rollout is not None:
                total_reflection_bonus += _compute_reflection_adjustment(previous_rollout, rollout, training_config)
        if pass_idx < passes - 1:
            current_prompt = _build_reflection_prompt(task, rollout)
            current_context = None
            current_replay_type = "reflection"
            previous_rollout = rollout
        else:
            final_rollout = rollout
    if final_rollout is None:
        raise RuntimeError("reflection passes failed to produce rollout")
    return final_rollout, total_reflection_bonus


def _extract_numbers(text: str) -> List[int]:
    return [int(match.group()) for match in re.finditer(r"-?\d+", text)]


def _number_near_token(text: str, keyword: str) -> Optional[int]:
    pattern = re.compile(rf"{keyword}\s*[:=]?\s*(-?\d+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None




def rollout_reasoning_episode(
    model: Gpt2MetaTransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    task: ReasoningTask,
    training_config: Gpt2MetaTrainingConfig,
    *,
    sample_gates: bool,
    prompt_override: Optional[str] = None,
    replay_context: Optional[dict[str, object]] = None,
    force_open_gates: bool = False,
) -> ReasoningRollout:
    base_model = _unwrap_model(model)
    device = base_model.device
    prompt_text = prompt_override if prompt_override is not None else task.prompt
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True).input_ids.to(device)
    attention_mask = torch.ones_like(encoded, device=device)
    prompt_len = encoded.size(1)
    max_steps = max(1, min(training_config.max_new_tokens, task.max_new_tokens))
    log_prob_accum: Optional[torch.Tensor]
    if sample_gates:
        log_prob_accum = torch.zeros((), device=device)
    else:
        log_prob_accum = None
    entropy_accum = torch.zeros((), device=device)
    steps = 0
    controller_state = None
    workspace_state = None
    gate_history: List[torch.Tensor] = []

    for _ in range(max_steps):
        outputs = model(
            encoded,
            attention_mask=attention_mask,
            sample_gates=sample_gates and not force_open_gates,
            return_gate_details=True,
            controller_state=controller_state,
            return_controller_state=True,
            workspace_state=workspace_state,
            return_workspace_state=True,
            record_workspace_trace=True,
            force_open_gates=force_open_gates,
        )
        logits, gate_details, controller_state, workspace_state = outputs
        if gate_details.probs is not None:
            entropy_accum = entropy_accum + _bernoulli_entropy(gate_details.probs).mean()
            gate_history.append(gate_details.probs.detach().to(torch.float32))
        if sample_gates:
            if gate_details.log_probs is None:
                raise RuntimeError("Expected log-probabilities when sampling gates")
            log_prob_accum = log_prob_accum + gate_details.log_probs.sum()
        next_token_logits = logits[:, -1, :].detach()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and steps < training_config.min_decoding_steps:
            next_token_logits[:, eos_id] = float('-inf')
        next_token = _select_next_token(next_token_logits, training_config.sample_temperature)
        encoded = torch.cat([encoded, next_token], dim=-1)
        attention_mask = torch.ones_like(encoded, device=device)
        steps += 1
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    completion_tokens = encoded[:, prompt_len:]
    completion_raw = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    completion = _ensure_structured_completion(completion_raw, task)
    reward = task.score_completion(completion)
    entropy = entropy_accum / max(steps, 1)
    gate_drift = 0.0
    if len(gate_history) >= 2:
        diffs = []
        for idx in range(1, len(gate_history)):
            diffs.append(torch.mean(torch.abs(gate_history[idx] - gate_history[idx - 1])))
        gate_drift = torch.stack(diffs).mean().item()
    monitoring_score = 0.0
    if task.monitoring_markers and task.min_monitoring_mentions > 0:
        text = completion.lower()
        hits = sum(text.count(marker.lower()) for marker in task.monitoring_markers)
        monitoring_score = min(1.0, hits / task.min_monitoring_mentions)
    outcome = _outcome_correct(task, completion)
    plan_score = _plan_consistency(completion)
    diagnosis_score, diagnosis_text_pred = _diagnosis_alignment(completion, outcome)
    plan_lines = _extract_plan_lines(completion)
    plan_only_lines = _extract_plan_only(completion)
    reasoning_lines = _extract_reasoning_steps(completion)
    monitor_lines = _extract_monitor_lines(completion)
    entity_tokens = _entity_tokens(task)
    plan_coverage = _coverage_ratio(plan_lines, entity_tokens)
    plan_repetition = _line_repetition_ratio(plan_lines)
    monitor_coverage = _coverage_ratio(monitor_lines, entity_tokens)
    workspace_trace = workspace_state.trace if workspace_state is not None else None
    summary_vec, introspection_tokens, raw_attentions, _ = base_model.get_introspection_state(encoded, attention_mask)
    focus_terms = base_model.extract_focus_terms(encoded, attention_mask, tokenizer)
    diagnosis_logit = base_model.diagnosis_prediction(summary_vec)
    diagnosis_pred_prob = torch.sigmoid(diagnosis_logit).item()
    judge_logit = base_model.judge_prediction(summary_vec)
    judge_prob = torch.sigmoid(judge_logit).item()
    teacher_vec = _build_teacher_attention_vector(
        getattr(task, "attention_hints", []),
        summary_vec.size(0),
        summary_vec.device,
    )
    if teacher_vec is not None:
        student_norm = summary_vec / (summary_vec.norm(p=2) + 1e-6)
        teacher_norm = teacher_vec / (teacher_vec.norm(p=2) + 1e-6)
        teacher_attention_alignment = float(
            torch.clamp(torch.dot(student_norm, teacher_norm), -1.0, 1.0).item()
        )
        teacher_attention_loss_tensor = F.mse_loss(student_norm, teacher_norm)
    else:
        teacher_attention_alignment = 0.0
        teacher_attention_loss_tensor = torch.zeros((), device=summary_vec.device)
    base_model.workspace.ingest_structure(
        workspace_state,
        plan_lines=plan_only_lines,
        step_lines=reasoning_lines,
        monitor_lines=monitor_lines,
        source="completion",
    )
    report_text = ""
    report_score = 0.0
    report_logits: Optional[torch.Tensor] = None
    alignment_score = 0.0
    focus_cov = 0.0
    confidence_value = None
    confidence_alignment = 0.0
    trace_alignment = 0.0
    report_embedding = None
    try:
        report_logits, report_tokens = base_model.generate_introspection_report(
            encoded,
            attention_mask=attention_mask,
            temperature=training_config.sample_temperature,
        )
    except RuntimeError:
        fallback_logits, fallback_tokens = base_model.report_head(
            summary_vec,
            temperature=training_config.sample_temperature,
        )
        report_logits = fallback_logits
        if fallback_tokens.dim() == 1:
            report_tokens = fallback_tokens.unsqueeze(0)
        else:
            report_tokens = fallback_tokens
    report_text = tokenizer.decode(report_tokens[0], skip_special_tokens=True)
    if not report_text.strip():
        report_text = task.gold_report or "Report: introspection fallback."
    if focus_terms and "trace summary" not in report_text.lower():
        report_text = report_text.rstrip() + "\nTrace Summary: " + ", ".join(focus_terms)
    report_score = _score_report(task, report_text, tokenizer, base_model.model)
    report_embedding = _encode_report_embedding(report_text, tokenizer, base_model.model)
    focus_cov = _focus_coverage(report_text, focus_terms)
    alignment_score = _descriptor_alignment_score(summary_vec, report_embedding)
    trace_alignment = _trace_alignment_score(workspace_trace, report_embedding, base_model.device)
    confidence_value = _extract_confidence(report_text)
    confidence_alignment = _confidence_alignment(confidence_value, outcome)
    trace_logits = base_model.trace_classifier(summary_vec.detach())
    trace_dim = trace_logits.size(-1)
    trace_target = torch.zeros(trace_dim, device=base_model.device)
    for term in focus_terms:
        idx = _trace_hash(term, trace_dim)
        trace_target[idx] = 1.0
    trace_loss = F.binary_cross_entropy_with_logits(trace_logits, trace_target)
    trace_probs = torch.sigmoid(trace_logits.detach())
    report_vector = torch.zeros(trace_dim, device=base_model.device)
    lowered_report = report_text.lower()
    for term in focus_terms:
        if term and term.lower() in lowered_report:
            idx = _trace_hash(term, trace_dim)
            report_vector[idx] = 1.0
    trace_agreement = 0.0
    if report_vector.sum() > 0:
        trace_agreement = float((trace_probs * report_vector).sum().item() / report_vector.sum().item())
    report_repetition = _report_repetition_ratio(report_text)
    graph_nodes = base_model.workspace.serialize_nodes(workspace_state)
    graph_ops = base_model.workspace.serialize_ops(workspace_state)
    graph_coverage = _graph_report_coverage(report_text, graph_nodes)
    return ReasoningRollout(
        reward=reward,
        log_prob=log_prob_accum,
        entropy=entropy,
        completion=completion,
        steps=steps,
        gate_drift=gate_drift,
        monitoring_score=monitoring_score,
        workspace_trace=workspace_trace,
        plan_score=plan_score,
        plan_coverage=plan_coverage,
        plan_repetition=plan_repetition,
        diagnosis_score=diagnosis_score,
        report_text=report_text,
        report_score=report_score,
        report_repetition=report_repetition,
        outcome=outcome,
        alignment_score=alignment_score,
        focus_terms=focus_terms,
        focus_score=focus_cov,
        monitor_coverage=monitor_coverage,
        trace_alignment=trace_alignment,
        trace_classifier_loss=trace_loss.item(),
        trace_agreement=trace_agreement,
        diagnosis_logit=diagnosis_logit,
        diagnosis_pred_prob=diagnosis_pred_prob,
        diagnosis_text_pred=diagnosis_text_pred,
        judge_logit=judge_logit,
        judge_prob=judge_prob,
        confidence_value=confidence_value,
        confidence_alignment=confidence_alignment,
        summary_vec=summary_vec,
        is_replay=replay_context is not None,
        replay_context=replay_context,
        report_logits=report_logits,
        graph_nodes=graph_nodes,
        graph_ops=graph_ops,
        teacher_attention_loss=float(teacher_attention_loss_tensor.item()),
        teacher_attention_alignment=teacher_attention_alignment,
        teacher_attention_loss_tensor=teacher_attention_loss_tensor,
        graph_coverage=graph_coverage,
    )


def build_model_and_optimizer(
    model_config: Gpt2MetaConfig, training_config: Gpt2MetaTrainingConfig
) -> Tuple[Gpt2MetaTransformerLM, Optimizer]:
    model = Gpt2MetaTransformerLM(model_config)
    params: List[nn.Parameter] = list(model.controller.parameters())
    if training_config.train_base_model:
        params += list(model.model.parameters())
    elif training_config.train_lm_head and hasattr(model.model, "lm_head"):
        for param in model.model.lm_head.parameters():
            param.requires_grad_(True)
        params += list(model.model.lm_head.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    return model, optimizer


def train_gpt2_meta(
    model: Gpt2MetaTransformerLM,
    optimizer: Optimizer,
    tokenizer: PreTrainedTokenizerBase,
    tasks: Sequence[ReasoningTask],
    training_config: Gpt2MetaTrainingConfig,
    *,
    trainable_params: Optional[Sequence[nn.Parameter]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    epoch_end_cb: Optional[Callable[[int, List[float]], None]] = None,
    progress_recorder: Optional[ProgressRecorder] = None,
    base_training_enabled: bool = True,
    is_main_process: bool = True,
) -> List[float]:
    if not tasks:
        raise RuntimeError("At least one reasoning task is required for training")
    base_model = _unwrap_model(model)
    model.train()
    rewards: List[float] = []
    params = (
        list(trainable_params)
        if trainable_params is not None
        else list(base_model.controller.parameters())
    )
    rng = random.Random(training_config.seed)
    baseline_value = 0.0
    episode_idx = 0
    report_root: Optional[Path] = (
        Path(training_config.run_dir) if (training_config.run_dir and is_main_process) else None
    )
    memory = EpisodeMemory(training_config.run_dir, training_config.global_rank)
    evaluator = EpisodeEvaluator(training_config)

    def _emit(msg: str) -> None:
        if not is_main_process:
            return
        print(msg, flush=True)
        if log_fn is not None:
            log_fn(msg)

    def _maybe_unfreeze_base(current_epoch: int) -> None:
        if not training_config.train_base_model or not base_training_enabled:
            return
        start = training_config.base_unfreeze_start_epoch
        if start is None:
            start = 0
        if current_epoch < start:
            return
        total_steps = training_config.base_unfreeze_steps
        if total_steps <= 0:
            total_steps = 1
        step_idx = max(0, current_epoch - start + 1)
        fraction = min(1.0, step_idx / total_steps)
        _set_transformer_grad_fraction(base_model, training_config, fraction)

    for epoch in range(training_config.num_epochs):
        _maybe_unfreeze_base(epoch)
        epoch_rewards: List[float] = []
        for _ in range(training_config.episodes_per_epoch):
            task = rng.choice(tasks)
            prior = memory.lookup(task.prompt)
            use_correction_replay = prior is not None and rng.random() < training_config.replay_probability
            use_trace_replay = prior is not None and rng.random() < training_config.trace_replay_probability
            prompt_override = None
            replay_type: Optional[str] = None
            if use_trace_replay and prior is not None:
                prompt_override = _build_trace_replay_prompt(task, prior)
                replay_type = "trace"
            elif use_correction_replay and prior is not None:
                prompt_override = _build_replay_prompt(task, prior)
                replay_type = "correction"

            force_open_gates = episode_idx < training_config.gate_warmup_episodes
            rollout, reflection_bonus_total = _execute_rollout_passes(
                model,
                tokenizer,
                task,
                training_config,
                prior,
                prompt_override,
                replay_type,
                sample_gates=not force_open_gates,
                force_open_gates=force_open_gates,
            )
            text_reward = rollout.reward * training_config.reward_scale
            monitor_struct = 0.5 * rollout.monitoring_score + 0.5 * rollout.monitor_coverage
            monitor_bonus = training_config.monitoring_bonus * monitor_struct
            if monitor_struct <= 0:
                monitor_bonus -= training_config.monitoring_penalty
            drift_bonus = training_config.gate_drift_bonus * rollout.gate_drift
            plan_struct = 0.5 * rollout.plan_score + 0.5 * rollout.plan_coverage
            plan_bonus = training_config.plan_bonus * plan_struct
            if plan_struct <= 0:
                plan_bonus -= training_config.plan_penalty
            diagnosis_bonus = training_config.diagnosis_bonus * rollout.diagnosis_score
            if rollout.diagnosis_score <= 0:
                diagnosis_bonus -= training_config.diagnosis_penalty
            report_bonus = training_config.report_bonus * rollout.report_score
            report_repeat_penalty = training_config.report_repetition_penalty * rollout.report_repetition
            plan_repeat_penalty = training_config.plan_repetition_penalty * rollout.plan_repetition
            focus_bonus = training_config.focus_bonus * rollout.focus_score
            focus_penalty_value = training_config.focus_penalty if rollout.focus_score <= 1e-6 else 0.0
            monitor_entity_penalty = training_config.monitor_entity_penalty if rollout.monitor_coverage <= 1e-6 else 0.0
            alignment_bonus = training_config.alignment_bonus * rollout.alignment_score
            confidence_bonus = training_config.confidence_bonus * rollout.confidence_alignment
            trace_bonus = training_config.trace_bonus * rollout.trace_alignment
            trace_classifier_bonus = training_config.trace_classifier_bonus * rollout.trace_agreement
            graph_bonus = training_config.graph_coverage_weight * rollout.graph_coverage
            replay_bonus = _compute_replay_bonus(prior, rollout, training_config)
            trace_replay_bonus = _compute_trace_replay_bonus(prior, rollout, training_config)
            reflection_bonus = reflection_bonus_total
            structure_gate = (plan_struct >= training_config.plan_gate) and (monitor_struct >= training_config.monitor_gate)
            if not structure_gate:
                text_reward = 0.0
                plan_bonus = 0.0
                monitor_bonus = 0.0
            classifier_bonus = 0.0
            classifier_pred = None
            if rollout.diagnosis_pred_prob is not None:
                classifier_pred = rollout.diagnosis_pred_prob >= 0.5
            if rollout.diagnosis_text_pred is not None and classifier_pred is not None:
                classifier_bonus = (
                    training_config.diagnosis_pred_bonus
                    if classifier_pred == rollout.diagnosis_text_pred
                    else -training_config.diagnosis_pred_bonus
                )
            memory_bonus = 0.0
            if prior is not None and rollout.outcome is not None and prior.get("outcome") is not None:
                if prior["outcome"] == rollout.outcome and rollout.diagnosis_score > 0:
                    memory_bonus = training_config.memory_bonus
                else:
                    memory_bonus = -training_config.memory_bonus
            reward_value = (
                text_reward
                + monitor_bonus
                + drift_bonus
                + plan_bonus
                + diagnosis_bonus
                + report_bonus
                + focus_bonus
                + alignment_bonus
                + confidence_bonus
                + trace_bonus
                + trace_classifier_bonus
                + reflection_bonus
                + trace_replay_bonus
                + replay_bonus
                + classifier_bonus
                + graph_bonus
                + memory_bonus
                - report_repeat_penalty
                - plan_repeat_penalty
                - monitor_entity_penalty
                - focus_penalty_value
            )
            raw_reward_value = reward_value
            evaluator_penalty = 0.0
            evaluator_payload = evaluator.evaluate(task, rollout.completion, episode_idx + 1)
            judge_target_value = 1.0 if rollout.outcome else 0.0
            if evaluator_payload and evaluator_payload.get("invalid"):
                evaluator_penalty = raw_reward_value
                reward_value = 0.0
                judge_target_value = 0.0
            if not structure_gate:
                judge_target_value = 0.0
            judge_bonus_value = 0.0
            if training_config.judge_bonus != 0.0:
                agreement = 1.0 - abs(judge_target_value - rollout.judge_prob)
                judge_bonus_value = training_config.judge_bonus * (2.0 * agreement - 1.0)
                reward_value = reward_value + judge_bonus_value
            advantage = reward_value - baseline_value
            if rollout.log_prob is not None:
                loss = -advantage * rollout.log_prob
            else:
                loss = torch.zeros((), device=base_model.device)
            if training_config.trace_classifier_weight > 0:
                loss = loss + training_config.trace_classifier_weight * rollout.trace_classifier_loss
            if (
                training_config.teacher_attention_weight > 0
                and rollout.teacher_attention_loss_tensor is not None
            ):
                loss = loss + training_config.teacher_attention_weight * rollout.teacher_attention_loss_tensor
            if rollout.outcome is not None and rollout.diagnosis_logit is not None:
                target = torch.tensor(
                    [[1.0 if rollout.outcome else 0.0]],
                    device=base_model.device,
                )
                diag_loss = F.binary_cross_entropy_with_logits(
                    rollout.diagnosis_logit.unsqueeze(0), target
                )
                loss = loss + training_config.diagnosis_pred_weight * diag_loss
            if training_config.judge_loss_weight > 0 and rollout.judge_logit is not None:
                judge_logit = rollout.judge_logit.view(1, -1)
                judge_target_tensor = torch.full_like(judge_logit, judge_target_value)
                judge_loss = F.binary_cross_entropy_with_logits(judge_logit, judge_target_tensor)
                loss = loss + training_config.judge_loss_weight * judge_loss
            if training_config.report_supervision_weight > 0 and task.gold_report and rollout.report_logits is not None:
                gold_ids = tokenizer(
                    task.gold_report,
                    return_tensors="pt",
                    truncation=True,
                    max_length=rollout.report_logits.size(0),
                ).input_ids.to(base_model.device)
                report_len = rollout.report_logits.size(0)
                if gold_ids.size(1) < report_len:
                    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                    pad = torch.full(
                        (gold_ids.size(0), report_len - gold_ids.size(1)),
                        pad_id,
                        dtype=gold_ids.dtype,
                        device=gold_ids.device,
                    )
                    gold_ids = torch.cat([gold_ids, pad], dim=1)
                else:
                    gold_ids = gold_ids[:, :report_len]
                pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                logits = rollout.report_logits.to(base_model.device)  # [L, V]
                ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    gold_ids.view(-1),
                    ignore_index=pad_token,
                )
                rollout.gold_loss = ce.item()
                loss = loss + training_config.report_supervision_weight * ce
            if training_config.entropy_coef > 0:
                loss = loss - training_config.entropy_coef * rollout.entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(params, training_config.grad_clip)
            optimizer.step()
            baseline_value = (
                training_config.baseline_momentum * baseline_value
                + (1.0 - training_config.baseline_momentum) * reward_value
            )
            rewards.append(reward_value)
            epoch_rewards.append(reward_value)
            episode_idx += 1
            memory_entry = memory.record(
                task.prompt,
                rollout.outcome,
                rollout.diagnosis_score,
                rollout.plan_score,
                rollout.report_text,
                rollout.summary_vec,
                rollout.focus_terms,
                rollout.workspace_trace,
                rollout.completion,
                rollout.confidence_value,
                rollout.graph_nodes,
                rollout.graph_ops,
            )
            if report_root is not None and rollout.report_text:
                _record_report(
                    report_root,
                    episode_idx,
                    task.prompt,
                    rollout.completion,
                    rollout.report_text,
                    rollout.focus_terms,
                    rollout.confidence_value,
                    memory_entry,
                )
            if training_config.log_every > 0 and (episode_idx % training_config.log_every) == 0:
                trace_len = len(rollout.workspace_trace) if rollout.workspace_trace is not None else 0
                _emit(
                    f"[epoch {epoch} episode {episode_idx}] "
                    f"reward={reward_value:.3f} base={text_reward:.3f} "
                    f"monitor={monitor_bonus:.3f} monitor_cov={rollout.monitor_coverage:.2f} drift={drift_bonus:.3f} "
                    f"plan={plan_bonus:.3f} plan_cov={rollout.plan_coverage:.2f} plan_rep_pen={plan_repeat_penalty:.3f} "
                    f"diagnosis={diagnosis_bonus:.3f} "
                    f"report={report_bonus:.3f} rep_pen={report_repeat_penalty:.3f} focus={focus_bonus:.3f} focus_pen={focus_penalty_value:.3f} align={alignment_bonus:.3f} "
                    f"confidence={confidence_bonus:.3f} trace={trace_bonus:.3f} trace_cls={trace_classifier_bonus:.3f} "
                    f"reflection={reflection_bonus:.3f} trace_replay={trace_replay_bonus:.3f} "
                    f"classifier={classifier_bonus:.3f} judge={rollout.judge_prob:.3f} teacher={rollout.teacher_attention_alignment:.3f} "
                    f"memory={memory_bonus:.3f} advantage={advantage:.3f} "
                    f"entropy={rollout.entropy.item():.3f} trace={trace_len} "
                    f"steps={rollout.steps}"
                )
            if (
                progress_recorder is not None
                and training_config.progress_every > 0
                and (episode_idx % training_config.progress_every) == 0
            ):
                validator_pass = None
                if task.validator:
                    validator_pass = run_validator(task.validator, task.validator_payload, rollout.completion)
                components = {
                    "text": text_reward,
                    "monitor": monitor_bonus,
                    "plan": plan_bonus,
                    "diagnosis": diagnosis_bonus,
                    "report": report_bonus,
                    "focus": focus_bonus,
                    "alignment": alignment_bonus,
                    "confidence": confidence_bonus,
                    "trace": trace_bonus,
                    "trace_classifier": trace_classifier_bonus,
                    "graph": graph_bonus,
                    "gate_drift": drift_bonus,
                    "reflection": reflection_bonus,
                    "trace_replay": trace_replay_bonus,
                    "replay": replay_bonus,
                    "classifier": classifier_bonus,
                    "judge_bonus": judge_bonus_value,
                    "memory": memory_bonus,
                    "penalty_report_repeat": -report_repeat_penalty,
                    "penalty_plan_repeat": -plan_repeat_penalty,
                    "penalty_focus": -focus_penalty_value,
                    "penalty_monitor_entity": -monitor_entity_penalty,
                    "penalty_evaluator": -evaluator_penalty,
                    "judge_prob": rollout.judge_prob,
                    "graph_plan_nodes": len(rollout.graph_nodes),
                    "graph_ops": len(rollout.graph_ops),
                    "graph_coverage": rollout.graph_coverage,
                    "teacher_attention": rollout.teacher_attention_alignment,
                }
                if validator_pass is not None:
                    components["validator_pass"] = 1.0 if validator_pass else 0.0
                if evaluator_payload is not None:
                    components["evaluator_prefix_quality"] = evaluator_payload.get("prefix_quality", 0.0)
                    components["evaluator_plan_cov"] = evaluator_payload.get("plan_coverage", 0.0)
                    components["evaluator_monitor_cov"] = evaluator_payload.get("monitor_coverage", 0.0)
                progress_recorder.record_episode(
                    epoch=epoch,
                    episode=episode_idx,
                    reward=reward_value,
                    components=components,
                    plan_score=rollout.plan_score,
                    plan_coverage=rollout.plan_coverage,
                    monitor_score=rollout.monitoring_score,
                    monitor_coverage=rollout.monitor_coverage,
                    structure_gate=structure_gate,
                    completion=rollout.completion,
                    report=rollout.report_text,
                    outcome=rollout.outcome,
                    confidence=rollout.confidence_value,
                    replay_type=rollout.replay_type,
                    reflection_pass=rollout.pass_index,
                    focus_terms=rollout.focus_terms,
                    evaluator=evaluator_payload,
                    judge_prob=rollout.judge_prob,
                    judge_target=judge_target_value,
                    graph_nodes=rollout.graph_nodes,
                    graph_ops=rollout.graph_ops,
                    validator_name=task.validator,
                    validator_payload=task.validator_payload,
                    validator_result=validator_pass,
                    teacher_alignment=rollout.teacher_attention_alignment,
                    graph_coverage=rollout.graph_coverage,
                )
        if epoch_end_cb is not None:
            epoch_end_cb(epoch, epoch_rewards)
    return rewards


def evaluate_gpt2_meta(
    model: Gpt2MetaTransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    tasks: Sequence[ReasoningTask],
    training_config: Gpt2MetaTrainingConfig,
) -> float:
    if not tasks:
        raise RuntimeError("Cannot evaluate without reasoning tasks")
    model.eval()
    rewards: List[float] = []
    evaluator = EpisodeEvaluator(training_config)
    with torch.no_grad():
        for idx in range(min(training_config.eval_episodes, len(tasks))):
            task = tasks[idx % len(tasks)]
            rollout, reflection_bonus_total = _execute_rollout_passes(
                model,
                tokenizer,
                task,
                training_config,
                prior_entry=None,
                initial_prompt=None,
                initial_replay_type=None,
                sample_gates=False,
            )
            text_reward = rollout.reward * training_config.reward_scale
            monitor_struct = 0.5 * rollout.monitoring_score + 0.5 * rollout.monitor_coverage
            monitor_bonus = training_config.monitoring_bonus * monitor_struct
            drift_bonus = training_config.gate_drift_bonus * rollout.gate_drift
            plan_struct = 0.5 * rollout.plan_score + 0.5 * rollout.plan_coverage
            plan_bonus = training_config.plan_bonus * plan_struct
            diagnosis_bonus = training_config.diagnosis_bonus * rollout.diagnosis_score
            report_bonus = training_config.report_bonus * rollout.report_score
            focus_bonus = training_config.focus_bonus * rollout.focus_score
            report_repeat_penalty = training_config.report_repetition_penalty * rollout.report_repetition
            plan_repeat_penalty = training_config.plan_repetition_penalty * rollout.plan_repetition
            focus_penalty_value = training_config.focus_penalty if rollout.focus_score <= 1e-6 else 0.0
            monitor_entity_penalty = training_config.monitor_entity_penalty if rollout.monitor_coverage <= 1e-6 else 0.0
            alignment_bonus = training_config.alignment_bonus * rollout.alignment_score
            confidence_bonus = training_config.confidence_bonus * rollout.confidence_alignment
            trace_bonus = training_config.trace_bonus * rollout.trace_alignment
            classifier_bonus = 0.0
            classifier_pred = rollout.diagnosis_pred_prob >= 0.5
            if rollout.diagnosis_text_pred is not None:
                classifier_bonus = (
                    training_config.diagnosis_pred_bonus
                    if classifier_pred == rollout.diagnosis_text_pred
                    else -training_config.diagnosis_pred_bonus
                )
            structure_gate = (plan_struct >= training_config.plan_gate) and (monitor_struct >= training_config.monitor_gate)
            judge_target_value = 1.0 if rollout.outcome else 0.0
            if not structure_gate:
                text_reward = 0.0
                plan_bonus = 0.0
                monitor_bonus = 0.0
                judge_target_value = 0.0
            reward_value = (
                text_reward
                + monitor_bonus
                + drift_bonus
                + plan_bonus
                + diagnosis_bonus
                + report_bonus
                + focus_bonus
                + alignment_bonus
                + confidence_bonus
                + trace_bonus
                + trace_bonus
                + reflection_bonus_total
                + classifier_bonus
                - report_repeat_penalty
                - plan_repeat_penalty
                - focus_penalty_value
                - monitor_entity_penalty
            )
            evaluator_payload = evaluator.evaluate(task, rollout.completion, idx + 1)
            if evaluator_payload and evaluator_payload.get("invalid"):
                reward_value = 0.0
                judge_target_value = 0.0
            if training_config.judge_bonus != 0.0:
                agreement = 1.0 - abs(judge_target_value - rollout.judge_prob)
                judge_bonus_value = training_config.judge_bonus * (2.0 * agreement - 1.0)
                reward_value = reward_value + judge_bonus_value
            rewards.append(reward_value)
    return float(sum(rewards) / max(len(rewards), 1))


def _ensure_run_dir(run_root: Optional[str]) -> Optional[Path]:
    if not run_root:
        return None
    base = Path(run_root).expanduser().resolve()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_config(run_dir: Path, training_cfg: Gpt2MetaTrainingConfig, model_cfg: Gpt2MetaConfig) -> None:
    snapshot = {
        "training": asdict(training_cfg),
        "model": {
            "base_model_name": model_cfg.base_model_name,
            "descriptor_dim": model_cfg.descriptor_dim,
            "controller": asdict(model_cfg.controller),
            "device": model_cfg.device,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _build_file_logger(run_dir: Optional[Path]) -> Optional[Callable[[str], None]]:
    if run_dir is None:
        return None
    log_path = run_dir / "train.log"

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("=== GPT-2 Meta RL Training Log ===\n")
        return _log
    except OSError:
        return None


def _save_checkpoint(
    model: Gpt2MetaTransformerLM,
    checkpoint_dir: Optional[Path],
    epoch_idx: int,
    include_lm_head: bool,
) -> Optional[Path]:
    if checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"epoch_{epoch_idx:04d}.pt"
    payload = {
        "epoch": epoch_idx,
        "controller": model.controller.state_dict(),
    }
    if include_lm_head and hasattr(model.model, "lm_head"):
        payload["lm_head"] = model.model.lm_head.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def _record_report(
    report_root: Path,
    episode_idx: int,
    prompt: str,
    completion: str,
    report_text: str,
    focus_terms: Sequence[str],
    confidence_value: Optional[float],
    memory_entry: Dict[str, object],
) -> None:
    report_dir = report_root / "reports"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    report_path = report_dir / f"episode_{episode_idx:05d}.txt"
    focus_line = ", ".join(focus_terms) if focus_terms else "None"
    content = [
        "=== Prompt ===",
        prompt.strip(),
        "",
        "=== Completion ===",
        completion.strip(),
        "",
        "=== Introspection Report ===",
        report_text.strip(),
        "",
        f"=== Focus Terms ===\n{focus_line}",
        "",
        f"=== Confidence ===\n{confidence_value if confidence_value is not None else 'N/A'}",
        "",
        "=== Memory Entry ===",
        json.dumps(memory_entry, indent=2),
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")


class ProgressRecorder:
    def __init__(self, run_dir: Optional[str], sample_chars: int) -> None:
        self.sample_chars = max(0, sample_chars)
        self.path = None
        if run_dir:
            base = Path(run_dir)
            base.mkdir(parents=True, exist_ok=True)
            self.path = base / "progress.jsonl"
            try:
                if not self.path.exists():
                    self.path.touch()
            except OSError:
                self.path = None

    def _sample_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = " ".join(text.strip().split())
        if self.sample_chars and len(normalized) > self.sample_chars:
            return normalized[: self.sample_chars] + "..."
        return normalized

    def _write(self, payload: dict) -> None:
        if self.path is None:
            return
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def record_imitation(self, epoch: int, total_epochs: int, loss: float, lm_loss: float, report_loss: float) -> None:
        self._write(
            {
                "phase": "imitation",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": loss,
                "lm_loss": lm_loss,
                "report_loss": report_loss,
            }
        )

    def record_episode(
        self,
        *,
        epoch: int,
        episode: int,
        reward: float,
        components: Dict[str, float],
        plan_score: float,
        plan_coverage: float,
        monitor_score: float,
        monitor_coverage: float,
        structure_gate: bool,
        completion: str,
        report: str,
        outcome: Optional[bool],
        confidence: Optional[float],
        replay_type: Optional[str],
        reflection_pass: int,
        focus_terms: Sequence[str],
        evaluator: Optional[Dict[str, object]] = None,
        judge_prob: Optional[float] = None,
        judge_target: Optional[float] = None,
        graph_nodes: Optional[List[Dict[str, object]]] = None,
        graph_ops: Optional[List[Dict[str, object]]] = None,
        validator_name: Optional[str] = None,
        validator_payload: Optional[Dict[str, object]] = None,
        validator_result: Optional[bool] = None,
        teacher_alignment: Optional[float] = None,
        graph_coverage: Optional[float] = None,
    ) -> None:
        entry = {
            "phase": "rl",
            "epoch": epoch,
            "episode": episode,
            "reward": reward,
            "components": components,
            "plan_score": plan_score,
            "plan_coverage": plan_coverage,
            "monitor_score": monitor_score,
            "monitor_coverage": monitor_coverage,
            "structure_gate": structure_gate,
            "outcome": outcome,
            "confidence": confidence,
            "replay_type": replay_type,
            "reflection_pass": reflection_pass,
            "focus_terms": list(focus_terms),
            "completion_excerpt": self._sample_text(completion),
            "report_excerpt": self._sample_text(report),
        }
        if evaluator is not None:
            entry["evaluator"] = evaluator
        if judge_prob is not None:
            entry["judge_prob"] = judge_prob
        if judge_target is not None:
            entry["judge_target"] = judge_target
        if graph_nodes:
            entry["graph_nodes"] = graph_nodes
        if graph_ops:
            entry["graph_ops"] = graph_ops
        if graph_coverage is not None:
            entry["graph_coverage"] = graph_coverage
        if validator_name:
            entry["validator"] = validator_name
            if validator_payload is not None:
                entry["validator_payload"] = validator_payload
        if validator_result is not None:
            entry["validator_pass"] = validator_result
        if teacher_alignment is not None:
            entry["teacher_alignment"] = teacher_alignment
        self._write(entry)

    def record_structure_check(
        self,
        *,
        index: int,
        total: int,
        plan_struct: float,
        monitor_struct: float,
        passed: bool,
        completion: str,
        report: str,
    ) -> None:
        entry = {
            "phase": "structure_check",
            "index": index,
            "total": total,
            "plan_struct": plan_struct,
            "monitor_struct": monitor_struct,
            "passed": passed,
            "completion_excerpt": self._sample_text(completion),
            "report_excerpt": self._sample_text(report),
        }
        self._write(entry)


class EpisodeMemory:
    def __init__(self, base_dir: Optional[str], rank: int = 0) -> None:
        suffix = "episode_memory.jsonl" if rank == 0 else f"episode_memory_rank{rank}.jsonl"
        self.path = None if base_dir is None else Path(base_dir) / suffix
        self.cache: dict[str, dict[str, object]] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    prompt = entry.get("prompt")
                    if prompt:
                        self.cache[prompt] = entry
                except json.JSONDecodeError:
                    continue

    def lookup(self, prompt: str) -> Optional[dict[str, object]]:
        return self.cache.get(prompt)


    def record(
        self,
        prompt: str,
        outcome: Optional[bool],
        diagnosis_score: float,
        plan_score: float,
        report_text: str,
        summary_vec: torch.Tensor,
        focus_terms: Sequence[str],
        workspace_trace: Optional[List[torch.Tensor]],
        completion: str,
        confidence: Optional[float],
        graph_nodes: Optional[List[Dict[str, object]]] = None,
        graph_ops: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        entry = {
            "prompt": prompt,
            "outcome": outcome,
            "diagnosis_score": diagnosis_score,
            "plan_score": plan_score,
            "report": report_text,
            "summary": summary_vec.detach().cpu().tolist(),
            "focus_terms": list(focus_terms),
            "workspace_trace_shapes": [tuple(int(dim) for dim in t.shape) for t in (workspace_trace or [])],
            "completion": completion,
            "confidence": confidence,
        }
        if graph_nodes:
            entry["graph_nodes"] = graph_nodes
        if graph_ops:
            entry["graph_ops"] = graph_ops
        self.cache[prompt] = entry
        if self.path is None:
            return entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry


class EpisodeEvaluator:
    def __init__(self, config: Gpt2MetaTrainingConfig) -> None:
        self.config = config
        self.enabled = bool(config.use_evaluator)
        if not self.enabled:
            self.tokenizer = None
            self.model = None
            self.teacher_tokenizer = None
            self.teacher_model = None
            return
        self.tokenizer, self.model = evaluator_maybe_load_lm(
            config.evaluator_prefix_lm,
            config.evaluator_device,
        )
        self.teacher_tokenizer = None
        self.teacher_model = None
        self.teacher_max_length = 0

    def evaluate(
        self,
        task: ReasoningTask,
        completion: str,
        episode_id: int,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled or not completion.strip():
            return None
        plan_markers = task.plan_markers if task.plan_markers else EVAL_DEFAULT_PLAN_MARKERS
        monitor_markers = task.monitoring_markers if task.monitoring_markers else EVAL_DEFAULT_MONITOR_MARKERS
        diag_markers = task.diagnosis_markers if task.diagnosis_markers else EVAL_DEFAULT_DIAG_MARKERS
        plan_min = task.min_plan_steps if task.min_plan_steps > 0 else (1 if plan_markers else 0)
        monitor_min = (
            task.min_monitoring_mentions
            if task.min_monitoring_mentions > 0
            else (1 if monitor_markers else 0)
        )
        diag_min = 1 if task.require_diagnosis else 0
        metrics = evaluator_evaluate_episode(
            {"episode": episode_id, "completion": completion},
            self.tokenizer,
            self.model,
            self.config.evaluator_prefix_cap,
            plan_markers,
            monitor_markers,
            diag_markers,
            plan_min,
            monitor_min,
            diag_min,
            self.teacher_tokenizer,
            self.teacher_model,
            self.teacher_max_length,
        )
        if metrics is None:
            return {
                "episode": episode_id,
                "invalid": True,
                "invalid_reasons": ["missing-completion"],
            }
        invalid_reasons = list(metrics.invalid_reasons)
        if metrics.plan_coverage < self.config.evaluator_plan_threshold and "plan-coverage" not in invalid_reasons:
            invalid_reasons.append("plan-coverage")
        if (
            metrics.monitor_coverage < self.config.evaluator_monitor_threshold
            and "monitor-coverage" not in invalid_reasons
        ):
            invalid_reasons.append("monitor-coverage")
        if (
            diag_min > 0
            and metrics.diagnosis_coverage < self.config.evaluator_diagnosis_threshold
            and "diagnosis-coverage" not in invalid_reasons
        ):
            invalid_reasons.append("diagnosis-coverage")
        payload: Dict[str, object] = {
            "episode": episode_id,
            "prefix_quality": metrics.prefix_quality,
            "prefix_perplexity": metrics.prefix_perplexity,
            "prefix_distance_score": metrics.prefix_distance_score,
            "plan_coverage": metrics.plan_coverage,
            "monitor_coverage": metrics.monitor_coverage,
            "diagnosis_coverage": metrics.diagnosis_coverage,
            "invalid_reasons": invalid_reasons,
            "invalid": bool(invalid_reasons),
        }
        return payload


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reinforcement learn GPT-2 meta-attention controller")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episodes-per-epoch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--gate-drift-bonus", type=float, default=0.1)
    parser.add_argument("--monitoring-bonus", type=float, default=0.1)
    parser.add_argument("--plan-bonus", type=float, default=0.05)
    parser.add_argument("--diagnosis-bonus", type=float, default=0.05)
    parser.add_argument("--report-bonus", type=float, default=0.1)
    parser.add_argument("--memory-bonus", type=float, default=0.05)
    parser.add_argument("--focus-bonus", type=float, default=0.05)
    parser.add_argument("--focus-penalty", type=float, default=0.0)
    parser.add_argument("--plan-repetition-penalty", type=float, default=0.0)
    parser.add_argument("--monitor-entity-penalty", type=float, default=0.0)
    parser.add_argument("--alignment-bonus", type=float, default=0.05)
    parser.add_argument("--confidence-bonus", type=float, default=0.05)
    parser.add_argument("--trace-bonus", type=float, default=0.05)
    parser.add_argument("--trace-replay-probability", type=float, default=0.0)
    parser.add_argument("--trace-replay-bonus", type=float, default=0.05)
    parser.add_argument("--trace-replay-penalty", type=float, default=0.05)
    parser.add_argument("--reflection-passes", type=int, default=1)
    parser.add_argument("--reflection-penalty", type=float, default=0.05)
    parser.add_argument("--reflection-bonus", type=float, default=0.05)
    parser.add_argument("--reflection-trace-bonus", type=float, default=0.05)
    parser.add_argument("--reflection-repeat-penalty", type=float, default=0.05)
    parser.add_argument("--gate-warmup-episodes", type=int, default=0)
    parser.add_argument("--min-decoding-steps", type=int, default=0)
    parser.add_argument("--report-repetition-penalty", type=float, default=0.0)
    parser.add_argument("--plan-penalty", type=float, default=0.0)
    parser.add_argument("--monitoring-penalty", type=float, default=0.0)
    parser.add_argument("--diagnosis-penalty", type=float, default=0.0)
    parser.add_argument("--teacher-attention-weight", type=float, default=0.0, help="Weight for teacher attention distillation loss.")
    parser.add_argument("--graph-coverage-weight", type=float, default=0.0, help="Weight for graph coverage reward component.")
    parser.add_argument("--imitation-steps", type=int, default=0)
    parser.add_argument("--imitation-batch-size", type=int, default=1)
    parser.add_argument("--plan-gate", type=float, default=0.0)
    parser.add_argument("--monitor-gate", type=float, default=0.0)
    parser.add_argument("--imitation-report-weight", type=float, default=1.0)
    parser.add_argument("--diagnosis-pred-weight", type=float, default=0.1)
    parser.add_argument("--diagnosis-pred-bonus", type=float, default=0.05)
    parser.add_argument("--judge-bonus", type=float, default=0.0, help="Bonus/penalty scalar applied to the self-judge verdict (positive rewards when judge predicts pass).")
    parser.add_argument("--judge-loss-weight", type=float, default=0.0, help="Cross-entropy weight for supervising the self-judge head (target = evaluator or outcome).")
    parser.add_argument("--trace-classifier-weight", type=float, default=0.0)
    parser.add_argument("--trace-classifier-bonus", type=float, default=0.0)
    parser.add_argument("--report-supervision-weight", type=float, default=0.0)
    parser.add_argument("--replay-probability", type=float, default=0.25)
    parser.add_argument("--replay-bonus", type=float, default=0.1)
    parser.add_argument("--replay-penalty", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--train-base-model", action="store_true")
    parser.add_argument("--base-unfreeze-start-epoch", type=int, default=None)
    parser.add_argument("--base-unfreeze-steps", type=int, default=0)
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/meta_autolog",
        help="Root directory for run artifacts; pass an empty string to disable filesystem logging.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write a progress.jsonl entry every N episodes (0 disables progress logging).",
    )
    parser.add_argument(
        "--progress-sample-chars",
        type=int,
        default=256,
        help="Maximum number of characters to keep per completion/report excerpt in progress.jsonl.",
    )
    parser.add_argument(
        "--use-evaluator",
        action="store_true",
        help="Enable automated evaluator gating per episode.",
    )
    parser.add_argument(
        "--evaluator-prefix-lm",
        default="none",
        help="HF model name for evaluator prefix perplexity (set to 'none' to disable).",
    )
    parser.add_argument(
        "--evaluator-device",
        default="cpu",
        help="Device for evaluator LM (cpu/cuda).",
    )
    parser.add_argument(
        "--evaluator-prefix-cap",
        type=int,
        default=160,
        help="Maximum number of characters allowed before 'Plan:' in completions before penalties apply.",
    )
    parser.add_argument(
        "--evaluator-plan-threshold",
        type=float,
        default=0.75,
        help="Minimum plan coverage required to retain reward (0-1).",
    )
    parser.add_argument(
        "--evaluator-monitor-threshold",
        type=float,
        default=0.75,
        help="Minimum monitor coverage required to retain reward (0-1).",
    )
    parser.add_argument(
        "--evaluator-diagnosis-threshold",
        type=float,
        default=1.0,
        help="Minimum diagnosis coverage required to retain reward when diagnoses are mandatory (0-1).",
    )
    parser.add_argument(
        "--structure-check-episodes",
        type=int,
        default=4,
        help="Number of imitation validation rollouts required to pass the plan/monitor gate check.",
    )
    parser.add_argument(
        "--enable-gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the base GPT-2 to reduce memory usage while keeping full traces.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reasoning-tasks", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--use-process-state", dest="use_process_state", action="store_true")
    parser.add_argument("--disable-process-state", dest="use_process_state", action="store_false")
    parser.set_defaults(use_process_state=True)
    parser.add_argument("--process-state-dim", type=int, default=128)
    return parser


def main(cli_args: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(cli_args)
    device_str, is_dist, rank, world_size, local_rank = _init_distributed(args.device)
    is_main = rank == 0
    training_cfg = Gpt2MetaTrainingConfig(
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        reward_scale=args.reward_scale,
        entropy_coef=args.entropy_coef,
        baseline_momentum=args.baseline_momentum,
        gate_drift_bonus=args.gate_drift_bonus,
        monitoring_bonus=args.monitoring_bonus,
        plan_bonus=args.plan_bonus,
        diagnosis_bonus=args.diagnosis_bonus,
        report_bonus=args.report_bonus,
        memory_bonus=args.memory_bonus,
        focus_bonus=args.focus_bonus,
        focus_penalty=args.focus_penalty,
        alignment_bonus=args.alignment_bonus,
        confidence_bonus=args.confidence_bonus,
        diagnosis_pred_weight=args.diagnosis_pred_weight,
        diagnosis_pred_bonus=args.diagnosis_pred_bonus,
        judge_bonus=args.judge_bonus,
        judge_loss_weight=args.judge_loss_weight,
        trace_classifier_weight=args.trace_classifier_weight,
        trace_classifier_bonus=args.trace_classifier_bonus,
        replay_probability=args.replay_probability,
        replay_bonus=args.replay_bonus,
        replay_penalty=args.replay_penalty,
        report_supervision_weight=args.report_supervision_weight,
        trace_bonus=args.trace_bonus,
        trace_replay_probability=args.trace_replay_probability,
        trace_replay_bonus=args.trace_replay_bonus,
        trace_replay_penalty=args.trace_replay_penalty,
        reflection_passes=args.reflection_passes,
        reflection_penalty=args.reflection_penalty,
        reflection_bonus=args.reflection_bonus,
        reflection_trace_bonus=args.reflection_trace_bonus,
        reflection_repeat_penalty=args.reflection_repeat_penalty,
        gate_warmup_episodes=args.gate_warmup_episodes,
        min_decoding_steps=args.min_decoding_steps,
        report_repetition_penalty=args.report_repetition_penalty,
        plan_repetition_penalty=args.plan_repetition_penalty,
        monitor_entity_penalty=args.monitor_entity_penalty,
        imitation_steps=args.imitation_steps,
        imitation_batch_size=args.imitation_batch_size,
        plan_gate=args.plan_gate,
        monitor_gate=args.monitor_gate,
        imitation_report_weight=args.imitation_report_weight,
        plan_penalty=args.plan_penalty,
        monitoring_penalty=args.monitoring_penalty,
        diagnosis_penalty=args.diagnosis_penalty,
        teacher_attention_weight=args.teacher_attention_weight,
        graph_coverage_weight=args.graph_coverage_weight,
        progress_every=args.progress_every,
        progress_sample_chars=args.progress_sample_chars,
        structure_check_episodes=args.structure_check_episodes,
        use_evaluator=args.use_evaluator,
        evaluator_prefix_lm=args.evaluator_prefix_lm,
        evaluator_device=args.evaluator_device,
        evaluator_prefix_cap=args.evaluator_prefix_cap,
        evaluator_plan_threshold=args.evaluator_plan_threshold,
        evaluator_monitor_threshold=args.evaluator_monitor_threshold,
        evaluator_diagnosis_threshold=args.evaluator_diagnosis_threshold,
        seed=args.seed,
        log_every=args.log_every,
        tokenizer_name=args.tokenizer_name,
        base_model_name=args.base_model,
        train_lm_head=args.train_lm_head,
        train_base_model=args.train_base_model,
        base_unfreeze_start_epoch=args.base_unfreeze_start_epoch,
        base_unfreeze_steps=args.base_unfreeze_steps,
        run_dir=args.run_dir,
        device=device_str,
        reasoning_tasks_path=args.reasoning_tasks,
        sample_temperature=args.sample_temperature,
        eval_episodes=args.eval_episodes,
        use_process_state=args.use_process_state,
        process_state_dim=args.process_state_dim,
        distributed=is_dist,
        global_rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    set_seed(training_cfg.seed)
    run_dir_path: Optional[Path] = None
    if is_main:
        run_dir_path = _ensure_run_dir(training_cfg.run_dir)
    if is_dist:
        buffer: List[Optional[str]] = [str(run_dir_path) if run_dir_path is not None else None]
        dist.broadcast_object_list(buffer, src=0)
        shared_path = buffer[0]
        run_dir_path = Path(shared_path) if shared_path else None
    if run_dir_path is not None:
        training_cfg.run_dir = str(run_dir_path)
    checkpoint_dir = run_dir_path / "checkpoints" if run_dir_path is not None else None
    tokenizer = _prepare_tokenizer(training_cfg.tokenizer_name)
    tasks = load_reasoning_tasks(training_cfg.reasoning_tasks_path)
    controller_cfg = MetaControllerConfig(
        descriptor_dim=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_process_state=training_cfg.use_process_state,
        process_hidden_size=training_cfg.process_state_dim,
    )
    model_cfg = Gpt2MetaConfig(
        base_model_name=training_cfg.base_model_name,
        descriptor_dim=4,
        controller=controller_cfg,
        device=training_cfg.device,
        gradient_checkpointing=args.enable_gradient_checkpointing,
    )
    base_model, optimizer = build_model_and_optimizer(model_cfg, training_cfg)
    if training_cfg.distributed:
        model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        model._set_static_graph()
    else:
        model = base_model
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=training_cfg.device)
        missing, unexpected = _unwrap_model(model).load_state_dict(state, strict=False)
        if is_main:
            msg = (
                f"Loaded checkpoint {args.init_checkpoint}"
                f" missing={missing} unexpected={unexpected}"
            )
            print(msg)
    if run_dir_path is not None and is_main:
        _write_run_config(run_dir_path, training_cfg, model_cfg)
    file_logger = _build_file_logger(run_dir_path) if is_main else None
    progress_recorder = (
        ProgressRecorder(training_cfg.run_dir, training_cfg.progress_sample_chars) if is_main else None
    )
    _run_imitation_phase(
        model,
        optimizer,
        tokenizer,
        tasks,
        training_cfg,
        log_fn=file_logger,
        progress_recorder=progress_recorder,
        is_main_process=is_main,
    )
    base_training_ready = True
    failure_msg = ""
    if training_cfg.train_base_model and is_main:
        try:
            _structure_gate_check(
                model,
                tokenizer,
                tasks,
                training_cfg,
                progress_recorder=progress_recorder,
                log_fn=file_logger,
            )
        except RuntimeError as exc:
            base_training_ready = False
            failure_msg = str(exc)
    if training_cfg.distributed:
        flag = torch.tensor(1 if base_training_ready else 0, device=training_cfg.device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            if failure_msg:
                raise RuntimeError(failure_msg)
            raise RuntimeError("structure-check failed on primary process")
    if not base_training_ready:
        raise RuntimeError(failure_msg or "structure-check failed")

    def _epoch_log(epoch_idx: int, epoch_rewards: List[float]) -> None:
        if not is_main:
            return
        if epoch_rewards and file_logger is not None:
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            file_logger(f"[epoch {epoch_idx}] avg_reward={avg_reward:.4f}")
        if checkpoint_dir is not None:
            ckpt_path = _save_checkpoint(
                _unwrap_model(model),
                checkpoint_dir,
                epoch_idx,
                training_cfg.train_lm_head,
            )
            if ckpt_path is not None:
                msg = f"[epoch {epoch_idx}] checkpoint saved to {ckpt_path}"
                print(msg)
                if file_logger is not None:
                    file_logger(msg)

    if training_cfg.train_base_model:
        _set_transformer_grad_fraction(_unwrap_model(model), training_cfg, 0.0)
    base_for_params = _unwrap_model(model)
    params_for_clip = list(base_for_params.controller.parameters())
    if training_cfg.train_base_model:
        params_for_clip += list(base_for_params.model.parameters())
    elif training_cfg.train_lm_head and hasattr(base_for_params.model, "lm_head"):
        params_for_clip += list(base_for_params.model.lm_head.parameters())
    rewards = train_gpt2_meta(
        model,
        optimizer,
        tokenizer,
        tasks,
        training_cfg,
        trainable_params=params_for_clip,
        log_fn=file_logger,
        epoch_end_cb=_epoch_log,
        progress_recorder=progress_recorder,
        base_training_enabled=base_training_ready,
        is_main_process=is_main,
    )
    if is_main:
        message = (
            f"Completed GPT-2 meta RL training, final reward {rewards[-1]:.4f}"
            if rewards
            else "Completed GPT-2 meta RL training"
        )
        print(message)
        if file_logger is not None:
            file_logger(message)
        avg_reward = evaluate_gpt2_meta(_unwrap_model(model), tokenizer, tasks, training_cfg)
        val_msg = f"Evaluation average reward {avg_reward:.4f}"
        print(val_msg)
        if file_logger is not None:
            file_logger(val_msg)
    if training_cfg.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
