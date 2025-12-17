from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn

from meta_transformer.config import MetaWorkspaceConfig


@dataclass
class MetaWorkspaceState:
    """Holds persistent workspace slots and optional trace history."""

    slots: torch.Tensor
    step: int = 0
    trace: Optional[List[torch.Tensor]] = None
    graph_nodes: List["WorkspaceGraphNode"] = field(default_factory=list)
    graph_ops: List["WorkspaceGraphOperation"] = field(default_factory=list)
    next_node_id: int = 0


@dataclass
class WorkspaceGraphNode:
    node_id: int
    kind: str
    label: str
    parent_id: Optional[int]
    score: float
    source: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "parent": self.parent_id,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class WorkspaceGraphOperation:
    op: str
    node_id: int
    kind: str
    label: str
    parent_id: Optional[int]
    step: int
    source: str
    targets: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "op": self.op,
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "parent": self.parent_id,
            "step": self.step,
            "source": self.source,
        }
        if self.targets:
            payload["targets"] = list(self.targets)
        return payload


@dataclass
class MetaWorkspaceOutput:
    summary: torch.Tensor
    state: MetaWorkspaceState
    trace_vector: Optional[torch.Tensor] = None


class MetaWorkspace(nn.Module):
    """Aggregates per-layer descriptors into a global workspace summary."""

    def __init__(self, config: MetaWorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.token_proj = nn.Linear(config.descriptor_dim, config.slot_dim)
        self.slot_gru = nn.GRUCell(config.slot_dim, config.slot_dim)
        self.slot_norm = nn.LayerNorm(config.slot_dim)
        self.summary_proj = nn.Linear(config.slot_dim, config.summary_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        descriptors: torch.Tensor,
        state: Optional[MetaWorkspaceState] = None,
        *,
        record_trace: bool = False,
    ) -> MetaWorkspaceOutput:
        if descriptors.dim() != 3:
            raise ValueError("descriptors must have shape [num_layers, num_heads, D]")
        device = descriptors.device
        dtype = descriptors.dtype
        global_token = descriptors.mean(dim=(0, 1))
        slot_input = self.activation(self.token_proj(global_token))
        num_slots = self.config.num_slots
        if state is None:
            prev_slots = torch.zeros(num_slots, self.config.slot_dim, device=device, dtype=slot_input.dtype)
            prev_trace: Optional[List[torch.Tensor]] = None
            step = 0
            prev_nodes: List[WorkspaceGraphNode] = []
            prev_ops: List[WorkspaceGraphOperation] = []
            next_node_id = 0
        else:
            prev_slots = state.slots.to(device=device, dtype=slot_input.dtype)
            prev_trace = state.trace
            step = state.step
            prev_nodes = list(state.graph_nodes)
            prev_ops = list(state.graph_ops)
            next_node_id = state.next_node_id
        slot_inputs = slot_input.unsqueeze(0).expand(num_slots, -1).contiguous()
        updated_slots = self.slot_gru(slot_inputs, prev_slots)
        normalized_slots = self.slot_norm(updated_slots)
        summary = self.summary_proj(normalized_slots.mean(dim=0))
        next_state = MetaWorkspaceState(
            slots=normalized_slots.detach(),
            step=step + 1,
            trace=prev_trace,
            graph_nodes=prev_nodes,
            graph_ops=prev_ops,
            next_node_id=next_node_id,
        )
        trace_vector = None
        if record_trace and self.config.track_trace:
            trace_vector = torch.cat(
                [summary.detach().cpu(), global_token.detach().cpu().to(summary.dtype)],
                dim=0,
            )
            history = [] if prev_trace is None else list(prev_trace)
            history.append(trace_vector)
            next_state.trace = history
        elif record_trace:
            next_state.trace = prev_trace
        else:
            next_state.trace = prev_trace
        return MetaWorkspaceOutput(summary=summary, state=next_state, trace_vector=trace_vector)

    def init_state(self, device: torch.device, dtype: torch.dtype = torch.float32) -> MetaWorkspaceState:
        slots = torch.zeros(self.config.num_slots, self.config.slot_dim, device=device, dtype=dtype)
        return MetaWorkspaceState(slots=slots, step=0, trace=[], graph_nodes=[], graph_ops=[], next_node_id=0)

    # Graph utilities -----------------------------------------------------

    def add_graph_node(
        self,
        state: MetaWorkspaceState,
        *,
        kind: str,
        label: str,
        parent_id: Optional[int] = None,
        score: float = 0.0,
        source: str = "auto",
    ) -> Optional[WorkspaceGraphNode]:
        if not self.config.track_graph or state is None:
            return None
        node_id = state.next_node_id
        state.next_node_id += 1
        node = WorkspaceGraphNode(
            node_id=node_id,
            kind=kind,
            label=label.strip(),
            parent_id=parent_id,
            score=score,
            source=source,
        )
        state.graph_nodes.append(node)
        state.graph_ops.append(
            WorkspaceGraphOperation(
                op="add",
                node_id=node_id,
                kind=kind,
                label=node.label,
                parent_id=parent_id,
                step=state.step,
                source=source,
            )
        )
        return node

    def merge_graph_nodes(
        self,
        state: MetaWorkspaceState,
        node_ids: List[int],
        *,
        label: str,
        kind: str = "summary",
        source: str = "merge",
    ) -> Optional[WorkspaceGraphNode]:
        if not self.config.track_graph or state is None or len(node_ids) < 2:
            return None
        merged = self.add_graph_node(state, kind=kind, label=label, parent_id=None, source=source)
        if merged is None:
            return None
        state.graph_ops.append(
            WorkspaceGraphOperation(
                op="merge",
                node_id=merged.node_id,
                kind=kind,
                label=label,
                parent_id=None,
                step=state.step,
                source=source,
                targets=list(node_ids),
            )
        )
        return merged

    def serialize_nodes(self, state: Optional[MetaWorkspaceState]) -> List[Dict[str, object]]:
        if state is None or not self.config.track_graph:
            return []
        return [node.to_dict() for node in state.graph_nodes]

    def serialize_ops(self, state: Optional[MetaWorkspaceState]) -> List[Dict[str, object]]:
        if state is None or not self.config.track_graph:
            return []
        return [op.to_dict() for op in state.graph_ops]

    def ingest_structure(
        self,
        state: Optional[MetaWorkspaceState],
        *,
        plan_lines: List[str],
        step_lines: List[str],
        monitor_lines: List[str],
        source: str = "completion",
    ) -> None:
        if state is None or not self.config.track_graph:
            return
        plan_ids: List[int] = []
        for line in plan_lines:
            node = self.add_graph_node(state, kind="plan", label=line, parent_id=None, source=source)
            if node is not None:
                plan_ids.append(node.node_id)
        for idx, line in enumerate(step_lines):
            parent = plan_ids[idx] if idx < len(plan_ids) else None
            self.add_graph_node(state, kind="step", label=line, parent_id=parent, source=source)
        for line in monitor_lines:
            self.add_graph_node(state, kind="monitor", label=line, parent_id=None, source=source)
        if len(plan_ids) >= 2:
            self.merge_graph_nodes(
                state,
                plan_ids,
                label="plan_chain",
                kind="plan",
                source=source,
            )
