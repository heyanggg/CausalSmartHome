"""把挖掘出的因果先验整理成生成提示的辅助工具。

这个模块里的早期入口可以学习 device-level prior，把数字设备 ID 映射成可读
设备名，并渲染 prompt 文本。当前主流程主要使用
``causal_relation_prior_source`` 和 ``causal_gss_reweight``，但这里的函数仍
适合做交互式检查、边表展示和旧格式 prompt 辅助生成。
"""

from __future__ import annotations

import ast
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causal_prior import CausalPrior
from .event_tensor import EventTensorizer
from .causal_relation_adapter import CausalRelationAdapter
from .schema import BehaviorSequence, load_numeric_sequences


SOFT_CAUSAL_HINT_INTRO = (
    "The following device-level causal patterns are common in the user's historical behavior. "
    "When these devices appear together, preserve their usual temporal order unless the target "
    "context explicitly suggests a change."
)


@dataclass(frozen=True)
class DeviceCausalEdge:
    """带设备名的 device-level GCAD 边。

    ``source``/``target`` 保留机器可读通道键，例如 ``d:13``；
    ``source_name``/``target_name`` 用于人读报告和 prompt。
    """

    source: str
    target: str
    source_name: str
    target_name: str
    source_index: int
    target_index: int
    weight: float
    lag: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pickle_sequences(path: str | Path) -> list[BehaviorSequence]:
    """读取 Gen pickle，并统一转换为 ``BehaviorSequence`` 对象。"""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return load_numeric_sequences(raw)


def learn_device_causal_relation_prior(
    sequences: Sequence[BehaviorSequence],
    lag: int = 4,
    epochs: int = 80,
    sparse_threshold: float = 0.001,
    batch_size: int = 64,
    hidden: int = 16,
    sample_limit: int | None = None,
) -> CausalPrior:
    """从源上下文正常序列中挖掘 device-level GCAD prior。"""
    tensorized = EventTensorizer(level="device", count_mode="binary", decay=0.1).fit_transform(sequences)
    if not tensorized.channel_to_key:
        raise ValueError("no device channels found in source training data")
    return CausalRelationAdapter().mine_event_prior(
        tensorized.tensor,
        tensorized.channel_to_key,
        lag=lag,
        epochs=epochs,
        hidden=hidden,
        sparse_threshold=sparse_threshold,
        batch_size=batch_size,
        sample_limit=sample_limit,
    )


def load_id_name_mapping(path: str | Path | None, preferred_names: Sequence[str] | None = None) -> dict[int, str]:
    """从 JSON 或 Gen ``dictionary.py`` 风格文件中读取设备 ID 到名称的映射。"""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _mapping_from_json(payload, preferred_names=preferred_names)
    return _mapping_from_python(path, preferred_names=preferred_names)


def _mapping_from_json(payload: Any, preferred_names: Sequence[str] | None = None) -> dict[int, str]:
    if isinstance(payload, Mapping):
        candidates: list[Mapping[Any, Any]] = []
        if preferred_names:
            for name in preferred_names:
                value = payload.get(name)
                if isinstance(value, Mapping):
                    candidates.append(value)
        if not candidates:
            candidates = [payload]
        for candidate in candidates:
            normalized = _normalize_name_to_id(candidate)
            if normalized:
                return normalized
    raise ValueError("JSON mapping must be an object containing name->id or id->name entries")


def _mapping_from_python(path: Path, preferred_names: Sequence[str] | None = None) -> dict[int, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    names = list(preferred_names or [])
    names.extend(name for name in assignments if name not in names)
    for name in names:
        value = assignments.get(name)
        if isinstance(value, Mapping):
            normalized = _normalize_name_to_id(value)
            if normalized:
                return normalized
    raise ValueError(f"no usable mapping dictionary found in {path}")


def _normalize_name_to_id(mapping: Mapping[Any, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(value, int):
            out[int(value)] = str(key)
        else:
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
    return out


def device_key_to_name(key: str, device_id_to_name: Mapping[int, str] | None = None) -> str:
    """把 ``d:3`` 这类标准通道键转换成可读设备名。"""
    prefix = "d:"
    if key.startswith(prefix):
        raw = key[len(prefix) :]
        try:
            device_id = int(raw)
        except ValueError:
            return raw
        if device_id_to_name and device_id in device_id_to_name:
            return device_id_to_name[device_id]
        return f"device_{device_id}"
    return key


def map_device_edges(
    prior: CausalPrior,
    device_id_to_name: Mapping[int, str] | None = None,
    top_k_edges: int = 20,
    min_weight: float | None = None,
) -> list[DeviceCausalEdge]:
    """把 prior 矩阵中的行列索引转换成带设备名称的有向边列表。"""
    return [
        DeviceCausalEdge(
            source=str(edge["source"]),
            target=str(edge["target"]),
            source_name=device_key_to_name(str(edge["source"]), device_id_to_name),
            target_name=device_key_to_name(str(edge["target"]), device_id_to_name),
            source_index=int(edge["source_index"]),
            target_index=int(edge["target_index"]),
            weight=float(edge["weight"]),
            lag=int(edge["lag"]),
        )
        for edge in prior.top_edges(k=top_k_edges, min_weight=min_weight, include_self=False)
    ]


def build_causal_hints_payload(
    edges: Sequence[DeviceCausalEdge],
    prior: CausalPrior,
    source_train_pkl: str | Path,
    level: str = "device",
) -> dict[str, Any]:
    """构造描述软因果提示的 JSON payload。"""
    return {
        "hint_type": "causal_gss_device_prior",
        "level": level,
        "constraint_strength": "soft",
        "intro": SOFT_CAUSAL_HINT_INTRO,
        "interpretation": (
            "Each edge means the source device is a common historical predecessor or causal-relation-style "
            "causal signal for the target device. Treat these as generation hints, not hard rules."
        ),
        "source_train_pkl": str(source_train_pkl),
        "lag": prior.lag,
        "sparse_threshold": prior.sparse_threshold,
        "method": prior.method,
        "top_causal_edges": [edge.to_dict() for edge in edges],
        "meta": prior.meta or {},
    }


def format_causal_hints_for_prompt(payload: Mapping[str, Any]) -> str:
    """把因果边渲染成人类可读的 prompt 文本。"""
    lines = [
        "",
        "causal-relation-guided GSS device-level causal hints (soft constraints):",
        str(payload["intro"]),
        "",
    ]
    edges = list(payload.get("top_causal_edges", []))
    if not edges:
        lines.append("- No non-zero device-level causal edges passed the configured threshold.")
    else:
        for index, edge in enumerate(edges, start=1):
            lines.append(
                f"{index}. If {edge['source_name']} and {edge['target_name']} appear together, "
                f"{edge['source_name']} usually precedes {edge['target_name']} "
                f"(causal relation weight={float(edge['weight']):.6f}, lag={int(edge['lag'])})."
            )
    lines.extend(
        [
            "",
            "Use these patterns as causal generation guidance only; keep the target context and the original Gen GSS hints authoritative when they provide stronger evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def render_edges_markdown(edges: Sequence[DeviceCausalEdge], payload: Mapping[str, Any]) -> str:
    """把 top causal edges 渲染成 Markdown 审计表。"""
    lines = [
        "# Top Device-Level causal relation Causal Edges",
        "",
        payload.get("intro", SOFT_CAUSAL_HINT_INTRO),
        "",
        "| rank | source | target | source_key | target_key | weight | lag |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for rank, edge in enumerate(edges, start=1):
        lines.append(
            f"| {rank} | {edge.source_name} | {edge.target_name} | `{edge.source}` | `{edge.target}` | "
            f"{edge.weight:.6f} | {edge.lag} |"
        )
    return "\n".join(lines) + "\n"
