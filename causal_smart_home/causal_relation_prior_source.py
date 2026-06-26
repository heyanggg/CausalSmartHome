from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .causal_prior import CausalPrior
from .event_tensor import EventTensorizer
from .causal_relation_adapter import CausalRelationAdapter
from .schema import BehaviorSequence, load_numeric_sequences


@dataclass
class ResolvedCausalRelationPrior:
    """Unified, JSON-serializable view of a causal relation prior.

    The object records where the causal matrix came from.  When the project has
    to use the local compact adapter fallback, ``causal_relation_source`` says so
    explicitly.
    """

    causal_relation_source: str
    level: str
    lag: int
    sparse_threshold: float
    channels: list[str]
    matrix: list[list[float]]
    top_causal_edges: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def np_matrix(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float32)

    def to_causal_prior(self) -> CausalPrior:
        return CausalPrior(
            matrix=self.matrix,
            channel_to_key=self.channels,
            lag=self.lag,
            sparse_threshold=self.sparse_threshold,
            method=self.causal_relation_source,
            meta={"resolved_config": self.config, **(self.meta or {})},
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


_PREFIX_BY_LEVEL = {"device": "d", "action": "a", "device_action": "da"}


def resolve_causal_relation_prior(
    prior_json: str | None = None,
    prior_matrix_path: str | None = None,
    source_pkl: str | None = None,
    out_dir: str | None = None,
    adapter_mode: str = "existing",
    level: str = "device",
    lag: int = 4,
    sparse_threshold: float = 0.001,
    seed: int = 2024,
) -> ResolvedCausalRelationPrior:
    """Resolve a causal relation prior from an existing artifact or adapter call.

    Priority is deliberately simple: existing prior JSON, then existing matrix,
    then the current project's causal relation adapter over source_pkl.
    No new causal discovery algorithm is implemented here.
    """

    config = {
        "prior_json": str(prior_json) if prior_json else None,
        "prior_matrix_path": str(prior_matrix_path) if prior_matrix_path else None,
        "source_pkl": str(source_pkl) if source_pkl else None,
        "out_dir": str(out_dir) if out_dir else None,
        "adapter_mode": adapter_mode,
        "level": level,
        "lag": lag,
        "sparse_threshold": sparse_threshold,
        "seed": seed,
    }

    if level not in {"device", "action", "device_action"}:
        raise ValueError("level must be one of: device, action, device_action")
    if adapter_mode not in {"existing", "compact_fallback"}:
        raise ValueError("adapter_mode must be 'existing' or 'compact_fallback' in this integration layer")

    if prior_json:
        resolved = _resolved_from_prior_json(Path(prior_json), level=level, fallback_lag=lag, fallback_threshold=sparse_threshold, config=config)
        _maybe_save(resolved, out_dir)
        return resolved

    if prior_matrix_path:
        resolved = _resolved_from_matrix_path(
            Path(prior_matrix_path),
            level=level,
            lag=lag,
            sparse_threshold=sparse_threshold,
            config=config,
        )
        _maybe_save(resolved, out_dir)
        return resolved

    if source_pkl:
        if adapter_mode == "compact_fallback":
            resolved = _resolved_from_transition_fallback(
                source_pkl=Path(source_pkl),
                level=level,
                lag=lag,
                sparse_threshold=sparse_threshold,
                config=config,
            )
        else:
            resolved = _resolved_from_existing_adapter(
                source_pkl=Path(source_pkl),
                out_dir=Path(out_dir) if out_dir else None,
                level=level,
                lag=lag,
                sparse_threshold=sparse_threshold,
                seed=seed,
                config=config,
            )
        _maybe_save(resolved, out_dir)
        return resolved

    raise ValueError("resolve_causal_relation_prior requires prior_json, prior_matrix_path, or source_pkl; no prior source was provided")


def _resolved_from_prior_json(
    path: Path,
    level: str,
    fallback_lag: int,
    fallback_threshold: float,
    config: dict[str, Any],
) -> ResolvedCausalRelationPrior:
    if not path.exists():
        raise FileNotFoundError(f"prior_json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "matrix" not in payload:
        raise ValueError(f"prior_json does not contain a matrix field: {path}")

    channels = list(payload.get("channels") or payload.get("channel_to_key") or [])
    matrix = _matrix_to_list(payload["matrix"])
    if not channels:
        channels = _default_channels(len(matrix), level)

    lag = int(payload.get("lag", fallback_lag))
    threshold = float(payload.get("sparse_threshold", fallback_threshold))
    source = str(
        payload.get("causal_relation_source")
        or payload.get("method")
        or payload.get("meta", {}).get("causal_relation_source")
        or "existing_prior_json"
    )
    top_edges = payload.get("top_causal_edges")
    if top_edges is None:
        top_edges = _top_edges_from_matrix(matrix, channels, lag)
    else:
        top_edges = _standardize_edges(top_edges, channels=channels, lag=lag)

    return ResolvedCausalRelationPrior(
        causal_relation_source=source,
        level=str(payload.get("level") or level),
        lag=lag,
        sparse_threshold=threshold,
        channels=channels,
        matrix=matrix,
        top_causal_edges=top_edges,
        config={**config, "resolved_from": str(path)},
        meta={"input_format": "prior_json", "original_keys": sorted(str(k) for k in payload.keys())},
    )


def _resolved_from_matrix_path(
    path: Path,
    level: str,
    lag: int,
    sparse_threshold: float,
    config: dict[str, Any],
) -> ResolvedCausalRelationPrior:
    if not path.exists():
        raise FileNotFoundError(f"prior_matrix_path not found: {path}")
    channels: list[str] | None = None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            matrix_obj = payload.get("matrix") or payload.get("causal_matrix")
            if matrix_obj is None:
                raise ValueError(f"matrix JSON must contain matrix or causal_matrix: {path}")
            channels = list(payload.get("channels") or payload.get("channel_to_key") or []) or None
            matrix = _matrix_to_list(matrix_obj)
            lag = int(payload.get("lag", lag))
            sparse_threshold = float(payload.get("sparse_threshold", sparse_threshold))
        else:
            matrix = _matrix_to_list(payload)
    elif path.suffix.lower() == ".npy":
        matrix = _matrix_to_list(np.load(path))
    elif path.suffix.lower() in {".csv", ".txt"}:
        matrix = _read_csv_matrix(path)
    else:
        raise ValueError("prior_matrix_path must be .json, .npy, .csv or .txt")
    channels = channels or _default_channels(len(matrix), level)
    return ResolvedCausalRelationPrior(
        causal_relation_source="prior_matrix_path",
        level=level,
        lag=lag,
        sparse_threshold=sparse_threshold,
        channels=channels,
        matrix=matrix,
        top_causal_edges=_top_edges_from_matrix(matrix, channels, lag),
        config={**config, "resolved_from": str(path)},
        meta={"input_format": "prior_matrix_path"},
    )


def _resolved_from_existing_adapter(
    source_pkl: Path,
    out_dir: Path | None,
    level: str,
    lag: int,
    sparse_threshold: float,
    seed: int,
    config: dict[str, Any],
) -> ResolvedCausalRelationPrior:
    if not source_pkl.exists():
        raise FileNotFoundError(f"source_pkl not found: {source_pkl}")
    sequences = _load_behavior_sequences_from_pickle(source_pkl)
    if not sequences:
        raise ValueError(f"source_pkl contains no valid flattened behavior sequences: {source_pkl}")

    tensorized = EventTensorizer(level=level, count_mode="binary", decay=0.1).fit_transform(sequences)
    if tensorized.tensor.shape[1] == 0:
        raise ValueError(f"source_pkl yielded no {level} channels: {source_pkl}")

    adapter = CausalRelationAdapter()
    prior = adapter.mine_event_prior(
        tensorized.tensor,
        tensorized.channel_to_key,
        lag=lag,
        epochs=int(config.get("epochs") or 2),
        hidden=int(config.get("hidden") or 16),
        sparse_threshold=sparse_threshold,
        batch_size=int(config.get("batch_size") or 64),
        sample_limit=config.get("sample_limit"),
    )
    source = "existing_adapter_compact_fallback"
    matrix = _matrix_to_list(prior.matrix)
    channels = list(prior.channel_to_key)
    resolved = ResolvedCausalRelationPrior(
        causal_relation_source=source,
        level=level,
        lag=prior.lag,
        sparse_threshold=prior.sparse_threshold,
        channels=channels,
        matrix=matrix,
        top_causal_edges=_standardize_edges(prior.top_edges(k=50, include_self=False), channels=channels, lag=prior.lag),
        config={**config, "resolved_from": str(source_pkl), "causal_relation_source": source},
        meta={
            "input_format": "source_pkl_via_existing_adapter",
            "sequence_count": len(sequences),
            "tensor_shape": list(tensorized.tensor.shape),
            "adapter_note": "CausalRelationAdapter.mine_event_prior calls the compact causal-relation-style fallback for event tensors.",
            **(prior.meta or {}),
        },
    )
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        prior.save(out_dir / "causal_prior_adapter_raw.json")
    return resolved


def _resolved_from_transition_fallback(
    source_pkl: Path,
    level: str,
    lag: int,
    sparse_threshold: float,
    config: dict[str, Any],
) -> ResolvedCausalRelationPrior:
    if not source_pkl.exists():
        raise FileNotFoundError(f"source_pkl not found: {source_pkl}")
    sequences = _load_behavior_sequences_from_pickle(source_pkl)
    if not sequences:
        raise ValueError(f"source_pkl contains no valid flattened behavior sequences: {source_pkl}")

    channels = sorted({event.key(level) for seq in sequences for event in seq.events})
    index = {channel: i for i, channel in enumerate(channels)}
    matrix = np.zeros((len(channels), len(channels)), dtype=np.float32)
    for seq in sequences:
        keys = [event.key(level) for event in seq.events]
        for i, source_key in enumerate(keys):
            for offset in range(1, max(1, int(lag)) + 1):
                j = i + offset
                if j >= len(keys):
                    break
                target_key = keys[j]
                if source_key == target_key:
                    continue
                matrix[index[source_key], index[target_key]] += 1.0 / float(offset)
    if matrix.size and float(matrix.max()) > 0.0:
        matrix = matrix / float(matrix.max())
    matrix[matrix < sparse_threshold] = 0.0
    matrix_list = _matrix_to_list(matrix)
    return ResolvedCausalRelationPrior(
        causal_relation_source="transition_count_compact_fallback",
        level=level,
        lag=lag,
        sparse_threshold=sparse_threshold,
        channels=channels,
        matrix=matrix_list,
        top_causal_edges=_top_edges_from_matrix(matrix_list, channels, lag),
        config={**config, "resolved_from": str(source_pkl), "causal_relation_source": "transition_count_compact_fallback"},
        meta={
            "input_format": "source_pkl_transition_count_fallback",
            "sequence_count": len(sequences),
            "note": "Lightweight non-torch fallback for GPT package prompt construction; not a downstream result.",
        },
    )


def _load_behavior_sequences_from_pickle(path: Path) -> list[BehaviorSequence]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    cleaned: list[Sequence[int]] = []
    for item in list(raw):
        candidate = _extract_flat_sequence(item)
        if candidate is not None:
            cleaned.append(candidate)
    return load_numeric_sequences(cleaned)


def _extract_flat_sequence(item: Any) -> list[int] | None:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, tuple):
        for part in item:
            candidate = _extract_flat_sequence(part)
            if candidate is not None:
                return candidate
        return None
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) > 0 and len(item) % 4 == 0:
            try:
                return [int(x) for x in item]
            except Exception:
                return None
    return None


def _matrix_to_list(matrix: Any) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"causal relation prior matrix must be square 2-D, got shape {arr.shape}")
    return [[float(v) for v in row] for row in arr.tolist()]


def _read_csv_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append([float(value) for value in row])
    return _matrix_to_list(rows)


def _default_channels(n: int, level: str) -> list[str]:
    prefix = _PREFIX_BY_LEVEL.get(level, "c")
    return [f"{prefix}:{i}" for i in range(n)]


def _top_edges_from_matrix(matrix: Sequence[Sequence[float]], channels: Sequence[str], lag: int, k: int = 50) -> list[dict[str, Any]]:
    edges: list[tuple[float, int, int]] = []
    arr = np.asarray(matrix, dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i == j:
                continue
            weight = float(arr[i, j])
            if weight > 0:
                edges.append((weight, i, j))
    edges.sort(key=lambda x: x[0], reverse=True)
    return [
        _edge_from_indices(i, j, weight, channels=channels, lag=lag)
        for weight, i, j in edges[:k]
    ]


def _standardize_edges(edges: Iterable[Mapping[str, Any]], channels: Sequence[str], lag: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for edge in edges:
        source = str(edge.get("source") or edge.get("source_key") or "")
        target = str(edge.get("target") or edge.get("target_key") or "")
        source_id = edge.get("source_id", edge.get("source_index"))
        target_id = edge.get("target_id", edge.get("target_index"))
        if not source and source_id is not None:
            source = _channel_by_index(channels, int(source_id))
        if not target and target_id is not None:
            target = _channel_by_index(channels, int(target_id))
        if source_id is None:
            source_id = _index_by_channel(channels, source)
        if target_id is None:
            target_id = _index_by_channel(channels, target)
        weight = float(edge.get("weight", edge.get("raw_weight", edge.get("guarded_weight", 0.0))))
        row = dict(edge)
        row.update(
            {
                "source": source,
                "target": target,
                "source_id": int(source_id) if source_id is not None else None,
                "target_id": int(target_id) if target_id is not None else None,
                "source_index": int(source_id) if source_id is not None else None,
                "target_index": int(target_id) if target_id is not None else None,
                "weight": weight,
                "lag": int(edge.get("lag", lag)),
            }
        )
        out.append(row)
    out.sort(key=lambda item: float(item.get("weight", 0.0)), reverse=True)
    return out


def _edge_from_indices(i: int, j: int, weight: float, channels: Sequence[str], lag: int) -> dict[str, Any]:
    return {
        "source": _channel_by_index(channels, i),
        "target": _channel_by_index(channels, j),
        "source_id": int(i),
        "target_id": int(j),
        "source_index": int(i),
        "target_index": int(j),
        "weight": float(weight),
        "lag": int(lag),
    }


def _channel_by_index(channels: Sequence[str], index: int) -> str:
    if 0 <= index < len(channels):
        return str(channels[index])
    return f"c:{index}"


def _index_by_channel(channels: Sequence[str], channel: str) -> int | None:
    try:
        return list(channels).index(channel)
    except ValueError:
        return None


def _maybe_save(resolved: ResolvedCausalRelationPrior, out_dir: str | None) -> None:
    if out_dir:
        resolved.save(Path(out_dir) / "resolved_causal_relation_prior.json")
