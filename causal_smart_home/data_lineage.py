"""Runtime data-lineage recording and zero-target access gates.

The recorder uses Python's audit-hook ``open`` events, so entries describe
files actually opened by the process rather than merely declared CLI paths.
It is activated for subprocesses through ``CSH_LINEAGE_*`` environment values.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


DATA_SUFFIXES = {".pkl", ".json", ".jsonl", ".npy", ".npz", ".csv", ".pth", ".txt", ".md", ".sha256"}
EVALUATION_PURPOSES = {"offline_generation_quality_evaluation", "downstream_ad_evaluation"}


class TargetDataLeakageError(RuntimeError):
    """Raised immediately when a protected stage opens target behavior data."""


class DataLineageRecorder:
    def __init__(
        self,
        *,
        out_path: str | Path,
        stage: str,
        variant: str,
        dataset: str,
        scenario: str,
        seed: int,
        purpose: str,
        target_normal_files: Iterable[str | Path] = (),
        target_attack_files: Iterable[str | Path] = (),
        allow_target_normal: bool = False,
        allow_target_attack: bool = False,
    ) -> None:
        self.out_path = _normalize(out_path)
        self.stage = stage
        self.variant = variant
        self.dataset = dataset
        self.scenario = scenario
        self.seed = int(seed)
        self.purpose = purpose
        self.target_normal_files = {_normalize(path) for path in target_normal_files}
        self.target_attack_files = {_normalize(path) for path in target_attack_files}
        self.allow_target_normal = bool(allow_target_normal)
        self.allow_target_attack = bool(allow_target_attack)
        self.events: list[dict[str, Any]] = []
        self.active = False
        self._written = False

    def install(self) -> "DataLineageRecorder":
        if self.active:
            return self
        self.active = True
        sys.addaudithook(self._audit_hook)
        atexit.register(self.write)
        return self

    def _audit_hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not self.active or event != "open" or not args:
            return
        raw_path = args[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        path = _normalize(os.fsdecode(raw_path))
        if Path(path).suffix.lower() not in DATA_SUFFIXES:
            return
        mode = args[1] if len(args) > 1 else "r"
        if isinstance(mode, str):
            mode_text = mode
            operation = "write" if any(flag in mode_text for flag in ("w", "a", "x", "+")) else "read"
        else:
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            operation = "write" if flags & write_flags else "read"
            mode_text = f"os_flags={flags}"
        target_normal = path in self.target_normal_files
        target_attack = path in self.target_attack_files
        row = {
            "event_index": len(self.events),
            "operation": operation,
            "path": path,
            "mode": mode_text,
            "target_normal": target_normal,
            "target_attack": target_attack,
        }
        self.events.append(row)
        if operation == "read" and target_normal and not self.allow_target_normal:
            self.write(status="blocked_target_normal_access", blocked_event=row)
            raise TargetDataLeakageError(
                f"zero-target gate blocked target normal read in stage={self.stage}: {path}"
            )
        if operation == "read" and target_attack and not self.allow_target_attack:
            self.write(status="blocked_target_attack_access", blocked_event=row)
            raise TargetDataLeakageError(
                f"pre-evaluation gate blocked target attack read in stage={self.stage}: {path}"
            )

    def payload(self, *, status: str = "success", blocked_event: dict[str, Any] | None = None) -> dict[str, Any]:
        consumed = _ordered_unique(row["path"] for row in self.events if row["operation"] == "read")
        produced = _ordered_unique(row["path"] for row in self.events if row["operation"] == "write")
        target_normal = _ordered_unique(
            row["path"] for row in self.events if row["operation"] == "read" and row["target_normal"]
        )
        target_attack = _ordered_unique(
            row["path"] for row in self.events if row["operation"] == "read" and row["target_attack"]
        )
        if self.out_path not in produced:
            produced.append(self.out_path)
        return {
            "schema_version": 1,
            "stage": self.stage,
            "variant": self.variant,
            "dataset": self.dataset,
            "scenario": self.scenario,
            "seed": self.seed,
            "consumed_files": consumed,
            "produced_files": produced,
            "target_normal_consumed": bool(target_normal),
            "target_normal_files": target_normal,
            "target_attack_consumed": bool(target_attack),
            "target_attack_files": target_attack,
            "purpose": self.purpose,
            "target_normal_allowed": self.allow_target_normal,
            "target_attack_allowed": self.allow_target_attack,
            "status": status,
            "blocked_event": blocked_event,
            "open_events": self.events,
        }

    def write(self, *, status: str = "success", blocked_event: dict[str, Any] | None = None) -> None:
        if self._written and status == "success":
            return
        self.active = False
        payload = self.payload(status=status, blocked_event=blocked_event)
        output = Path(self.out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            try:
                previous = json.loads(output.read_text(encoding="utf-8"))
            except Exception:
                previous = None
            if isinstance(previous, dict) and previous.get("stage") == self.stage:
                payload = merge_lineage_payloads([previous, payload], stage=self.stage, variant=self.variant)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._written = True


def merge_lineage_payloads(
    payloads: Iterable[dict[str, Any]], *, stage: str, variant: str
) -> dict[str, Any]:
    rows = list(payloads)
    if not rows:
        raise ValueError("at least one lineage payload is required")
    first = rows[0]
    events = []
    for row in rows:
        for event in row.get("open_events", []):
            copied = dict(event)
            copied["event_index"] = len(events)
            events.append(copied)
    consumed = _ordered_unique(path for row in rows for path in row.get("consumed_files", []))
    produced = _ordered_unique(path for row in rows for path in row.get("produced_files", []))
    target_normal = _ordered_unique(path for row in rows for path in row.get("target_normal_files", []))
    target_attack = _ordered_unique(path for row in rows for path in row.get("target_attack_files", []))
    return {
        "schema_version": 1,
        "stage": stage,
        "variant": variant,
        "dataset": first.get("dataset"),
        "scenario": first.get("scenario"),
        "seed": first.get("seed"),
        "consumed_files": consumed,
        "produced_files": produced,
        "target_normal_consumed": bool(target_normal),
        "target_normal_files": target_normal,
        "target_attack_consumed": bool(target_attack),
        "target_attack_files": target_attack,
        "purpose": first.get("purpose"),
        "target_normal_allowed": all(bool(row.get("target_normal_allowed")) for row in rows),
        "target_attack_allowed": all(bool(row.get("target_attack_allowed")) for row in rows),
        "status": "success" if all(row.get("status") == "success" for row in rows) else "mixed_or_blocked",
        "open_events": events,
    }


def install_from_environment() -> DataLineageRecorder | None:
    out = os.environ.get("CSH_LINEAGE_OUT")
    if not out:
        return None
    recorder = DataLineageRecorder(
        out_path=out,
        stage=os.environ.get("CSH_LINEAGE_STAGE", "unknown"),
        variant=os.environ.get("CSH_LINEAGE_VARIANT", "unknown"),
        dataset=os.environ.get("CSH_LINEAGE_DATASET", "unknown"),
        scenario=os.environ.get("CSH_LINEAGE_SCENARIO", "unknown"),
        seed=int(os.environ.get("CSH_LINEAGE_SEED", "0")),
        purpose=os.environ.get("CSH_LINEAGE_PURPOSE", "unknown"),
        target_normal_files=_json_env_list("CSH_TARGET_NORMAL_FILES"),
        target_attack_files=_json_env_list("CSH_TARGET_ATTACK_FILES"),
        allow_target_normal=_bool_env("CSH_ALLOW_TARGET_NORMAL"),
        allow_target_attack=_bool_env("CSH_ALLOW_TARGET_ATTACK"),
    )
    return recorder.install()


def _json_env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON list") from exc
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return [str(item) for item in value]


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _normalize(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
