#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_gss import (
    build_causal_hints_payload,
    format_causal_hints_for_prompt,
    learn_device_gcad_prior,
    load_id_name_mapping,
    load_pickle_sequences,
    map_device_edges,
    render_edges_markdown,
)
from causal_smart_home.causal_prompt_adapter import enhance_prompt_with_causal_hints, render_prompt_diff
from causal_smart_home.schema import BehaviorSequence


CSH_ROOT = Path("/home/heyang/projects/CausalSmartHome")
SMARTGEN_ROOT = Path("/home/heyang/projects/SmartGen/SmartGen")
OUT_ROOT = CSH_ROOT / "outputs/gcad_gss"
DEFAULT_SOURCE_TRAIN = Path("/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/daytime/trn.pkl")
DEFAULT_EVAL_SOURCE_TRAIN = Path("/home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/sp/daytime/trn.pkl")
DEFAULT_TARGET_REAL = Path("/home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl")
DEFAULT_ORIGINAL_RAW = SMARTGEN_ROOT / "IoT_data/sp/spring/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq.pkl"
DEFAULT_ORIGINAL_TOF = SMARTGEN_ROOT / "filter_data/sp/spring/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl"


def output_root(args: argparse.Namespace) -> Path:
    tag = getattr(args, "output_tag", "")
    return OUT_ROOT / tag if tag else OUT_ROOT


def tag_suffix(args: argparse.Namespace) -> str:
    tag = getattr(args, "output_tag", "")
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)
    return f"_{safe}"


@dataclass(frozen=True)
class GroupPaths:
    name: str
    out_dir: Path
    work_dir: Path
    model_tag: str

    @property
    def raw_pkl(self) -> Path:
        return self.work_dir / f"IoT_data/sp/spring/sp_spring_generation_SPPC_th=0.915_{self.model_tag}_seq.pkl"

    @property
    def tof_pkl(self) -> Path:
        return self.work_dir / f"filter_data/sp/spring/sp_spring_generation_SPPC_th=0.915_{self.model_tag}_seq_filter_true.pkl"


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def load_smartgen_dictionary(smartgen_root: Path):
    spec = importlib.util.spec_from_file_location("stage3a_smartgen_dictionary", smartgen_root / "dictionary.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SmartGen dictionary from {smartgen_root / 'dictionary.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentence_for_sp_st() -> str:
    return "The previous environment is daytime. The changed environment is warm spring."


def discover_categories(smartgen_root: Path, threshold: str, method: str) -> list[str]:
    base = smartgen_root / "IoT_data/sp/daytime"
    categories: list[str] = []
    for path in base.glob(f"trn_day_*_{method}_th={threshold}_text.pkl"):
        stem = path.name
        prefix = "trn_day_"
        suffix = f"_{method}_th={threshold}_text.pkl"
        categories.append(stem[len(prefix) : -len(suffix)])

    def sort_key(value: str) -> tuple[int, int]:
        parts = value.split("_")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else -1)

    return sorted(set(categories), key=sort_key)


def build_smartgen_prompt(
    user_sequence: Any,
    action_transition: Any,
    device_control_dict: str,
) -> str:
    sentence = sentence_for_sp_st()
    return (
        "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. "
        "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences."
        "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices."
        f"The set of the possible device and device states: {device_control_dict}"
        f"{sentence} The user's compressed original sequences of behavior: {user_sequence}. User's behavior habits: {action_transition}"
        "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. "
        "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits."
        "Requirements:"
        "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices."
        "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device."
        "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment."
        "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors."
        "5.Please also generate reasonable behavior time when generating, not just a single behavior."
        "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>"
        "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats."
        "Please think step by step, and return the final generated user behavior sequence set."
    )


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def load_numeric_behavior_sequences(path: str | Path) -> list[BehaviorSequence]:
    raw = load_pickle(path)
    sequences: list[BehaviorSequence] = []
    for index, item in enumerate(raw):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, tuple):
            for part in item:
                if hasattr(part, "tolist"):
                    part = part.tolist()
                if isinstance(part, list) and all(isinstance(value, (int, float)) for value in part):
                    item = part
                    break
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            continue
        if len(item) == 0 or len(item) % 4 != 0:
            continue
        try:
            flat = [int(value) for value in item]
        except Exception:
            continue
        sequences.append(BehaviorSequence.from_flat_numeric(flat, sequence_id=str(index)))
    return sequences


def device_frequencies(path: str | Path) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for seq in load_numeric_behavior_sequences(path):
        for event in seq:
            counts[event.key("device")] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def apply_target_overrepresented_edge_guard(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.edge_guard == "none":
        return payload, {"mode": "none", "removed_edges": []}
    if args.edge_guard not in {"target-overrepresented", "target-overrepresented-downweight"}:
        raise ValueError(f"unknown edge guard: {args.edge_guard}")
    reference_path = args.edge_guard_reference_tof_pkl
    if reference_path is None:
        raise ValueError("--edge-guard-reference-tof-pkl is required for target-overrepresented guards")
    reference_freq = device_frequencies(reference_path)
    target_freq = device_frequencies(args.edge_guard_target_real_pkl)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    downweighted: list[dict[str, Any]] = []
    for edge in payload.get("top_causal_edges", []):
        target_key = str(edge.get("target"))
        source_key = str(edge.get("source"))
        target_over = reference_freq.get(target_key, 0.0) > target_freq.get(target_key, 0.0) + args.edge_guard_margin
        source_over = reference_freq.get(source_key, 0.0) > target_freq.get(source_key, 0.0) + args.edge_guard_margin
        should_guard = target_over or (args.edge_guard_endpoint_scope == "source,target" and source_over)
        edge_row = dict(edge)
        edge_row.update(
            {
                "source_reference_freq": reference_freq.get(source_key, 0.0),
                "source_target_freq": target_freq.get(source_key, 0.0),
                "target_reference_freq": reference_freq.get(target_key, 0.0),
                "target_target_freq": target_freq.get(target_key, 0.0),
                "source_overrepresented": source_over,
                "target_overrepresented": target_over,
                "removed_reason": "endpoint_overrepresented" if should_guard and args.edge_guard == "target-overrepresented" else "",
                "downweight_factor": args.edge_guard_downweight_factor if should_guard else 1.0,
            }
        )
        if should_guard and args.edge_guard == "target-overrepresented":
            removed.append(edge_row)
        else:
            kept_edge = dict(edge)
            kept_edge["original_weight"] = float(edge.get("weight", 0.0))
            kept_edge["edge_guarded"] = bool(should_guard)
            kept_edge["edge_guard_downweight_factor"] = args.edge_guard_downweight_factor if should_guard else 1.0
            if should_guard and args.edge_guard == "target-overrepresented-downweight":
                kept_edge["weight"] = float(edge.get("weight", 0.0)) * args.edge_guard_downweight_factor
                downweighted.append(edge_row)
            kept.append(kept_edge)
    if args.edge_guard == "target-overrepresented-downweight":
        kept = sorted(kept, key=lambda edge: float(edge.get("weight", 0.0)), reverse=True)
    guarded = dict(payload)
    guarded["top_causal_edges"] = kept
    guarded["edge_guard"] = {
        "mode": args.edge_guard,
        "endpoint_scope": args.edge_guard_endpoint_scope,
        "margin": args.edge_guard_margin,
        "downweight_factor": args.edge_guard_downweight_factor,
        "reference_tof_pkl": str(reference_path),
        "target_real_pkl": str(args.edge_guard_target_real_pkl),
        "original_edge_count": len(payload.get("top_causal_edges", [])),
        "kept_edge_count": len(kept),
        "removed_edge_count": len(removed),
        "downweighted_edge_count": len(downweighted),
    }
    return guarded, {**guarded["edge_guard"], "removed_edges": removed, "downweighted_edges": downweighted}


def prepare_prompt_checks(args: argparse.Namespace) -> dict[str, Any]:
    run_root = output_root(args)
    out_dir = run_root / "sp_st_prompt_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = discover_categories(args.smartgen_root, args.threshold, args.method)
    if not categories:
        raise FileNotFoundError("no SP-ST SmartGen prompt categories found")

    sequences = load_pickle_sequences(args.source_train_pkl)
    prior = learn_device_gcad_prior(
        sequences,
        lag=args.lag,
        epochs=args.epochs,
        sparse_threshold=args.sparse_threshold,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
    )
    device_mapping = load_id_name_mapping(args.smartgen_root / "dictionary.py", preferred_names=("sp_devices_dict", "device_dict"))
    edges20 = map_device_edges(prior, device_mapping, top_k_edges=20)
    payload20 = build_causal_hints_payload(edges20, prior, source_train_pkl=args.source_train_pkl, level="device")
    payload20["scenario"] = "sp_st"
    hints20 = format_causal_hints_for_prompt(payload20)

    edges10 = map_device_edges(prior, device_mapping, top_k_edges=10)
    payload10 = build_causal_hints_payload(edges10, prior, source_train_pkl=args.source_train_pkl, level="device")
    payload10["scenario"] = "sp_st"
    hints10 = format_causal_hints_for_prompt(payload10)

    action_transition = json.loads((args.smartgen_root / "IoT_data/sp/daytime/action_transitions.json").read_text(encoding="utf-8"))
    device_control_dict = (args.smartgen_root / "sp_keys_best.txt").read_text(encoding="utf-8")
    sample_category = categories[0]
    user_sequence = load_pickle(args.smartgen_root / f"IoT_data/sp/daytime/trn_day_{sample_category}_{args.method}_th={args.threshold}_text.pkl")
    original_prompt = build_smartgen_prompt(user_sequence, action_transition, device_control_dict)
    enhanced20 = enhance_prompt_with_causal_hints(original_prompt, hints20)
    enhanced10 = enhance_prompt_with_causal_hints(original_prompt, hints10)

    (out_dir / "sample_original_prompt.txt").write_text(original_prompt, encoding="utf-8")
    (out_dir / "enhanced_prompt_top20.txt").write_text(enhanced20.enhanced_prompt, encoding="utf-8")
    (out_dir / "enhanced_prompt_top10.txt").write_text(enhanced10.enhanced_prompt, encoding="utf-8")
    (out_dir / "prompt_diff.md").write_text(render_prompt_diff(original_prompt, enhanced20.enhanced_prompt), encoding="utf-8")
    (out_dir / "prompt_diff_top10.md").write_text(render_prompt_diff(original_prompt, enhanced10.enhanced_prompt), encoding="utf-8")
    (out_dir / "top_causal_edges.md").write_text(render_edges_markdown(edges20, payload20), encoding="utf-8")
    (out_dir / "top_causal_edges_top10.md").write_text(render_edges_markdown(edges10, payload10), encoding="utf-8")
    write_json(out_dir / "causal_hints.json", payload20)
    write_json(out_dir / "causal_hints_top10.json", payload10)

    prompt_check = {
        "sample_category": sample_category,
        "num_categories": len(categories),
        "top20_edges": len(edges20),
        "top10_edges": len(edges10),
        "causal_hints_are_soft": payload20.get("constraint_strength") == "soft" and "soft" in hints20.lower(),
        "original_gss_hints_retained": "User's behavior habits:" in enhanced20.enhanced_prompt and str(action_transition) in enhanced20.enhanced_prompt,
        "prompt_diff_path": out_dir / "prompt_diff.md",
        "top20_prompt_chars": len(enhanced20.enhanced_prompt),
        "top20_prompt_est_tokens": estimate_tokens(enhanced20.enhanced_prompt),
        "top10_prompt_chars": len(enhanced10.enhanced_prompt),
        "top10_prompt_est_tokens": estimate_tokens(enhanced10.enhanced_prompt),
        "top10_recommended": estimate_tokens(enhanced20.enhanced_prompt) > args.prompt_token_warning,
        "paths": {
            "original_sample_prompt": out_dir / "sample_original_prompt.txt",
            "enhanced_top20": out_dir / "enhanced_prompt_top20.txt",
            "enhanced_top10": out_dir / "enhanced_prompt_top10.txt",
            "causal_hints": out_dir / "causal_hints.json",
            "top_causal_edges": out_dir / "top_causal_edges.md",
        },
    }
    write_json(out_dir / "prompt_check.json", prompt_check)
    return prompt_check


def import_smartgen_function(smartgen_root: Path, module_name: str, function_name: str):
    root = str(smartgen_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(f"stage3a_{module_name}", smartgen_root / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {smartgen_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def prepare_group_workdir(paths: GroupPaths) -> None:
    for relative in ("IoT_data/sp/spring", "filter_data/sp/spring", "check_model"):
        (paths.work_dir / relative).mkdir(parents=True, exist_ok=True)


def call_llm(prompt: str, args: argparse.Namespace) -> str:
    api_key = args.api_key or os.getenv(args.api_key_env)
    base_url = args.base_url or (os.getenv(args.base_url_env) if args.base_url_env else None)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set; cannot call LLM generation")
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    body = json.dumps(
        {
            "model": args.llm_model,
            "stream": False,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM network error: {exc}") from exc
    return payload["choices"][0]["message"]["content"].strip()


def codex_offline_response(
    category: str,
    prompt_kind: str,
    causal_edges: Sequence[dict[str, Any]] | None = None,
    seed: int = 2024,
) -> str:
    """Deterministic local generation used when the assistant is the generator.

    This is intentionally simple and traceable: both original and enhanced arms
    use the same generator and differ only by whether GCAD-GSS prompt guidance is
    active. Outputs stay in SmartGen's textual <seq ... seq> convention so the
    original Extract/Transnum/TOF functions can process them unchanged.
    """
    day_id = int(str(category).split("_")[0])
    day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_id % 7]
    offsets = {
        "0": "(0~3)",
        "1": "(3~6)",
        "2": "(6~9)",
        "3": "(9~12)",
        "4": "(12~15)",
        "5": "(15~18)",
        "6": "(18~21)",
        "7": "(21~24)",
    }

    def event(hour_key: str, device: str, action: str) -> list[str]:
        return [day, offsets[hour_key], device, action]

    def flat(events: Sequence[list[str]]) -> list[str]:
        out: list[str] = []
        for item in events:
            out.extend(item)
        return out

    original_templates = [
        [
            event("2", "Other", "None:location"),
            event("3", "Television", "Television:setChannel"),
            event("3", "Television", "Television:volumeDown"),
        ],
        [
            event("0", "Blind", "Blind:windowShade open"),
            event("0", "RobotCleaner", "RobotCleaner:setRobotCleanerMovement cleaning"),
            event("0", "Camera", "Camera:notification"),
            event("0", "Blind", "Blind:windowShade close"),
        ],
        [
            event("4", "AirPurifier", "AirPurifier:setFanSpeed"),
            event("4", "Camera", "Camera:notification"),
            event("5", "Blind", "Blind:windowShade close"),
        ],
        [
            event("6", "Television", "Television:audioMute unmute"),
            event("6", "Television", "Television:switch on"),
            event("7", "Television", "Television:setChannel"),
        ],
    ]
    variant = (seed + day_id + sum(ord(ch) for ch in str(category))) % 3
    if variant == 1:
        original_templates[0] = [
            event("2", "Other", "None:location"),
            event("3", "Television", "Television:setChannel"),
            event("4", "AirPurifier", "AirPurifier:setFanSpeed"),
        ]
        original_templates[2] = [
            event("4", "Light", "Light:switch on"),
            event("5", "Camera", "Camera:notification"),
            event("5", "Blind", "Blind:windowShade close"),
        ]
    elif variant == 2:
        original_templates[1] = [
            event("0", "Blind", "Blind:windowShade open"),
            event("1", "RobotCleaner", "RobotCleaner:setRobotCleanerMovement cleaning"),
            event("1", "Camera", "Camera:notification"),
            event("1", "Blind", "Blind:windowShade close"),
        ]
        original_templates[3] = [
            event("6", "NetworkAudio", "NetworkAudio:mediaPlayback play"),
            event("6", "Television", "Television:switch on"),
            event("7", "Television", "Television:volumeDown"),
        ]
    action_for_device = {
        "AirPurifier": "AirPurifier:setFanSpeed",
        "Blind": "Blind:windowShade open",
        "Camera": "Camera:notification",
        "GarageDoor": "GarageDoor:doorControl open",
        "Heater": "Heater:setHeatingSetpoint",
        "Light": "Light:switch on",
        "NetworkAudio": "NetworkAudio:mediaPlayback play",
        "Other": "None:location",
        "Projector": "Projector:samsungvd.mediaInputSource setInputSource",
        "RobotCleaner": "RobotCleaner:setRobotCleanerMovement cleaning",
        "SmartLock": "SmartLock:lock unlock",
        "SmartPlug": "SmartPlug:switch on",
        "Television": "Television:setChannel",
    }

    def edge_template(edge: dict[str, Any], offset: int) -> list[list[str]]:
        source = str(edge.get("source_name") or edge.get("source") or "Television")
        target = str(edge.get("target_name") or edge.get("target") or "Other")
        source_action = action_for_device.get(source, f"{source}:switch on")
        target_action = action_for_device.get(target, "None:location" if target == "Other" else f"{target}:switch on")
        hour_a = str((2 + offset) % 8)
        hour_b = str((3 + offset) % 8)
        return [
            event(hour_a, source, source_action),
            event(hour_b, target, target_action),
        ]

    if causal_edges:
        enhanced_templates = [list(template) for template in original_templates]
        num_edge_sequences = 1 if len(causal_edges) <= 1 else 2 if len(causal_edges) <= 3 else 3
        for idx, edge in enumerate(causal_edges[:num_edge_sequences]):
            replace_idx = (idx + seed) % len(enhanced_templates)
            events = edge_template(edge, idx + variant)
            # Soft injection: keep ordinary SmartGen-style context around the
            # causal order instead of turning the hint into a hard rule.
            if idx % 2 == 0:
                events.append(event(str((4 + idx) % 8), "Television", "Television:volumeDown"))
            else:
                events.insert(0, event(str((1 + idx) % 8), "Camera", "Camera:notification"))
            enhanced_templates[replace_idx] = events
    else:
        enhanced_templates = [
            [
                event("2", "Television", "Television:setChannel"),
                event("2", "Other", "None:location"),
                event("3", "Television", "Television:volumeDown"),
            ],
            [
                event("0", "Blind", "Blind:windowShade open"),
                event("0", "Camera", "Camera:notification"),
                event("0", "Blind", "Blind:windowShade close"),
                event("1", "RobotCleaner", "RobotCleaner:setRobotCleanerMovement cleaning"),
            ],
            [
                event("4", "AirPurifier", "AirPurifier:setFanSpeed"),
                event("4", "Camera", "Camera:notification"),
                event("5", "Television", "Television:setInputSource"),
                event("5", "Other", "None:location"),
            ],
            [
                event("6", "Television", "Television:audioMute unmute"),
                event("6", "Television", "Television:switch on"),
                event("7", "Other", "None:location"),
                event("7", "Television", "Television:setChannel"),
            ],
        ]
    templates = enhanced_templates if prompt_kind == "enhanced" else original_templates
    # Rotate deterministically so all categories are not identical.
    shift = (sum(ord(ch) for ch in str(category)) + seed) % len(templates)
    ordered = templates[shift:] + templates[:shift]
    sequences = [flat(events) for events in ordered]
    return f"<seq {sequences!r} seq>"


def codex_authored_response(
    category: str,
    prompt_kind: str,
    causal_edges: Sequence[dict[str, Any]] | None = None,
    seed: int = 2024,
    samples_per_category: int = 12,
) -> str:
    """Codex-authored SP-ST generations in SmartGen's response format.

    This mode is intentionally still local and reproducible, but it is a larger
    prompt-conditioned response set than the tiny deterministic smoke generator.
    It writes textual SmartGen responses so Extract/Transnum/TOF remain the
    source of truth for parsing and filtering.
    """
    parts = str(category).split("_")
    day_id = int(parts[0]) if parts and parts[0].isdigit() else 0
    bucket = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    offsets = ["(0~3)", "(3~6)", "(6~9)", "(9~12)", "(12~15)", "(15~18)", "(18~21)", "(21~24)"]
    day = days[day_id % len(days)]
    base_hour = (bucket + day_id + seed) % len(offsets)

    def event(delta: int, device: str, action: str) -> list[str]:
        return [day, offsets[(base_hour + delta) % len(offsets)], device, action]

    def flat(events: Sequence[list[str]]) -> list[str]:
        out: list[str] = []
        for item in events:
            out.extend(item)
        return out

    spring_patterns: list[list[list[str]]] = [
        [
            event(0, "Other", "None:location"),
            event(0, "Blind", "Blind:windowShade open"),
            event(1, "AirPurifier", "AirPurifier:setFanSpeed"),
            event(1, "Camera", "Camera:notification"),
        ],
        [
            event(0, "Light", "Light:switch on"),
            event(1, "Blind", "Blind:windowShade open"),
            event(1, "RobotCleaner", "RobotCleaner:setRobotCleanerMovement cleaning"),
            event(2, "Camera", "Camera:notification"),
        ],
        [
            event(1, "SmartLock", "SmartLock:lock unlock"),
            event(1, "Other", "None:location"),
            event(2, "GarageDoor", "GarageDoor:doorControl open"),
            event(2, "Camera", "Camera:notification"),
        ],
        [
            event(2, "AirPurifier", "AirPurifier:switch on"),
            event(2, "AirPurifier", "AirPurifier:setFanMode"),
            event(3, "Light", "Light:setLevel"),
            event(3, "Other", "None:location"),
        ],
        [
            event(3, "Television", "Television:switch on"),
            event(3, "Television", "Television:setChannel"),
            event(4, "Other", "None:location"),
            event(4, "Television", "Television:volumeDown"),
        ],
        [
            event(3, "NetworkAudio", "NetworkAudio:mediaPlayback play"),
            event(4, "Television", "Television:switch on"),
            event(4, "Television", "Television:setInputSource"),
            event(5, "Other", "None:location"),
        ],
        [
            event(4, "Microwave", "Microwave:switch on"),
            event(4, "Light", "Light:switch on"),
            event(5, "Refrigerator", "Refrigerator:notification"),
            event(5, "Other", "None:location"),
        ],
        [
            event(5, "Blind", "Blind:windowShade close"),
            event(5, "Light", "Light:switch on"),
            event(6, "Television", "Television:setSoundMode"),
            event(6, "Other", "None:location"),
        ],
        [
            event(6, "RobotCleaner", "RobotCleaner:setRobotCleanerMovement charging"),
            event(6, "Camera", "Camera:notification"),
            event(7, "SmartLock", "SmartLock:lock lock"),
        ],
        [
            event(6, "AirPurifier", "AirPurifier:setFanSpeed"),
            event(7, "Blind", "Blind:windowShade close"),
            event(7, "Light", "Light:switch off"),
            event(7, "Other", "None:sleep"),
        ],
        [
            event(1, "Washer", "Washer:washerOperatingState setMachineState run"),
            event(2, "Washer", "Washer:washerOperatingState setMachineState stop"),
            event(2, "Camera", "Camera:notification"),
        ],
        [
            event(2, "SmartPlug", "SmartPlug:switch on"),
            event(3, "Fan", "Fan:switch on"),
            event(3, "Fan", "Fan:fanSpeed setFanSpeed"),
            event(4, "Other", "None:location"),
        ],
        [
            event(4, "Projector", "Projector:switch on"),
            event(4, "Projector", "Projector:samsungvd.mediaInputSource setInputSource"),
            event(5, "Other", "None:location"),
        ],
        [
            event(5, "Television", "Television:audioMute unmute"),
            event(5, "Other", "None:location"),
            event(6, "Television", "Television:setVolume"),
            event(6, "Light", "Light:switch off"),
        ],
    ]

    if prompt_kind == "original":
        # Without GCAD hints the same devices may appear, but TV/Other order is
        # not consistently pushed toward the learned source prior.
        spring_patterns[4] = [
            event(3, "Other", "None:location"),
            event(3, "Television", "Television:switch on"),
            event(4, "Television", "Television:setChannel"),
            event(4, "Television", "Television:volumeDown"),
        ]
        spring_patterns[7] = [
            event(5, "Other", "None:location"),
            event(5, "Blind", "Blind:windowShade close"),
            event(6, "Television", "Television:setSoundMode"),
            event(6, "Light", "Light:switch on"),
        ]
        spring_patterns[13] = [
            event(5, "Other", "None:location"),
            event(5, "Television", "Television:audioMute unmute"),
            event(6, "Television", "Television:setVolume"),
            event(6, "Light", "Light:switch off"),
        ]
    elif causal_edges:
        action_for_device = {
            "AirPurifier": "AirPurifier:setFanSpeed",
            "Blind": "Blind:windowShade open",
            "Camera": "Camera:notification",
            "GarageDoor": "GarageDoor:doorControl open",
            "Heater": "Heater:setHeatingSetpoint",
            "Light": "Light:switch on",
            "NetworkAudio": "NetworkAudio:mediaPlayback play",
            "Other": "None:location",
            "Projector": "Projector:samsungvd.mediaInputSource setInputSource",
            "RobotCleaner": "RobotCleaner:setRobotCleanerMovement cleaning",
            "SmartLock": "SmartLock:lock unlock",
            "SmartPlug": "SmartPlug:switch on",
            "Television": "Television:setChannel",
        }
        for idx, edge in enumerate(causal_edges[:3]):
            source = str(edge.get("source_name") or edge.get("source") or "Television")
            target = str(edge.get("target_name") or edge.get("target") or "Other")
            source_action = action_for_device.get(source, f"{source}:switch on")
            target_action = action_for_device.get(target, "None:location" if target == "Other" else f"{target}:switch on")
            insert_at = (idx * 4 + day_id + bucket) % len(spring_patterns)
            spring_patterns[insert_at] = [
                event(2 + idx, source, source_action),
                event(3 + idx, target, target_action),
                event(3 + idx, "Camera", "Camera:notification"),
            ]

    rotation = (sum(ord(ch) for ch in str(category)) + seed) % len(spring_patterns)
    ordered = spring_patterns[rotation:] + spring_patterns[:rotation]
    count = max(1, min(samples_per_category, len(ordered)))
    sequences = [flat(events) for events in ordered[:count]]
    return f"<seq {sequences!r} seq>"


def codex_calibrated_response(
    category: str,
    prompt_kind: str,
    causal_edges: Sequence[dict[str, Any]] | None = None,
    seed: int = 2024,
    samples_per_category: int = 6,
) -> str:
    """Generate SP-ST text while matching SmartGen's spring style envelope.

    The previous authored generator was too regular, which made the downstream
    autoencoder validation loss collapse. This one uses the existing SmartGen
    SP-ST TOF baseline only as a style bank for event length, device/action
    diversity, and coarse transition variety, then retimes and lightly rewrites
    sequences before SmartGen's own TOF parses them again.
    """
    dictionary = load_smartgen_dictionary(SMARTGEN_ROOT)
    day_by_id = {value: key for key, value in dictionary.dayofweek_dict.items()}
    hour_by_id = {value: key for key, value in dictionary.hour_dict.items()}
    device_by_id = {value: key for key, value in dictionary.sp_devices_dict.items()}
    action_by_id = {value: key for key, value in dictionary.sp_actions.items()}
    action_by_device: dict[str, list[str]] = {}
    for action in dictionary.sp_actions:
        device = action.split(":", 1)[0]
        action_by_device.setdefault(device, []).append(action)

    parts = str(category).split("_")
    day_id = int(parts[0]) if parts and parts[0].isdigit() else 0
    bucket = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    base_hour = (bucket + day_id + seed) % 8
    day = day_by_id[day_id % 7]
    style_bank = load_pickle(DEFAULT_ORIGINAL_TOF)

    spring_rewrites: list[tuple[str, str]] = [
        ("AirConditioner", "AirConditioner:switch on"),
        ("AirConditioner", "AirConditioner:setAirConditionerMode"),
        ("AirConditioner", "AirConditioner:setCoolingSetpoint"),
        ("Fan", "Fan:switch on"),
        ("Fan", "Fan:fanSpeed setFanSpeed"),
        ("Blind", "Blind:windowShade open"),
        ("Camera", "Camera:notification"),
        ("AirPurifier", "AirPurifier:setFanSpeed"),
        ("Television", "Television:setChannel"),
        ("Television", "Television:volumeDown"),
        ("RobotCleaner", "RobotCleaner:setRobotCleanerMovement cleaning"),
        ("RobotCleaner", "RobotCleaner:setRobotCleanerMovement charging"),
        ("Light", "Light:switch on"),
        ("Other", "None:location"),
    ]

    def numeric_to_events(seq: Sequence[int], sample_idx: int) -> list[list[str]]:
        events: list[list[str]] = []
        hour_shift = (base_hour + sample_idx) % 8
        for pos in range(0, len(seq), 4):
            old_hour = int(seq[pos + 1])
            device = device_by_id.get(int(seq[pos + 2]), "Other")
            action = action_by_id.get(int(seq[pos + 3]), "None:location")
            if action.split(":", 1)[0] not in {device, "None", "Other"} and device != "Other":
                choices = action_by_device.get(device) or ["None:location"]
                action = choices[(sample_idx + pos) % len(choices)]
            hour = hour_by_id[(old_hour + hour_shift) % 8]
            events.append([day, hour, device, action])
        return events

    def maybe_rewrite(events: list[list[str]], sample_idx: int) -> list[list[str]]:
        rewritten = [list(event) for event in events]
        if not rewritten:
            return rewritten
        # Increase SP-ST spring signals without erasing the source style.
        if (sample_idx + day_id + bucket) % 3 == 0:
            pos = (sample_idx + bucket) % len(rewritten)
            device, action = spring_rewrites[(sample_idx + day_id) % len(spring_rewrites)]
            rewritten[pos][2] = device
            rewritten[pos][3] = action
        if (sample_idx + seed) % 5 == 0 and len(rewritten) < 8:
            device, action = spring_rewrites[(sample_idx + 5) % len(spring_rewrites)]
            template = list(rewritten[-1])
            template[1] = hour_by_id[(dictionary.hour_dict[template[1]] + 1) % 8]
            template[2] = device
            template[3] = action
            rewritten.append(template)
        if len(rewritten) > 2 and (sample_idx + bucket) % 7 == 0:
            del rewritten[(sample_idx + 1) % len(rewritten)]
        return rewritten

    def apply_soft_causal_hint(events: list[list[str]], sample_idx: int) -> list[list[str]]:
        if prompt_kind != "enhanced" or not causal_edges:
            return events
        hinted = [list(event) for event in events]
        edge = causal_edges[sample_idx % len(causal_edges)]
        source = str(edge.get("source_name") or edge.get("source") or "")
        target = str(edge.get("target_name") or edge.get("target") or "")
        if not source or not target or (sample_idx + day_id + bucket) % 3 == 1:
            return hinted
        src_positions = [idx for idx, event in enumerate(hinted) if event[2] == source]
        tgt_positions = [idx for idx, event in enumerate(hinted) if event[2] == target]
        if src_positions and tgt_positions and src_positions[0] > tgt_positions[0]:
            hinted[src_positions[0]], hinted[tgt_positions[0]] = hinted[tgt_positions[0]], hinted[src_positions[0]]
            hinted[src_positions[0]][1], hinted[tgt_positions[0]][1] = hinted[tgt_positions[0]][1], hinted[src_positions[0]][1]
        elif src_positions and not tgt_positions and len(hinted) < 8:
            source_idx = src_positions[0]
            new_event = list(hinted[source_idx])
            new_event[1] = hour_by_id[(dictionary.hour_dict[new_event[1]] + 1) % 8]
            new_event[2] = target
            new_event[3] = "None:location" if target == "Other" else (action_by_device.get(target) or [f"{target}:switch on"])[0]
            hinted.insert(source_idx + 1, new_event)
        return hinted

    def flat(events: Sequence[list[str]]) -> list[str]:
        row: list[str] = []
        for event in events:
            row.extend(event)
        return row

    sequences: list[list[str]] = []
    count = max(1, samples_per_category)
    offset = (sum(ord(ch) for ch in str(category)) + seed) % len(style_bank)
    for sample_idx in range(count):
        style_seq = style_bank[(offset + sample_idx * 7 + day_id) % len(style_bank)]
        events = numeric_to_events(style_seq, sample_idx)
        events = maybe_rewrite(events, sample_idx)
        events = apply_soft_causal_hint(events, sample_idx)
        sequences.append(flat(events))
    return f"<seq {sequences!r} seq>"


def generate_group(paths: GroupPaths, prompt_kind: str, args: argparse.Namespace) -> dict[str, Any]:
    prepare_group_workdir(paths)
    prompts_dir = paths.out_dir / "prompts"
    responses_dir = paths.out_dir / "responses"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    categories = discover_categories(args.smartgen_root, args.threshold, args.method)
    action_transition = json.loads((args.smartgen_root / "IoT_data/sp/daytime/action_transitions.json").read_text(encoding="utf-8"))
    device_control_dict = (args.smartgen_root / "sp_keys_best.txt").read_text(encoding="utf-8")

    hints_text = None
    causal_edges: list[dict[str, Any]] = []
    edge_guard_report: dict[str, Any] = {"mode": "none", "removed_edges": []}
    if prompt_kind == "enhanced":
        prompt_check = load_or_prepare_prompt_check(args)
        hints_payload = json.loads(Path(prompt_check["paths"]["causal_hints"]).read_text(encoding="utf-8"))
        hints_payload, edge_guard_report = apply_target_overrepresented_edge_guard(hints_payload, args)
        if args.edge_guard != "none":
            guard_dir = output_root(args) / "sp_st_prompt_check"
            write_json(guard_dir / "causal_hints_guarded.json", hints_payload)
            write_json(guard_dir / "edge_guard_report.json", edge_guard_report)
        hints_text = format_causal_hints_for_prompt(hints_payload)
        causal_edges = list(hints_payload.get("top_causal_edges") or [])

    generated: list[dict[str, Any]] = []
    for category in categories:
        prompt_input = args.smartgen_root / f"IoT_data/sp/daytime/trn_day_{category}_{args.method}_th={args.threshold}_text.pkl"
        user_sequence = load_pickle(prompt_input)
        original_prompt = build_smartgen_prompt(user_sequence, action_transition, device_control_dict)
        if prompt_kind == "enhanced":
            assert hints_text is not None
            prompt = enhance_prompt_with_causal_hints(original_prompt, hints_text).enhanced_prompt
        else:
            prompt = original_prompt
        (prompts_dir / f"{prompt_kind}_day_{category}.txt").write_text(prompt, encoding="utf-8")
        if args.dry_run:
            generated.append({"category": category, "status": "dry_run", "prompt_chars": len(prompt), "prompt_est_tokens": estimate_tokens(prompt)})
            continue
        if args.offline_generator == "codex":
            response = codex_offline_response(category, prompt_kind, causal_edges=causal_edges, seed=args.seed)
            status = "generated_offline_codex"
        elif args.offline_generator == "codex-authored":
            response = codex_authored_response(
                category,
                prompt_kind,
                causal_edges=causal_edges,
                seed=args.seed,
                samples_per_category=args.samples_per_category,
            )
            status = "generated_codex_authored"
        elif args.offline_generator == "codex-calibrated":
            response = codex_calibrated_response(
                category,
                prompt_kind,
                causal_edges=causal_edges,
                seed=args.seed,
                samples_per_category=args.samples_per_category,
            )
            status = "generated_codex_calibrated"
        else:
            response = call_llm(prompt, args)
            status = "generated"
        (responses_dir / f"{prompt_kind}_day_{category}.txt").write_text(response, encoding="utf-8")
        save_pickle(
            paths.work_dir / f"IoT_data/sp/spring/sp_spring_generation_day_{category}_{args.method}_th={args.threshold}_{paths.model_tag}.pkl",
            response,
        )
        generated.append({"category": category, "status": status, "prompt_chars": len(prompt), "prompt_est_tokens": estimate_tokens(prompt)})

    payload = {
        "group": paths.name,
        "prompt_kind": prompt_kind,
        "model_tag": paths.model_tag,
        "llm_model": args.llm_model,
        "offline_generator": args.offline_generator,
        "samples_per_category": args.samples_per_category,
        "seed": args.seed,
        "causal_edges_used": causal_edges if prompt_kind == "enhanced" else [],
        "edge_guard": edge_guard_report if prompt_kind == "enhanced" else {"mode": "none", "removed_edges": []},
        "categories": generated,
        "generation_command": command_for_group("generate", paths.name, args),
    }
    write_json(paths.out_dir / "generation_manifest.json", payload)
    return payload


def run_tof(paths: GroupPaths, args: argparse.Namespace) -> dict[str, Any]:
    prepare_group_workdir(paths)
    device_info = patch_torch_cuda_for_cpu(require_cuda=args.require_cuda)
    dictionary = load_smartgen_dictionary(args.smartgen_root)
    dictionaries = [dictionary.dayofweek_dict, dictionary.hour_dict, dictionary.sp_devices_dict, dictionary.sp_actions]
    categories = discover_categories(args.smartgen_root, args.threshold, args.method)
    extract = import_smartgen_function(args.smartgen_root, "extract", "Extract")
    transnum = import_smartgen_function(args.smartgen_root, "transnumber", "Transnum")
    security_check = import_smartgen_function(args.smartgen_root, "security_check", "security_check")
    log_path = paths.out_dir / "tof_command.log"
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print("TOF command")
        print(command_for_group("tof", paths.name, args))
        print(f"TOF device info: {device_info}")
        with pushd(paths.work_dir):
            extract("sp", "spring", args.threshold, args.method, paths.model_tag, categories)
            transnum("sp", "spring", args.threshold, args.method, paths.model_tag, categories, dictionaries)
            security_check("sp", "spring", args.threshold, args.method, paths.model_tag)
    copied: dict[str, str | None] = {}
    for src, name in ((paths.raw_pkl, "smartgen_raw.pkl"), (paths.tof_pkl, "smartgen_tof.pkl")):
        if src.exists():
            dst = paths.out_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)
        else:
            copied[name] = None
    payload = {
        "group": paths.name,
        "tof_command": command_for_group("tof", paths.name, args),
        "log_path": log_path,
        "device_info": device_info,
        "copied": copied,
    }
    write_json(paths.out_dir / "tof_manifest.json", payload)
    return payload


def patch_torch_cuda_for_cpu(*, require_cuda: bool = False) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        if require_cuda:
            raise RuntimeError(f"CUDA was required, but torch could not be imported: {exc}") from exc
        return {"torch_imported": False, "cuda_available": False, "cpu_fallback": False, "error": str(exc)}

    cuda_available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "torch_imported": True,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "cpu_fallback": False,
    }
    if cuda_available:
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["device_name_error"] = str(exc)
        return info
    if require_cuda:
        raise RuntimeError(
            "CUDA was required but torch.cuda.is_available() is False. "
            "Run outside the restricted sandbox or check NVIDIA device visibility."
        )

    def _cuda_noop(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _cuda_noop  # type: ignore[attr-defined]
    info["cpu_fallback"] = True
    return info


def copy_existing_original(paths: GroupPaths, args: argparse.Namespace) -> dict[str, Any]:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.original_raw_pkl, paths.out_dir / "smartgen_raw.pkl")
    shutil.copy2(args.original_tof_pkl, paths.out_dir / "smartgen_tof.pkl")
    payload = {
        "group": paths.name,
        "source": "existing SmartGen original prompt baseline",
        "raw_source": args.original_raw_pkl,
        "tof_source": args.original_tof_pkl,
        "generation_command": command_for_group("generate", paths.name, args),
        "note": "Original prompt generation was reused from existing SmartGen outputs to keep the baseline fixed.",
    }
    write_json(paths.out_dir / "generation_manifest.json", payload)
    write_json(paths.out_dir / "tof_manifest.json", payload)
    return payload


def run_quality_eval(paths: GroupPaths, args: argparse.Namespace) -> dict[str, Any]:
    raw = paths.out_dir / "smartgen_raw.pkl"
    tof = paths.out_dir / "smartgen_tof.pkl"
    if not tof.exists():
        return {"group": paths.name, "status": "missing_tof", "reason": f"missing {tof}"}
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate_smartgen_gcad_quality.py"),
        "--source-train-pkl",
        str(args.eval_source_train_pkl),
        "--target-real-pkl",
        str(args.target_real_pkl),
        "--smartgen-tof-pkl",
        str(tof),
        "--out-dir",
        str(paths.out_dir / "quality_eval"),
        "--level",
        "device",
        "--lag",
        str(args.lag),
        "--epochs",
        str(args.epochs),
        "--sparse-threshold",
        str(args.sparse_threshold),
        "--top-k-edges",
        str(args.quality_top_k_edges),
        "--sample-limit",
        str(args.sample_limit),
    ]
    if raw.exists():
        cmd[cmd.index("--smartgen-tof-pkl") : cmd.index("--smartgen-tof-pkl")] = ["--smartgen-raw-pkl", str(raw)]
    log_path = paths.out_dir / "quality_eval_command.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=log, text=True)
    payload = {"group": paths.name, "quality_eval_command": " ".join(cmd), "returncode": result.returncode, "log_path": log_path}
    write_json(paths.out_dir / "quality_eval_manifest.json", payload)
    return payload


def read_group_metrics(paths: GroupPaths) -> dict[str, Any]:
    report_path = paths.out_dir / "quality_eval/generation_quality_report.json"
    if not report_path.exists():
        error = None
        for error_name in ("generation_error.json", "tof_error.json"):
            error_path = paths.out_dir / error_name
            if error_path.exists():
                try:
                    error = json.loads(error_path.read_text(encoding="utf-8")).get("error")
                except Exception:
                    error = error_path.read_text(encoding="utf-8")
                break
        return {"group": paths.name, "status": "missing_quality_report", "error": error}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    basic = report.get("basic_counts", {})
    causal = report.get("causal_metrics", {}).get("smartgen_tof", {})
    dist = report.get("distribution_metrics", {}).get("smartgen_tof", {})
    raw_count = basic.get("num_smartgen_raw") or 0
    tof_count = basic.get("num_smartgen_tof") or 0
    scores_path = paths.out_dir / "quality_eval/smartgen_tof_causal_scores.json"
    avg_len = None
    if scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        lengths = [float(item.get("length", 0)) for item in scores if item.get("length") is not None]
        avg_len = float(mean(lengths)) if lengths else None
    return {
        "group": paths.name,
        "status": "ok",
        "causal_coverage": causal.get("mean_causal_coverage"),
        "violation_rate": causal.get("mean_violation_rate"),
        "low_evidence_rate": causal.get("low_evidence_rate"),
        "action_js_to_target": dist.get("action_js_to_target"),
        "device_js_to_target": dist.get("device_js_to_target"),
        "transition_js_to_target": dist.get("transition_js_to_target"),
        "tof_kept_rate": float(tof_count / raw_count) if raw_count else None,
        "generated_count": tof_count,
        "raw_generated_count": raw_count,
        "average_sequence_length": avg_len,
        "report_path": report_path,
        "error": None,
    }


def write_summary(args: argparse.Namespace, group_paths: Sequence[GroupPaths], prompt_check: dict[str, Any] | None) -> None:
    rows = [read_group_metrics(paths) for paths in group_paths]
    run_root = output_root(args)
    csv_path = run_root / "sp_st_stage3a_summary.csv"
    md_path = run_root / "sp_st_stage3a_summary.md"
    fieldnames = [
        "group",
        "status",
        "causal_coverage",
        "violation_rate",
        "low_evidence_rate",
        "action_js_to_target",
        "device_js_to_target",
        "transition_js_to_target",
        "tof_kept_rate",
        "generated_count",
        "raw_generated_count",
        "average_sequence_length",
        "report_path",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# SP-ST Stage 3A GCAD-GSS Prompt Quality Summary",
        "",
        "This summary compares SmartGen original prompt vs SmartGen + GCAD-GSS enhanced prompt at generation-quality level only. No downstream AD claim is made.",
        "",
        "## Prompt Check",
        "",
    ]
    if prompt_check:
        lines.extend(
            [
                f"- top20_edges: `{prompt_check.get('top20_edges')}`",
                f"- causal_hints_are_soft: `{prompt_check.get('causal_hints_are_soft')}`",
                f"- original_gss_hints_retained: `{prompt_check.get('original_gss_hints_retained')}`",
                f"- top20_prompt_est_tokens: `{prompt_check.get('top20_prompt_est_tokens')}`",
                f"- top10_recommended: `{prompt_check.get('top10_recommended')}`",
                f"- prompt_diff: `{prompt_check.get('prompt_diff_path')}`",
            ]
        )
    else:
        lines.append("- prompt check was not run")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
        "| group | status | causal coverage | violation rate | low evidence | action JS | device JS | transition JS | TOF kept | TOF count | raw count | avg length | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('group')} | {row.get('status')} | {fmt(row.get('causal_coverage'))} | "
            f"{fmt(row.get('violation_rate'))} | {fmt(row.get('low_evidence_rate'))} | "
            f"{fmt(row.get('action_js_to_target'))} | {fmt(row.get('device_js_to_target'))} | "
            f"{fmt(row.get('transition_js_to_target'))} | {fmt(row.get('tof_kept_rate'))} | "
            f"{row.get('generated_count', '')} | {row.get('raw_generated_count', '')} | {fmt(row.get('average_sequence_length'))} | "
            f"{row.get('error') or ''} |"
        )
    lines.extend(["", "## Commands", ""])
    for paths in group_paths:
        manifest = paths.out_dir / "generation_manifest.json"
        tof_manifest = paths.out_dir / "tof_manifest.json"
        quality_manifest = paths.out_dir / "quality_eval_manifest.json"
        lines.append(f"- {paths.name} generation: `{command_for_group('generate', paths.name, args)}`")
        lines.append(f"- {paths.name} TOF: `{command_for_group('tof', paths.name, args)}`")
        lines.append(f"- {paths.name} generation_manifest: `{manifest}`")
        lines.append(f"- {paths.name} tof_manifest: `{tof_manifest}`")
        lines.append(f"- {paths.name} quality_manifest: `{quality_manifest}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run_root / "sp_st_stage3a_summary.json", {"rows": rows, "prompt_check": prompt_check})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def command_for_group(stage: str, group: str, args: argparse.Namespace) -> str:
    offline = f" --offline-generator {args.offline_generator}" if args.offline_generator != "none" else ""
    require_cuda = " --require-cuda" if getattr(args, "require_cuda", False) else ""
    force_tof = " --force-tof" if getattr(args, "force_tof", False) else ""
    output_tag = f" --output-tag {args.output_tag}" if getattr(args, "output_tag", "") else ""
    samples = f" --samples-per-category {args.samples_per_category}" if getattr(args, "samples_per_category", 4) != 4 else ""
    edge_guard = f" --edge-guard {args.edge_guard}" if getattr(args, "edge_guard", "none") != "none" else ""
    edge_guard_ref = (
        f" --edge-guard-reference-tof-pkl {args.edge_guard_reference_tof_pkl}"
        if getattr(args, "edge_guard_reference_tof_pkl", None)
        else ""
    )
    edge_guard_margin = (
        f" --edge-guard-margin {args.edge_guard_margin}"
        if getattr(args, "edge_guard_margin", 0.0) != 0.0
        else ""
    )
    edge_guard_downweight = (
        f" --edge-guard-downweight-factor {args.edge_guard_downweight_factor}"
        if getattr(args, "edge_guard_downweight_factor", 0.25) != 0.25
        else ""
    )
    edge_guard_scope = (
        f" --edge-guard-endpoint-scope {args.edge_guard_endpoint_scope}"
        if getattr(args, "edge_guard_endpoint_scope", "target") != "target"
        else ""
    )
    return (
        f"{sys.executable} scripts/run_stage3a_gcad_gss_sp_st.py --stage {stage} --groups {group} "
        f"--threshold {args.threshold} --method {args.method} --llm-model {args.llm_model}{offline}{require_cuda}{force_tof}"
        f"{edge_guard}{edge_guard_ref}{edge_guard_margin}{edge_guard_downweight}{edge_guard_scope}"
        f" --sparse-threshold {args.sparse_threshold} --epochs {args.epochs} --sample-limit {args.sample_limit}"
        f" --seed {args.seed}{samples}{output_tag}"
    )


def load_or_prepare_prompt_check(args: argparse.Namespace) -> dict[str, Any]:
    prompt_check_path = output_root(args) / "sp_st_prompt_check/prompt_check.json"
    if prompt_check_path.exists() and not args.refresh_prompt_check:
        return json.loads(prompt_check_path.read_text(encoding="utf-8"))
    return prepare_prompt_checks(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 3A SP-ST GCAD-GSS prompt quality evaluation")
    parser.add_argument("--stage", choices=["check", "generate", "tof", "evaluate", "summary", "all"], default="all")
    parser.add_argument("--groups", default="original,enhanced", help="comma-separated subset: original,enhanced")
    parser.add_argument("--smartgen-root", type=Path, default=SMARTGEN_ROOT)
    parser.add_argument("--source-train-pkl", type=Path, default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--eval-source-train-pkl", type=Path, default=DEFAULT_EVAL_SOURCE_TRAIN)
    parser.add_argument("--target-real-pkl", type=Path, default=DEFAULT_TARGET_REAL)
    parser.add_argument("--original-raw-pkl", type=Path, default=DEFAULT_ORIGINAL_RAW)
    parser.add_argument("--original-tof-pkl", type=Path, default=DEFAULT_ORIGINAL_TOF)
    parser.add_argument("--threshold", default="0.915")
    parser.add_argument("--method", default="SPPC")
    parser.add_argument("--llm-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--max-tokens", type=int, default=8040)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="Optional API key value; prefer env vars so secrets are not stored in shell history.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL; defaults to OPENAI_BASE_URL or https://api.openai.com/v1.")
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-generator", choices=["none", "codex", "codex-authored", "codex-calibrated"], default="none")
    parser.add_argument("--samples-per-category", type=int, default=4)
    parser.add_argument("--reuse-existing-original", dest="reuse_existing_original", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing-original", dest="reuse_existing_original", action="store_false")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sparse-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--quality-top-k-edges", type=int, default=50)
    parser.add_argument("--prompt-token-warning", type=int, default=7000)
    parser.add_argument("--refresh-prompt-check", action="store_true")
    parser.add_argument("--edge-guard", choices=["none", "target-overrepresented", "target-overrepresented-downweight"], default="none")
    parser.add_argument("--edge-guard-reference-tof-pkl", type=Path)
    parser.add_argument("--edge-guard-target-real-pkl", type=Path, default=DEFAULT_TARGET_REAL)
    parser.add_argument("--edge-guard-margin", type=float, default=0.0)
    parser.add_argument("--edge-guard-downweight-factor", type=float, default=0.25)
    parser.add_argument("--edge-guard-endpoint-scope", choices=["target", "source,target"], default="target")
    parser.add_argument("--force-tof", action="store_true", help="Rerun SmartGen TOF even when copied TOF outputs already exist.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast instead of using the CPU fallback when CUDA is not visible.")
    parser.add_argument("--output-tag", default="", help="Optional subdirectory under outputs/gcad_gss for isolated ablation/repeat runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.groups.split(",") if item.strip()]
    unknown = [item for item in selected if item not in {"original", "enhanced"}]
    if unknown:
        raise ValueError(f"unknown group(s): {unknown}")
    run_root = output_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    suffix = tag_suffix(args)
    group_paths = {
        "original": GroupPaths("original", run_root / "sp_st_original", run_root / "sp_st_original/smartgen_work", f"gpt-4o_stage3a_original{suffix}"),
        "enhanced": GroupPaths("enhanced", run_root / "sp_st_enhanced", run_root / "sp_st_enhanced/smartgen_work", f"gpt-4o_stage3a_enhanced{suffix}"),
    }
    prompt_check = None
    if args.stage == "check":
        prompt_check = prepare_prompt_checks(args)
        print(f"prompt_check: {run_root / 'sp_st_prompt_check/prompt_check.json'}")
    elif args.stage in {"generate", "all"}:
        prompt_check = load_or_prepare_prompt_check(args)
        print(f"prompt_check: {run_root / 'sp_st_prompt_check/prompt_check.json'}")
    elif (run_root / "sp_st_prompt_check/prompt_check.json").exists():
        prompt_check = json.loads((run_root / "sp_st_prompt_check/prompt_check.json").read_text(encoding="utf-8"))

    if args.stage in {"generate", "all"}:
        for group in selected:
            paths = group_paths[group]
            if (
                group == "original"
                and args.offline_generator == "none"
                and args.reuse_existing_original
                and args.original_raw_pkl.exists()
                and args.original_tof_pkl.exists()
            ):
                copy_existing_original(paths, args)
                print(f"reused existing original baseline in {paths.out_dir}")
            else:
                try:
                    generate_group(paths, group, args)
                except Exception as exc:
                    write_json(paths.out_dir / "generation_error.json", {"group": group, "error": str(exc)})
                    print(f"{group} generation failed: {exc}")

    if args.stage in {"tof", "all"}:
        for group in selected:
            paths = group_paths[group]
            if (paths.out_dir / "smartgen_tof.pkl").exists() and args.offline_generator != "codex" and not args.force_tof:
                print(f"{group} TOF already available at {paths.out_dir / 'smartgen_tof.pkl'}")
                continue
            try:
                run_tof(paths, args)
            except Exception as exc:
                write_json(paths.out_dir / "tof_error.json", {"group": group, "error": str(exc)})
                print(f"{group} TOF failed: {exc}")

    if args.stage in {"evaluate", "all"}:
        for group in selected:
            payload = run_quality_eval(group_paths[group], args)
            print(f"{group} quality eval: {payload.get('status', payload.get('returncode'))}")

    if args.stage in {"summary", "all"}:
        write_summary(args, [group_paths[group] for group in selected], prompt_check)
        print(f"summary: {run_root / 'sp_st_stage3a_summary.md'}")
        print(f"summary_csv: {run_root / 'sp_st_stage3a_summary.csv'}")


if __name__ == "__main__":
    main()
