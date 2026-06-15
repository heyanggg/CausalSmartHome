from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence

from .schema import BehaviorSequence, load_numeric_sequences


class SmartGenAdapter:
    """Non-invasive adapter around SmartGen code and data conventions."""

    def __init__(self, smartgen_root: str | None = None) -> None:
        self.smartgen_root = Path(smartgen_root).resolve() if smartgen_root else None
        if self.smartgen_root:
            sys.path.insert(0, str(self.smartgen_root / "SmartGen"))

    @staticmethod
    def load_pkl_sequences(path: str | Path) -> list[BehaviorSequence]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return load_numeric_sequences(data)

    @staticmethod
    def save_pkl_sequences(path: str | Path, sequences: Sequence[BehaviorSequence]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump([seq.to_flat_numeric() for seq in sequences], f)

    @staticmethod
    def build_transition_hints(sequences: Sequence[BehaviorSequence], top_k: int = 5) -> dict[str, Any]:
        counts: dict[str, Counter] = defaultdict(Counter)
        for seq in sequences:
            keys = [ev.key("action") for ev in seq]
            for a, b in zip(keys, keys[1:]):
                counts[a][b] += 1
        hints: dict[str, Any] = {}
        for src, counter in counts.items():
            hints[src] = [
                {"next_action": tgt, "count": int(c)} for tgt, c in counter.most_common(top_k)
            ]
        return hints

    def run_original_tss_if_available(self, dataset: str, ori_env: str, need_split: int = 1) -> bool:
        """Call SmartGen.Split if the full original project and data layout exist."""
        try:
            from split import Split  # type: ignore
        except Exception:
            return False
        Split(dataset, ori_env, need_split)
        return True
