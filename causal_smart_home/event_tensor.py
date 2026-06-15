from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, Tuple, List
import numpy as np

from .schema import BehaviorSequence, BehaviorEvent, load_numeric_sequences, SLOTS_PER_DAY, DAYS_PER_WEEK


@dataclass
class TensorizedEvents:
    tensor: np.ndarray
    channel_to_key: list[str]
    key_to_channel: dict[str, int]
    sequence_offsets: list[tuple[int, int]]


class EventTensorizer:
    """Bridge user behavior sequences to GCAD-style multivariate time series.

    SmartGuard/SmartGen operate on discrete behavior sequences. GCAD operates on
    multivariate time series. This tensorizer is the glue bridge: every action,
    device or device-action pair becomes a channel, and every 3-hour bin becomes
    a time step. The output is a dense [T, C] array usable by GCAD-style causal
    mining, without changing any of the original projects.
    """

    def __init__(
        self,
        level: str = "action",
        bin_count: int = SLOTS_PER_DAY * DAYS_PER_WEEK,
        count_mode: str = "count",
        decay: float = 0.0,
        min_frequency: int = 1,
    ) -> None:
        if level not in {"action", "device", "device_action"}:
            raise ValueError("level must be action, device or device_action")
        if count_mode not in {"count", "binary"}:
            raise ValueError("count_mode must be count or binary")
        self.level = level
        self.bin_count = bin_count
        self.count_mode = count_mode
        self.decay = decay
        self.min_frequency = min_frequency
        self.key_to_channel: dict[str, int] = {}
        self.channel_to_key: list[str] = []

    def fit(self, sequences: Sequence[BehaviorSequence]) -> "EventTensorizer":
        counts: dict[str, int] = {}
        for seq in sequences:
            for ev in seq:
                key = ev.key(self.level)
                counts[key] = counts.get(key, 0) + 1
        kept = sorted(k for k, v in counts.items() if v >= self.min_frequency)
        self.channel_to_key = kept
        self.key_to_channel = {k: i for i, k in enumerate(kept)}
        return self

    def transform(self, sequences: Sequence[BehaviorSequence], concatenate: bool = True) -> TensorizedEvents:
        if not self.key_to_channel:
            self.fit(sequences)
        tensors: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for seq in sequences:
            arr = np.zeros((self.bin_count, len(self.channel_to_key)), dtype=np.float32)
            for ev in seq:
                key = ev.key(self.level)
                channel = self.key_to_channel.get(key)
                if channel is None:
                    continue
                t = max(0, min(self.bin_count - 1, ev.time_index % self.bin_count))
                if self.count_mode == "binary":
                    arr[t, channel] = 1.0
                else:
                    arr[t, channel] += 1.0
            if self.decay > 0:
                arr = self._apply_decay(arr)
            tensors.append(arr)
            offsets.append((cursor, cursor + len(arr)))
            cursor += len(arr)
        if concatenate:
            tensor = np.concatenate(tensors, axis=0) if tensors else np.empty((0, len(self.channel_to_key)))
        else:
            tensor = np.stack(tensors, axis=0) if tensors else np.empty((0, self.bin_count, len(self.channel_to_key)))
        return TensorizedEvents(tensor=tensor, channel_to_key=self.channel_to_key, key_to_channel=self.key_to_channel, sequence_offsets=offsets)

    def fit_transform(self, sequences: Sequence[BehaviorSequence], concatenate: bool = True) -> TensorizedEvents:
        return self.fit(sequences).transform(sequences, concatenate=concatenate)

    def _apply_decay(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for t in range(1, arr.shape[0]):
            out[t] += out[t - 1] * self.decay
        return out

    @classmethod
    def from_numeric_sequences(cls, numeric_sequences: Iterable[Sequence[int]], **kwargs) -> tuple["EventTensorizer", TensorizedEvents]:
        seqs = load_numeric_sequences(numeric_sequences)
        tensorizer = cls(**kwargs)
        return tensorizer, tensorizer.fit_transform(seqs)
