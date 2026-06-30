"""把离散的 Gen 行为序列转换成 GCAD 可用的时间序列张量。

GCAD/causal-relation 类方法通常输入多变量时间序列 ``[time, channel]``，
而 Gen 生成的是短的符号事件序列。本模块负责做桥接：选择通道粒度
（device/action/device_action），把事件落到周内 3 小时时间槽上，并可选地
加入简单衰减，让近期事件在后续时间槽中保留弱影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

from .schema import BehaviorSequence, load_numeric_sequences, SLOTS_PER_DAY, DAYS_PER_WEEK


@dataclass
class TensorizedEvents:
    """``EventTensorizer`` 的返回结果。

    ``tensor`` 是实际的数值时间序列；``channel_to_key`` 和 ``key_to_channel``
    用来解释因果矩阵的行列含义；``sequence_offsets`` 记录每条原始行为序列在
    拼接后的大张量中对应的起止位置，便于调试和溯源。
    """

    tensor: np.ndarray
    channel_to_key: list[str]
    key_to_channel: dict[str, int]
    sequence_offsets: list[tuple[int, int]]


class EventTensorizer:
    """把用户行为序列桥接到 causal-relation 风格的多变量时间序列。

    Gen 侧处理的是离散行为序列，causal relation/GCAD 侧处理的是多变量时间序列。
    这里把每个 action、device 或 device-action pair 变成一个通道，把每个 3
    小时时间段变成一个时间步，最终输出稠密的 ``[T, C]`` 数组。这样无需修改
    Gen 或 GCAD 原始项目，就能在二者之间传递结构信息。
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
        """统计事件键频次，并确定哪些符号键会成为张量通道。"""
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
        """把行为序列投影到“时间槽 x 通道”的稠密数组。"""
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
                # 使用周内固定时间槽坐标，而不是让不同序列无限增长时间轴；
                # 这样所有序列都会落在同一个 7 天 x 8 槽的可比较坐标系里。
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
        """把前一时间槽的一部分活动量衰减传递到后一时间槽。"""
        out = arr.copy()
        for t in range(1, arr.shape[0]):
            out[t] += out[t - 1] * self.decay
        return out

    @classmethod
    def from_numeric_sequences(cls, numeric_sequences: Iterable[Sequence[int]], **kwargs) -> tuple["EventTensorizer", TensorizedEvents]:
        seqs = load_numeric_sequences(numeric_sequences)
        tensorizer = cls(**kwargs)
        return tensorizer, tensorizer.fit_transform(seqs)
