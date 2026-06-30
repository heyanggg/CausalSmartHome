"""CausalSmartHome 各阶段共用的 Gen 序列数据结构。

原始 Gen/SmartGen 代码把一条智能家居行为序列保存成扁平四元组列表：

    [day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]

本模块给这种格式套了一层很薄的类型封装。项目内部可以用事件、设备、动作、
时间槽来表达逻辑；和 Gen 的 pickle 文件交互时，又能无损转换回原始扁平
数字格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any, Optional


HOURS_PER_SLOT = 3
SLOTS_PER_DAY = 8
DAYS_PER_WEEK = 7
HOURS_PER_WEEK = 24 * DAYS_PER_WEEK


@dataclass(frozen=True)
class BehaviorEvent:
    """一条智能家居行为事件。

    Gen 的数字格式使用 ``[day, hour_slot, device_id, action_id]`` 表示一个
    事件，其中 ``hour_slot`` 通常代表一天中的 3 小时时间段。若上游生成的是
    文本名称，也可以先填入名称，再在导出数字格式前完成 ID 映射。
    """

    day: int
    hour_slot: int
    device: int | str
    action: int | str

    @property
    def time_index(self) -> int:
        """返回周内时间槽编号，用于后续事件张量化和 GCAD 挖掘。"""
        return int(self.day) * SLOTS_PER_DAY + int(self.hour_slot)

    @property
    def hour_of_week(self) -> int:
        """返回周内小时编号，分析 night/time attack 这类时间攻击时更直观。"""
        return int(self.day) * 24 + int(self.hour_slot) * HOURS_PER_SLOT

    def to_numeric_quad(self) -> list[int]:
        """导出 Gen 原始代码期望的四元组格式。"""
        if not isinstance(self.device, int) or not isinstance(self.action, int):
            raise TypeError("device and action must be integers for numeric export")
        return [int(self.day), int(self.hour_slot), int(self.device), int(self.action)]

    def key(self, level: str = "action") -> str:
        """生成因果挖掘和分布 guard 使用的通道键。"""
        if level == "action":
            return f"a:{self.action}"
        if level == "device":
            return f"d:{self.device}"
        if level == "device_action":
            return f"d:{self.device}|a:{self.action}"
        raise ValueError(f"unknown level: {level}")


@dataclass
class BehaviorSequence:
    """一条 Gen 行为序列，以及可选的序列 ID 和溯源元数据。"""

    events: list[BehaviorEvent]
    sequence_id: Optional[str] = None
    meta: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @classmethod
    def from_flat_numeric(cls, flat: Sequence[int], sequence_id: Optional[str] = None) -> "BehaviorSequence":
        """把 Gen 的扁平数字列表解析为事件对象列表。"""
        if len(flat) % 4 != 0:
            raise ValueError(f"flat sequence length must be divisible by 4, got {len(flat)}")
        events: list[BehaviorEvent] = []
        for i in range(0, len(flat), 4):
            day, hour, device, action = flat[i : i + 4]
            events.append(BehaviorEvent(int(day), int(hour), int(device), int(action)))
        return cls(events=events, sequence_id=sequence_id)

    def to_flat_numeric(self) -> list[int]:
        """把事件对象重新展平成 Gen pickle 使用的原始数字格式。"""
        out: list[int] = []
        for ev in self.events:
            out.extend(ev.to_numeric_quad())
        return out

    def action_ids(self) -> list[int | str]:
        return [ev.action for ev in self.events]

    def device_ids(self) -> list[int | str]:
        return [ev.device for ev in self.events]


def load_numeric_sequences(obj: Iterable[Sequence[int]]) -> list[BehaviorSequence]:
    """批量把 Gen 扁平数字序列转换为 ``BehaviorSequence``。"""
    return [BehaviorSequence.from_flat_numeric(seq, sequence_id=str(i)) for i, seq in enumerate(obj)]


def dump_numeric_sequences(sequences: Iterable[BehaviorSequence]) -> list[list[int]]:
    return [seq.to_flat_numeric() for seq in sequences]
