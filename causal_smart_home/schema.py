from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Dict, Any, Optional


HOURS_PER_SLOT = 3
SLOTS_PER_DAY = 8
DAYS_PER_WEEK = 7
HOURS_PER_WEEK = 24 * DAYS_PER_WEEK


@dataclass(frozen=True)
class BehaviorEvent:
    """One smart-home behavior event.

    The original Gen numeric format stores each behavior as
    [day, hour_slot, device_id, action_id], where hour_slot usually denotes a
    3-hour bin. Textual Gen outputs can be converted to the same logical
    object by filling names instead of ids.
    """

    day: int
    hour_slot: int
    device: int | str
    action: int | str

    @property
    def time_index(self) -> int:
        return int(self.day) * SLOTS_PER_DAY + int(self.hour_slot)

    @property
    def hour_of_week(self) -> int:
        return int(self.day) * 24 + int(self.hour_slot) * HOURS_PER_SLOT

    def to_numeric_quad(self) -> list[int]:
        if not isinstance(self.device, int) or not isinstance(self.action, int):
            raise TypeError("device and action must be integers for numeric export")
        return [int(self.day), int(self.hour_slot), int(self.device), int(self.action)]

    def key(self, level: str = "action") -> str:
        if level == "action":
            return f"a:{self.action}"
        if level == "device":
            return f"d:{self.device}"
        if level == "device_action":
            return f"d:{self.device}|a:{self.action}"
        raise ValueError(f"unknown level: {level}")


@dataclass
class BehaviorSequence:
    events: list[BehaviorEvent]
    sequence_id: Optional[str] = None
    meta: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @classmethod
    def from_flat_numeric(cls, flat: Sequence[int], sequence_id: Optional[str] = None) -> "BehaviorSequence":
        if len(flat) % 4 != 0:
            raise ValueError(f"flat sequence length must be divisible by 4, got {len(flat)}")
        events: list[BehaviorEvent] = []
        for i in range(0, len(flat), 4):
            day, hour, device, action = flat[i : i + 4]
            events.append(BehaviorEvent(int(day), int(hour), int(device), int(action)))
        return cls(events=events, sequence_id=sequence_id)

    def to_flat_numeric(self) -> list[int]:
        out: list[int] = []
        for ev in self.events:
            out.extend(ev.to_numeric_quad())
        return out

    def action_ids(self) -> list[int | str]:
        return [ev.action for ev in self.events]

    def device_ids(self) -> list[int | str]:
        return [ev.device for ev in self.events]


def load_numeric_sequences(obj: Iterable[Sequence[int]]) -> list[BehaviorSequence]:
    """Convert an iterable of Gen flattened numeric sequences."""
    return [BehaviorSequence.from_flat_numeric(seq, sequence_id=str(i)) for i, seq in enumerate(obj)]


def dump_numeric_sequences(sequences: Iterable[BehaviorSequence]) -> list[list[int]]:
    return [seq.to_flat_numeric() for seq in sequences]
