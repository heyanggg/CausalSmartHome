from __future__ import annotations

import random
from typing import Tuple

from .schema import BehaviorEvent, BehaviorSequence


def make_toy_normal_sequences(n: int = 80, seed: int = 7) -> list[BehaviorSequence]:
    """Create toy routines with stable causal order.

    Action ids in the toy data:
    10: arrive home, 11: unlock, 12: turn light on, 13: close curtain,
    14: cook, 15: eat, 16: wash dishes, 20: noise action.
    """
    rng = random.Random(seed)
    seqs = []
    for i in range(n):
        day = rng.randrange(7)
        start = rng.choice([5, 6])
        actions = [10, 11, 12, 13, 14, 15, 16]
        if rng.random() < 0.25:
            actions.insert(rng.randrange(1, len(actions)), 20)
        events = []
        for j, act in enumerate(actions):
            hour = min(7, start + j // 2)
            device = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 20: 8}[act]
            events.append(BehaviorEvent(day, hour, device, act))
        seqs.append(BehaviorSequence(events, sequence_id=f"normal_{i}"))
    return seqs


def make_toy_generated_candidates(seed: int = 13) -> list[BehaviorSequence]:
    rng = random.Random(seed)
    candidates: list[BehaviorSequence] = []
    good_orders = [
        [10, 11, 12, 13, 14, 15, 16],
        [10, 11, 13, 12, 14, 15, 16],
        [10, 11, 12, 14, 15, 16],
    ]
    bad_orders = [
        [16, 15, 14, 13, 12, 11, 10],
        [10, 16, 11, 15, 12, 14, 13],
    ]
    for i, actions in enumerate(good_orders + bad_orders):
        events = []
        for j, act in enumerate(actions):
            device = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 20: 8}.get(act, 0)
            events.append(BehaviorEvent(day=i % 7, hour_slot=min(7, 5 + j // 2), device=device, action=act))
        candidates.append(BehaviorSequence(events, sequence_id=f"candidate_{i}"))
    return candidates
