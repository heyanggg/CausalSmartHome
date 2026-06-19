from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptEnhancementResult:
    original_prompt: str
    enhanced_prompt: str
    insertion_index: int
    insertion_label: str


GSS_INSERTION_MARKERS = (
    "User's behavior habits:",
    "user behavior habits:",
    "GSS hints:",
    "gss hints:",
    "Graph Structure",
    "graph structure",
    "transition_hints",
    "smartgen_transition_hints",
)


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def enhance_prompt_with_causal_hints(original_prompt: str, causal_hints_text: str) -> PromptEnhancementResult:
    insertion_index, label = find_gss_insertion_point(original_prompt)
    block = "\n" + causal_hints_text.strip() + "\n"
    enhanced = original_prompt[:insertion_index] + block + original_prompt[insertion_index:]
    return PromptEnhancementResult(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced,
        insertion_index=insertion_index,
        insertion_label=label,
    )


def find_gss_insertion_point(prompt: str) -> tuple[int, str]:
    marker_hits = [(prompt.find(marker), marker) for marker in GSS_INSERTION_MARKERS if prompt.find(marker) >= 0]
    if marker_hits:
        start, marker = min(marker_hits, key=lambda item: item[0])
        next_section = _next_section_after(prompt, start + len(marker))
        if next_section is not None:
            return next_section, marker
        return _sentence_end_after(prompt, start + len(marker)), marker
    requirements = prompt.find("Requirements:")
    if requirements >= 0:
        return requirements, "before Requirements:"
    task = prompt.find("Your task:")
    if task >= 0:
        return task, "before Your task:"
    return len(prompt), "end of prompt"


def _next_section_after(text: str, start: int) -> int | None:
    candidates = []
    for marker in ("Your task:", "Requirements:", "Note that"):
        index = text.find(marker, start)
        if index >= 0:
            candidates.append(index)
    return min(candidates) if candidates else None


def _sentence_end_after(text: str, start: int) -> int:
    for index in range(start, len(text)):
        if text[index] in ".\n":
            return index + 1
    return len(text)


def render_prompt_diff(original_prompt: str, enhanced_prompt: str) -> str:
    diff = difflib.unified_diff(
        original_prompt.splitlines(),
        enhanced_prompt.splitlines(),
        fromfile="original_prompt",
        tofile="enhanced_prompt",
        lineterm="",
    )
    return "\n".join(diff) + "\n"
