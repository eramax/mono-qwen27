#!/usr/bin/env python3
"""Helpers for parsing mono27b and llama-style trace dumps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_FLOAT_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')


DebugKey = Tuple[str, int, int, int, str]


def parse_debug_tsv(path: str | Path) -> Dict[DebugKey, List[float]]:
    """Parse mono27b debug TSV into keyed tensor dumps.

    The file contains a mix of tensor dumps and pointer diagnostics. We only
    keep rows that look like full tensor dumps.
    """

    out: Dict[DebugKey, List[float]] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("phase\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            if not parts[1].lstrip("-").isdigit() or not parts[2].lstrip("-").isdigit() or not parts[3].lstrip("-").isdigit():
                continue
            if not parts[5].lstrip("-").isdigit():
                continue
            values = parts[10].strip()
            if not values:
                continue
            key: DebugKey = (parts[0], int(parts[1]), int(parts[2]), int(parts[3]), parts[4])
            out[key] = [float(v) for v in values.split(",") if v]
    return out


def get_debug_tensor(
    dumps: Dict[DebugKey, List[float]],
    phase: str,
    step: int,
    pos: int,
    tok: int,
    label: str,
) -> Optional[List[float]]:
    return dumps.get((phase, step, pos, tok, label))


def extract_ref_tensor(text: str, tensor_name: str, occurrence: int = 0) -> Optional[List[float]]:
    """Extract a tensor dump from llama-style debug text.

    The reference file has repeated tensor blocks. We return the `occurrence`-th
    matching block, where 0 means the first one.
    """

    pattern = re.compile(r"^common_debug_cb_eval:\s+" + re.escape(tensor_name) + r"\b.*$", re.M)
    matches = list(pattern.finditer(text))
    if occurrence < 0 or occurrence >= len(matches):
        return None

    start = matches[occurrence].end()
    end_match = re.search(r"^common_debug_cb_eval:\s+", text[start:], re.M)
    end = start + end_match.start() if end_match else len(text)
    block = text[start:end]

    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("sum ="):
            break
        lines.append(line)

    nums = _FLOAT_RE.findall("\n".join(lines))
    return [float(n) for n in nums] if nums else None


def extract_ref_tensor_occurrences(text: str, tensor_name: str) -> List[List[float]]:
    """Return all matching tensor dumps for a llama-style debug name."""

    pattern = re.compile(r"^common_debug_cb_eval:\s+" + re.escape(tensor_name) + r"\b.*$", re.M)
    matches = list(pattern.finditer(text))
    out: List[List[float]] = []
    for match in matches:
        start = match.end()
        end_match = re.search(r"^common_debug_cb_eval:\s+", text[start:], re.M)
        end = start + end_match.start() if end_match else len(text)
        block = text[start:end]
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("sum ="):
                break
            lines.append(line)
        nums = _FLOAT_RE.findall("\n".join(lines))
        if nums:
            out.append([float(n) for n in nums])
    return out


def max_abs_diff(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return max(abs(a[i] - b[i]) for i in range(n))
