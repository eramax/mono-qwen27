#!/usr/bin/env python3
"""Verify the qwen35 text M-RoPE path against the expected sectioned rotation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from trace_utils import get_debug_tensor, parse_debug_tsv


ROPE_THETA = 10_000_000.0
N_ROT_DIMS = 64
ROPE_SECTIONS = (11, 11, 10, 0)


def apply_text_mrope(head: list[float], pos: int) -> list[float]:
    if len(head) < N_ROT_DIMS:
        raise ValueError(f"need at least {N_ROT_DIMS} values, got {len(head)}")

    out = head[:N_ROT_DIMS]
    sec0, sec1, sec2, sec3 = ROPE_SECTIONS
    sec01 = sec0 + sec1
    sec012 = sec01 + sec2
    half = N_ROT_DIMS // 2

    for p in range(half):
        if p < sec0:
            stream_pos = pos
        elif p < sec01:
            stream_pos = pos
        elif p < sec012:
            stream_pos = pos
        else:
            stream_pos = 0

        theta = ROPE_THETA ** (-2.0 * float(p) / float(N_ROT_DIMS))
        c = math.cos(theta * float(stream_pos))
        s = math.sin(theta * float(stream_pos))
        i0 = p
        i1 = p + half
        v0 = out[i0]
        v1 = out[i1]
        out[i0] = v0 * c - v1 * s
        out[i1] = v0 * s + v1 * c

    return out


def main() -> int:
    if len(sys.argv) > 2:
        debug_path = Path(sys.argv[2])
    elif len(sys.argv) > 1:
        debug_path = Path(sys.argv[1])
    else:
        debug_path = Path("/tmp/mono_verify.debug.tsv")
    dumps = parse_debug_tsv(debug_path)

    checked = 0
    worst = 0.0

    for key, q_rope in dumps.items():
        phase, step, pos, tok, label = key
        if phase != "attn" or label != "q_rope":
            continue
        q_norm = get_debug_tensor(dumps, phase, step, pos, tok, "q_norm")
        if not q_norm:
            q_norm = get_debug_tensor(dumps, phase, step, pos, tok, "q_proj")
        if not q_norm:
            continue

        expected = apply_text_mrope(q_norm, pos)
        n = min(len(q_rope), len(expected))
        diff = max(abs(q_rope[i] - expected[i]) for i in range(n))
        worst = max(worst, diff)
        checked += 1

        print(
            f"layer={step} pos={pos} tok={tok} "
            f"max_diff={diff:.3e} "
            f"first8_ref={[f'{v:.6f}' for v in expected[:8]]} "
            f"first8_got={[f'{v:.6f}' for v in q_rope[:8]]}"
        )

    if checked == 0:
        print("no q_rope/q_proj dumps found")
        return 2

    print(f"checked={checked} worst_diff={worst:.3e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
