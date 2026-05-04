#!/usr/bin/env python3
"""Compare local mono27b debug output with the checked-in reference trace.

This is self-contained: it only uses the repo fixtures, not a live llama.cpp
build.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from trace_utils import extract_ref_tensor, get_debug_tensor, parse_debug_tsv


def _summarize(label: str, values: list[float] | None) -> str:
    if not values:
        return "NOT FOUND"
    return (
        f"{len(values)} vals, first 16: {[f'{v:.6f}' for v in values[:16]]}\n"
        f"  min={min(values):.6f} max={max(values):.6f} "
        f"L2={math.sqrt(sum(v * v for v in values)):.6f}"
    )


if __name__ == "__main__":
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("debug/verify/ref_intermediates.txt")
    gpu_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/mono_fresh.debug.tsv")

    ref_text = ref_path.read_text(encoding="utf-8", errors="replace")
    gpu_data = parse_debug_tsv(gpu_path)

    print(
        "Note: this compares a debug-path trace file against the reference trace file.\n"
        "Use run_all_checks.py for the component parity gates and compare_e2e.py for logits."
    )

    comparisons = [
        ("model.input_embed", ("embed", "h"), "Embedding"),
        ("attn_norm-0", ("ssm", "attn_norm"), "RMS norm"),
        ("conv_output_silu-0", ("ssm", "conv"), "SSM conv+SiLU"),
        ("attn_output-0", ("ssm", "deltanet"), "DeltaNet output"),
        ("final_output-0", ("ssm", "layer_out"), "Final reshape/output"),
        ("linear_attn_out-0", ("ssm", "ssm_out"), "SSM output projection"),
        ("l_out-0", ("ssm", "post_ffn"), "Layer output"),
    ]

    for ref_name, (phase, gpu_name), desc in comparisons:
        ref_vals = extract_ref_tensor(ref_text, ref_name)
        gpu_vals = None
        for key, vals in gpu_data.items():
            if key[0] == phase and key[4] == gpu_name and key[1] == 0 and key[2] == 0 and key[3] == 44883:
                gpu_vals = vals
                break

        print(f"\n=== {desc} ===")
        print(f"Ref tensor: {ref_name}")
        print(f"  Ref: {_summarize(ref_name, ref_vals)}")
        print(f"  GPU: {_summarize(gpu_name, gpu_vals)}")

        if ref_vals and gpu_vals:
            if len(ref_vals) != len(gpu_vals):
                print(f"  SIZE MISMATCH: ref={len(ref_vals)} gpu={len(gpu_vals)}")
            else:
                max_diff = max(abs(ref_vals[i] - gpu_vals[i]) for i in range(len(ref_vals)))
                print(f"  Max diff: {max_diff:.6f}")

    gpu_logits = get_debug_tensor(gpu_data, "out", -1, 0, 44883, "logits")
    if gpu_logits:
        print("\n=== GPU Logits ===")
        print(f"First 16: {[f'{v:.4f}' for v in gpu_logits[:16]]}")
        top1 = max(range(len(gpu_logits)), key=lambda i: gpu_logits[i])
        print(f"Top1: {top1} (logit={gpu_logits[top1]:.4f})")
