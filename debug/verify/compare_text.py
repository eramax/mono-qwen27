#!/usr/bin/env python3
"""Compare generated text output between mono27b and llama.cpp reference.

Usage:
    python3 compare_text.py <our_text.txt> <ref_text.txt>

Reports:
    - EXACT: whether the full text matches exactly
    - First divergence position (character index)
    - Common prefix up to divergence point
    - Character-level edit distance (Levenshtein) as a rough similarity measure
"""

import os
import sys


def levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1, 1):
        curr_row = [i]
        for j, c2 in enumerate(s2, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def find_divergence(a, b):
    """Return index of first character where strings differ, or -1 if match."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_text.py <our_text> <ref_text>", file=sys.stderr)
        sys.exit(1)

    our_path = sys.argv[1]
    ref_path = sys.argv[2]

    if not os.path.exists(our_path):
        print(f"Error: our text file not found: {our_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ref_path):
        print(f"Error: ref text file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    with open(our_path, "r", encoding="utf-8", errors="replace") as f:
        our_text = f.read().rstrip("\n")
    with open(ref_path, "r", encoding="utf-8", errors="replace") as f:
        ref_text = f.read().rstrip("\n")

    exact = our_text == ref_text
    diverge_idx = find_divergence(our_text, ref_text)

    # Compute prefix match (whitespace-normalized)
    our_norm = " ".join(our_text.split())
    ref_norm = " ".join(ref_text.split())
    norm_diverge = find_divergence(our_norm, ref_norm)

    # Edit distance for similarity measure
    max_len = max(len(our_text), len(ref_text), 1)
    dist = levenshtein(our_text, ref_text)
    similarity = 1.0 - (dist / max_len)

    print()
    print("=" * 78)
    print("  E2E TEXT COMPARISON")
    print("=" * 78)
    print(f"  {'Exact match:':<28} {'YES' if exact else 'NO'}")
    print(f"  {'Similarity (1-LD/len):':<28} {similarity:.6f}")
    print(f"  {'Our text length:':<28} {len(our_text)} chars")
    print(f"  {'Ref text length:':<28} {len(ref_text)} chars")

    if diverge_idx >= 0:
        ctx = 60
        start = max(0, diverge_idx - ctx)
        our_snip = our_text[start:diverge_idx + ctx]
        ref_snip = ref_text[start:diverge_idx + ctx]
        # Replace non-printables for display
        def clean(s):
            return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        print(f"  {'First divergence:':<28} char {diverge_idx}")
        print(f"  {'  Our  context:':<28} ...{clean(our_snip)}...")
        print(f"  {'  Ref  context:':<28} ...{clean(ref_snip)}...")
        if norm_diverge >= 0:
            print(f"  {'Norm 1st diverge:':<28} char {norm_diverge}")
            start_n = max(0, norm_diverge - ctx)
            our_nsnip = our_norm[start_n:norm_diverge + ctx]
            ref_nsnip = ref_norm[start_n:norm_diverge + ctx]
            print(f"  {'  Our  norm ctx:':<28} ...{clean(our_nsnip)}...")
            print(f"  {'  Ref  norm ctx:':<28} ...{clean(ref_nsnip)}...")

    if not exact:
        common = our_text[:diverge_idx] if diverge_idx >= 0 else our_text
        print(f"\n  Common prefix ({len(common)} chars):")
        preview = common[-200:] if len(common) > 200 else common
        print(f"  ...{clean(preview)}")

    print()
    print("=" * 78)
    if exact:
        print("  Status: MATCH")
        ret = 0
    else:
        print("  Status: DIFFERS")
        ret = 1
    print("=" * 78)
    print()

    return ret


if __name__ == "__main__":
    sys.exit(main())
