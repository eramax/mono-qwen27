#!/usr/bin/env python3
"""Parse mono27b timing output and compare with llama.cpp benchmark."""
import sys, re, subprocess, json, os

def parse_mono27b_timing(path):
    with open(path) as f:
        text = f.read()
    gen_m = re.search(r'Gen:\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tok\s+=\s+([\d.]+)\s+tok/s', text)
    gen_ms, gen_tok, gen_tps = (float(gen_m.group(1)), int(gen_m.group(2)), float(gen_m.group(3))) if gen_m else (0, 0, 0)
    prefill_m = re.search(r'Prefill:\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tok\s+=\s+([\d.]+)\s+tok/s', text)
    prefill_ms, prefill_tok, prefill_tps = (float(prefill_m.group(1)), int(prefill_m.group(2)), float(prefill_m.group(3))) if prefill_m else (0, 0, 0)

    m = re.search(r'Timing breakdown \((\d+) tokens, total ([\d.]+) ms/tok\) =+\n(.*?)={3,}', text, re.DOTALL)
    if not m:
        return None
    tokens = int(m.group(1))
    total_ms = float(m.group(2))
    section = m.group(3)
    rows = []
    for line in section.strip().split('\n')[1:]:
        parts = line.split()
        if len(parts) >= 4:
            rows.append((parts[0], float(parts[1]), float(parts[2].rstrip('%')), int(parts[3])))

    return {
        'prefill_ms': prefill_ms, 'prefill_tok': prefill_tok, 'prefill_tps': prefill_tps,
        'gen_ms': gen_ms, 'gen_tok': gen_tok, 'gen_tps': gen_tps,
        'tokens': tokens, 'total_ms': total_ms, 'rows': rows
    }

def run_llama_bench(model_path, n_prompt=4, n_gen=30):
    bench = '/mnt/data1/projects/llm/llama.cpp/build/bin/llama-bench'
    if not os.path.exists(bench):
        return None
    try:
        out = subprocess.run([bench, '-m', model_path, '-p', str(n_prompt), '-n', str(n_gen)],
                             capture_output=True, text=True, timeout=120)
        for line in out.stdout.split('\n'):
            if 'tg' in line and '±' in line:
                parts = line.split('|')
                if len(parts) >= 8:
                    tps_str = parts[7].strip().split()[0]
                    return float(tps_str)
    except Exception as e:
        print(f"llama-bench failed: {e}", file=sys.stderr)
    return None

def categorize(rows):
    categories = {
        'Attention Q/K/V/O': ['wq', 'wk', 'wv', 'wo', 'qkn'],
        'Attention RoPE+KV': ['mq', 'mk', 'kvc', 'gt'],
        'Attention norms': ['rms', 'res1'],
        'FFN gate+up': ['fg+fu'],
        'FFN swiglu+down': ['mul', 'fd'],
        'FFN norm': ['porm'],
        'SSM wqkv+gate': ['re-g', 'g_silu'],
        'SSM norms+mul': ['grms', 'gmul2'],
        'SSM state+out': ['ssmo', 'scp', 'sadd'],
        'Embed+head': ['emb', 'post-loop'],
        'Other': ['pre']
    }
    cat_times = {k: 0.0 for k in categories}
    for label, ms, pct, calls in rows:
        found = False
        for cat, labels in categories.items():
            if label in labels:
                cat_times[cat] += ms
                found = True
                break
        if not found:
            cat_times['Other'] += ms
    return cat_times

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <timing_output_file> [model_path]")
        sys.exit(1)

    data = parse_mono27b_timing(sys.argv[1])
    if not data:
        print("No timing data found")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else '/mnt/mydata/projects2/mono27b/model/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    llama_tps = run_llama_bench(model_path, n_gen=data['gen_tok'])

    print("=" * 70)
    print("MONO27B vs llama.cpp Performance Comparison")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Mono27B':>15} {'llama.cpp':>15} {'Gap':>10}")
    print("-" * 70)
    print(f"{'Token generation (tok/s)':<30} {data['gen_tps']:>15.2f} {llama_tps or 0:>15.2f} {(llama_tps/data['gen_tps'] if llama_tps and data['gen_tps'] else 0):>10.2f}x")
    print()

    cats = categorize(data['rows'])
    total = sum(cats.values())
    print("GPU Time Breakdown (Mono27B, ms/tok):")
    print(f"{'Category':<30} {'Mono27B':>12} {'%':>8} {'Est. llama':>12} {'Gap':>10}")
    print("-" * 70)
    ratio = llama_tps / data['gen_tps'] if llama_tps and data['gen_tps'] else 1.0
    for cat, ms in sorted(cats.items(), key=lambda x: -x[1]):
        pct = ms / total * 100
        est_llama = ms / ratio
        gap = ms - est_llama
        print(f"{cat:<30} {ms:>12.2f} {pct:>7.1f}% {est_llama:>12.2f} {gap:>10.2f}")
    print(f"{'Total GPU':<30} {total:>12.2f} {'100.0%':>8} {total/ratio:>12.2f} {total-total/ratio:>10.2f}")
    print()

    wall_ms = data['gen_ms'] / data['gen_tok'] if data['gen_tok'] else 0
    overhead = wall_ms - total
    print(f"Wall clock ms/tok:         {wall_ms:.2f}")
    print(f"GPU work ms/tok:           {total:.2f}")
    if wall_ms:
        print(f"CPU/overhead ms/tok:       {overhead:.2f}  ({overhead/wall_ms*100:.1f}%)")
    else:
        print(f"CPU/overhead ms/tok:       {overhead:.2f}  (N/A)")
    print()

    print("Top 10 Individual Kernels:")
    print(f"{'Kernel':<20} {'ms/tok':>10} {'%':>8} {'calls/tok':>10}")
    print("-" * 70)
    for label, ms, pct, calls in data['rows'][:10]:
        print(f"{label:<20} {ms:>10.3f} {pct:>7.1f}% {calls:>10}")
    print()

    print("=" * 70)
    print("Recommendations (based on largest gaps):")
    print("=" * 70)
    if cats.get('SSM state+out', 0) > 5:
        print(f"1. SSM output matvec (ssmo): {cats['SSM state+out']:.2f} ms/tok — largest single cost")
    if cats.get('SSM wqkv+gate', 0) > 5:
        print(f"2. SSM wqkv_gate matvec: {cats['SSM wqkv+gate']:.2f} ms/tok — 2nd largest")
    if cats.get('FFN gate+up', 0) > 2:
        print(f"3. FFN gate+up matvec: {cats['FFN gate+up']:.2f} ms/tok — fuse into single kernel")
    if cats.get('FFN swiglu+down', 0) > 1:
        print(f"4. FFN down matvec: {cats['FFN swiglu+down']:.2f} ms/tok — MMQ path for large output")
    if overhead / wall_ms > 0.2:
        print(f"5. CPU/overhead: {overhead:.2f} ms/tok ({overhead/wall_ms*100:.1f}%) — D2H copy + sampling")
    print()

if __name__ == '__main__':
    main()
