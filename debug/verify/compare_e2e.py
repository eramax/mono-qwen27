#!/usr/bin/env python3
"""Compare reference logits vs our logits, print E2E comparison table."""
import struct, math, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ref_bin = os.path.join(SCRIPT_DIR, 'ref_logits.bin')
our_bin = os.path.join(SCRIPT_DIR, 'our_logits.bin')

if not os.path.exists(ref_bin) or not os.path.exists(our_bin):
    print("Error: binary logit files not found. Run 'make e2e' first.")
    sys.exit(1)

with open(ref_bin, 'rb') as f:
    ref = struct.unpack(f'<248320f', f.read(248320*4))
with open(our_bin, 'rb') as f:
    our = struct.unpack(f'<248320f', f.read(248320*4))

ref_top5 = sorted([(ref[i], i) for i in range(248320)], reverse=True)[:5]
our_top5 = sorted([(our[i], i) for i in range(248320)], reverse=True)[:5]

n = 248320
rm = sum(ref)/n
om = sum(our)/n
corr = sum((ref[i]-rm)*(our[i]-om) for i in range(n))
corr /= math.sqrt(sum((v-rm)**2 for v in ref) * sum((v-om)**2 for v in our))
mse = sum((ref[i]-our[i])**2 for i in range(n))/n

print()
print('=' * 62)
print('  E2E LOGIT COMPARISON')
print('=' * 62)
print(f'  {"Token":>8} {"Ref Logit":>10} {"Our Logit":>10} {"Diff":>10}')
print('  ' + '-' * 42)
all_tokens = set([i for _,i in ref_top5] + [i for _,i in our_top5] + [728, 11, 220, 10074])
for tok in sorted(all_tokens):
    print(f'  {tok:>8} {ref[tok]:>10.2f} {our[tok]:>10.2f} {ref[tok]-our[tok]:>10.2f}')
print('  ' + '-' * 42)
print(f'  Reference top5:')
for v,i in ref_top5:
    print(f'    token {i:>6d}  logit {v:>8.2f}')
print(f'  Our top5:')
for v,i in our_top5:
    print(f'    token {i:>6d}  logit {v:>8.2f}')
print(f'  Correlation: {corr:.6f}  (1.0 = perfect)')
print(f'  MSE:         {mse:.4f}')
print('=' * 62)
