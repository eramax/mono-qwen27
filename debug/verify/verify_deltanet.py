# DeltaNet verification script
# Compare our k_deltanet kernel logic against a Python reference computation
# Uses actual data from the GPU debug dump

import struct, math

def delta_net_ref(q, k, v, gate, beta, state, hk, ng, dr, hv):
    """Reference DeltaNet computation matching reference's gated_delta_net_cuda kernel.
    
    Args:
        q: Q vector [ng * hk] = [16 * 128] = [2048]  (grouped keys)
        k: K vector [ng * hk] = [16 * 128] = [2048]
        v: V vector [dr * hv] = [48 * 128] = [6144]  (per rank values)
        gate: alpha gate [dr] = [48]  (dt_rank per rank)
        beta: beta per rank [dr] = [48]
        state: initial state [dr * hv * hk] = [48 * 128 * 128] = [786432]
        hk: head_k dimension = 128
        ng: number of Q/K groups = 16
        dr: number of ranks = 48
        hv: head_v dimension = 128
    
    Returns:
        output: [dr * hv] = [6144]  (attention output)
        new_state: [dr * hv * hk] = [786432]  (updated state)
    """
    output = []
    
    for r_idx in range(dr):
        g_idx = r_idx // (dr // ng)
        
        # gate for this rank
        g_raw = gate[r_idx]
        gv = math.exp(g_raw)
        bv = beta[r_idx]
        
        # Q and K for this group
        qg = q[g_idx * hk : (g_idx + 1) * hk]
        kg = k[g_idx * hk : (g_idx + 1) * hk]
        
        for col in range(hv):
            # V for this rank, column
            vr = v[r_idx * hv + col]
            
            # State for this (r_idx, col): S[i] = state[r_idx][col][i]
            # state layout: [r_idx * hv * hk + col * hk + i]
            S_base = r_idx * hv * hk + col * hk
            
            # kv = sum_i S[i] * kg[i]
            kv = 0.0
            for i in range(hk):
                kv += state[S_base + i] * kg[i]
            
            # delta = (v - gv * kv) * bv
            delta = (vr - gv * kv) * bv
            
            # Update state and compute attention
            attn = 0.0
            for i in range(hk):
                sn = gv * state[S_base + i] + kg[i] * delta
                state[S_base + i] = sn
                attn += sn * qg[i]
            
            # Output = attn / sqrt(hv)
            output.append(attn / math.sqrt(hv))
    
    return output, state

# Test with actual data from GPU debug
# We need the actual v, q, k, gate, beta values from the layer 0 computation

# Read debug file and find the specific values we need
debug_path = '/tmp/mono_v2.debug.tsv'
with open(debug_path, 'r') as f:
    content = f.read()

# Extract all intermediate values for layer 0
data = {}
for line in content.split('\n'):
    if line.startswith('ssm\t0\t0\t44883\t'):
        parts = line.strip().split('\t')
        label = parts[4]
        n_elems = int(parts[5])
        values = [float(v) for v in parts[10].split(',')]
        data[label] = values

# The h2 (attn_norm) for conv simulation
h2 = data.get('attn_norm', [])
print(f"h2: {len(h2)} elements")

# The wqkv output for conv simulation
conv_raw = data.get('conv_raw', [])
conv = data.get('conv', [])
print(f"conv_raw: {len(conv_raw)} elements, first 8: {conv_raw[:8]}")
print(f"conv: {len(conv)} elements, first 8: {conv[:8]}")

# Split conv into Q, K, V
NG = 16
HK = 128
DR = 48
HV = 128
QK_DIM = NG * HK  # 2048
V_DIM = DR * HV  # 6144

# From the conv output (after SiLU):
# Q = conv[0:2048]
# K = conv[2048:4096]
# V = conv[4096:10240]

if len(conv) >= QK_DIM * 2 + V_DIM:
    q = conv[0:QK_DIM]
    k = conv[QK_DIM:2*QK_DIM]
    v = conv[2*QK_DIM:2*QK_DIM+V_DIM]
    
    # After L2 norm, Q and K should have each group normalized to 1.0
    # Let's check group norms
    print(f"\nQ norms before L2:")
    for g in range(min(4, NG)):
        g_start = g * HK
        sq = sum(q[g_start + i]**2 for i in range(HK))
        print(f"  Group {g}: L2^2={sq:.6f}, after L2={1.0 if abs(sq) > 1e-12 else 'NAN'}")
    
    # Gate and beta from debug
    gate = data.get('dt', [])
    beta = data.get('beta', [])
    
    print(f"\ngate: {len(gate)} elements, first 8: {gate[:8]}")
    print(f"beta: {len(beta)} elements, first 8: {beta[:8]}")
    
    # NOTE: The debug conv values are the ACTUAL conv+SiLU values.
    # But in the real computation, they undergo L2 NORM for Q and K.
    # The debug dump shows the L2-normed Q/K values (since L2 norm acts in-place on sb).
    # So the 'conv' dump already contains L2-normed Q and K.
    
    # Now compute DeltaNet with reference formula
    # Initialize zero state
    state_size = DR * HV * HK  # 48 * 128 * 128 = 786432
    state = [0.0] * state_size
    
    ref_output, ref_state = delta_net_ref(q, k, v, gate, beta, state.copy(), 
                                          HK, NG, DR, HV)
    
    # Compare with GPU deltanet output from debug
    gpu_deltanet = data.get('deltanet', [])
    
    print(f"\nReference DeltaNet first 16: {[f'{v:.8f}' for v in ref_output[:16]]}")
    print(f"GPU deltanet first 16:       {[f'{v:.8f}' for v in gpu_deltanet[:16]]}")
    
    max_diff = max(abs(ref_output[i] - gpu_deltanet[i]) for i in range(len(ref_output)))
    print(f"\nMax diff: {max_diff:.10f}")
    print(f"MATCH: {'YES' if max_diff < 1e-4 else 'NO'}")
    
else:
    print(f"Not enough data: have {len(conv)} but need {QK_DIM * 2 + V_DIM}")
