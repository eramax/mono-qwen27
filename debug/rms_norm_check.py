"""Check RMS norm computation by verifying attn_norm output against embedding."""
import math

# From debug output
emb = [0.015738368, 0.00514060259, 0.0192709565, 0.0334013104, 
       -0.0195875168, 0.00867319107, 0.015738368, -0.00898975134,
       0.0192709565, 0.00160801411, -0.00192457438, -0.00898975134,
       -0.00545716286, 0.00514060259, 0.015738368, 0.00160801411]

attn_norm = [1.27643144, 0.374890596, 1.38591099, 2.44649267,
             -1.37789428, 0.61596179, 1.28732598, -0.619606495,
             1.38555038, 0.11215391, -0.141722396, -0.638443589,
             -0.390421271, 0.357579082, 1.16954684, 0.114921659]

# Compute RMS from the full L2 norm
l2 = 0.930995183
n = 5120
sum_sq = l2 * l2
mean_sq = sum_sq / n
inv_rms = 1.0 / math.sqrt(mean_sq + 1e-6)

# The output should be: out[i] = emb[i] * inv_rms * weight[i]
# So weight[i] = out[i] / (emb[i] * inv_rms)
print("Verifying RMS norm output:")
print(f"inv_rms = {inv_rms}")
for i in range(16):
    expected_no_weight = emb[i] * inv_rms
    implied_weight = attn_norm[i] / expected_no_weight
    print(f"  [{i}]: emb={emb[i]:.8f}, expected_no_w={expected_no_weight:.6f}, actual={attn_norm[i]:.8f}, implied_w={implied_weight:.6f}")
