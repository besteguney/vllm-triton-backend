import math
from scipy.stats import qmc

import torch
import triton
import triton.language as tl
import random
import itertools
from triton_gemm import matmul 


def round_to_closest_power_of_two(x):
    if x <= 0:
        return 0

    lower = 2 ** math.floor(math.log2(x))
    upper = 2 ** math.ceil(math.log2(x))

    return lower if abs(x - lower) < abs(x - upper) else upper
    # return 2 ** math.ceil(math.log2(x))

DEVICE="cuda"
N_SAMPLES = 1000
random.seed(42)
features = ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages', 'M', 'N', 'K']
block_size = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))

discrete_values = [block_size, block_size, block_size, warp_size, stage_size]
discrete_samples = []
for i, vals in enumerate(discrete_values):
    repeats = (N_SAMPLES + len(vals) - 1) // len(vals)  # ceil division
    repeated_vals = vals * repeats
    repeated_vals = repeated_vals[:N_SAMPLES]
    random.shuffle(repeated_vals)
    discrete_samples.append(repeated_vals)



sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_SAMPLES)

l_bounds = [ 1, 1, 1]
u_bounds = [ 8192, 8192, 8192]

sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

final_samples = []
cont_idx = 0
for j in range(N_SAMPLES):
    sample_row = []
    for i in range(len(features)):
        if i < 5:
            sample_row.append(discrete_samples[i][j])
        else:
            val = sample_scaled[j, cont_idx]
            sample_row.append(int(math.ceil(val)))  # or floor, round as appropriate
            cont_idx += 1
    cont_idx = 0
    final_samples.append(sample_row)
    


print(final_samples[0][5])
index = 0
for ex in final_samples:
    config =  triton.Config({
        'BLOCK_SIZE_M': ex[0],
        'BLOCK_SIZE_N': ex[1],
        'BLOCK_SIZE_K': ex[2],
        'GROUP_SIZE_M': 8},
        num_warps= ex[3],
        num_stages= ex[4])
    
    a = torch.randn((ex[5], ex[7]), device=DEVICE, dtype=torch.float16)
    b = torch.randn((ex[7], ex[6]), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, [config, config]), quantiles=quantiles)
    print(f"It took {ms}, {min_ms}, {max_ms}")
