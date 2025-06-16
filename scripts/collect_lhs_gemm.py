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
N_PROGRAMS = 100
random.seed(42)
features = ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages']
block_size = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))

param_values = [block_size, block_size, block_size, warp_size, stage_size]
all_combinations = list(itertools.product(*param_values))
sampled_configs = random.sample(all_combinations, 50)
config_list = [triton.Config({
        'BLOCK_SIZE_M': ex[0],
        'BLOCK_SIZE_N': ex[1],
        'BLOCK_SIZE_K': ex[2],
        'GROUP_SIZE_M': 8},
        num_warps= ex[3],
        num_stages= ex[4]) for ex in sampled_configs]




sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_PROGRAMS)

l_bounds = [ 1, 1, 1]
u_bounds = [ 8192, 8192, 8192]

sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
features = ['M', 'N', 'K']
final_samples = []
cont_idx = 0
for j in range(N_PROGRAMS):
    sample_row = []
    for i in range(len(features)):
        val = sample_scaled[j, cont_idx]
        sample_row.append(int(math.ceil(val)))  # or floor, round as appropriate
        cont_idx += 1
    cont_idx = 0
    final_samples.append(sample_row)

index = 0
for ex in final_samples:
    a = torch.randn((ex[0], ex[2]), device=DEVICE, dtype=torch.float16)
    b = torch.randn((ex[2], ex[1]), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, config_list), quantiles=quantiles)
    print(f"It took {ms}, {min_ms}, {max_ms}")
