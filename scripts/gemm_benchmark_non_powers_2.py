import sys
import os
import numpy as np
import torch

import triton
import triton.language as tl
import triton_dejavu
import random
import itertools
from triton_gemm import matmul 


DEVICE = "cuda"

dimensions = [257, 300, 333, 384, 411, 512, 555, 600, 701, 768,
 803, 900, 1024, 1111, 1200, 1303, 1400, 1537, 1600,
 1729, 1801, 1920, 2048, 2177, 2305, 2401, 2561, 2703,
 2880, 3073, 3201, 3457, 3600, 3841, 4096, 4353, 4609,
 4800, 5121, 5377, 5760, 6145, 6667, 7001, 7681, 8001, 8191]


all_dim_combinations = list(itertools.product(dimensions, repeat=3))

# Randomly sample 100 unique combinations
random.seed(42)  # for reproducibility
sampled_dim_combinations = random.sample(all_dim_combinations, 50)

group_size = [1,2,4,8,16]
block_size_m = [16, 32, 64, 128, 256]
block_size_n = [16, 32, 64, 128, 256]
block_size_k = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))

configurations = []
all_combinations = list(itertools.product(block_size_m, block_size_n, block_size_k, group_size, warp_size, stage_size))
all_combinations_filtered = [
    combo for combo in all_combinations
    if not ((combo[1] > 128 and combo[2] > 128)) 
]

for m, n, k in sampled_dim_combinations:
    print("Starting to create the matrices")
    a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
    b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
    assert a.is_cuda, "Matrix a is not on gpu"
    assert b.is_cuda, "Matrix b is not on gpu"
    print("Matrices are created")
    random.shuffle(all_combinations_filtered)
    if os.getenv("CONFIG_COUNT", "0") != "0":
        sampled_configurations = random.sample(all_combinations_filtered, min(int(os.getenv("CONFIG_COUNT", "0")), len(all_combinations_filtered)))
        for sample in sampled_configurations:
            configurations.append(triton.Config({'BLOCK_SIZE_M': sample[0], 'BLOCK_SIZE_N': sample[1], 'BLOCK_SIZE_K': sample[2], 'GROUP_SIZE_M': sample[3]}, num_stages=sample[5],
                        num_warps=sample[4]))
    else:
        for sample in all_combinations_filtered:
            configurations.append(triton.Config({'BLOCK_SIZE_M': sample[0], 'BLOCK_SIZE_N': sample[1], 'BLOCK_SIZE_K': sample[2], 'GROUP_SIZE_M': sample[3]}, num_stages=sample[5],
                            num_warps=sample[4]))
    print(f"Starting to matrix multiplication with config count {len(configurations)}")
    triton_output = matmul(a, b, configurations)
    print("Ending matrix multiplication")
    configurations = []
    # print(f"m={m}, n={n}, k={k}")