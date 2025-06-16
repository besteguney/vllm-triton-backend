import sys
import os
import numpy as np
import torch

import triton
import triton.language as tl
import random
import itertools
from triton_gemm import matmul 

import pandas as pd

DEVICE = 'cuda'
random.seed(71)
df = pd.read_csv('gemm_data_v100.csv')

# Group by 'm', 'n', 'k' and count appearances
grouped = df.groupby(['M', 'N', 'K']).size().reset_index(name='count')

# Convert to list of tuples
result = list(grouped.itertuples(index=False, name=None))

# configurations = [
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         # Good config for fp8 inputs.
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4)
#     ]

group_size = [1,2,4,8,16]
block_size_m = [16, 32, 64, 128, 256]
block_size_n = [16, 32, 64, 128, 256]
block_size_k = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))

configurations = []
for i in range(100):
    config =  triton.Config({
        'BLOCK_SIZE_M': random.choice(block_size_m),
        'BLOCK_SIZE_N': random.choice(block_size_n),
        'BLOCK_SIZE_K': random.choice(block_size_k),
        'GROUP_SIZE_M': random.choice(group_size)},
        num_warps= random.choice(warp_size),
        num_stages= random.choice(stage_size))
    configurations.append(config)


for m, n, k, count in result:
    print("Starting to create the matrices")
    a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
    b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
    assert a.is_cuda, "Matrix a is not on gpu"
    assert b.is_cuda, "Matrix b is not on gpu"
    print(f"Starting to matrix multiplication with config count {len(configurations)}")
    triton_output = matmul(a, b, configurations)
    print("Ending matrix multiplication")
    # print(f"m={m}, n={n}, k={k}")