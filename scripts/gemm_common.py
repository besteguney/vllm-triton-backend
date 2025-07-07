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

problems = [[1, 16, 256], [512, 4096, 16], [16, 4, 2],
             [128, 64, 16], [3351, 7816, 7254], [8087, 5459, 3997], [8192, 64, 4096],
               [256, 2048, 512], [8, 256, 4], [5797, 7114, 5181], [1529, 6535, 19], [7340, 1495, 1319],
                 [5329, 1032, 3131], [4, 16, 2], [4268, 1021, 231], [5410, 7736, 1650], [128, 32, 512], 
                 [5244, 8017, 1787], [3633, 3910, 2335], [2048, 2, 1024]]

configurations = [(32, 128, 16, 8, 2, 1), (32, 16, 32, 1, 8, 1), (32, 16, 32, 16, 2, 5), 
                  (256, 128, 32, 1, 16, 5), (32, 16, 32, 4, 4, 6), (16, 16, 256, 4, 8, 0), (128, 64, 32, 8, 2, 7), 
                  (128, 128, 64, 8, 2, 5), (32, 64, 32, 4, 8, 1), (64, 128, 64, 1, 2, 6), (16, 32, 256, 4, 1, 7), (16, 64, 64, 2, 2, 7),
                    (128, 256, 16, 8, 4, 0), (256, 32, 64, 4, 2, 1), (16, 128, 64, 8, 32, 3), (256, 256, 128, 8, 8, 2), (16, 256, 16, 4, 1, 4),
                      (16, 32, 16, 2, 4, 2), (256, 128, 256, 1, 1, 5), (16, 64, 16, 16, 16, 5), (128, 256, 128, 4, 4, 3), (256, 128, 64, 2, 2, 0), 
                      (128, 16, 32, 16, 1, 2), (128, 256, 32, 2, 8, 4), (64, 32, 16, 4, 1, 5), (256, 128, 128, 2, 32, 7), (256, 16, 16, 2, 4, 6),
                        (64, 256, 128, 8, 1, 5), (32, 64, 128, 8, 1, 7), (64, 256, 128, 2, 2, 2), (256, 64, 256, 16, 1, 3), (32, 32, 128, 1, 16, 4),
                          (256, 128, 256, 16, 32, 3), (32, 32, 16, 2, 32, 3), (128, 256, 16, 8, 32, 1), (128, 32, 128, 4, 8, 6), (256, 128, 32, 8, 4, 2), 
                          (64, 32, 256, 2, 2, 7), (128, 32, 64, 1, 16, 3), (32, 64, 128, 4, 32, 2), (64, 64, 128, 4, 8, 4), (64, 128, 256, 2, 2, 7), 
                          (128, 128, 128, 1, 16, 2), (128, 128, 16, 4, 16, 2), (128, 256, 128, 8, 1, 0), (256, 64, 64, 2, 8, 3), (64, 16, 16, 8, 32, 6), 
                          (16, 64, 32, 2, 2, 3), (32, 256, 16, 2, 4, 3), (16, 128, 16, 8, 16, 3)]

triton_configs = []
for sample in configurations:
    triton_configs.append(triton.Config({'BLOCK_SIZE_M': sample[0], 'BLOCK_SIZE_N': sample[1], 'BLOCK_SIZE_K': sample[2], 'GROUP_SIZE_M': sample[3]}, num_stages=sample[5],
                            num_warps=sample[4]))

for prob in problems:
    m = prob[0]
    n = prob[1]
    k = prob[2]
    print("Starting to create the matrices")
    try:
        a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
        b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
    except RuntimeError as e:
        print(f"Tensor failed {e}")
        a, b = None, None
    if a is None or b is None:
        continue
    assert a.is_cuda, "Matrix a is not on gpu"
    assert b.is_cuda, "Matrix b is not on gpu"
    print("Matrices are created")
    triton_output = matmul(a, b, triton_configs)
    print("Ending matrix multiplication")
    del a, b, triton_output