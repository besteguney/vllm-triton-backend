import sys
import os
import numpy as np
import torch

import triton
import triton.language as tl
import triton_dejavu
from triton_gemm import matmul 

DEVICE = "cuda"

dimensions = [2**i for i in range(8,14)]
for m in dimensions:
    for n in dimensions:
        for k in dimensions:
            a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
            b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
            triton_output = matmul(a, b)
            