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
# torch.manual_seed(0)

# torch_output = torch.matmul(a, b)
# print(f"triton_output_with_fp16_inputs={triton_output}")
# print(f"torch_output_with_fp16_inputs={torch_output}")
# # Bigger tolerance for AMD CDNA2 devices.
# # CDNA2 devices use reduced precision fp16 and bf16 and flush input and
# # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
# rtol = 0
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")