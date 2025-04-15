import torch
from triton_swiglu import fused_silu_and_mul


tokens = [2**i for i in range(4,14)]
heads = [2**i for i in range(4, 14)]

for token in tokens:
    for d in heads:
        input = torch.randn(token, 2 * d, device="cuda", dtype=torch.float16)
        fused_silu_and_mul(input)
