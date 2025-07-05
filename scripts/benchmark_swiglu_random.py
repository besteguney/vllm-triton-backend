import torch
import triton
import triton.language as tl
from triton import Config
import random
import triton_dejavu
from triton_swiglu import fused_silu_and_mul_cfg
from lhs import LatinHypercubeSampler
import itertools

random.seed(71)
random.seed(71)

# Problem dimensions
# heads = range(16, 2**14+1)
# seqlen = range(16, 1024)
# heads = range(16, 2**14+1)
# seqlen = range(16, 1024)
# max_values = [0.01, 0.1, 1.0]
# batch = range(1, 128)
# batch = range(1, 128)
# block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# num_warps = [2**i for i in range(6)]
# num_stages = [i for i in range(6)]


heads = [2**i for i in range(4,15)]
seqlen = [2**i for i in range(4,11)]
max_values = [0.01, 0.1, 1.0]
batch = [2**i for i in range(9)]
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_warps = [2**i for i in range(6)]
num_stages = [i for i in range(6)]


heads = [2**i for i in range(4,15)]
seqlen = [2**i for i in range(4,11)]
max_values = [0.01, 0.1, 1.0]
batch = [2**i for i in range(9)]
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_warps = [2**i for i in range(6)]
num_stages = [i for i in range(6)]

# all_problems = list(itertools.product(heads, seqlen, max_values, batch))
all_combinations = list(itertools.product(block_sizes, num_warps, num_stages))

def swiglu_random_sampler():
    sampled_configurations = random.sample(all_combinations, 10)
    configurations = []
    for sample in sampled_configurations:
        configurations.append(triton.Config({'BLOCK_SIZE': sample[0]}, num_stages=sample[2],
                    num_warps=sample[1]))
        
    return configurations


# print(final_samples)
for i in range(50):
    seq = random.sample(seqlen, 1)
    b = random.sample(batch, 1)
    num_tokens = seq[0] * b[0]
    d = random.sample(heads, 1)
    d = d[0]
    max_value = random.sample(max_values, 1)
    max_value = max_value[0]
    configs = swiglu_random_sampler()
    try:
        x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda').uniform_(-1 * max_value, max_value)
    except RuntimeError as e:
        print(f"Could not allocated the size because of {e}")
        continue
    fused_silu_and_mul_cfg(x, configs)
    del x

    # a = torch.randn((ex['m'], ex['k']), device=DEVICE, dtype=torch.float16)
    # b = torch.randn((ex['k'], ex['n']), device=DEVICE, dtype=torch.float16)
    # quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, ex['cfgs']), quantiles=quantiles)
    # print(f"It took {ms}, {min_ms}, {max_ms}")
