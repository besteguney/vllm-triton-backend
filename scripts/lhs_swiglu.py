import torch
import triton
import triton.language as tl
from triton import Config
import random
import triton_dejavu
from triton_swiglu import fused_silu_and_mul_cfg
from lhs import LatinHypercubeSampler

random.seed(0)

# Problem dimensions
# heads = range(16, 2**14+1)
# seqlen = range(16, 1024)
# max_values = [0.01, 0.1, 1.0]
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



def swiglu_lhs_sampler(n_samples_prob=10, n_samples_cfg=10, n_samples=10, is_combined=False):
    if is_combined:
        search_dict = {
            'd': heads,
            'seqlen': seqlen,
            'max_vals': max_values,
            'batch_size': batch,
            'block_size': block_sizes,
            'num_warps': num_warps,
            'num_stages': num_stages,
        }
        lhs = LatinHypercubeSampler(search_dict)
        samples = lhs.generate_new_categorical_samples(n_samples)
        return samples
    else:
        ## Sampling in the problem size dimension
        search_dict_prob =  {
            'd': heads,
            'seqlen': seqlen,
            'max_vals': max_values,
            'batch_size': batch,
        }
        lhs = LatinHypercubeSampler(search_dict_prob)
        samples_prob = lhs.generate_new_categorical_samples(n_samples_prob)

        search_dict_cfg =  {
            'block_size': block_sizes,
            'num_warps': num_warps,
            'num_stages': num_stages,
        }

        lhs = LatinHypercubeSampler(search_dict_cfg)
        samples_cfg = lhs.generate_new_categorical_samples(n_samples_cfg)
        samples = []
        for s in samples_prob:
            sample = {**s}
            sample['cfgs'] = []
            for cfg in samples_cfg:
                sample['cfgs'].append(triton.Config({'BLOCK_SIZE': cfg['block_size']}, 
                                                    num_stages=cfg['num_stages'],
                                                    num_warps=cfg['num_warps']))
            samples.append(sample)
        return samples

final_samples = swiglu_lhs_sampler(50, 10)
# print(final_samples)
for ex in final_samples:
    num_tokens = ex['seqlen'] * ex['batch_size']
    d = ex['d']
    max_value = ex['max_vals']
    try:
        x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda').uniform_(-1 * max_value, max_value)
    except RuntimeError as e:
        print(f"Could not allocated the size because of {e}")
        continue
    fused_silu_and_mul_cfg(x, ex['cfgs'])
    del x
    # a = torch.randn((ex['m'], ex['k']), device=DEVICE, dtype=torch.float16)
    # b = torch.randn((ex['k'], ex['n']), device=DEVICE, dtype=torch.float16)
    # quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, ex['cfgs']), quantiles=quantiles)
    # print(f"It took {ms}, {min_ms}, {max_ms}")
