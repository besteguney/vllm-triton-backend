from lhs import LatinHypercubeSampler

import torch

import triton
import triton.language as tl
import triton_dejavu
import random
import itertools
from collections import defaultdict
from triton_gemm import matmul 

DEVICE = 'cuda'
seed = 42

## GEMM Search Space Dimensions
sequence_length = [16, 32, 64, 128, 512, 1024, 2048, 4096]
block_m = [16, 32, 64, 128, 256, 512]
block_n = [16, 32, 64, 128, 256, 512]
warp_size = [2, 4, 8]
stage_size = [1, 2, 4, 6, 8]

def attention_lhs_sampler(n_samples_prob=10, n_samples_cfg=10, n_samples=10, is_combined=False):
    
    ## Sampling in the problem size dimension
    search_dict_prob = {
        'MAX_SEQ_Q': sequence_length,
        'MAX_SEQ_K': sequence_length,
        'AVG_SEQ_Q': sequence_length,
        'AVG_SEQ_K': sequence_length
    }
    lhs = LatinHypercubeSampler(search_dict_prob, seed)
    samples_prob = lhs.generate_new_categorical_samples(n_samples_prob)

    print(samples_prob)
    search_dict_cfg = {
        'block_m': block_m,
        'block_n': block_n,
        'num_warps': warp_size,
        'num_stages': stage_size
    }

    lhs = LatinHypercubeSampler(search_dict_cfg, seed)
    samples_cfg = lhs.generate_new_categorical_samples(n_samples_cfg)
    print(samples_cfg)
    # samples = []
    # for s in samples_prob:
    #     sample = {**s}
    #     sample['cfgs'] = []
    #     for cfg in samples_cfg:
    #         sample['cfgs'].append(triton.Config({'BLOCK_M': cfg['block_m'], 'BLOCK_N': cfg['block_n']}, 
    #                                             num_stages=cfg['num_stages'],
    #                                             num_warps=cfg['num_warps']))
    #     samples.append(sample)
    return samples_cfg

final_samples = attention_lhs_sampler(80, 17)
# for ex in final_samples:
#     print(ex)

    # try:
    #     a = torch.randn((ex['m'], ex['k']), device=DEVICE, dtype=torch.float16)
    #     b = torch.randn((ex['k'], ex['n']), device=DEVICE, dtype=torch.float16)
    # except RuntimeError as e:
    #     print(f"Could not allocate because of {e}")
    #     continue
    # quantiles = [0.5, 0.2, 0.8]
    # matmul(a, b, ex['cfgs'])
    # del a, b
    # try: 
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, ex['cfgs']), quantiles=quantiles)
    # except RuntimeError as e:
    #     print(f"Coult not run the benchmark because of {e}")
    #     continue
    # del a,b
    # print(f"It took {ms}, {min_ms}, {max_ms}")
