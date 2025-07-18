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
random.seed(71)

## GEMM Search Space Dimensions
# problem_dimension = [2**i for i in range(15)]
problem_dimension = range(1, 8193)
block_sizes = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))
group_size = [1,2,4,8,16]

def gemm_lhs_sampler(n_samples_prob=10, n_samples_cfg=10, n_samples=10, is_combined=False):
    if is_combined:
        search_dict = {
            'm': problem_dimension,
            'n': problem_dimension,
            'k': problem_dimension,
            'block_size_m': block_sizes,
            'block_size_n': block_sizes,
            'block_size_k': block_sizes,
            'group_size_m': group_size,
            'num_warps': warp_size,
            'num_stages': stage_size
        }
        lhs = LatinHypercubeSampler(search_dict)
        samples_combined = lhs.generate_new_categorical_samples(n_samples)
        grouped = defaultdict(list)
        for d in samples_combined:
            key = (d['m'], d['n'], d['k'])
            # Remove the shared keys for cfg
            cfg = {k: v for k, v in d.items() if k not in ['m', 'n', 'k']}
            configuration = triton.Config({'BLOCK_SIZE_M': cfg['block_size_m'], 'BLOCK_SIZE_N': cfg['block_size_n'], 'BLOCK_SIZE_K': cfg['block_size_k'], 'GROUP_SIZE_M': cfg['group_size_m']}, 
                                                    num_stages=cfg['num_stages'],
                                                    num_warps=cfg['num_warps'])
            grouped[key].append(configuration)
            grouped[key].append(configuration)
        combined = []
        for (m, n, k), cfgs in grouped.items():
            combined.append({
                'm': m, 'n': n, 'k': k,
                'cfgs': cfgs
            })
        return combined
    else:
        ## Sampling in the problem size dimension
        search_dict_prob = {
            'm': problem_dimension,
            'n': problem_dimension,
            'k': problem_dimension
        }
        lhs = LatinHypercubeSampler(search_dict_prob)
        samples_prob = lhs.generate_new_categorical_samples(n_samples_prob)

        search_dict_cfg = {
            'block_size_m': block_sizes,
            'block_size_n': block_sizes,
            'block_size_k': block_sizes,
            'group_size_m': group_size,
            'num_warps': warp_size,
            'num_stages': stage_size
        }

        lhs = LatinHypercubeSampler(search_dict_cfg)
        samples_cfg = lhs.generate_new_categorical_samples(n_samples_cfg)
        samples = []
        for s in samples_prob:
            sample = {**s}
            sample['cfgs'] = []
            for cfg in samples_cfg:
                sample['cfgs'].append(triton.Config({'BLOCK_SIZE_M': cfg['block_size_m'], 'BLOCK_SIZE_N': cfg['block_size_n'], 'BLOCK_SIZE_K': cfg['block_size_k'], 'GROUP_SIZE_M': cfg['group_size_m']}, 
                                                    num_stages=cfg['num_stages'],
                                                    num_warps=cfg['num_warps']))
            samples.append(sample)
        return samples

final_samples = gemm_lhs_sampler(50, 10)
for ex in final_samples:
    try:
        a = torch.randn((ex['m'], ex['k']), device=DEVICE, dtype=torch.float16)
        b = torch.randn((ex['k'], ex['n']), device=DEVICE, dtype=torch.float16)
    except RuntimeError as e:
        print(f"Could not allocate because of {e}")
        continue
    quantiles = [0.5, 0.2, 0.8]

    try: 
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, ex['cfgs']), quantiles=quantiles)
    except RuntimeError as e:
        print(f"Coult not run the benchmark because of {e}")
        continue
    del a,b
    print(f"It took {ms}, {min_ms}, {max_ms}")
