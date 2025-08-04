from lhs import LatinHypercubeSampler

import torch

import triton
import triton.language as tl
import triton_dejavu
import random
import itertools
from collections import defaultdict
from triton_gemm import matmul 


# ## GEMM Search Space Dimensions
seed = 0
# sequence_length = [16, 32, 64, 128, 512, 1024, 2048, 4096]
block_m = [16, 32, 64, 128, 256, 512]
block_n = [16, 32, 64, 128, 256, 512]
warp_size = [2, 4, 8]
stage_size = [1, 2, 4, 6, 8]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SEQUENCE_LENGTHS = [16, 32, 64, 128, 512, 1024, 2048, 4096]
PROMPT_PATTERNS = [0,1]
PREFIX_PREFILL_SHARE_OF_DECODE = [0.0, 0.5]
# PREFIX_PREFILL_SHARE_OF_DECODE = [0.5]
PREFIX_PREFILL_SHARE_OF_PARTIAL_PREFILL = [0.0, 0.5]

def attention_lhs_sampler(n_samples_prob=10, n_samples_cfg=10, n_samples=10, is_combined=False):
    
    ## Sampling in the problem size dimension
    search_dict_prob = {
        # 'max_seq_q': unique_values['max_seq_q'],
        # 'max_seq_k': unique_values['max_seq_k'],
        # 'avg_seq_q': unique_values['avg_seq_q'],
        # 'avg_seq_k': unique_values['avg_seq_k']
        'batch_size':BATCH_SIZES,
        'seq_len':SEQUENCE_LENGTHS,
        'prompt':PROMPT_PATTERNS,
        'prefix_prefill':PREFIX_PREFILL_SHARE_OF_DECODE,
        'partial_prefill':PREFIX_PREFILL_SHARE_OF_PARTIAL_PREFILL
    }
    lhs = LatinHypercubeSampler(search_dict_prob, seed)
    samples_prob = lhs.generate_new_categorical_samples(n_samples_prob)

    for sample in samples_prob:
        # mask = (df[list(sample)] == pd.Series(sample)).all(axis=1)

        # # Filter the DataFrame
        # matching_rows = df[mask]
        # print(matching_rows.shape)
        # print("****")
        print(sample)

    search_dict_cfg = {
        'BLOCK_M': block_m,
        'BLOCK_N': block_n,
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
    return samples_prob

final_samples = attention_lhs_sampler(10, 17)
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
