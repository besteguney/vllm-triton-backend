# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch
import triton
import triton.language as tl

import os
import triton_dejavu
import functools


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1

# not as lambda, for python3.9
def fallback_heuristic_dt2(key):
    tpa_test_q = key[1]
    tpa_test_k = key[2]
    # Model trained on max
    if tpa_test_q < 1024:
        BLOCK_M = 16
    else:
        BLOCK_M = 64

    if tpa_test_k < 64:
        if tpa_test_k < 32:
            BLOCK_N = 16
        else:
            BLOCK_N = 32
    else:
        if tpa_test_q < 256:
            BLOCK_N = 128
        else:
            BLOCK_N = 64
    ret = triton.Config(
        {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, 
        num_stages=4, num_warps=8)
    return ret


def informed_fallback_next(key, cache):
    # key[27] = MAX_SEQLENS_Q
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[2]))]
    return ret


def prepare_informed_fallback(cache):
    # key[2] = max k
    ret = {int(k[2]): c for k, c in cache.items()}
    return ret
    
# gpu_name = torch.cuda.get_device_name()
# # print(gpu_name)
# 
# 
# # @functools.lru_cache
# def prefill_heuristics_2d_BLOCK_M(MAX_SEQ_Q, MAX_SEQ_K):
#     if "NVIDIA H100" in gpu_name:
#         BLOCK_M = 16 if MAX_SEQ_Q <= 192 else 128
#     elif "AMD Instinct MI300" in gpu_name:
#         if MAX_SEQ_Q <= 384:
#             if MAX_SEQ_K > 384 and MAX_SEQ_Q > 192:
#                 BLOCK_M = 32
#             else:
#                 BLOCK_M = 16
#         else: 
#             BLOCK_M = 64
#     else:
#         BLOCK_M = 64 if MAX_SEQ_Q > 1 else 16
#     # print(f"MAX_SEQ_Q {MAX_SEQ_Q}, MAX_SEQ_K {MAX_SEQ_K}")
#     # print("BLOCK_M: ", BLOCK_M)
#     return BLOCK_M
# 
# 
# # @functools.lru_cache
# def prefill_heuristics_2d_BLOCK_N(MAX_SEQ_Q, MAX_SEQ_K):
#     if "NVIDIA H100" in gpu_name:
#         BLOCK_N = 32 if MAX_SEQ_K <= 192 else 128
#     elif "AMD Instinct MI300" in gpu_name:
#         if MAX_SEQ_Q <= 384:
#             if 96 < MAX_SEQ_K <= 192 and MAX_SEQ_Q <= 96:
#                 BLOCK_N = 128
#             elif MAX_SEQ_K > 384:
#                 BLOCK_N = 256
#             else:
#                 BLOCK_N = 32
#         else:
#             if MAX_SEQ_K <= 768:
#                 BLOCK_N = 16
#             else:
#                 BLOCK_N = 64
#     else:
#         BLOCK_N = 16 if MAX_SEQ_K < 128 else 64
#     # print("BLOCK_N: ", BLOCK_N)
#     return BLOCK_N
# 
# 
# # @functools.lru_cache
# def prefill_heuristics_2d_WARPS(MAX_SEQ_Q, MAX_SEQ_K):
#     if "NVIDIA H100" in gpu_name:
#         num_warps = 4 if MAX_SEQ_K <= 96 else 8
#     elif "AMD Instinct MI300" in gpu_name:
#         if MAX_SEQ_Q <= 384:
#             if 96 < MAX_SEQ_K <= 192 and MAX_SEQ_Q <= 96:
#                 num_warps = 8
#             else:
#                 num_warps = 4
#         else:
#             if MAX_SEQ_K <= 768:
#                 num_warps = 4
#             else:
#                 num_warps = 2
#     else:
#         num_warps = 4  # default
#     # print("num_warps: ", num_warps)
#     return num_warps 
# 
# 
# # @functools.lru_cache
# def prefill_heuristics_2d_STAGES(MAX_SEQ_Q, MAX_SEQ_K):
#     if "NVIDIA H100" in gpu_name:
#         if MAX_SEQ_K <= 96:
#             num_stages = 4
#         else:
#             if MAX_SEQ_Q <= 192:
#                 if MAX_SEQ_K <= 1536:
#                     num_stages = 2
#                 else:
#                     num_stages = 8
#             else:
#                 num_stages = 1
#     elif "AMD Instinct MI300" in gpu_name:
#         if MAX_SEQ_Q <= 192:
#             # if 96 < MAX_SEQ_K <= 192 and MAX_SEQ_Q <= 96:
#             if 192 < MAX_SEQ_K < 1536:
#                 num_stages = 2
#             elif MAX_SEQ_K >= 1536:
#                 num_stages = 1
#             else:
#                 num_stages = 4
#         else:
#             if MAX_SEQ_K <= 768:
#                 if MAX_SEQ_K > 384 and MAX_SEQ_Q <= 384:
#                     num_stages = 1
#                 else:
#                     num_stages = 4
#             else:
#                 num_stages = 1
# 
#     else:
#         num_stages = 3  # default
#     # print("num_stages: ", num_stages)
#     return num_stages


@functools.lru_cache
def prefill_heuristics_2d(MAX_SEQ_Q, MAX_SEQ_K):
    gpu_name = torch.cuda.get_device_name()
    # print(f"MAX_SEQ_Q {MAX_SEQ_Q}, MAX_SEQ_K {MAX_SEQ_K}")
    if "NVIDIA H100" in gpu_name:
        # # TPA original heuristic
        # if MAX_SEQ_Q < 1024:
        #     BLOCK_M = 16
        # else:
        #     BLOCK_M = 64
        # if MAX_SEQ_K < 64:
        #     if MAX_SEQ_K < 32:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 32
        # else:
        #     if MAX_SEQ_Q < 256:
        #         BLOCK_N = 128
        #     else:
        #         BLOCK_N = 64
        # config = {'num_stages': 3, 'num_warps': 4,
        #           'BLOCK_N': BLOCK_N, 'BLOCK_M': BLOCK_M}
        # dejavu with microbenchmarks
        if MAX_SEQ_K <= 96:
            config = {'num_stages' : 4, 'num_warps': 4, 
                      'BLOCK_N' : 32, 'BLOCK_M' : 16}
        else:
            if MAX_SEQ_Q <= 192:
                if MAX_SEQ_K <= 1536:
                    config = {'num_stages' : 2, 'num_warps': 8, 
                              'BLOCK_N' : 128, 'BLOCK_M' : 16}
                else:
                    config = {'num_stages' : 8, 'num_warps': 8, 
                              'BLOCK_N' : 128, 'BLOCK_M' : 16}
            else:
                config = {'num_stages' : 1, 'num_warps': 8, 
                          'BLOCK_N' : 128, 'BLOCK_M' : 128}
    elif "AMD Instinct MI300" in gpu_name:
        if MAX_SEQ_Q <= 384:
            if MAX_SEQ_K <= 96:
                config = {"num_stages": 4, "num_warps": 4, "BLOCK_N": 32, "BLOCK_M": 16}
            else:
                if MAX_SEQ_K <= 192:
                    if MAX_SEQ_Q <= 96:
                        config = {"num_stages": 2, "num_warps": 8, "BLOCK_N": 128, "BLOCK_M": 16}
                    else:
                        config = {"num_stages": 4, "num_warps": 4, "BLOCK_N": 32, "BLOCK_M": 16}
                else:
                    if MAX_SEQ_Q <= 128:
                        config = {"num_stages": 4, "num_warps": 4, "BLOCK_N": 32, "BLOCK_M": 16}
                    else:
                        if MAX_SEQ_K <= 384:
                            config = {"num_stages": 4, "num_warps": 4, "BLOCK_N": 32, "BLOCK_M": 16}
                        else:
                            config = { "num_stages": 1, "num_warps": 4, "BLOCK_N": 256, "BLOCK_M": 32}
        else:
            if MAX_SEQ_K <= 768:
                config = {"num_stages": 4, "num_warps": 4, "BLOCK_N": 16, "BLOCK_M": 64}
            else:
                config = {"num_stages": 1, "num_warps": 2, "BLOCK_N": 64, "BLOCK_M": 64}
    else:
        # default
        config = {
            'BLOCK_M': 64 if MAX_SEQ_Q > 1 else 16,
            'BLOCK_N': 16 if MAX_SEQ_K < 128 else 64,
            'num_warps': 4,
            'num_stages': 3,
        }
    # print(config)
    return config

# @triton_dejavu.jitcache(
#     # this list is shorter, since it will be called only within one model
#     check_keys=["MAX_SEQ_Q", "MAX_SEQ_K", "AVG_SEQ_Q", "AVG_SEQ_K", 
#                 "stride_k_cache_3", "stride_v_cache_3"],
#     check_specialization=["num_seqs"],
#     assume_const=[
#         "scale",
#         "k_scale",
#         "v_scale",
#         "query_stride_1",
#         "output_stride_1",
#         "stride_k_cache_0",
#         "stride_k_cache_1",
#         "stride_k_cache_2",
#         "stride_k_cache_4",
#         "stride_v_cache_0",
#         "stride_v_cache_1",
#         "stride_v_cache_2",
#     ],
#     autotuner_args=["BLOCK_N", "BLOCK_M"],
# )
@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            'BLOCK_N': [16, 32, 64, 128, 256, 512],
            'BLOCK_M': [16, 32, 64, 128, 256, 512]
        },
    num_warps=[2, 4, 8],
    num_stages=[1, 2, 4, 6, 8],
    ),
    # this list is longer, since it would be used for multiple models
    key = ["MAX_SEQ_Q", "MAX_SEQ_K", "AVG_SEQ_Q", "AVG_SEQ_K",
           "num_query_heads", "num_queries_per_kv", 
           "BLOCK_SIZE", "HEAD_SIZE", "HEAD_SIZE_PADDED",
           "SLIDING_WINDOW",
           "stride_k_cache_3", "stride_v_cache_3"
           ],
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "unified_attention")),
    use_cuda_graph=True,
    # use_bo=True,
    # search_max_search_t=360,
    # informed_fallback=informed_fallback_next,
    # prepare_informed_fallback=prepare_informed_fallback,
    # fallback_heuristic=fallback_heuristic_dt2,
    ignore_dtypes=True,
)
# @triton.heuristics(
#        {
#            "BLOCK_M": lambda args: prefill_heuristics_2d(args['MAX_SEQ_Q'], args['MAX_SEQ_K'])['BLOCK_M'],
#            "BLOCK_N": lambda args: prefill_heuristics_2d(args['MAX_SEQ_Q'], args['MAX_SEQ_K'])['BLOCK_N'],
#            "num_warps": lambda args: prefill_heuristics_2d(args['MAX_SEQ_Q'], args['MAX_SEQ_K'])['num_warps'],
#            "num_stages": lambda args: prefill_heuristics_2d(args['MAX_SEQ_Q'], args['MAX_SEQ_K'])['num_stages'],
#         } 
# )
@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    # used as input to the autotuner/heuristics
    MAX_SEQ_Q: tl.constexpr,
    MAX_SEQ_K: tl.constexpr,
    AVG_SEQ_Q: tl.constexpr,
    AVG_SEQ_K: tl.constexpr,
    # autotuner args
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    BLOCK_Q: tl.constexpr = BLOCK_M // num_queries_per_kv

    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q + mid
        if mid_val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid

    seq_idx = left - 1
    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_Q * num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_Q * num_queries_per_kv],
                float("-inf"),
                dtype=tl.float32)
    L = tl.full([BLOCK_Q * num_queries_per_kv], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * num_queries_per_kv, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
        BLOCK_M - 1) // num_queries_per_kv + 1

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    offs_n = tl.arange(0, BLOCK_N)

    # iterate through tiles (below the mask)
    for start_n in range(0,
                         max_seq_prefix_len,
                         BLOCK_N):

        start_n = tl.multiple_of(start_n, BLOCK_N)

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                     (start_n + offs_n) // BLOCK_SIZE,
                                     mask=(start_n + offs_n) < seq_len,
                                     other=0)

        v_offset = (physical_block_idx[:, None] * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 +
                    (offs_n[:, None] % BLOCK_SIZE) * stride_v_cache_1)

        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None] * stride_k_cache_3 +
                    (offs_n[None, :] % BLOCK_SIZE) * stride_k_cache_1)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None],
                         other=0.0)

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = start_n + tl.arange(0, BLOCK_N)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_Q * num_queries_per_kv, BLOCK_N,)
        S = tl.zeros(shape=(BLOCK_Q * num_queries_per_kv, BLOCK_N),
                     dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                     S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset)
                         < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (BLOCK_Q * num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_Q * num_queries_per_kv, BLOCK_N,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_Q * num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_Q * num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_N,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_N,)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# @triton.jit
# def kernel_unified_attention_3d(
#     segm_output_ptr,
#     # [num_tokens, num_query_heads, num_segments, head_size]
#     segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
#     segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
#     query_ptr,  # [num_tokens, num_query_heads, head_size]
#     key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
#     value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
#     block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
#     seq_lens_ptr,  # [num_seqs]
#     alibi_slopes_ptr,  # [num_query_heads]
#     scale,  # float32
#     k_scale,  # float32
#     v_scale,  # float32
#     softcap,  # float32
#     num_query_heads: tl.constexpr,  # int
#     num_queries_per_kv: tl.constexpr,  # int
#     block_table_stride: tl.int64,  # int
#     query_stride_0: tl.int64,  # int
#     query_stride_1: tl.int64,  # int, should be equal to head_size
#     BLOCK_SIZE: tl.constexpr,  # int
#     HEAD_SIZE: tl.constexpr,  # int
#     HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
#     USE_ALIBI_SLOPES: tl.constexpr,  # bool
#     USE_SOFTCAP: tl.constexpr,  # bool
#     SLIDING_WINDOW: tl.constexpr,  # int
#     stride_k_cache_0: tl.int64,  # int
#     stride_k_cache_1: tl.int64,  # int
#     stride_k_cache_2: tl.int64,  # int
#     stride_k_cache_3: tl.constexpr,  # int
#     stride_v_cache_0: tl.int64,  # int
#     stride_v_cache_1: tl.int64,  # int
#     stride_v_cache_2: tl.int64,  # int
#     stride_v_cache_3: tl.constexpr,  # int
#     query_start_len_ptr,  # [num_seqs+1]
#     BLOCK_Q: tl.constexpr,  # int
#     num_seqs: tl.int32,
#     BLOCK_M: tl.constexpr,  # int
#     NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
# ):
#     q_block_global_idx = tl.program_id(0)
#     kv_head_idx = tl.program_id(1)
#     segm_idx = tl.program_id(2)

#     seq_idx = find_seq_idx(
#         query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
#     )

#     q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

#     q_block_local_idx = q_block_global_idx - q_block_start_idx

#     cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
#     cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

#     cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

#     if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
#         return

#     # sequence len for this particular sequence
#     seq_len = tl.load(seq_lens_ptr + seq_idx)

#     # number of segments for this particular sequence
#     num_segments = NUM_SEGMENTS_PER_SEQ
#     blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

#     if segm_idx * blocks_per_segment * BLOCK_SIZE >= seq_len:
#         return

#     offs_m = tl.arange(0, BLOCK_M)
#     offs_d = tl.arange(0, HEAD_SIZE_PADDED)

#     query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

#     query_offset_0 = cur_batch_in_all_start_index + query_pos
#     query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

#     query_offset = (
#         query_offset_0[:, None] * query_stride_0
#         + query_offset_1[:, None] * query_stride_1
#         + offs_d[None, :]
#     )

#     dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
#     query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
#     query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

#     # Q : (BLOCK_M, HEAD_SIZE_PADDED)
#     Q = tl.load(
#         query_ptr + query_offset,
#         mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
#         other=0.0,
#     )

#     block_table_offset = seq_idx * block_table_stride

#     M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
#     L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
#     acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

#     # context length for this particular sequences
#     context_len = seq_len - cur_batch_query_len

#     # alibi slope for this head
#     if USE_ALIBI_SLOPES:
#         alibi_slope = tl.load(
#             alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
#         )

#     num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

#     # iterate through tiles within current segment
#     for j in range(
#         segm_idx * blocks_per_segment,
#         min((segm_idx + 1) * blocks_per_segment, num_blocks),
#     ):
#         physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

#         offs_n = tl.arange(0, BLOCK_SIZE)

#         v_offset = (
#             physical_block_idx * stride_v_cache_0
#             + kv_head_idx * stride_v_cache_2
#             + offs_d[None, :] * stride_v_cache_3
#             + offs_n[:, None] * stride_v_cache_1
#         )

#         k_offset = (
#             physical_block_idx * stride_k_cache_0
#             + kv_head_idx * stride_k_cache_2
#             + offs_d[:, None] * stride_k_cache_3
#             + offs_n[None, :] * stride_k_cache_1
#         )

#         # K : (HEAD_SIZE, BLOCK_SIZE)
#         K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)

#         if K_load.dtype.is_fp8():
#             if Q.dtype.is_fp8():
#                 K = K_load
#             else:
#                 K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
#         else:
#             K = K_load

#         # V : (BLOCK_SIZE, HEAD_SIZE)
#         V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)

#         if V_load.dtype.is_fp8():
#             if Q.dtype.is_fp8():
#                 V = V_load
#             else:
#                 V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
#         else:
#             V = V_load

#         seq_offset = j * BLOCK_SIZE + offs_n

#         seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

#         # S : (BLOCK_M, BLOCK_SIZE)
#         S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

#         S += scale * tl.dot(Q, K)

#         if USE_SOFTCAP:
#             S = apply_softcap(S, softcap)

#         S = tl.where(
#             query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
#         )

#         if SLIDING_WINDOW > 0:
#             S = tl.where(
#                 (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
#                 S,
#                 float("-inf"),
#             )

#         if USE_ALIBI_SLOPES:
#             S += alibi_slope[:, None] * (seq_offset - context_len)

#         # compute running maximum
#         # m_j : (BLOCK_M,)
#         m_j = tl.maximum(M, tl.max(S, axis=1))
#         # For sliding window there's a chance the max is -inf due to masking of
#         # the entire row. In this case we need to set m_j 0 to avoid NaN
#         m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

#         # P : (BLOCK_M, BLOCK_SIZE,)
#         P = tl.exp(S - m_j[:, None])

#         # l_j : (BLOCK_M,)
#         l_j = tl.sum(P, axis=1)

#         # alpha : (BLOCK_M, )
#         alpha = tl.exp(M - m_j)

#         # acc : (BLOCK_M, HEAD_SIZE_PADDED)
#         acc = acc * alpha[:, None]

#         # update constants
#         L = L * alpha + l_j
#         M = m_j

#         # acc : (BLOCK_M, HEAD_SIZE_PADDED)
#         acc += tl.dot(P.to(V.dtype), V)

#     segm_output_offset = (
#         query_offset_0[:, None].to(tl.int64)
#         * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
#         + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
#         + segm_idx * HEAD_SIZE_PADDED
#         + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
#     )
#     tl.store(
#         segm_output_ptr + segm_output_offset,
#         acc,
#         mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
#     )
#     segm_offset = (
#         query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
#         + query_offset_1 * NUM_SEGMENTS_PER_SEQ
#         + segm_idx
#     )
#     tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
#     tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# @triton.jit
# def reduce_segments(
#     output_ptr,  # [num_tokens, num_query_heads, head_size]
#     segm_output_ptr,
#     # [num_tokens, num_query_heads, max_num_segments, head_size]
#     segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
#     segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
#     seq_lens_ptr,  # [num_seqs]
#     num_seqs,  # int
#     num_query_heads: tl.constexpr,  # int
#     output_stride_0: tl.int64,  # int
#     output_stride_1: tl.int64,  # int, should be equal to head_size
#     block_table_stride: tl.int64,  # int
#     BLOCK_SIZE: tl.constexpr,  # int
#     HEAD_SIZE: tl.constexpr,  # int, must be power of 2
#     HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
#     query_start_len_ptr,  # [num_seqs+1]
#     BLOCK_Q: tl.constexpr,  # int
#     NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
# ):
#     query_token_idx = tl.program_id(0)
#     query_head_idx = tl.program_id(1)

#     seq_idx = find_seq_idx(
#         query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
#     )

#     # sequence len for this particular sequence
#     seq_len = tl.load(seq_lens_ptr + seq_idx)

#     # number of segments for this particular sequence
#     num_segments = NUM_SEGMENTS_PER_SEQ
#     blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

#     # create masks for subsequent loads
#     act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
#     segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
#         [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
#     )
#     dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

#     # load segment maxima
#     segm_offset = (
#         query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
#         + query_head_idx * NUM_SEGMENTS_PER_SEQ
#         + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
#     )
#     segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
#     overall_max = tl.max(segm_max)

#     # load and rescale segment exp sums
#     segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
#     segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
#     overall_expsum = tl.sum(segm_expsum)

#     # load, rescale, and add segment attention outputs
#     segm_output_offset = (
#         query_token_idx.to(tl.int64)
#         * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
#         + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
#         + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
#         + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
#     )
#     segm_output = tl.load(
#         segm_output_ptr + segm_output_offset,
#         mask=segm_mask[:, None] & dim_mask[None, :],
#         other=0.0,
#     )
#     segm_output *= tl.exp(segm_max - overall_max)[:, None]
#     acc_sum = tl.sum(segm_output, axis=0)
#     # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
#     acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

#     # write result
#     output_offset = (
#         query_token_idx * output_stride_0
#         + query_head_idx * output_stride_1
#         + tl.arange(0, HEAD_SIZE_PADDED)
#     )
#     tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    avg_seqlen_q=None,
    avg_seqlen_k=None,
    force_selection=None,  # None, 2, 3 to select kernel
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # if batch contains a prefill
    if (
        max_seqlen_q > 1
        or total_num_q_blocks * num_kv_heads > 128
        or force_selection == 2
    ) and force_selection != 3:

        # '''
        # BLOCK_M = 16 if avg_seqlen_q < 8 else (16 if avg_seqlen_k < 128 else 64)

        # if avg_seqlen_q < 8:
        #     if avg_seqlen_k < 64:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 128
        # else:
        #     if avg_seqlen_k < 32:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 32
        # '''

        # tpa_test_q = triton.next_power_of_2(int(max_seqlen_q))
        # tpa_test_k = triton.next_power_of_2(int(max_seqlen_k))

        # '''
        # # Model trained on avg
        # BLOCK_M = 16 if tpa_test_q < 8 else (16 if tpa_test_k < 128 else 64)

        # if tpa_test_q < 8:
        #     if tpa_test_k < 64:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 128
        # else:
        #     if tpa_test_k < 32:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 32
        # '''

        # # Model trained on max
        # if tpa_test_q < 1024:
        #     BLOCK_M = 16
        # else:
        #     BLOCK_M = 64

        # if tpa_test_k < 64:
        #     if tpa_test_k < 32:
        #         BLOCK_N = 16
        #     else:
        #         BLOCK_N = 32
        # else:
        #     if tpa_test_q < 256:
        #         BLOCK_N = 128
        #     else:
        #         BLOCK_N = 64

        # '''
        # m_factor = 1 if tpa_test_q < 1024 else 4

        # BLOCK_M : tl.constexpr = 16 * m_factor
        # BLOCK_N : tl.constexpr = max(16, min(tpa_test_k, 128) // m_factor)
        # '''
        # # grid = (q.shape[0] // (BLOCK_M // num_queries_per_kv)
        # #     + num_seqs, num_kv_heads)
        
        MAX_SEQ_Q = triton.next_power_of_2(int(max_seqlen_q))
        MAX_SEQ_K = triton.next_power_of_2(int(max_seqlen_k))
        AVG_SEQ_Q = triton.next_power_of_2(int(avg_seqlen_q))
        AVG_SEQ_K = triton.next_power_of_2(int(avg_seqlen_k))

        grid = lambda META : (q.shape[0] // (META['BLOCK_M'] // num_queries_per_kv)
                                + num_seqs, num_kv_heads)


        kernel_unified_attention_2d[grid](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_SOFTCAP=(softcap > 0),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            MAX_SEQ_Q=MAX_SEQ_Q,
            MAX_SEQ_K=MAX_SEQ_K,
            AVG_SEQ_Q=AVG_SEQ_Q,
            AVG_SEQ_K=AVG_SEQ_K,
            # tpa_test_q=tpa_test_q,
            # tpa_test_k=tpa_test_k,
            # BLOCK_M=BLOCK_M,
            # BLOCK_N=BLOCK_N,
        )
    else:
        # for initial version, NUM_SEGMENTS = 16 is chosen as a default
        # value that showed good performance in tests
        NUM_SEGMENTS = 16

        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_SOFTCAP=(softcap > 0),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        )

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        )
