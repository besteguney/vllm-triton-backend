
import os
import torch
import triton
import triton.language as tl
from triton import Config
import triton_dejavu

# import numpy as np


def fallback_heuristic(key):
    ret = Config({'BLOCK_SIZE': 2048 if key[1] <= 128 else 4096}, num_warps=16, num_stages=2)
    return ret

def informed_fallback_next(key, cache):
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[2]))]
    return ret

def prepare_informed_fallback(cache):
    ret = {int(k[2]): c for k, c in cache.items()}
    return ret

# lazy functions for paper evals
use_bo = lambda: os.getenv('NGL_EXP_USE_BO', '0') == '1'
use_random = lambda: os.getenv('NGL_EXP_USE_RANDOM_SEARCH', '0') == '1'
bo_time = lambda: int(os.getenv('NGL_EXP_BO_TIME', '360'))


def _select_informed_fallback():
    fallback_mode = os.getenv('NGL_EXP_FALLBACK', 'none')
    if fallback_mode == 'static':
        return None, None
    if fallback_mode == 'next':
        return informed_fallback_next, prepare_informed_fallback
    return informed_fallback_next, prepare_informed_fallback

# defaults to work without env
select_fallback_heuristic = lambda: fallback_heuristic if os.getenv('NGL_EXP_FALLBACK', 'none') == 'static' else None
select_informed_fallback = lambda: _select_informed_fallback()[0]
select_prepare_informed_fallback = lambda: _select_informed_fallback()[1]


def make_swiglu_kernel(configurations):
    @triton_dejavu.autotune(
    configs=configurations,
    key=['D', 'num_tokens', 'n_elements'], 
    use_cuda_graph=True,
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "swiglu_data_bao_lhs_stop_10_power_of_two_3")
    ),)
    @triton.jit
    def fused_silu_and_mul_kernel_other(
        x_ptr, 
        out_ptr,
        n_elements: tl.constexpr, 
        D: tl.constexpr,
        num_tokens: tl.constexpr,  # only for the autotuner
        BLOCK_SIZE: tl.constexpr):
        """An activation function for SwiGLU.

        The function computes x,y -> x * sigmoid(x) * y 

        Shapes:
            x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
            return: (batch_size, seq_len, d) or (num_tokens, d)
        """
        pid = tl.program_id(axis=0)
        bid = tl.program_id(axis=1)
        # num_threads_per_block = tl.num_programs(axis=1)
        row_start = pid * 2 * D
        block_start = row_start + bid * BLOCK_SIZE
        
        offsets_x = block_start + tl.arange(0, BLOCK_SIZE)
        # max_x = block_start + min(BLOCK_SIZE, D)
        # max_x = block_start + D
        max_x = row_start + min((bid+1) * BLOCK_SIZE, D)
        mask_x = offsets_x < n_elements and offsets_x < max_x
        x = tl.load(x_ptr + offsets_x, mask=mask_x).to(tl.float32)
        
        offsets_y = block_start + D + tl.arange(0, BLOCK_SIZE)
        # max_y = block_start + min(BLOCK_SIZE, D) + D
        # max_y = block_start + D + D
        max_y = row_start + D + min((bid+1) * BLOCK_SIZE, D)
        mask_y = offsets_y < n_elements and offsets_y < max_y
        y = tl.load(x_ptr + offsets_y, mask=mask_y).to(tl.float32)
        
        output = (x / (1.0 + tl.exp(-x))) * y
        # output = (x / (1.0 + tl.math.exp(-x))) * y
        
        out_cast = output.to(tl.float16)
        row_out_start = pid * D
        block_start_out = row_out_start + bid * BLOCK_SIZE
        offsets_o = block_start_out + tl.arange(0, BLOCK_SIZE)
        # max_o = (bid+1) * BLOCK_SIZE + pid*D
        # max_o = block_start_out + min(BLOCK_SIZE, D)
        # max_o = block_start_out + D
        max_o = row_out_start + min((bid+1) * BLOCK_SIZE, D)
        mask_o = offsets_o < ((n_elements+1)//2) and offsets_o < max_o
        # mask_o = offsets_o < max_o
        tl.store(out_ptr + offsets_o, out_cast, mask=mask_o)
        return
    return fused_silu_and_mul_kernel_other


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {'BLOCK_SIZE': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        num_warps=[2**i for i in range(6)],
        num_stages=[i for i in range(1, 9)],
        num_ctas=[1],
    ),
    key=['D', 'num_tokens', 'n_elements'], 
    use_cuda_graph=True,
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "swiglu_data_autotuned")
    ),
)
@triton.jit
def fused_silu_and_mul_kernel(
    x_ptr, 
    out_ptr,
    n_elements: tl.constexpr, 
    D: tl.constexpr,
    num_tokens: tl.constexpr,  # only for the autotuner
    BLOCK_SIZE: tl.constexpr):
    """An activation function for SwiGLU.

    The function computes x,y -> x * sigmoid(x) * y 

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # num_threads_per_block = tl.num_programs(axis=1)
    row_start = pid * 2 * D
    block_start = row_start + bid * BLOCK_SIZE
    
    offsets_x = block_start + tl.arange(0, BLOCK_SIZE)
    # max_x = block_start + min(BLOCK_SIZE, D)
    # max_x = block_start + D
    max_x = row_start + min((bid+1) * BLOCK_SIZE, D)
    mask_x = offsets_x < n_elements and offsets_x < max_x
    x = tl.load(x_ptr + offsets_x, mask=mask_x).to(tl.float32)
    
    offsets_y = block_start + D + tl.arange(0, BLOCK_SIZE)
    # max_y = block_start + min(BLOCK_SIZE, D) + D
    # max_y = block_start + D + D
    max_y = row_start + D + min((bid+1) * BLOCK_SIZE, D)
    mask_y = offsets_y < n_elements and offsets_y < max_y
    y = tl.load(x_ptr + offsets_y, mask=mask_y).to(tl.float32)
    
    output = (x / (1.0 + tl.exp(-x))) * y
    # output = (x / (1.0 + tl.math.exp(-x))) * y
    
    out_cast = output.to(tl.float16)
    row_out_start = pid * D
    block_start_out = row_out_start + bid * BLOCK_SIZE
    offsets_o = block_start_out + tl.arange(0, BLOCK_SIZE)
    # max_o = (bid+1) * BLOCK_SIZE + pid*D
    # max_o = block_start_out + min(BLOCK_SIZE, D)
    # max_o = block_start_out + D
    max_o = row_out_start + min((bid+1) * BLOCK_SIZE, D)
    mask_o = offsets_o < ((n_elements+1)//2) and offsets_o < max_o
    # mask_o = offsets_o < max_o
    tl.store(out_ptr + offsets_o, out_cast, mask=mask_o)
    return

@triton.jit
def fused_silu_and_mul_kernel2(
    x_ptr, 
    out_ptr,
    n_elements: tl.constexpr, 
    D: tl.constexpr,
    num_tokens: tl.constexpr,  # only for the autotuner
    BLOCK_SIZE: tl.constexpr):
    """An activation function for SwiGLU.

    The function computes x,y -> x * sigmoid(x) * y 

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # num_threads_per_block = tl.num_programs(axis=1)
    row_start = pid * 2 * D
    block_start = row_start + bid * BLOCK_SIZE
    
    offsets_x = block_start + tl.arange(0, BLOCK_SIZE)
    # max_x = block_start + min(BLOCK_SIZE, D)
    # max_x = block_start + D
    max_x = row_start + min((bid+1) * BLOCK_SIZE, D)
    mask_x = offsets_x < n_elements and offsets_x < max_x
    x = tl.load(x_ptr + offsets_x, mask=mask_x).to(tl.float32)
    
    offsets_y = block_start + D + tl.arange(0, BLOCK_SIZE)
    # max_y = block_start + min(BLOCK_SIZE, D) + D
    # max_y = block_start + D + D
    max_y = row_start + D + min((bid+1) * BLOCK_SIZE, D)
    mask_y = offsets_y < n_elements and offsets_y < max_y
    y = tl.load(x_ptr + offsets_y, mask=mask_y).to(tl.float32)
    
    output = (x / (1.0 + tl.exp(-x))) * y
    # output = (x / (1.0 + tl.math.exp(-x))) * y
    
    out_cast = output.to(tl.float16)
    row_out_start = pid * D
    block_start_out = row_out_start + bid * BLOCK_SIZE
    offsets_o = block_start_out + tl.arange(0, BLOCK_SIZE)
    # max_o = (bid+1) * BLOCK_SIZE + pid*D
    # max_o = block_start_out + min(BLOCK_SIZE, D)
    # max_o = block_start_out + D
    max_o = row_out_start + min((bid+1) * BLOCK_SIZE, D)
    mask_o = offsets_o < ((n_elements+1)//2) and offsets_o < max_o
    # mask_o = offsets_o < max_o
    tl.store(out_ptr + offsets_o, out_cast, mask=mask_o)
    return

def fused_silu_and_mul(xy: torch.Tensor):
    d = xy.shape[-1] // 2
    output_shape = (xy.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=xy.dtype, device=xy.device)
    # num_tokens = xy.shape[0] # not for 3D!
    num_tokens = xy.numel() // xy.shape[-1]
    n_elements = xy.numel()
        
    # grid = lambda meta: (int(num_tokens), triton.cdiv(d, (min(d, meta['BLOCK_SIZE']))))
    # number of blocks, threads per block (?)
    # grid = lambda meta: (int(num_tokens), int((d + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])) 
    # grid = (int(num_tokens), int(d/1024)+1, )
    grid = lambda meta: (int(num_tokens), triton.cdiv(d, meta['BLOCK_SIZE'])) 
    # print(f'expected grid: {num_tokens}, {np.ceil(d/1024)}')
    # print(f'd % block_size: {d%2048}')
    fused_silu_and_mul_kernel[grid](xy, out, n_elements, d, num_tokens)
    return out


def fused_silu_and_mul_cfg(xy: torch.Tensor, configurations):
    d = xy.shape[-1] // 2
    output_shape = (xy.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=xy.dtype, device=xy.device)
    # num_tokens = xy.shape[0] # not for 3D!
    num_tokens = xy.numel() // xy.shape[-1]
    n_elements = xy.numel()
        
    # grid = lambda meta: (int(num_tokens), triton.cdiv(d, (min(d, meta['BLOCK_SIZE']))))
    # number of blocks, threads per block (?)
    # grid = lambda meta: (int(num_tokens), int((d + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])) 
    # grid = (int(num_tokens), int(d/1024)+1, )
    grid = lambda meta: (int(num_tokens), triton.cdiv(d, meta['BLOCK_SIZE'])) 
    # print(f'expected grid: {num_tokens}, {np.ceil(d/1024)}')
    # print(f'd % block_size: {d%2048}')
    swiglu_kernel = make_swiglu_kernel(configurations)
    try:
        swiglu_kernel[grid](xy, out, n_elements, d, num_tokens)
    except Exception as e:
        print(f"Could not run the swiglu kernel exception is {e}", )
    return out

def fused_silu_and_mul_given_out(out: torch.Tensor, xy: torch.Tensor):
    d = xy.shape[-1] // 2
    num_tokens = xy.numel() // xy.shape[-1]
    n_elements = xy.numel()

    # grid = lambda meta: (int(num_tokens), triton.cdiv(d, (min(d, meta['BLOCK_SIZE']))))
    # number of blocks, threads per block (?)
    # grid = lambda meta: (int(num_tokens), int((d + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])) 
    # grid = (int(num_tokens), int(d/1024)+1, )
    grid = lambda meta: (int(num_tokens), triton.cdiv(d, meta['BLOCK_SIZE'])) 
    # print(f'expected grid: {num_tokens}, {np.ceil(d/1024)}')
    # print(f'd % block_size: {d%2048}')
    fused_silu_and_mul_kernel[grid](xy, out, n_elements, d, num_tokens)
    # return nothing
