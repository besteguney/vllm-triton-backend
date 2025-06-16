import torch
from triton_swiglu import fused_silu_and_mul


SEQUENCE_LENGTH = [16, 32, 44, 54, 64, 511, 512, 513, 648, 912, 1024]
BATCH_SIZES = [1, 2, 3, 4, 6, 8, 9, 12, 16, 22, 25, 28, 32, 54, 58, 64, 72, 84, 96, 128]
HEAD_SIZES = [32, 64, 128]
NUM_HEADS = [32]
heads = [2**i for i in range(4, 14)]
MAX_VALUES = [0.01, 0.1, 1.0]

for seqlen in SEQUENCE_LENGTH:
    for batch_size in BATCH_SIZES:
        num_tokens = seqlen * batch_size
        for d in heads:
            for max_value in MAX_VALUES:
                x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda').uniform_(-1 * max_value, max_value)
                fused_silu_and_mul(x)

# tokens = [2**i for i in range(4,14)]
# heads = [2**i for i in range(4, 14)]

# for token in tokens:
#     for d in heads:
#         input = torch.randn(token, 2 * d, device="cuda", dtype=torch.float16)
#         fused_silu_and_mul(input)
