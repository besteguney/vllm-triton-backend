import random
import itertools

random.seed(0)
# Generate all powers of 2 from 1 to 8192
powers_of_two = [2 ** i for i in range(14)]  # 2^0 to 2^13

# Generate 10 lists with powers of two only
power_of_two_lists = [
    random.sample(powers_of_two, 3) for _ in range(10)
]

# Generate 10 completely random lists
random_lists = [
    [random.randint(1, 8192) for _ in range(3)] for _ in range(10)
]

# Combine and shuffle them
final_list = power_of_two_lists + random_lists
random.shuffle(final_list)

# Print the result
print(final_list)
# for sublist in final_list:
#     print(sublist)

print("********")
group_size = [1,2,4,8,16]
block_size_m = [16, 32, 64, 128, 256]
block_size_n = [16, 32, 64, 128, 256]
block_size_k = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))

configurations = []
all_combinations = list(itertools.product(block_size_m, block_size_n, block_size_k, group_size, warp_size, stage_size))
all_combinations_filtered = [
    combo for combo in all_combinations
    if not ((combo[1] > 128 and combo[2] > 128)) 
]

sampled_configurations = random.sample(all_combinations_filtered, min(50, len(all_combinations_filtered)))
## 'BLOCK_SIZE_M,'BLOCK_SIZE_N': sample[1], 'BLOCK_SIZE_K', 'GROUP_SIZE_M': sample[3], num_warps, num_stages
# for config in sampled_configurations:
#     print(config)
print(sampled_configurations)