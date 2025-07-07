from lhs import LatinHypercubeSampler
from enum import Enum
import ast
import json
import random
import numpy as np
import pandas as pd
import os
from pathlib import Path
from functools import partial
from skopt import Optimizer

import torch
import triton.language as tl

import lightgbm as lgb
from sklearn.metrics import ndcg_score
from skopt import gp_minimize
from skopt.space.space import Categorical, Integer

import triton
from triton.runtime.errors import OutOfResources
from triton.compiler.errors import CompilationError
from triton_gemm import matmul, matmul_kernel2

## Global Variables
random.seed(0)
DEVICE = 'cuda'
model_params = {
    'objective':'lambdarank',
    'metric':'ndcg',
    'boosting_type':'gbdt',
    'n_estimators':200,
    'learning_rate':0.3,
    'label_gain':[i for i in range(1001)],
    'eval_at':[1, 3, 5],
}
categorical_features = ['BLOCK_SIZE_M', 'BLOCK_SIZE_K', 'BLOCK_SIZE_N', 'GROUP_SIZE_M', 'num_warps', 'num_stages']
numerical_features = ['M', 'N', 'K']
collected_data = [] ## The data that has been collected with bao so far

# problem_dimension = range(1,8192)
# problem_sizes = [2**i for i in range(14)]
block_sizes = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(5)]
stage_size = list(range(4))
group_size = [1,2,4,8,16]

config_count = 50
iteration = 0
config_list = []
no_improvement_rounds_config = 0
n_rounds_no_improve = 10
no_improvement_rounds = 0
best_ms = np.inf
best_ndcg = -np.inf

data_frame = pd.DataFrame({
    'BLOCK_SIZE_M': pd.Series(dtype='int'),
    'BLOCK_SIZE_N': pd.Series(dtype='int'),
    'BLOCK_SIZE_K': pd.Series(dtype='int'),
    'GROUP_SIZE_M': pd.Series(dtype='int'),
    'num_warps': pd.Series(dtype='int'),
    'num_stages': pd.Series(dtype='int'),
    'runtime': pd.Series(dtype='float'),
    'M': pd.Series(dtype='int'),
    'N': pd.Series(dtype='int'),
    'K': pd.Series(dtype='int')
})

def is_power_of_two(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0

def get_group_sizes(g):
    _, sizes = np.unique(g, return_counts=True)
    return sizes.tolist()

def process_data(file_path, gpu=None):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"There is no csv in {file_path}")
        return None
    df.drop_duplicates(subset=['M', 'N', 'K', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps', 'num_stages'], inplace=True)
    df['M'] = df['M'].astype(int)
    df['N'] = df['N'].astype(int)
    df['K'] = df['K'].astype(int)
    if gpu is not None:
        df['GPU'] = gpu
        
    df['program'] = df['M'].astype(str) + '_' + df['N'].astype(str) + '_' + df['K'].astype(str) + '_' + df['GPU'].astype(str)

    df = (
        df
        .groupby('program', group_keys=False)
        .apply(lambda x: x.sample(n=1000) if len(x) > 1000 else x)
    ).reset_index(drop=True)

    df['runtime'] = df['runtime'].fillna(np.inf)
    df['rank'] = df.groupby('program')['runtime'].rank(ascending=False)
    df['rank'] = np.ceil(df['rank']).astype(int)
    df['grid'] = np.ceil(df['M'] / df['BLOCK_SIZE_M'].astype(int)) * np.ceil(df['N'] / df['BLOCK_SIZE_N'].astype(int))
    df[categorical_features] = df[categorical_features].astype('category')

    ## A label column for non_powers_of_two
    df['problem_volume'] = df['M'] * df['N'] * df['K']
    df['problem_volume'] = df['problem_volume'].astype(int)
    df['power_of_two'] = (((df['problem_volume'] > 0) & ((df['problem_volume'] & (df['problem_volume'] - 1)) == 0))).astype(int)
    return df

def read_json(file_path):
    data = None
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def parse_config(config_str, runtime, compile_time):
    pairs = [item.strip() for item in config_str.split(',')]
    config = {}
    for pair in pairs:
        key, val = pair.split(':')
        key = key.strip()
        val = val.strip()
        config[key] = int(val) if val.isdigit() else val
    config['runtime'] = runtime
    config['compile_time'] = compile_time
    return config

def create_data_frame_gemm(file_path):
    df = read_json(file_path)
    df = df['timings']

    new_df = pd.DataFrame()
    for key, value in df.items():
        values = [parse_config(val['config'], val['runtime'], val['compile_time']) for val in value]
        key_config = ast.literal_eval(key)
        m = int(key_config[0])
        n = int(key_config[1])
        k = int(key_config[2])
        cur_df = pd.DataFrame(values)
        cur_df['M'] = m
        cur_df['N'] = n
        cur_df['K'] = k
        new_df = pd.concat([new_df, cur_df])
    return new_df


def find_all_json_files(root_dir):
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)
        if base.startswith("bao_data"):
            # print(base)
            base_path = Path(base)
            all_json_files = base_path.rglob('all*.json')
            for json_file in all_json_files:
                caller = create_data_frame_gemm
                df_new = caller(json_file)
                if df_new is None:
                    continue
                df_new['GPU'] = gpu
                result.append(df_new)
    return result

# Wrapper Function for Kernel Launch
def matrix_mul(a, b, config):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    def ceil_div(first, second):
        return (first + second - 1) // second
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    try:
        matmul_kernel2[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            **config
        )
        # print("DKAFJKJFD")
    except (RuntimeError, OutOfResources, CompilationError) as e:
        print(f'Could not run the grid configuration {grid} because of {e}')
        return None
    return c

def objective_function_cfg(config, a, b):
    print("Trying another configuration")
    global data_frame, no_improvement_rounds_config, best_ms
    test_config = {'BLOCK_SIZE_M': config[0], 'BLOCK_SIZE_N': config[1], 'BLOCK_SIZE_K': config[2], 'GROUP_SIZE_M': config[3],'num_warps': config[4], 'num_stages': config[5]}
    print(f"shapes are {a.shape[0]}, {b.shape[0]}, {b.shape[1]}")
    print(test_config)
    print("hahah")
    output = {}
    def wrapped_gemm():
        res = matrix_mul(a, b, test_config)
        output['result'] = res
        return res
    
    ms, min_ms, max_ms = triton.testing.do_bench(wrapped_gemm, quantiles=quantiles)
    runtime = np.inf if output['result'] is None else ms
    row = {**test_config, 'runtime':runtime, 'M': a.shape[0], 'N': b.shape[1], 'K': a.shape[1]}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], ignore_index=True)
    if ms < best_ms - 1e-6:
        best_ms = ms
        no_improvement_rounds_config = 0
    else:
        no_improvement_rounds_config += 1
    return ms

## The objective function maximizes the model performance on the given test set
def objective_function(config, test_programs):
    global iteration, data_frame, n_rounds_no_improve, best_ndcg, no_improvement_rounds
    collected_data.append(config)

    try:
        a = torch.randn((config[0], config[1]), device=DEVICE, dtype=torch.float16)
        b = torch.randn((config[1], config[2]), device=DEVICE, dtype=torch.float16)
    except RuntimeError as e:
        print(f"Could not allocate because of {e}")

    max_number_configs = 50
    search_space = [Categorical(block_sizes), Categorical(block_sizes), Categorical(block_sizes), Categorical(group_size), Categorical(warp_size), Categorical(stage_size)]

    # --- BO Loop with Custom Stopping ---
    opt2= Optimizer(search_space, base_estimator="GP", acq_func="EI", random_state=42)

    for i in range(max_number_configs):
        x1 = opt2.ask()
        y1 = objective_function_cfg(x1, a, b)
        opt2.tell(x1, y1)

        if no_improvement_rounds_config >= n_rounds_no_improve:
            print(f"Stopped after {i+1} iterations due to no improvement.")
            break

    del a,b

    data = data_frame
    data['GPU'] = gpu
    print(f'All the data shape before processing {data.shape}')

    data.drop_duplicates(subset=['M', 'N', 'K', 
                            'BLOCK_SIZE_M', 
                            'BLOCK_SIZE_N',
                            'BLOCK_SIZE_K',
                            'GROUP_SIZE_M',
                            'num_warps',
                            'num_stages',
                            'GPU'], inplace=True)
    
    print(f'The data shape after dropping the duplicates {data.shape}')

    ## When the runtime is nan, replace with np.inf
    data['runtime'] = data['runtime'].fillna(np.inf)

    ## Remove all the columns where number of warps is not power of two
    data = data[data['num_warps'].apply(is_power_of_two)]
    data['program'] = data['M'].astype(str) + '_' + data['N'].astype(str) + '_' + data['K'].astype(str) + '_' + str(gpu)
    data['rank'] = data.groupby('program')['runtime'].rank(ascending=False)
    data['rank'] = np.ceil(data['rank']).astype(int)
    data[categorical_features] = data[categorical_features].astype('category')
    
    train_programs = data['program'].unique()
    train_programs = [prog for prog in train_programs if prog not in test_programs]
    train_df = data[data['program'].isin(train_programs)]
    test_df = df_full[df_full['program'].isin(test_programs)]

    # # Prepare features and target
    X_train = train_df[categorical_features + numerical_features]
    y_train = train_df['rank']
    group_train = train_df['program']

    X_test = test_df[categorical_features + numerical_features]
    y_test = test_df['rank']
    group_test = test_df['program']

    train_group_sizes = get_group_sizes(group_train)
    test_group_sizes = get_group_sizes(group_test)

    train_dataset = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
    test_dataset = lgb.Dataset(X_test, label=y_test, group=test_group_sizes)
    ranker = lgb.train(model_params, train_dataset, num_boost_round=100, 
        valid_sets=test_dataset,
        valid_names=['test data'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
        ])
    y_pred = ranker.predict(X_test)

    # Compute NDCG for the test set (example for the first query group)
    ndcg = ndcg_score([y_test], [y_pred])

    results.append(ndcg)
    ranker.save_model(f'ranker_model_cfg_{iteration}.json')
    if ndcg > best_ndcg + 1e-6:
        best_ndcg = ndcg
        no_improvement_rounds = 0
    else:
        no_improvement_rounds += 1

    return -ndcg

quantiles = [0.5, 0.2, 0.8]
df_full = process_data('all_gemm.csv')
gpu = 'A100'
df_full = df_full[df_full['GPU'] == gpu]

results = []

problem_sizes = [2**i for i in range(15)]
search_space = [Categorical(problem_sizes), Categorical(problem_sizes), Categorical(problem_sizes)]
# problem_sizes = Integer(1, 8192)
# search_space = [problem_sizes, problem_sizes, problem_sizes]
opt = Optimizer(search_space, base_estimator="GP", acq_func="EI", random_state=42)

for i in range(50):
    ## Creating the test programs
    test_programs = df_full['program'].unique()
    test_programs = random.sample(list(test_programs), 50)

    x = opt.ask()
    y = objective_function(x, test_programs)
    opt.tell(x, y)

    if no_improvement_rounds >= n_rounds_no_improve:
        print(f"Stopped after {i+1} iterations due to no improvement.")
        break

data_frame.to_csv('gemm_data_a100_bao_power_of_two_1.csv')