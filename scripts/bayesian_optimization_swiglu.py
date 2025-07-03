from lhs import LatinHypercubeSampler
from enum import Enum
import ast
import json
import random
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import torch

import lightgbm as lgb
from sklearn.metrics import ndcg_score
from skopt import gp_minimize
from skopt.space.space import Categorical

import triton
import triton.language as tl
import triton_dejavu
import random
import itertools
from collections import defaultdict
from triton_swiglu import fused_silu_and_mul_cfg 

## Global Variables
random.seed(0)
model_params = {
    'objective':'lambdarank',
    'metric':'ndcg',
    'boosting_type':'gbdt',
    'n_estimators':200,
    'learning_rate':0.3,
    'label_gain':[i for i in range(1001)],
    'eval_at':[1, 3, 5],
}
numerical_features = ['log_d', 'log_tokens']
categorical_features = ['BLOCK_SIZE', 'num_warps', 'num_stages']

quantiles = [0.5, 0.2, 0.8]
heads = (16, 2**14+1)
seqlen = (16, 1024)
max_values = [0.01, 0.1, 1.0]
batch = (1, 128)
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_warps = [2**i for i in range(6)]
num_stages = [i for i in range(6)]

iteration = 0

collected_data = [] ## The data that has been collected with bao so far
config_count = 10
search_dict_cfg =  {
    'block_size': block_sizes,
    'num_warps': num_warps,
    'num_stages': num_stages,
}

lhs = LatinHypercubeSampler(search_dict_cfg)
samples_cfg = lhs.generate_new_categorical_samples(config_count)
config_list = []

for cfg in samples_cfg:
    print(f"prinitng the cfg is {cfg}")
    config_list.append(triton.Config({'BLOCK_SIZE': cfg['block_size']}, 
                                        num_stages=cfg['num_stages'],
                                        num_warps=cfg['num_warps']))
def is_power_of_two(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0

def get_group_sizes(g):
    _, sizes = np.unique(g, return_counts=True)
    return sizes.tolist()

def process_data(file_path, gpu=None):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"There is no csv in {file_path}")
        return None
    
    data.drop_duplicates(subset=['d', 'tokens', 'BLOCK_SIZE', 'num_warps', 'num_stages', 'GPU'], inplace=True)

    # Adding some features
    data['runtime'] = data['runtime'].fillna(np.inf) ## If the runtime failed, it means that it is not a suitable configuration. We should treat it infinity
    data['program'] = data['d'].astype(str) + '_' + data['tokens'].astype(str) + '_' + data['GPU'].astype(str)
    # Giving the rank information
    data['rank'] = data.groupby('program')['runtime'].rank(ascending=False)
    data['rank'] = np.ceil(data['rank']).astype(int)

    data['grid'] = data['tokens'] * (data['d'] / data['BLOCK_SIZE'])
    scaler = StandardScaler()
    grid_vals = data['grid'].values.reshape(-1,1)
    scaler.fit(grid_vals)
    new_grid_vals = scaler.transform(grid_vals)
    data['grid'] = new_grid_vals

    # Log scale works better
    data[['log_d', 'log_tokens', 'log_runtime']] = data[['d', 'tokens', 'runtime']].apply(np.log2)

    numerical_features = ['log_d', 'log_tokens']
    categorical_features = ['BLOCK_SIZE', 'num_warps', 'num_stages']
    data[categorical_features] = data[categorical_features].astype('category')
    return data
    
df_full = process_data('all_swiglu.csv')

gpu = 'V100'
df_full = df_full[df_full['GPU'] == gpu]

## Creating the test programs
test_programs = df_full['program'].unique()
test_programs = random.sample(list(test_programs), 50)

results = []

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

## Creating the data frame from the json results
def create_data_frame_swiglu(file_path):
    df = read_json(file_path)
    df = df['timings']

    new_df = pd.DataFrame()
    for key, value in df.items():
        values = [parse_config(val['config'], val['runtime'], val['compile_time']) for val in value]
        key_config = ast.literal_eval(key)
        tokens = int(key_config[0])
        d = int(key_config[1])
        cur_df = pd.DataFrame(values)
        cur_df['tokens'] = tokens
        cur_df['d'] = d
        new_df = pd.concat([new_df, cur_df])
    return new_df

def find_all_json_files(root_dir):
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)
        if base.startswith("bao_data_swiglu"):
            # print(base)
            base_path = Path(base)
            all_json_files = base_path.rglob('all*.json')
            for json_file in all_json_files:
                caller = create_data_frame_swiglu
                df_new = caller(json_file)
                if df_new is None:
                    continue
                df_new['GPU'] = gpu
                result.append(df_new)
    return result

# def fused_silu_and_mul(xy: torch.Tensor, config):
#     d = xy.shape[-1] // 2
#     output_shape = (xy.shape[:-1] + (d, ))
#     out = torch.empty(output_shape, dtype=xy.dtype, device=xy.device)
#     # num_tokens = xy.shape[0] # not for 3D!
#     num_tokens = xy.numel() // xy.shape[-1]
#     n_elements = xy.numel()
        
#     # grid = lambda meta: (int(num_tokens), triton.cdiv(d, (min(d, meta['BLOCK_SIZE']))))
#     # number of blocks, threads per block (?)
#     # grid = lambda meta: (int(num_tokens), int((d + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])) 
#     # grid = (int(num_tokens), int(d/1024)+1, )
#     grid = lambda meta: (int(num_tokens), triton.cdiv(d, meta['BLOCK_SIZE'])) 
#     # print(f'expected grid: {num_tokens}, {np.ceil(d/1024)}')
#     # print(f'd % block_size: {d%2048}')
#     try:
#         fused_silu_and_mul_kernel2[grid](xy, out, n_elements, d, num_tokens, **config)
#     except Exception as e:
#         print(f"Could not run the swiglu kernel exception is {e}", )
#     return out

## The objective function maximizes the model performance on the given test set
## Config should be seq len, batch size, d, max vals
def objective_function(config):
    global iteration, config_list
    collected_data.append(config)

    num_tokens = config[0] * config[1]
    d = config[2]
    max_value = config[3]
    print(f" The variables are {num_tokens}, {d}, {max_value}")
    try:
        x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda').uniform_(-1 * max_value, max_value)
    except RuntimeError as e:
        print(f"Could not allocated the size because of {e}")

    fused_silu_and_mul_cfg(x, config_list)
    del x 

    all_data_frames= find_all_json_files('.')
    data = pd.concat(all_data_frames, axis=0)
    print(f'All the data shape before processing {data.shape}')

    data.drop_duplicates(subset=['BLOCK_SIZE','num_warps', 'num_stages', 'd', 'tokens', 'GPU'], inplace=True)
    
    print(f'The data shape after dropping the duplicates {data.shape}')

    ## When the runtime is nan, replace with np.inf
    data['runtime'] = data['runtime'].fillna(np.inf)

    ## Remove all the columns where number of warps is not power of two
    data = data[data['num_warps'].apply(is_power_of_two)]
    data['program'] = data['d'].astype(str) + '_' + data['tokens'].astype(str) + '_' + data['GPU'].astype(str)  
    data['rank'] = data.groupby('program')['runtime'].rank(ascending=False)
    data['rank'] = np.ceil(data['rank']).astype(int)
    data[categorical_features] = data[categorical_features].astype('category')
    
    data[['log_d', 'log_tokens', 'log_runtime']] = data[['d', 'tokens', 'runtime']].apply(np.log2)

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
    ranker.save_model(f'ranker_model_swiglu_{iteration}.json')
    iteration += 1

    return -ndcg

print('Starting Bayesian Optimization')
res = gp_minimize(objective_function, [seqlen, batch, heads, Categorical(max_values)], n_calls=20)
print(results)