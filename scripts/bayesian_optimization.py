from lhs import LatinHypercubeSampler
from enum import Enum
import ast
import json
import random
import numpy as np
import pandas as pd
import os
from pathlib import Path

import torch

from ConfigSpace import ConfigurationSpace, Integer
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from smac.main.smbo import SMBO
from smac.runhistory.runhistory import RunHistory

import lightgbm as lgb
from sklearn.metrics import ndcg_score
from skopt import gp_minimize

import triton
import triton.language as tl
import triton_dejavu
import random
import itertools
from collections import defaultdict
from triton_gemm import matmul 

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
categorical_features = ['BLOCK_SIZE_M', 'BLOCK_SIZE_K', 'BLOCK_SIZE_N', 'GROUP_SIZE_M', 'num_warps', 'num_stages']
numerical_features = ['M', 'N', 'K']
collected_data = [] ## The data that has been collected with bao so far

problem_dimension = range(1,8192)
block_sizes = [16, 32, 64, 128, 256]
warp_size = [2** i for i in range(6)]
stage_size = list(range(8))
group_size = [1,2,4,8,16]

config_count = 10
iteration = 0
config_list = []
search_dict_cfg = {
    'block_size_m': block_sizes,
    'block_size_n': block_sizes,
    'block_size_k': block_sizes,
    'group_size_m': group_size,
    'num_warps': warp_size,
    'num_stages': stage_size
}

lhs = LatinHypercubeSampler(search_dict_cfg)
DEVICE = 'cuda'
samples_cfg = lhs.generate_new_categorical_samples(config_count)
for cfg in samples_cfg:
    config_list.append(triton.Config({'BLOCK_SIZE_M': cfg['block_size_m'], 'BLOCK_SIZE_N': cfg['block_size_n'], 'BLOCK_SIZE_K': cfg['block_size_k'], 'GROUP_SIZE_M': cfg['group_size_m']}, 
                                        num_stages=cfg['num_stages'],
                                        num_warps=cfg['num_warps']))

class KernelTypes(Enum):
    GEMM = 1
    SWIGLU = 2

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


df_full = process_data('all_gemm.csv')
gpu = 'V100'
df_full = df_full[df_full['GPU'] == gpu]

## Creating the test programs
test_programs = df_full['program'].unique()

power_of_two = df_full[df_full['power_of_two'] == 1]
non_power_of_two = df_full[df_full['power_of_two'] == 0]

test_programs1 = random.sample(list(power_of_two['program'].unique()), 25)
test_programs2 = random.sample(list(non_power_of_two['program'].unique()), 25)

test_programs = test_programs1 + test_programs2

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

## The objective function maximizes the model performance on the given test set
def objective_function(config):
    global iteration
    collected_data.append(config)

    try:
        a = torch.randn((config[0], config[1]), device=DEVICE, dtype=torch.float16)
        b = torch.randn((config[1], config[2]), device=DEVICE, dtype=torch.float16)
    except RuntimeError as e:
        print(f"Could not allocate because of {e}")
    quantiles = [0.5, 0.2, 0.8]

    try: 
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, config_list), quantiles=quantiles)
        print(f'The {config} took {ms} milliseconds')
    except RuntimeError as e:
        print(f"Could not run the benchmark because of {e}")
    del a,b

    all_data_frames= find_all_json_files('.')
    data = pd.concat(all_data_frames, axis=0)
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
    ranker.save_model(f'ranker_model_{iteration}.json')
    iteration += 1

    return -ndcg

print('Starting Bayesian Optimization')
res = gp_minimize(objective_function, [(1, 8192), (1, 8192), (1, 8192)], n_calls=10)
print(results)