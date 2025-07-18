## This script is developed to combine the benchmarking data together. ##
import os
import ast
import pandas as pd
import numpy as np
from pathlib import Path

import json

gpus = ['V100', 'A100', 'L40S', 'H100'] 

# Helper functions
def read_json(file_path):
    data = None
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f'Could not read the file {file_path} because of json error')

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
def create_data_frame_gemm(file_path):
    df = read_json(file_path)
    if df is None:
        return
    df = df['timings']

    new_df = pd.DataFrame()
    for key, value in df.items():
        values = [val for val in value if 'runtime' in val]
        values = [parse_config(val['config'], val['runtime'], val['compile_time']) for val in values]
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

def create_data_frame_swiglu(file_path):
    df = read_json(file_path)
    if df is None:
        return 
    df = df['timings']

    new_df = pd.DataFrame()
    for key, value in df.items():
        values = [val for val in value if 'runtime' in val]
        values = [parse_config(val['config'], val['runtime'], val['compile_time']) for val in values]
        key_config = ast.literal_eval(key)
        tokens = int(key_config[0])
        d = int(key_config[1])
        cur_df = pd.DataFrame(values)
        cur_df['tokens'] = tokens
        cur_df['d'] = d
        new_df = pd.concat([new_df, cur_df])
    return new_df

def is_power_of_two(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0

def find_all_json_files(root_dir, is_gemm):
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)
        if base.startswith("gemm_data"):
            # print("JDAKSJ")
            # print(base)
            # base_path = Path(base)
            # for subdir in base_path.rglob("*"):
            #     if "A100" in str(subdir):
            #         all_json_files = subdir.rglob('all*.json')
            #         for json_file in all_json_files:
            #             # print(json_file)
            #             caller = create_data_frame_gemm if is_gemm else create_data_frame_swiglu
            #             df_new = caller(json_file)
            #             if df_new is None:
            #                 continue
            #             for gpu in gpus:
            #                 if gpu in str(json_file):
            #                     df_new['GPU'] = gpu
            #                     break
            #             result.append(df_new)
            base_path = Path(base)
            all_json_files = base_path.rglob('all*.json')
            for json_file in all_json_files:
                caller = create_data_frame_gemm if is_gemm else create_data_frame_swiglu
                df_new = caller(json_file)
                if df_new is None:
                    continue
                for gpu in gpus:
                    if gpu in str(json_file):
                        df_new['GPU'] = gpu
                        break
                result.append(df_new)
    return result

if __name__ == "__main__":
    # Set this to the root directory where the search should begin
    search_root = "."
    is_gemm = True
    all_data_frames= find_all_json_files(search_root, is_gemm)

    data = pd.concat(all_data_frames, axis=0)
    print(f'All the data shape before processing {data.shape}')

    ## Remove the duplicates
    print(data.columns)
    if is_gemm:
        data.drop_duplicates(subset=['M', 'N', 'K', 
                                    'BLOCK_SIZE_M', 
                                    'BLOCK_SIZE_N',
                                    'BLOCK_SIZE_K',
                                    'GROUP_SIZE_M',
                                    'num_warps',
                                    'num_stages',
                                    'GPU'], inplace=True)
    else:
        data.drop_duplicates(subset=['BLOCK_SIZE', 
                                     'tokens','d',
                                    'num_warps',
                                    'num_stages',
                                    'GPU'], inplace=True)
    print(f'The data shape after dropping the duplicates {data.shape}')

    ## When the runtime is nan, replace with np.inf
    data['runtime'] = data['runtime'].fillna(np.inf)

    ## Remove all the columns where number of warps is not power of two
    data = data[data['num_warps'].apply(is_power_of_two)]
    print(f'The data shape after dropping the non power of two warps {data.shape}')
    # data = data.iloc[500:1001]
    print(f'The data shape after dropping the non power of two warps {data.shape}')
    csv_name = 'all_gemm.csv' if is_gemm else 'all_swiglu.csv'
    csv_name = 'all_gemm_data.csv'
    data.to_csv(csv_name)

