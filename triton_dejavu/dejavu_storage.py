#  /*******************************************************************************
#   * Copyright 2024 -- 2025 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

import sys
import os
import json
import torch
import triton
import hashlib
import time
import asyncio
from distutils.util import strtobool

from triton_dejavu import __version__ as dejavu_version
from .dejavu_utilities import (
    get_storage_identifier,
    create_dir_if_not_exist_recursive,
    flag_print_debug,
    flag_print_debug_verbose,
    get_storage_prefix,
    get_storage_tag,
)
from triton.runtime.jit import DependenciesFinder


def _create_tuple(k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        if e[0] == "'" or e[0] == '"':
            ret.append(e[1:-1])
        else:
            ret.append(e)
    ret_t = tuple(ret)
    return ret_t


__int_config_args__ = ["num_warps", "num_stages", "num_ctas"]
__int_or_none_config_args__ = ["maxnreg"]
__bool_config_args__ = ["enable_warp_specialization"]
__config_args__ = (
    ["pre_hook"]
    + __int_config_args__
    + __bool_config_args__
    + __int_or_none_config_args__
)
__skip_config_args__ = ["enable_persistent"]


def _get_cache_template(fn, configs_len=0, autotuner_keys=None):
    ret = {
        "signature": str(fn),
        "total_bench_time_s": 0.0,
        "total_configs": configs_len,
        "current_eval": {},
        "keys": autotuner_keys,
        "cache": {},
        "timings": {},
    }
    return ret

def _create_config_args(v):
    vlist = v.split(", ")
    ret = {"kwargs": {}}
    for e in vlist:
        sl = e.split(": ")
        if sl[0] in __skip_config_args__:
            continue
        if sl[0] in __config_args__:
            if sl[0] in __int_config_args__:
                ret[sl[0]] = int(sl[1])
            elif sl[0] in __bool_config_args__:
                ret[sl[0]] = bool(strtobool(sl[1]))
            elif sl[0] in __int_or_none_config_args__:
                try:
                    ret[sl[0]] = int(sl[1])
                except ValueError:
                    ret[sl[0]] = None
            else:
                ret[sl[0]] = sl[1]
        else:
            try:
                ret["kwargs"][sl[0]] = int(sl[1])
            except ValueError:
                try:
                    ret["kwargs"][sl[0]] = bool(strtobool(sl[1]))
                except ValueError:
                    print(
                        f"[triton-dejavu] WARNING: can't determine type of kwarg {sl[0]}: {sl[1]}"
                    )
                    ret["kwargs"][sl[0]] = sl[1]
    return ret


# async def _get_fn_hash(fn: triton.JITFunction):
#     # trigger JIT
#     test = fn.cache_key
#     from triton.runtime.jit import DependenciesFinder
#
#     while fn.hash is None:
#         await asyncio.sleep(0.1)
#     starting_line_number = str(fn.starting_line_number)
#     corrected_fn_hash = fn.hash[: -(len(starting_line_number))]
#     # assert fn.hash == corrected_fn_hash + starting_line_number
#     return corrected_fn_hash
#
# def _wait_fn_hash(fn):
#     # loop = asyncio.new_event_loop()
#     # task = loop.create_task(_get_fn_hash(fn))
#     # loop.run_until_complete(task)
#     # fn_hash = task.result()
#     fn_hash = _get_weak_fn_hash(fn)
#     return fn_hash


def _get_weak_fn_hash(fn: triton.JITFunction):
    # we are not a compiler, just an autotuner match, we don't need globals
    dependencies_finder = DependenciesFinder(name=fn.__name__, globals={}, src=fn.src)
    dependencies_finder.visit(fn.parse())
    return dependencies_finder.ret


def _get_folder_name(fn_name, fn_hash, configs_hash, key_hash, param_hash):
    storage_tag = get_storage_tag()
    fn_hash_256 = get_string_hash(f"{fn_hash}")
    folder_tree_name = f"{fn_name}/autotune_config-{param_hash}/code_version-{fn_hash_256}/tune_features-{key_hash}/kernel_configs-{configs_hash}/{storage_tag}"
    return folder_tree_name

def get_config_list_hash(configs):
    # sorted_configs = configs.sort()
    # order of config list should be the same if come from ConfigSpace
    # if manual, then can't be guaranteed
    #  (but may not matter in practice, since then also the source code may has changed?)
    s = "|"
    for c in configs:
        s += f"{c}|"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h


def get_list_hash(l):
    s = "|".join(l)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h


def get_string_hash(s):
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h


class DejavuStorage:
    def __init__(self) -> None:
        self.storage_prefix = get_storage_prefix()
        self.storage_identifier = get_storage_identifier()
        self.default_storage_path = os.path.abspath(
            os.path.join(self.storage_prefix, self.storage_identifier)
        )
        create_dir_if_not_exist_recursive(self.default_storage_path)
        self.fn_storage = {}
        self.all_storage = {}
        self.measured_timings = {}
        self._known_files = []
        self.used_configs = {}
        self.partial_results = []
        self.folder_name_to_storage_path = {}

    def __store__(self, is_all_config=False):
        storage = self.all_storage if is_all_config else self.fn_storage
        for folder_name in storage:
            file_name = self._get_cache_file_path(folder_name, is_all_config)
            dir_name = os.path.dirname(file_name)
            create_dir_if_not_exist_recursive(dir_name)
            if file_name not in self._known_files:
                self._known_files.append(file_name)
            with open(file_name, "w") as f:
                print(f"Dumping all {storage[folder_name]}")
                json.dump(storage[folder_name], f, indent=4)
            try:
                os.chmod(file_name, 0o0777)
            except PermissionError as e:
                print(f"can't set permission of cache file {file_name}: {e}")

    def add_cache_data_path_prefix(
        self, new_path, fn, configs_hash, key_hash, param_hash
    ):
        fn_hash = _get_weak_fn_hash(fn)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(
            fn_name, fn_hash, configs_hash, key_hash, param_hash
        )
        self.add_cache_data_path_prefix_for_folder(new_path, folder_name)

    def add_cache_data_path_prefix_for_folder(self, new_path, folder_name):
        if folder_name in self.folder_name_to_storage_path:
            raise Exception(
                f"[triton-dejavu] There exist already a custom dejavu storage path for {folder_name} ({self.folder_name_to_storage_path[folder_name]}), can't over write it."
            )
        self.folder_name_to_storage_path[folder_name] = os.path.abspath(
            os.path.join(new_path, self.storage_identifier)
        )
        if flag_print_debug:
            print(
                f"[triton-dejavu] Adding {self.folder_name_to_storage_path[folder_name]} as custom dejavu storage path for {folder_name}."
            )

    def _get_cache_file_prefix(self, folder_name):
        if folder_name in self.folder_name_to_storage_path:
            print(f"folder name in get cache file prefix {folder_name}")
            if flag_print_debug_verbose:
                print(
                    f"[triton-dejavu] Using {self.folder_name_to_storage_path[folder_name]} as custom dejavu storage path for {folder_name}."
                )
            return self.folder_name_to_storage_path[folder_name]
        return self.default_storage_path

    def _get_cache_file_path(self, folder_name, is_all_config=False):
        file_suffix = "all.json" if is_all_config else "cache.json"
        file_name = os.path.join(
            self._get_cache_file_prefix(folder_name), folder_name, file_suffix
        )
        return file_name

    def store_all_config_results(self, cache, all_timings, key, fn, configs_hash, key_hash, param_hash, configs_len, bench_time, use_cuda_graph, autotuner_keys):
        fn_hash = _get_weak_fn_hash(fn)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(
            fn_name, fn_hash, configs_hash, key_hash, param_hash
        )
        if folder_name not in self.all_storage:
            all_json = _get_cache_template(fn, configs_len, autotuner_keys)
            tmp_used_configs = []
        else:
            all_json = self.all_storage[folder_name]
            tmp_used_configs = self.used_configs[folder_name]
        changes_made = False
        if key not in cache.items() or len(cache[key]) < len(all_timings[key]):
            changes_made = True
            vals = all_timings[key] # A list of config and time 
            for item in vals:
                item['config'] = str(item['config'])
                if item['config'] not in tmp_used_configs:
                    tmp_used_configs.append(item['config'])
                    if flag_print_debug:
                        print(
                            f"[triton-dejavu] added {str(item['config'])} for {folder_name} and key {key}"
                        )
            all_json["timings"][str(key)] = vals
            all_json["total_configs"] = configs_len
            all_json["current_eval"][str(key)] = len(all_timings[key])
            
        if changes_made:
            all_json["total_bench_time_s"] += bench_time
            all_json["keys"] = autotuner_keys
            all_json["cuda_graphs"] = use_cuda_graph
            self.all_storage[folder_name] = all_json
            self.used_configs[folder_name] = tmp_used_configs
            print("Now we are about to store all configurations")
            self.__store__(is_all_config=True)
    
    def add_autotuner_cache(
        self,
        cache,
        fn,
        configs_hash,
        key_hash,
        param_hash,
        configs_len,
        timings,
        repetitiont,
        warmupt,
        bench_time,
        use_cuda_graph,
        autotuner_keys,
    ):
        fn_hash = _get_weak_fn_hash(fn)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(
            fn_name, fn_hash, configs_hash, key_hash, param_hash
        )
        if folder_name not in self.fn_storage:
            cache_json = _get_cache_template(fn, configs_len, autotuner_keys)
            tmp_used_configs = []
        else:
            # TODO: reload content to avoid overwriting in case of parallel processes?
            print("Entered in storage")
            cache_json = self.fn_storage[folder_name]
            tmp_used_configs = self.used_configs[folder_name]
        changes_made = False
        timings_data = None
        for key, config in cache.items():
            if str(key) in cache_json["cache"]:
                continue
            # compatability with cuda stream feature of triton 3
            vals = timings[key]
            if type(vals) is not list:
                vals = [vals]
            if timings_data is None:
                # it is protected by hash, so it is the same for all entries in the file
                if len(vals) == 1:
                    labels = ["ms"]
                else:
                    labels = ["ms", "min_ms", "max_ms"]
                timings_data = {
                    "labels": labels,
                    "rep_t_ms": repetitiont,
                    "warmup_t_ms": warmupt,
                    "cuda_graphs": use_cuda_graph,
                }
            if float("inf") in vals:
                continue
            cache_json["cache"][str(key)] = str(config)
            cache_json["timings"][str(key)] = vals
            cache_json["total_configs"] = configs_len
            if config not in tmp_used_configs:
                tmp_used_configs.append(config)
            changes_made = True
            if flag_print_debug:
                print(
                    f"[triton-dejavu] added {str(config)} for {folder_name} and key {key}"
                )
        if changes_made:
            cache_json["timings_data"] = timings_data
            cache_json["total_bench_time_s"] += bench_time
            cache_json["keys"] = autotuner_keys
            self.fn_storage[folder_name] = cache_json
            self.used_configs[folder_name] = tmp_used_configs
            print("Now we are about to store")
            self.__store__()

    def restore_autotuner_cache(
        self, fn, configs_hash, key_hash, param_hash, all_pre_hook=None
    ):
        # we need to consider dependencies as well, so we will wait for fn.hash
        fn_hash = _get_weak_fn_hash(fn)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(
            fn_name, fn_hash, configs_hash, key_hash, param_hash
        )
        cache_file = self._get_cache_file_path(folder_name)
        all_config_cache = self._get_cache_file_path(folder_name, True)
        if not os.path.isfile(cache_file):
            if flag_print_debug:
                print(f"[triton-dejavu] No configurations found for {folder_name}.")
            # create cache file early
            cache_json = _get_cache_template(fn)
            all_json = _get_cache_template(fn)
            self.fn_storage[folder_name] = cache_json
            self.all_storage[folder_name] = all_json
            self.used_configs[folder_name] = []
            self.__store__()
            self.__store__(True)
            return {}, {}
        if cache_file not in self._known_files:
            self._known_files.append(cache_file)
        if all_config_cache not in self._known_files:
            self._known_files.append(all_config_cache)
        with open(cache_file, "r") as f:
            cache_json = json.load(f)
        with open(all_config_cache, "r") as f:
            all_json = json.load(f)
        self.fn_storage[folder_name] = cache_json
        self.all_storage[folder_name] = all_json
        ret = {}
        tmp_used_configs = []
        partial_configs = {}
        key_list = []     

        for k, v in cache_json["cache"].items():
            kt = _create_tuple(k)
            key_list.append(kt)
            va = _create_config_args(v)
            if all_pre_hook is not None:
                va["pre_hook"] = all_pre_hook
            c = triton.Config(**va)
            ret[kt] = c
            if c not in tmp_used_configs:
                tmp_used_configs.append(c)
            if flag_print_debug_verbose:
                print(
                    f"[triton-dejavu] restored {str(c)} for {folder_name} and key {kt}"
                )
        self.used_configs[folder_name] = tmp_used_configs

        for k, v in all_json['timings'].items():
            kt = _create_tuple(k)
            if kt not in key_list:
                print("KT NOT THERERE")
                partial_configs[kt] = []
                for val in v:
                    va = _create_config_args(val['config'])
                    partial_configs[kt].append({'config': triton.Config(**va), 'runtime': val['runtime'], 'compile_time': val['compile_time']})
                    # partial_configs[kt].append(val)
        if flag_print_debug:
            print(
                f"[triton-dejavu] restored {len(ret)} configurations for {folder_name}."
            )
        return ret, partial_configs

    def get_used_configs(self, fn, configs_hash, key_hash, param_hash):
        fn_hash = _get_weak_fn_hash(fn)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(
            fn_name, fn_hash, configs_hash, key_hash, param_hash
        )
        return self.used_configs[folder_name]

    def print_storage_info(self):
        print(
            f"DejavuStorage path:\t\t{self.storage_prefix}\nDejavuStorage identifier:\t{self.storage_identifier}\n"
            f"\tknown storage keys: {list(self.fn_storage.keys())}"
        )

    def dump_storage(self, filter_timings=False):
        self.print_storage_info()
        if filter_timings:
            tmp_json = {}
            for k, v in self.fn_storage.items():
                tmp_d = {}
                for kk, vv in v.items():
                    if kk == "timings":
                        continue
                    tmp_d[kk] = vv
                tmp_json[k] = tmp_d
            print(json.dumps(tmp_json, indent=4))
        else:
            print(json.dumps(self.fn_storage, indent=4))
            print(json.dumps(self.all_storage, indent=4))


global_dejavu_storage = DejavuStorage()
