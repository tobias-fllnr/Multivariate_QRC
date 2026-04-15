"""
Aggregates per-seed results from Results_run/ into averaged results in Results_averaged/.

Usage:
    Set the 'dates' and 'name' variables below, then run:
        python average_runs_general.py
"""

import os
import pickle
import time
import numpy as np
from joblib import Parallel, delayed

PARAM_KEYS = ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma']


def aggregate_nested_dicts(dict_list):
    """Recursively averages nested dictionaries of arrays/scalars across seeds."""
    if not dict_list:
        return {}, {}
    if not isinstance(dict_list[0], dict):
        return np.mean(dict_list, axis=0), np.std(dict_list, axis=0)

    mean_dict = {}
    std_dict = {}
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    for k in all_keys:
        sub_list = [d[k] for d in dict_list if k in d]
        if sub_list:
            mean_dict[k], std_dict[k] = aggregate_nested_dicts(sub_list)
    return mean_dict, std_dict


def average_results(results):
    """Groups results by hyperparameters and computes mean/std across seeds."""
    grouped = {}
    for result in results:
        key = tuple(result[k] for k in PARAM_KEYS)
        grouped.setdefault(key, []).append(result)

    averaged_results = []
    for key, group in grouped.items():
        averaged = {k: key[i] for i, k in enumerate(PARAM_KEYS)}
        averaged['num_seeds'] = len(group)

        # Average regular keys (scalars/arrays)
        skip = set(PARAM_KEYS) | {'seed'}
        regular_keys = [k for k in group[0] if k not in skip and 'breakdown' not in k]
        for k in regular_keys:
            values = [entry[k] for entry in group if k in entry]
            if values:
                averaged[f"{k}_mean"] = np.mean(values, axis=0)
                averaged[f"{k}_std"] = np.std(values, axis=0)

        # Average breakdown keys (nested dictionaries)
        breakdown_keys = [k for k in group[0] if 'breakdown' in k]
        for b_key in breakdown_keys:
            b_dicts = [entry[b_key] for entry in group if b_key in entry]
            if b_dicts:
                b_mean, b_std = aggregate_nested_dicts(b_dicts)
                averaged[f"{b_key}_mean"] = b_mean
                averaged[f"{b_key}_std"] = b_std

        averaged_results.append(averaged)
    return averaged_results


def load_single_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_results_parallel(dates, name, base_dir="./Results_run", n_jobs=-1):
    """Loads pickle results from multiple date-stamped directories in parallel."""
    time_start = time.time()
    all_file_paths = []

    print("Scanning directories...")
    for date in dates:
        dir_path = os.path.join(base_dir, f"{name}_results_{date}")
        if os.path.exists(dir_path):
            files = [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".pkl")
            ]
            all_file_paths.extend(files)
        else:
            print(f"Warning: Directory not found: {dir_path}")

    print(f"Found {len(all_file_paths)} files across {len(dates)} directories.")
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_single_pickle)(fp) for fp in all_file_paths
    )
    print(f"Loaded in {time.time() - time_start:.2f} seconds.")
    return results


if __name__ == "__main__":
    # ========================================================================
    # Configuration: set the date stamps and experiment name
    # ========================================================================
    dates = ["04151700"]
    name = "mixing_capacity_qrc_gaussian"
    output_dir_name = f"{name}_results_{dates[-1]}"

    results_run_dir = "./Results_run"
    results_averaged_dir = "./Results_averaged"
    os.makedirs(results_averaged_dir, exist_ok=True)

    print(f"Looking for results in: {results_run_dir}")
    results = load_results_parallel(dates, name, base_dir=results_run_dir)

    if not results:
        print("No results found. Exiting.")
        exit()

    time_start = time.time()
    averaged_results = average_results(results)

    output_filepath = os.path.join(results_averaged_dir, f"{output_dir_name}_averaged.pkl")
    with open(output_filepath, 'wb') as f:
        pickle.dump(averaged_results, f)

    print(f"Averaging completed in {time.time() - time_start:.2f} seconds.")
    print(f"Saved averaged results to: {output_filepath}")
