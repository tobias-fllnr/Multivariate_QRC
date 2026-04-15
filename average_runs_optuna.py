"""
Combines per-trial Optuna JSON results into a single pickle file in Results_averaged/.

Usage:
    Set the 'dates' and 'name' variables below, then run:
        python average_runs_optuna.py
"""

import os
import json
import pickle
import time
from joblib import Parallel, delayed


def load_single_json(file_path):
    """Loads a single JSON file and flattens 'best_params' into the top level."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'best_params' in data:
                params = data.pop('best_params')
                data.update(params)
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def combine_results_parallel(dates, name, base_dir="./Results_run", n_jobs=-1):
    """Loads JSON results from multiple date-stamped directories in parallel."""
    time_start = time.time()
    all_file_paths = []

    print("Scanning directories...")
    for date in dates:
        dir_path = os.path.join(base_dir, f"{name}_results_{date}")
        if os.path.exists(dir_path):
            files = [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".json")
            ]
            all_file_paths.extend(files)
        else:
            print(f"Warning: Directory not found: {dir_path}")

    print(f"Found {len(all_file_paths)} JSON files across {len(dates)} directories.")
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_single_json)(fp) for fp in all_file_paths
    )
    results = [r for r in results if r is not None]
    print(f"Loaded in {time.time() - time_start:.2f} seconds.")
    return results


if __name__ == "__main__":
    # ========================================================================
    # Configuration: set the date stamps and experiment name
    # ========================================================================
    dates = ["03151101", "03081329"]
    name = "optuna_lorenz63_qrc_tilted_tfim"
    output_dir_name = f"{name}_results_{dates[-1]}"

    results_run_dir = "./Results_run"
    results_combined_dir = "./Results_averaged"
    os.makedirs(results_combined_dir, exist_ok=True)

    print(f"Looking for results in: {results_run_dir}")
    combined_results = combine_results_parallel(dates, name, base_dir=results_run_dir)

    if not combined_results:
        print("No results found. Exiting.")
        exit()

    time_start = time.time()
    output_filepath = os.path.join(results_combined_dir, f"{output_dir_name}_combined.pkl")
    with open(output_filepath, 'wb') as f:
        pickle.dump(combined_results, f)

    print(f"Saving completed in {time.time() - time_start:.2f} seconds.")
    print(f"Saved {len(combined_results)} combined results to: {output_filepath}")
