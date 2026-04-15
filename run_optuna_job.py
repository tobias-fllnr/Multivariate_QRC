import os

# Set thread limits before importing numerical libraries to prevent
# thread oversubscription when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import copy
import concurrent.futures
import json
import optuna
import numpy as np

from run_general_job import (
    run_mixing_capacity_qrc_tilted_tfim,
    run_mixing_capacity_qrc_gaussian,
    run_lorenz63_qrc_tilted_tfim,
    run_lorenz63_qrc_gaussian,
)


def run_single_seed(run_func, params, seed):
    """Helper function to run a single simulation in an isolated process."""
    local_params = copy.deepcopy(params)
    local_params['seed'] = seed
    try:
        return run_func(local_params)
    except Exception as e:
        print(f"Seed {seed} failed with error: {e}")
        return None


def evaluate(run_func, parameters, metric_key="first_moment_capacity_sum", n_seeds=10):
    """Evaluates a parameter set across multiple seeds in parallel (maximization)."""
    scores = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_seeds) as executor:
        futures = [
            executor.submit(run_single_seed, run_func, parameters, i)
            for i in range(n_seeds)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                return float('-inf'), 0.0
            val = result.get(metric_key)
            if val is None or np.isnan(val):
                return float('-inf'), 0.0
            scores.append(val)
    return np.mean(scores), np.std(scores)


def evaluate_lorenz63(run_func, parameters, metric_key="x_1", n_seeds=10):
    """Evaluates a parameter set across multiple seeds in parallel (minimization)."""
    scores = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_seeds) as executor:
        futures = [
            executor.submit(run_single_seed, run_func, parameters, i)
            for i in range(n_seeds)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                return float('inf'), 0.0
            val = get_nrmse_element(result, metric_key)
            if val is None or np.isnan(val):
                return float('inf'), 0.0
            scores.append(val)
    return np.mean(scores), np.std(scores)


def get_nrmse_element(data_dict, keyword):
    """Retrieves an NRMSE element from a result dictionary based on a keyword like 'x_1', 'all_1'."""
    try:
        prefix, num_str = keyword.split('_')
        row_idx = int(num_str) - 1
    except ValueError:
        raise ValueError(f"Invalid keyword format. Expected format like 'x_1', got: '{keyword}'")

    if prefix == 'all':
        return data_dict['first_moment_nrmse_test_average'][row_idx]
    elif prefix in ['x', 'y', 'z']:
        col_map = {'x': 0, 'y': 1, 'z': 2}
        return data_dict['first_moment_nrmse_test_list'][row_idx, col_map[prefix]]
    else:
        raise ValueError(f"Unknown prefix '{prefix}'. Must be 'x', 'y', 'z', or 'all'.")


def optimize_mixing_capacity_tilted_tfim(n, d, encoding_mode, outdir):
    def objective(trial):
        params = {
            'n': n, 'd': d, 'encoding_mode': encoding_mode, 'dt': 1.0,
            'encoding_strength': trial.suggest_float("encoding_strength", 0.001, 1.0, log=True),
            'coupling_strength': trial.suggest_float("coupling_strength", 0.0001, 10.0, log=True),
            'gamma': trial.suggest_float("gamma", 0.01, 100.0, log=True),
        }
        mean_score, std_score = evaluate(run_func=run_mixing_capacity_qrc_tilted_tfim, parameters=params, metric_key="first_moment_2")
        trial.set_user_attr("std", std_score)
        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    save_results(study, "qrc_tilted_tfim", n, d, outdir, encoding_mode=encoding_mode)


def optimize_mixing_capacity_gaussian(n, d, encoding_mode, outdir):
    def objective(trial):
        params = {
            'n': n, 'd': d, 'encoding_mode': encoding_mode, 'dt': 1.0,
            'encoding_strength': trial.suggest_float("encoding_strength", 0.001, 1.0, log=True),
            'coupling_strength': trial.suggest_float("coupling_strength", 0.0001, 10.0, log=True),
            'gamma': trial.suggest_float("gamma", 0.01, 100.0, log=True),
        }
        mean_score, std_score = evaluate(run_func=run_mixing_capacity_qrc_gaussian, parameters=params, metric_key="first_moment_2")
        trial.set_user_attr("std", std_score)
        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    save_results(study, "qrc_gaussian", n, d, outdir, encoding_mode=encoding_mode)


def optimize_lorenz63_qrc_tilted_tfim(n, d, target, encoding_mode, outdir):
    def objective(trial):
        params = {
            'n': n, 'd': d, 'encoding_mode': encoding_mode, 'dt': 1.0,
            'encoding_strength': trial.suggest_float("encoding_strength", 0.001, 1.0, log=True),
            'coupling_strength': trial.suggest_float("coupling_strength", 0.0001, 10.0, log=True),
            'gamma': trial.suggest_float("gamma", 0.01, 100.0, log=True),
        }
        mean_score, std_score = evaluate_lorenz63(run_func=run_lorenz63_qrc_tilted_tfim, parameters=params, metric_key=target)
        trial.set_user_attr("std", std_score)
        return mean_score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    save_results(study, "qrc_tilted_tfim", n, d, outdir, encoding_mode=encoding_mode, target=target)


def optimize_lorenz63_qrc_gaussian(n, d, target, encoding_mode, outdir):
    def objective(trial):
        params = {
            'n': n, 'd': d, 'encoding_mode': encoding_mode, 'dt': 1.0,
            'encoding_strength': trial.suggest_float("encoding_strength", 0.001, 1.0, log=True),
            'coupling_strength': trial.suggest_float("coupling_strength", 0.0001, 10.0, log=True),
            'gamma': trial.suggest_float("gamma", 0.01, 100.0, log=True),
        }
        mean_score, std_score = evaluate_lorenz63(run_func=run_lorenz63_qrc_gaussian, parameters=params, metric_key=target)
        trial.set_user_attr("std", std_score)
        return mean_score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    save_results(study, "qrc_gaussian", n, d, outdir, encoding_mode=encoding_mode, target=target)


def save_results(study, model_name, n, d, outdir, encoding_mode=None, target=None):
    os.makedirs(outdir, exist_ok=True)
    best_trial = study.best_trial
    results = {
        "best_params": best_trial.params,
        "best_score_mean": best_trial.value,
        "best_score_std": best_trial.user_attrs.get("std"),
        "n": n,
        "d": d,
        "encoding_mode": encoding_mode,
        "target": target
    }
    json_path = os.path.join(outdir, f"best_params_{model_name}_n{n}_d{d}_em{encoding_mode}_t{target}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args_full = sys.argv[1:]
    name = args_full[-1]
    args = args_full[:-1]
    outdir = args[-1]

    if name == "optuna_mixing_capacity_qrc_tilted_tfim":
        n, d, encoding_mode, outdir = args
        optimize_mixing_capacity_tilted_tfim(int(n), int(d), str(encoding_mode), str(outdir))

    elif name == "optuna_mixing_capacity_qrc_gaussian":
        n, d, encoding_mode, outdir = args
        optimize_mixing_capacity_gaussian(int(n), int(d), str(encoding_mode), str(outdir))

    elif name == "optuna_lorenz63_qrc_tilted_tfim":
        n, d, target, encoding_mode, outdir = args
        optimize_lorenz63_qrc_tilted_tfim(int(n), int(d), str(target), str(encoding_mode), str(outdir))

    elif name == "optuna_lorenz63_qrc_gaussian":
        n, d, target, encoding_mode, outdir = args
        optimize_lorenz63_qrc_gaussian(int(n), int(d), str(target), str(encoding_mode), str(outdir))

    else:
        print(f"Unknown task: {name}")
        sys.exit(1)
