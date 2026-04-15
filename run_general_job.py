import pickle
import sys
import os
import numpy as np
from utils.data import generate_random_sequence
from utils.prediction import Prediction
from utils.qrc_spin import SpinQRC
from utils.ipc import IPC
from utils.qrc_gaussian import GaussianQRC
from utils.lorenz63 import Lorenz63Generator


def run_mixing_capacity_qrc_tilted_tfim(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = 1000
    train_length = 10000
    total_length = washout + train_length
    data = generate_random_sequence(length=total_length, dimension=d, seed=seed)
    reservoir = SpinQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, model="TiltedTFIM", encoding_method=encoding_mode, observables="local_and_twoqubit", seed=seed)
    measurements, negativity, coherence = reservoir.run(data, return_negativity=True, return_coherence=True)
    measurements_local = measurements[:, :3*n]
    max_degree = min(d, 2)
    ipc_local = IPC(values=measurements_local, targets=data, washout=washout, train_length=train_length)
    mixing_capacity_breakdown_local, capacity_mixing_local = ipc_local.ipc(max_delay=50, max_degree=max_degree, return_ipc=False, return_capacity_mixing=True)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in capacity_mixing_local.items()})
    result.update({f"first_moment_breakdown_{k}": v for k, v in mixing_capacity_breakdown_local.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result['coherence'] = np.mean(coherence[washout:washout+train_length])
    return result


def run_mixing_capacity_qrc_gaussian(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = 1000
    train_length = 10000
    total_length = washout + train_length
    data = generate_random_sequence(length=total_length, dimension=d, seed=seed)
    reservoir = GaussianQRC(
        n=n,
        encoding_strength=encoding_strength,
        coupling_strength=coupling_strength,
        gamma=gamma,
        dt=dt,
        cov_measurements="q_only",
        encoding_mode=encoding_mode,
        seed=seed,
        return_fourth_moments=True
    )
    means, flat_covs, fourth_moments, negativity, purity, squeezing = reservoir.run(
        data, return_negativity=True, return_purity=True, return_squeezing=True
    )
    max_degree = min(d, 2)
    ipc_covs = IPC(values=flat_covs, targets=data, washout=washout, train_length=train_length)
    mixing_capacity_breakdown_covs, capacity_mixing_covs = ipc_covs.ipc(max_delay=50, max_degree=max_degree, return_ipc=False, return_capacity_mixing=True)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in capacity_mixing_covs.items()})
    result.update({f"first_moment_breakdown_{k}": v for k, v in mixing_capacity_breakdown_covs.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result['purity'] = np.mean(purity[washout:washout+train_length])
    result['squeezing'] = np.mean(squeezing[washout:washout+train_length])
    return result


def run_lorenz63_qrc_tilted_tfim(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = 1000
    train_length = 6000
    test_length = 4000
    total_length = washout + train_length + test_length
    generator = Lorenz63Generator(length=total_length, dt_int=0.005, dt_data=0.092, t_init_cutoff=20.0, seed=42)
    data = generator.generate()
    min_max_normalized_data = normalize_data_min_max(data.copy())
    data_in = min_max_normalized_data[:, :d]
    reservoir = SpinQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, model="TiltedTFIM", encoding_method=encoding_mode, observables="local_and_twoqubit", seed=seed)
    measurements, negativity, coherence = reservoir.run(data_in, return_negativity=True, return_coherence=False)
    measurements_local = measurements[:, :3*n]
    prediction_local = Prediction(observations=measurements_local, data=data, washout=washout, train_length=train_length, test_length=test_length, model="linear")
    pred_results_local = prediction_local.prediction_multi_step(max_steps=10)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in pred_results_local.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result['coherence'] = np.mean(coherence[washout:washout+train_length])
    return result


def run_lorenz63_qrc_gaussian(parameters: dict) -> dict:
    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed = (
        parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']
    )

    washout = 1000
    train_length = 6000
    test_length = 4000
    total_length = washout + train_length + test_length
    generator = Lorenz63Generator(length=total_length, dt_int=0.005, dt_data=0.092, t_init_cutoff=20.0, seed=42)
    data = generator.generate()
    min_max_normalized_data = normalize_data_min_max(data.copy())
    data_in = min_max_normalized_data[:, :d]
    reservoir = GaussianQRC(n=n, encoding_strength=encoding_strength, coupling_strength=coupling_strength, gamma=gamma, dt=dt, encoding_mode=encoding_mode, cov_measurements="q_only", return_fourth_moments=True, seed=seed)
    means, flat_covs, fourth_moments, negativity, purity, squeezing = reservoir.run(
        data_in, return_negativity=False, return_purity=False, return_squeezing=True
    )
    prediction_covs = Prediction(observations=flat_covs, data=data, washout=washout, train_length=train_length, test_length=test_length, model="linear")
    pred_results_covs = prediction_covs.prediction_multi_step(max_steps=10)
    result = {k: parameters[k] for k in ['n', 'd', 'encoding_mode', 'dt', 'encoding_strength', 'coupling_strength', 'gamma', 'seed']}
    result.update({f"first_moment_{k}": v for k, v in pred_results_covs.items()})
    result['negativity'] = np.mean(negativity[washout:washout+train_length])
    result['purity'] = np.mean(purity[washout:washout+train_length])
    result['squeezing'] = np.mean(squeezing[washout:washout+train_length])
    return result


def normalize_data_min_max(data: np.ndarray) -> np.ndarray:
    for dim_i in range(data.shape[1]):
        col = data[:, dim_i]
        d_min, d_max = col.min(), col.max()
        if d_max > d_min:
            data[:, dim_i] = (col - d_min) / (d_max - d_min)
    return data


# Maps task names to their run functions
TASK_FUNCTIONS = {
    "mixing_capacity_qrc_tilted_tfim": run_mixing_capacity_qrc_tilted_tfim,
    "mixing_capacity_qrc_gaussian": run_mixing_capacity_qrc_gaussian,
    "lorenz63_qrc_tilted_tfim": run_lorenz63_qrc_tilted_tfim,
    "lorenz63_qrc_gaussian": run_lorenz63_qrc_gaussian,
}


if __name__ == "__main__":
    args_full = sys.argv[1:]
    name = args_full[-1]
    args = args_full[:-1]

    n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed, outdir = args
    params = {
        'n': int(n),
        'd': int(d),
        'encoding_mode': encoding_mode,
        'dt': float(dt),
        'encoding_strength': float(encoding_strength),
        'coupling_strength': float(coupling_strength),
        'gamma': float(gamma),
        'seed': int(seed)
    }

    run_func = TASK_FUNCTIONS.get(name)
    if run_func is None:
        print(f"Unknown task: {name}")
        sys.exit(1)

    result = run_func(params)

    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"result_n{n}_d{d}_em{encoding_mode}_dt{dt}_es{encoding_strength}_cs{coupling_strength}_gamma{gamma}_seed{seed}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(result, f)