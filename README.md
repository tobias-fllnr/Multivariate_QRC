# Multivariate Quantum Reservoir Computing

Code for reproducing the results of the paper ["Multivariate quantum reservoir computing with discrete and continuous variable systems"](https://arxiv.org/abs/2604.08427) by Fellner, Merklinger, and Holm (2026). The repository implements and compares two types of quantum reservoirs:



- **Discrete-Variable (DV) QRC**: Spin-1/2 systems governed by a Tilted Transverse-Field Ising Model (TiltedTFIM), simulated via Lindblad master equation dynamics using [QuTiP](https://qutip.org/).
- **Continuous-Variable (CV) QRC**: Networks of coupled quantum harmonic oscillators, simulated via exact Gaussian-state (covariance matrix) dynamics.

Both reservoir types are evaluated on two tasks:
1. **Mixing Capacity**: Measures how well the reservoir mixes information from multiple input dimensions (information-processing capacity with cross-dimensional terms).
2. **Lorenz-63 Prediction**: Multi-step ahead prediction of the chaotic Lorenz-63 attractor.

Three multivariate encoding strategies are compared: local (here: `one_to_one`), clustered (here: `fill`), and global (here: `dense`).

## Repository Structure

```
.
├── run_general_job.py        # Core simulation functions and CLI entry point
├── run_optuna_job.py         # Hyperparameter optimization with Optuna
├── prepare_jobs_general.py   # Generates parameter grids and execution scripts
├── average_runs_general.py   # Aggregates per-seed results (pickle) into averages
├── average_runs_optuna.py    # Combines per-trial Optuna results (JSON) into pickle
├── run_general_job.sh        # Shell wrapper for run_general_job.py
├── run_optuna_job.sh         # Shell wrapper for run_optuna_job.py
├── requirements.txt          # Python dependencies
├── utils/                    # Core library modules
│   ├── qrc_spin.py           #   DV reservoir (SpinQRC)
│   ├── qrc_gaussian.py       #   CV reservoir (GaussianQRC)
│   ├── ipc.py                #   Information-processing capacity calculation
│   ├── prediction.py         #   Multi-step prediction with linear regression
│   ├── lorenz63.py           #   Lorenz-63 data generator (RK4 integration)
│   └── data.py               #   Random input sequence generator
├── Results_run/              # Raw per-seed results (one pickle file per job)
├── Results_averaged/         # Averaged results across seeds
├── PlottingNotebooks/        # Jupyter notebooks for generating paper figures
└── Plots/                    # Generated PDF figures
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tobias-fllnr/Multivariate_QRC
cd Multivariate_QRC

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Python version**: 3.10 or later is recommended. 3.12 was used to generate the results.

## Reproducing the Results

The pipeline consists of three stages: (1) running simulations, (2) aggregating results, and (3) plotting.

### Step 1: Configure and Generate Jobs

Edit `prepare_jobs_general.py` to select the experiment by setting the `name` variable at the top of the file. Available experiments:

| `name` | Description |
|---|---|
| `mixing_capacity_qrc_tilted_tfim` | Mixing capacity for the DV (spin) reservoir |
| `mixing_capacity_qrc_gaussian` | Mixing capacity for the CV (Gaussian) reservoir |
| `lorenz63_qrc_tilted_tfim` | Lorenz-63 prediction with the DV reservoir |
| `lorenz63_qrc_gaussian` | Lorenz-63 prediction with the CV reservoir |
| `optuna_mixing_capacity_qrc_tilted_tfim` | Optuna hyperparameter search for DV mixing capacity |
| `optuna_mixing_capacity_qrc_gaussian` | Optuna hyperparameter search for CV mixing capacity |
| `optuna_lorenz63_qrc_tilted_tfim` | Optuna hyperparameter search for DV Lorenz-63 |
| `optuna_lorenz63_qrc_gaussian` | Optuna hyperparameter search for CV Lorenz-63 |

Next, adjust the parameter grids for that experiment as needed. The parameters used in the experiments are detailed in `notes.txt`.

Then generate the job files:

```bash
python prepare_jobs_general.py
```

This creates several files in the `jobs/` directory:
- `args_<name>_<id>.txt` -- one line per job with all parameters
- `run_local_<name>_<id>.sh` -- bash script for local execution (no cluster needed)
- `submit_<name>_<id>.sub` -- HTCondor submit file
- `submit_<name>_<id>.slurm` -- Slurm array job script

### Step 2: Run the Simulations

Choose one of the three execution modes:

**Local (no cluster required):**
```bash
bash jobs/run_local_<name>_<id>.sh
```
This runs all jobs sequentially on the local machine. For parameter sweeps with many seeds this can take a long time, but it requires no cluster infrastructure.

**HTCondor:**
```bash
condor_submit jobs/submit_<name>_<id>.sub
```

**Slurm:**
```bash
sbatch --array=1-<N> jobs/submit_<name>_<id>.slurm
```
where `<N>` is the number of jobs (printed when running `prepare_jobs_general.py`).

Results are written as pickle files to `Results_run/<name>_results_<id>/`.

### Step 3: Aggregate Results

After all jobs complete, aggregate the per-seed results into averaged results.

For grid-search experiments (mixing capacity / Lorenz-63 prediction):
```bash
# Edit the 'dates' and 'name' variables in average_runs_general.py, then:
python average_runs_general.py
```

For Optuna hyperparameter optimization experiments:
```bash
# Edit the 'dates' and 'name' variables in average_runs_optuna.py, then:
python average_runs_optuna.py
```

Averaged/combined results are saved to `Results_averaged/`.

### Step 4: Generate Figures

Open the Jupyter notebooks in `PlottingNotebooks/` to reproduce the paper figures:

```bash
jupyter notebook PlottingNotebooks/
```

The notebooks load data from `Results_averaged/` and save figures to `Plots/`.

## Module Reference

### Quantum Reservoirs

- **`utils/qrc_spin.py` (SpinQRC)**: Simulates an `n`-qubit system with XX coupling and amplitude damping. The Hamiltonian includes a static longitudinal field (sigma_z) and a data-dependent transverse field (sigma_x). Evolution uses `qutip.mesolve`. Observables include local and two-qubit Pauli expectation values. Quantum properties tracked: negativity (entanglement) and l1-norm coherence.

- **`utils/qrc_gaussian.py` (GaussianQRC)**: Simulates `n` coupled harmonic oscillators via exact covariance-matrix propagation (Langevin dynamics with Lyapunov solver for the noise term). No Hilbert-space truncation is needed. Measurements are elements of the covariance matrix (with configurable subsets). Quantum properties tracked: negativity, purity, and squeezing (in dB).

### Encoding Modes

Both reservoirs support three encoding strategies for mapping a `d`-dimensional input to `n` reservoir nodes:

- **`one_to_one`**: Input dimension `i` drives node `i` directly. Requires `n >= d`.
- **`fill`**: Each input dimension is replicated to fill `n // d` nodes (block encoding with per-node noise).
- **`dense`**: A random weight matrix linearly combines all input dimensions onto all nodes (row-normalized for stability).

### Evaluation

- **`utils/ipc.py` (IPC)**: Computes information-processing capacity using Legendre polynomials. Includes an optimized implementation with precomputed pseudo-inverse, cached polynomial transformations, and vectorized delay processing. Reports per-degree capacity and a breakdown of cross-dimensional mixing terms.

- **`utils/prediction.py` (Prediction)**: Multi-step ahead prediction using linear regression. Reports RMSE and NRMSE for each prediction horizon and each output dimension.

### Data Generation

- **`utils/lorenz63.py` (Lorenz63Generator)**: Generates Lorenz-63 time series via RK4 integration with configurable integration and sampling time steps. Includes Lyapunov exponent calculation via Benettin's algorithm.

- **`utils/data.py`**: Generates uniform random input sequences in [0, 1].
