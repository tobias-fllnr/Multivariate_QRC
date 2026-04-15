"""
Generates job argument files and submission scripts for running QRC experiments.

Supports three execution modes:
  - Local:  Generates a bash script that runs all jobs sequentially.
  - Condor: Generates an HTCondor submit file.
  - Slurm:  Generates a Slurm array job script.

Usage:
    1. Set the 'name' variable below to the desired experiment.
    2. Adjust the parameter grids for that experiment as needed.
    3. Run: python prepare_jobs_general.py
    4. Execute using one of the generated scripts (see output).
"""

from datetime import datetime
import os
import numpy as np

# ============================================================================
# Configuration: select the experiment to run
# ============================================================================
name = "mixing_capacity_qrc_tilted_tfim"

# ============================================================================
# Parameter grids for each experiment
# ============================================================================
if name == "mixing_capacity_qrc_tilted_tfim":
    params_list = [
        (n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed)
        for n in [6]
        for d in [2]
        for encoding_mode in ["dense"]
        for dt in [1.0]
        for encoding_strength in np.logspace(-4, 0, 13)
        for coupling_strength in np.logspace(-5, 1, 19)
        for gamma in [3.0]
        for seed in range(20)
    ]

elif name == "mixing_capacity_qrc_gaussian":
    params_list = [
        (n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed)
        for n in [6]
        for d in [2]
        for encoding_mode in ["dense"]
        for dt in [1.0]
        for encoding_strength in np.logspace(-4, 0, 13)
        for coupling_strength in np.logspace(-5, 1, 19)
        for gamma in [3.0]
        for seed in range(20)
    ]

elif name == "lorenz63_qrc_tilted_tfim":
    params_list = [
        (n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed)
        for n in [6]
        for d, gamma in [(1, 5.0), (2, 4.0), (3, 5.0)]
        for encoding_mode in ["dense"]
        for dt in [1.0]
        for encoding_strength in np.logspace(-4, 0, 13)
        for coupling_strength in np.logspace(-5, 1, 19)
        for seed in range(20)
    ]

elif name == "lorenz63_qrc_gaussian":
    params_list = [
        (n, d, encoding_mode, dt, encoding_strength, coupling_strength, gamma, seed)
        for n in [6]
        for d, gamma in [(1, 5.0), (2, 10.0), (3, 20.0)]
        for encoding_mode in ["fill"]
        for dt in [1.0]
        for encoding_strength in np.logspace(-4, 0, 13)
        for coupling_strength in np.logspace(-5, 1, 19)
        for seed in range(20)
    ]

elif name == "optuna_mixing_capacity_qrc_tilted_tfim":
    params_list = [
        (n, d, encoding_mode)
        for n in [2, 3, 4, 5, 6]
        for d in [2, 3, 4, 5, 6]
        for encoding_mode in ["dense"]
        if n < d
    ]

elif name == "optuna_mixing_capacity_qrc_gaussian":
    params_list = [
        (n, d, encoding_mode)
        for n in [2, 3, 4, 5, 6, 7, 8]
        for d in [2, 3, 4, 5, 6, 7, 8]
        for encoding_mode in ["dense"]
        if n >= d
    ]

elif name == "optuna_lorenz63_qrc_tilted_tfim":
    params_list = [
        (n, d, target, encoding_mode)
        for n in [6]
        for d in [3]
        for target in ["x_1"]
        for encoding_mode in ["fill"]
    ]

elif name == "optuna_lorenz63_qrc_gaussian":
    params_list = [
        (n, d, target, encoding_mode)
        for n in [6]
        for d in [1, 2, 3]
        for target in ["x_1", "y_1", "z_1", "all_1"]
        for encoding_mode in ["fill"]
    ]

else:
    params_list = None

# ============================================================================
# Generate job files
# ============================================================================
if params_list:
    run_id = datetime.now().strftime("%m%d%H%M")
    relative_outdir = f"Results_run/{name}_results_{run_id}"

    # Create local directories
    os.makedirs(relative_outdir, exist_ok=True)
    os.makedirs("jobs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Filenames for this batch
    args_filename = f"jobs/args_{name}_{run_id}.txt"
    local_filename = f"jobs/run_local_{name}_{run_id}.sh"
    condor_filename = f"jobs/submit_{name}_{run_id}.sub"
    slurm_filename = f"jobs/submit_{name}_{run_id}.slurm"

    target_script = "run_optuna_job" if name.startswith("optuna_") else "run_general_job"

    # A. Write arguments file
    with open(args_filename, "w") as f:
        for params in params_list:
            line_items = [str(p) for p in params]
            line_items.append(relative_outdir)
            line_items.append(name)
            f.write(" ".join(line_items) + "\n")

    # B. Generate local run script (no cluster required)
    with open(local_filename, "w") as f:
        f.write(f"""\
#!/bin/bash
# Local execution script - runs all {len(params_list)} jobs sequentially.
# Usage: bash {local_filename}

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

TOTAL=$(wc -l < {args_filename})
COUNT=0

while IFS= read -r ARGS; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running: python {target_script}.py $ARGS"
    python {target_script}.py $ARGS
done < {args_filename}

echo "All $TOTAL jobs completed."
""")
    os.chmod(local_filename, 0o755)

    # C. Generate HTCondor submit file
    with open(condor_filename, "w") as f:
        f.write(f"""\
Executable = {target_script}.sh
Arguments = $(ARGS)
Log    = logs/$(Cluster)_$(Process).log
Output = logs/$(Cluster)_$(Process).out
Error  = logs/$(Cluster)_$(Process).err
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
request_cpus = 10
request_memory = 8000MB
Queue ARGS from {args_filename}
""")

    # D. Generate Slurm array job script
    with open(slurm_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=2:00:00

# Activate environment (adjust path to your virtual environment)
source .venv/bin/activate

OFFSET=${{OFFSET:-0}}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + OFFSET))

ARGS=$(sed -n "${{LINE_NUM}}p" {args_filename})

python {target_script}.py $ARGS
""")

    print(f"--- Generated {len(params_list)} jobs (run ID: {run_id}) ---")
    print(f"Arguments file  : {args_filename}")
    print(f"")
    print(f"To run locally (no cluster):")
    print(f"  bash {local_filename}")
    print(f"")
    print(f"To submit on HTCondor:")
    print(f"  condor_submit {condor_filename}")
    print(f"")
    print(f"To submit on Slurm:")
    print(f"  sbatch --array=1-{len(params_list)} {slurm_filename}")
else:
    print(f"Unknown experiment name: '{name}'")
