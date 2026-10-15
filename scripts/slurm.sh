#!/usr/bin/env bash
#SBATCH --job-name=geometa-paper
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
task_seeds=(7 19 31 43 59)
task_seed="${task_seeds[${SLURM_ARRAY_TASK_ID}]}"
# Set GEOMETA_PYTHON to an already installed environment on the cluster.
task_python="${GEOMETA_PYTHON:-python}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${task_python}" -m geometa.cli --config configs/paper.yaml --seed "${task_seed}" \
  --output "outputs/paper-seed${task_seed}-job${SLURM_JOB_ID}"

