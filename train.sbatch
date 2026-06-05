#!/bin/bash
# SLURM batch script for Five-Ten-K PPO training on the Stanford ICME cluster.
#
# Usage (override any directive on the command line):
#   sbatch train.sbatch
#   sbatch --cpus-per-task=32 --time=24:00:00 --export=ALL,EPISODES=40000 train.sbatch
#
# Resubmit the SAME command to resume: --resume picks up <out-dir>/train_state.pt.
#
#SBATCH --job-name=shengji
#SBATCH --output=runs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # one process per core; ICME gives 1 core/4GB per CPU
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00             # raise freely; resume makes long runs safe
#SBATCH --signal=B:USR1@120         # SIGUSR1 120s before timeout -> clean state checkpoint
#SBATCH --requeue

set -euo pipefail

# ---- environment (ADJUST these two lines for your ICME setup) ----
module load python/3.12 2>/dev/null || true
source "${VENV:-$HOME/envs/shengji/bin/activate}"

cd "${SLURM_SUBMIT_DIR:-.}"

OUT="runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

EPISODES="${EPISODES:-30000}"
# Leave one core for the learner; ~6 games/worker per PPO update is a good batch.
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
ROLLOUTS=$(( SLURM_CPUS_PER_TASK * 6 ))

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} workers=${WORKERS} rollouts=${ROLLOUTS} out=${OUT}"

# Keep each worker single-threaded (we parallelize across processes, not BLAS threads).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python train_ppo.py \
    --episodes "${EPISODES}" \
    --num-workers "${WORKERS}" \
    --rollouts "${ROLLOUTS}" \
    --device cpu \
    --out-dir "${OUT}" \
    --resume \
    --state-freq 50 \
    --seed "${SEED:-0}"
