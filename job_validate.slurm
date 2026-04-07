#!/bin/bash
#SBATCH --job-name=dmrg_validate
#SBATCH --output=logs/validate.%J.out
#SBATCH --error=logs/validate.%J.err
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C rome
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference

# Validate H and He only
export DMRG_SYSTEMS="H,He"
export DMRG_MAXBOND=200
export DMRG_NSWEEPS=15

echo "Validation job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Julia threads: $JULIA_NUM_THREADS"

cd /mnt/home/xgao1/work/tnn_reference

~/.juliaup/bin/julia --project=. \
    --threads=${JULIA_NUM_THREADS} \
    dmrg_ref.jl

echo "Done at $(date)"
