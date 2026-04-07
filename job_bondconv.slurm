#!/bin/bash
#SBATCH --job-name=dmrg_conv
#SBATCH --output=logs/bondconv.%J.out
#SBATCH --error=logs/bondconv.%J.err
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C rome
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference
export DMRG_SYSTEMS="H,He"

echo "Bond-dim convergence job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Julia threads: $JULIA_NUM_THREADS"

cd /mnt/home/xgao1/work/tnn_reference

~/.juliaup/bin/julia --project=. \
    --threads=${JULIA_NUM_THREADS} \
    dmrg_bondconv.jl

echo "Done at $(date)"
