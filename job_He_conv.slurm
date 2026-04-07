#!/bin/bash
#SBATCH --job-name=He_conv
#SBATCH --output=logs/He_conv.%J.out
#SBATCH --error=logs/He_conv.%J.err
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C rome|genoa
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference

echo "He convergence job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Threads: $JULIA_NUM_THREADS  CPU: $(lscpu | grep 'Model name' | head -1)"

cd /mnt/home/xgao1/work/tnn_reference

echo "--- Bond-dim convergence ---"
DMRG_MODE=bondconv ~/.juliaup/bin/julia --project=. --threads=${JULIA_NUM_THREADS} dmrg_conv.jl

echo "--- Grid convergence ---"
DMRG_MODE=gridconv ~/.juliaup/bin/julia --project=. --threads=${JULIA_NUM_THREADS} dmrg_conv.jl

echo "Done at $(date)"
