#!/bin/bash
#SBATCH --job-name=Li_dmrg
#SBATCH --output=logs/Li.%J.out
#SBATCH --error=logs/Li.%J.err
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C rome|genoa
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference
export DMRG_MODE=Li
export LI_MAXBOND=1000

echo "Li DMRG job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Threads: $JULIA_NUM_THREADS"

cd /mnt/home/xgao1/work/tnn_reference

~/.juliaup/bin/julia --project=. --threads=${JULIA_NUM_THREADS} dmrg_conv.jl

echo "Done at $(date)"
