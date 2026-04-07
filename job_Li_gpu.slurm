#!/bin/bash
#SBATCH --job-name=Li_gpu
#SBATCH --output=logs/Li_gpu.%J.out
#SBATCH --error=logs/Li_gpu.%J.err
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference
export DMRG_MODE=Li
export USE_GPU=true
export USE_QN=true
export LI_MAXBOND=1000

echo "Li GPU job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Threads: $JULIA_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /mnt/home/xgao1/work/tnn_reference

~/.juliaup/bin/julia --project=. --threads=${JULIA_NUM_THREADS} dmrg_conv.jl

echo "Done at $(date)"
