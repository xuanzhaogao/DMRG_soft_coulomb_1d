#!/bin/bash
#SBATCH --job-name=Li_gpu
#SBATCH --output=logs/Li_gpu.%J.out
#SBATCH --error=logs/Li_gpu.%J.err
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH -N1 -n1 -c 16 --gpus-per-task=1
#SBATCH --time=12:00:00

cd /mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export JULIA_NUM_THREADS=16
export JULIA_PROJECT=/mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export DMRG_SYSTEMS=Li
export DMRG_MAXBOND=1000
export DMRG_NSWEEPS=30
export USE_GPU=true

julia src/dmrg_gpu.jl
