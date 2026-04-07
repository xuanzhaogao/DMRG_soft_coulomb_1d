#!/bin/bash
#SBATCH --job-name=Li_cpu
#SBATCH --output=logs/Li_cpu.%J.out
#SBATCH --error=logs/Li_cpu.%J.err
#SBATCH -p ccm
#SBATCH -N1 -n1 -c 16
#SBATCH --time=24:00:00

cd /mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export JULIA_NUM_THREADS=16
export JULIA_PROJECT=/mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export DMRG_SYSTEMS=Li
export DMRG_MAXBOND=1000
export DMRG_ETOL=1e-8
export DMRG_MAXSWEEPS=200
export USE_GPU=false

julia src/dmrg_gpu.jl
