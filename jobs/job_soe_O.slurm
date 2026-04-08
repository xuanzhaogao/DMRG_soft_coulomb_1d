#!/bin/bash
#SBATCH --job-name=SOE_O
#SBATCH --output=logs/soe_O.%J.out
#SBATCH --error=logs/soe_O.%J.err
#SBATCH -p ccm
#SBATCH -C genoa
#SBATCH -N1 -n1 -c 16
#SBATCH --time=24:00:00

cd /mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export JULIA_NUM_THREADS=16
export JULIA_PROJECT=/mnt/home/xgao1/work/DMRG_soft_coulomb_1d

export DMRG_OTEM=O
export DMRG_MAXBOND=1000
export DMRG_ETOL=1e-8
export DMRG_MAXSWEEPS=200

julia src/dmrg_soe_single.jl
