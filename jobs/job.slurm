#!/bin/bash
#SBATCH --job-name=dmrg_ref
#SBATCH --output=logs/job.%J.out
#SBATCH --error=logs/job.%J.err
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C genoa
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

mkdir -p logs

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JULIA_PROJECT=/mnt/home/xgao1/work/tnn_reference

# Systems to run (comma-separated). Edit as needed.
export DMRG_SYSTEMS="H,He,Li,Be,B,C,N,O"
export DMRG_MAXBOND=1000
export DMRG_NSWEEPS=25

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Julia threads: $JULIA_NUM_THREADS"
echo "Systems: $DMRG_SYSTEMS"

cd /mnt/home/xgao1/work/tnn_reference

~/.juliaup/bin/julia --project=. \
    --threads=${JULIA_NUM_THREADS} \
    dmrg_ref.jl

echo "Job finished at $(date)"
