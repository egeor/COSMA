#!/bin/bash
#SBATCH --job-name=bf16-16n-prepacked
#SBATCH --partition=emr
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=/nfs_home/egeorgan/test_cosma/COSMA/build/bf16_16n_prepacked_%j.out
#SBATCH --error=/nfs_home/egeorgan/test_cosma/COSMA/build/bf16_16n_prepacked_%j.err
#SBATCH --exclusive
#SBATCH --constraint="c1state"

source /swtools/intel/mpi/latest/env/vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force

export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact

export FI_PROVIDER=psm3

export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

# 64-byte aligned COSMA buffers for AMX performance
export COSMA_CPU_MEMORY_ALIGNMENT=64

# BF16 A-reshuffle mode: 0=at leaf GEMM, 1=after allgather (overlap-friendly)
export COSMA_BF16_RESHUFFLE_MODE=${COSMA_BF16_RESHUFFLE_MODE:-0}

cd /nfs_home/egeorgan/test_cosma/COSMA

echo "=== BFloat16 16-node prepacked test ==="
echo "Date      : $(date)"
echo "Nodes     : ${SLURM_NNODES}"
echo "Hostlist  : ${SLURM_NODELIST}"
echo "COSMA_CPU_MEMORY_ALIGNMENT=${COSMA_CPU_MEMORY_ALIGNMENT}"
echo ""

mpirun -n ${SLURM_NTASKS} ./numactl_wrapper.sh ./build/test_bf16_mpi 32768 16384 32768 5

echo "=== Done ==="
