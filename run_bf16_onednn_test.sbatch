#!/bin/bash
#SBATCH --job-name=bf16-onednn
#SBATCH --partition=emr
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=/nfs_home/egeorgan/test_fixed_cosma/COSMA/build/bf16_onednn_%j.out
#SBATCH --error=/nfs_home/egeorgan/test_fixed_cosma/COSMA/build/bf16_onednn_%j.err
#SBATCH --exclusive
#SBATCH --constraint="c1state"

# Strong scaling: 32768 x 32768 x 32768
# Usage:  sbatch --nodes=N run_bf16_onednn_test.sbatch

NODES=${SLURM_NNODES}

source /swtools/intel/2025.3/oneapi-vars.sh --force

export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact
export FI_PROVIDER=psm3

# oneDNN library
export LD_LIBRARY_PATH=$HOME/onednn_2026/oneDNN/install/lib64:$LD_LIBRARY_PATH

# Select GEMM backend: "onednn" or "libxsmm" (default)
export COSMA_GEMM_BACKEND=${COSMA_GEMM_BACKEND:-onednn}

export COSMA_CPU_MEMORY_ALIGNMENT=64
export COSMA_BF16_RESHUFFLE_MODE=${COSMA_BF16_RESHUFFLE_MODE:-0}
export COSMA_OVERLAP_COMM_AND_COMP=${COSMA_OVERLAP_COMM_AND_COMP:-OFF}

M=${COSMA_M:-32768}
N=${COSMA_N:-32768}
K=${COSMA_K:-32768}
NREPS=${COSMA_NREPS:-5}

cd /nfs_home/egeorgan/test_fixed_cosma/COSMA

echo "=== BFloat16 strong scaling 32k^3 test ==="
echo "Date      : $(date)"
echo "Nodes     : ${NODES}"
echo "Ranks     : $((NODES * 2))"
echo "Hostlist  : ${SLURM_NODELIST}"
echo "Problem   : ${M} x ${N} x ${K}"
echo "Backend   : ${COSMA_GEMM_BACKEND}"
echo "COSMA_CPU_MEMORY_ALIGNMENT=${COSMA_CPU_MEMORY_ALIGNMENT}"
echo ""

mpirun -n $((NODES * 2)) ./numactl_wrapper.sh ./build/tests/test_bf16_mpi ${M} ${N} ${K} ${NREPS}

echo "=== Done ==="
