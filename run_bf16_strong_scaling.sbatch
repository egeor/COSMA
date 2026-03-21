#!/bin/bash
#SBATCH --job-name=bf16-strong
#SBATCH --partition=emr
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --constraint="c1state"

# Strong scaling: fixed 32768 x 32768 x 32768 across 2, 4, 8, 16 nodes
# Usage:  sbatch --nodes=N run_bf16_strong_scaling.sbatch

NODES=${SLURM_NNODES}

source /swtools/intel/mpi/latest/env/vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force

export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact
export FI_PROVIDER=psm3

export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

export COSMA_CPU_MEMORY_ALIGNMENT=64
export COSMA_BF16_RESHUFFLE_MODE=${COSMA_BF16_RESHUFFLE_MODE:-0}
export COSMA_OVERLAP_COMM_AND_COMP=${COSMA_OVERLAP_COMM_AND_COMP:-OFF}

M=32768
N=16384
K=8192
NREPS=5

cd /nfs_home/egeorgan/test_cosma/COSMA

echo "=== BFloat16 strong scaling test ==="
echo "Date      : $(date)"
echo "Nodes     : ${NODES}"
echo "Ranks     : $((NODES * 2))"
echo "Hostlist  : ${SLURM_NODELIST}"
echo "Problem   : ${M} x ${N} x ${K}"
echo "Reshuffle : mode ${COSMA_BF16_RESHUFFLE_MODE} (post-allgather)"
echo ""

mpirun -n $((NODES * 2)) ./numactl_wrapper.sh ./build/test_bf16_mpi ${M} ${N} ${K} ${NREPS}

echo "=== Done ==="
