#!/bin/bash
#SBATCH --job-name=bf16-scaling
#SBATCH --partition=emr
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --constraint="c1state"

# Usage:  sbatch run_bf16_scaling.sbatch <NODES>
#   or:   COSMA_BF16_RESHUFFLE_MODE=1 COSMA_OVERLAP_COMM_AND_COMP=ON \
#            sbatch --export=ALL run_bf16_scaling.sbatch <NODES>
#
# The problem size scales with nodes to keep per-rank work constant:
#   4 nodes  (8 ranks):  16384 x  8192 x 16384
#   8 nodes (16 ranks):  32768 x  8192 x 16384
#  16 nodes (32 ranks):  32768 x 16384 x 32768

NODES=${1:-${SLURM_NNODES}}

# Set output file names using node count and job id
#SBATCH --output=/nfs_home/egeorgan/test_cosma/COSMA/build/bf16_scaling_%j.out
#SBATCH --error=/nfs_home/egeorgan/test_cosma/COSMA/build/bf16_scaling_%j.err

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

# Select problem size based on node count
case ${NODES} in
    4)  M=16384; N=8192;  K=16384 ;;
    8)  M=32768; N=8192;  K=16384 ;;
    16) M=32768; N=16384; K=32768 ;;
    *)  echo "Unsupported node count: ${NODES}"; exit 1 ;;
esac

NREPS=5

cd /nfs_home/egeorgan/test_cosma/COSMA

echo "=== BFloat16 scaling test ==="
echo "Date      : $(date)"
echo "Nodes     : ${NODES}"
echo "Ranks     : $((NODES * 2))"
echo "Hostlist  : ${SLURM_NODELIST}"
echo "Problem   : ${M} x ${N} x ${K}"
echo "Reshuffle : mode ${COSMA_BF16_RESHUFFLE_MODE}"
echo "Overlap   : ${COSMA_OVERLAP_COMM_AND_COMP}"
echo ""

mpirun -n $((NODES * 2)) ./numactl_wrapper.sh ./build/test_bf16_mpi ${M} ${N} ${K} ${NREPS}

echo "=== Done ==="
