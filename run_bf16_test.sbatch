#!/bin/bash
#SBATCH --job-name=bf16-gemm-test
#SBATCH --partition=emr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=bf16_gemm_%j.out
#SBATCH --error=bf16_gemm_%j.err
#SBATCH --exclusive
#SBATCH --constraint="c1state"

source /swtools/intel/mpi/latest/env/vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force

export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact

export FI_PROVIDER=psm3

export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

WORKDIR=/data/nfs_home/egeorgan/cosma/COSMA
LIBXSMM_DIR=$HOME/sfc_ca_gemm_flat_act/sfc_ca_gemm/libxsmm
export LD_LIBRARY_PATH=${LIBXSMM_DIR}/lib:${LD_LIBRARY_PATH}

cd ${WORKDIR}

echo "=== BFloat16 GEMM via sfc_ca_gemm — Multi-MPI Test ==="
echo "Date      : $(date)"
echo "Nodes     : ${SLURM_NNODES}"
echo "Tasks     : ${SLURM_NTASKS}"
echo "CPUs/task : ${SLURM_CPUS_PER_TASK}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Hostlist  : ${SLURM_NODELIST}"
echo ""

# Verify CPU + NUMA binding
echo "--- CPU/NUMA Affinity Check ---"
mpirun -n ${SLURM_NTASKS} ./numactl_wrapper.sh bash -c 'echo "rank ${MPI_LOCALRANKID} on $(hostname): cpus=$(taskset -cp $$), numa=$(numactl --show | grep membind)"'
echo ""

# Run with 16k for perf (correctness is embedded in the test)
for SIZE in "16384 16384 16384 3"; do
    echo "--- Size: ${SIZE} ---"
    mpirun -n ${SLURM_NTASKS} ./numactl_wrapper.sh ./build/test_bf16_mpi ${SIZE}
    echo ""
done

echo "=== Done ==="
