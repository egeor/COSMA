#!/bin/bash
# Wrapper script for per-socket MPI rank binding.
# MPI rank 0 → socket 0 (cores 0-63, memory node 0)
# MPI rank 1 → socket 1 (cores 64-127, memory node 1)
#
# Uses taskset for CPU pinning (physical cores only, no HT) and
# numactl --membind for NUMA-local memory allocation.
# Usage: mpirun -n 2 ./numactl_wrapper.sh <program> [args...]

LOCAL_RANK=${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}

if [ "$LOCAL_RANK" -eq 0 ]; then
    exec taskset -c 0-63 numactl --membind=0 "$@"
elif [ "$LOCAL_RANK" -eq 1 ]; then
    exec taskset -c 64-127 numactl --membind=1 "$@"
else
    echo "WARNING: numactl_wrapper: unexpected LOCAL_RANK=$LOCAL_RANK, running without binding"
    exec "$@"
fi
