# BFloat16 Blocked-Communication GEMM in COSMA

This document describes the BF16 GEMM modes supported in the
`feature/bf16-blocked-comm` branch, their data layouts, configuration
variables, and how to run scaling experiments.

## Overview

COSMA's BF16 GEMM uses the `sfc_ca_gemm` kernel — a space-filling-curve,
cache-aware BRGEMM built on libxsmm AMX tile instructions.  The kernel
operates on blocked tile layouts with configurable tile sizes
(`bm×bn×bk`, default 32×32×32) and internal parameters `kbf` and
`K_layers` that control brgemm batching and K-layer decomposition.

Three execution modes are supported, selected automatically based on
the COSMA strategy:

| Mode | When selected | A layout during MPI | A layout at kernel | Pack/unpack per leaf call |
|------|---------------|--------------------|--------------------|--------------------------|
| **PREPACKED** | No parallel m/n splits | M-outer VNNI | M-outer VNNI (same) | None |
| **BLOCKED_COMM** | Parallel m or n splits exist | K-outer VNNI | M-outer VNNI (reshuffled) | Reshuffle A only |
| **STANDARD** | Fallback (not used currently) | Column-major | M-outer VNNI | Full pack A + B, unpack C |

---

## Data Layouts

### Matrix A — VNNI blocked format

Two tile orderings exist for A (both use the same intra-tile VNNI
packing `[bk/2][bm][2]`):

**M-outer** `[Mb][Kb][bk/2][bm][2]` — kernel native (`a_k_outer=0`).
Each M-block's K-tiles are contiguous, giving stride = `bm × bk × sizeof(bf16)`
for brgemm.

**K-outer** `[Kb][Mb][bk/2][bm][2]` — MPI communication format.
K outermost makes the layout compatible with COSMA's flat-array comm
operations:
- **Allgather expanding K** (split_n step): concatenation at the array
  end produces valid `[Kb_full][Mb][...]` — no interleaving needed.
- **K-split pointer offset** (split_k step): a contiguous K-slice is a
  simple pointer offset.

Before each leaf GEMM, K-outer A is **reshuffled** to M-outer by
transposing the tile-level `(Kb, Mb)` indices. Each tile (`bm × bk`
elements = 2 KB) is memcpy'd intact. Cost: ~1-3 ms for 128 MB of tiles
at memory bandwidth.

### Matrix B — blocked format

`[Nb][Kb][bn][bk]` — kernel native, no reshuffle needed.

When B is allgathered (split_m steps), N is outermost, so flat
concatenation of N-block chunks produces valid `[Nb_full][Kb][bn][bk]`.

### Matrix C — blocked format

`[Nb][Mb][bn][bm]` — kernel native, no reshuffle needed.

C is only involved in reduce_scatter (split_k steps), which sums
partial results element-wise — layout invariant under addition.

---

## Mode Details

### PREPACKED

Selected when the COSMA strategy has no parallel m or n splits (only
parallel k splits and/or sequential steps). All matrices are packed
once before the timed multiply loop and stay in blocked layout
throughout.

- A: M-outer VNNI `[Mb][Kb][bk/2][bm][2]`
- B: `[Nb][Kb][bn][bk]`
- C: `[Nb][Mb][bn][bm]`
- Zero per-call pack/unpack overhead.

### BLOCKED_COMM

Selected when parallel m or n splits exist (the common case for
multi-node runs). All matrices are packed once before the timed loop.
A uses K-outer layout for MPI compatibility.

Two **reshuffle modes** control when A is converted from K-outer to
M-outer:

#### Reshuffle Mode 0 — at leaf GEMM (default)

- A stays in K-outer layout during all MPI communication.
- At each leaf GEMM call (`blas.cpp`), A is reshuffled from K-outer to
  M-outer into a scratch buffer before calling the kernel.
- Shows up as `pack_A` time in the timing breakdown (~2-3 ms).
- Simple, always correct, no changes to COSMA's communication layer.

#### Reshuffle Mode 1 — after allgather (overlap-friendly)

- After each allgather that expands A (split_n steps in `multiply.cpp`),
  A is reshuffled in-place from K-outer to M-outer.
- In the overlap path (`one_sided_communicator.cpp`), each arriving
  chunk is reshuffled before `local_multiply` is called.
- The leaf GEMM receives M-outer A directly — zero `pack_A` time.
- The reshuffle cost is absorbed into the communication phase, where it
  can overlap with other ranks' computation.

---

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `COSMA_BF16_RESHUFFLE_MODE` | `0`, `1` | `0` | A-reshuffle placement (see above) |
| `COSMA_OVERLAP_COMM_AND_COMP` | `ON`, `OFF` | `OFF` | Enable COSMA communication-computation overlap |
| `COSMA_CPU_MEMORY_ALIGNMENT` | bytes | `0` | Buffer alignment (use `64` for AMX) |
| `LIBXSMM_X86_AMX_GEMM_STREAMING_A` | `0`, `1` | — | libxsmm AMX streaming hint for A |
| `LIBXSMM_X86_AMX_GEMM_STREAMING_B` | `0`, `1` | — | libxsmm AMX streaming hint for B |

### Kernel parameters (compile-time singleton)

Set in `sfc_gemm_wrapper.cpp` singleton:

```cpp
static blocked_layout_desc desc{32, 32, 32, 2, 3};
//                              bm  bn  bk  kbf K_layers
```

- `kbf=2`: K-blocking factor — divides the K-round into 2 sub-rounds.
- `K_layers=3`: number of K-layers — divides K into 3 accumulation layers
  with separate scratch buffers to increase ILP.

---

## Step-by-Step Setup, Build, and Run Guide

This section walks through every step from a clean checkout to running
BF16 GEMM experiments with both backends (libxsmm and oneDNN).

---

### Step 0 — Prerequisites

| Requirement | Version tested | Notes |
|-------------|---------------|-------|
| Intel oneAPI (compiler + MPI + MKL) | 2025.3 | Provides `mpiicpx`, MKL, Intel MPI |
| oneDNN (optional, for oneDNN backend) | 3.12.0 | Pre-built at `~/onednn_2026/oneDNN/install/` |
| CMake | ≥ 3.24 | |
| Slurm | any | Cluster job scheduler |
| SPR/EMR nodes with AMX | Sapphire Rapids / Emerald Rapids | Required for BF16 AMX instructions |

---

### Step 1 — Clone the repository

```bash
git clone -b feature/bf16-blocked-comm https://github.com/egeor/COSMA.git
cd COSMA
```

---

### Step 2 — Load the Intel toolchain

On the **login node** (adjust path if different on your cluster):

```bash
source /data/swtools/intel/2025.3/oneapi-vars.sh --force
```

> **Note:** On compute nodes the path is typically `/swtools/intel/2025.3/oneapi-vars.sh`
> (without `/data`). The sbatch scripts source the compute-node path automatically.

Verify the compiler and MPI are available:

```bash
which mpiicpx    # should print a path
echo $MKLROOT    # should be set (e.g. .../mkl/2025.3)
```

---

### Step 3 — Build oneDNN (if not already installed)

Skip this step if you already have a oneDNN installation.

```bash
cd ~
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx \
         -DCMAKE_C_COMPILER=icx \
         -DCMAKE_INSTALL_PREFIX=$HOME/onednn_install \
         -DCMAKE_BUILD_TYPE=Release \
         -DDNNL_CPU_RUNTIME=OMP
make -j$(nproc) && make install
cd ~
```

The install prefix (here `$HOME/onednn_install`) is passed to COSMA's
cmake as `ONEDNN_PATH`. It must contain `include/` and `lib64/`.

---

### Step 4 — Configure COSMA with CMake

```bash
cd COSMA          # back to COSMA repo root
mkdir -p build && cd build
```

**Both backends (libxsmm + oneDNN):**

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_BLAS=MKL \
  -DCOSMA_SCALAPACK=OFF \
  -DCOSMA_WITH_SFC_GEMM=ON \
  -DCOSMA_WITH_ONEDNN=ON \
  -DONEDNN_PATH=$HOME/onednn_install
```

**libxsmm only (no oneDNN):**

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_BLAS=MKL \
  -DCOSMA_SCALAPACK=OFF \
  -DCOSMA_WITH_SFC_GEMM=ON
```

**oneDNN only (no libxsmm):**

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_BLAS=MKL \
  -DCOSMA_SCALAPACK=OFF \
  -DCOSMA_WITH_ONEDNN=ON \
  -DONEDNN_PATH=$HOME/onednn_install
```

#### CMake options reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `COSMA_WITH_SFC_GEMM` | BOOL | OFF | Enable BF16 GEMM via sfc_ca_gemm (libxsmm AMX BRGEMM) |
| `COSMA_WITH_ONEDNN` | BOOL | OFF | Enable BF16 GEMM via oneDNN matmul |
| `ONEDNN_PATH` | STRING | `""` | Path to oneDNN install prefix (must contain `include/` and `lib64/`) |
| `COSMA_BLAS` | STRING | — | BLAS library: `MKL`, `OPENBLAS`, etc. (required) |
| `COSMA_SCALAPACK` | STRING | OFF | ScaLAPACK (not needed for BF16 tests) |

---

### Step 5 — Build the test binary

```bash
make -j$(nproc) test_bf16_mpi
```

The binary is at `build/tests/test_bf16_mpi`.

**Usage:**

```
mpirun -n P ./numactl_wrapper.sh ./build/tests/test_bf16_mpi M N K [nreps]
```

where `P` = total MPI ranks, `M N K` = problem dimensions, `nreps` = timed
iterations (default 5).

---

### Step 6 — Set runtime environment variables

These environment variables control which backend is used and how the
computation is configured:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `COSMA_GEMM_BACKEND` | `libxsmm`, `onednn` (or `dnnl`) | `libxsmm` | Selects the BF16 GEMM kernel (requires both backends compiled in) |
| `COSMA_BF16_RESHUFFLE_MODE` | `0`, `1` | `0` | When to reshuffle A from K-outer to M-outer: `0`=at leaf GEMM, `1`=after allgather |
| `COSMA_OVERLAP_COMM_AND_COMP` | `ON`, `OFF` | `OFF` | Enable COSMA communication-computation overlap |
| `COSMA_CPU_MEMORY_ALIGNMENT` | bytes | `0` | Buffer alignment (`64` recommended for AMX) |
| `OMP_NUM_THREADS` | integer | — | OpenMP threads per rank |
| `KMP_AFFINITY` | string | — | Intel thread affinity (e.g. `granularity=fine,compact`) |
| `FI_PROVIDER` | string | — | Libfabric provider (e.g. `psm3` for Intel MPI on OPA/OPX) |

**Example (shell):**

```bash
export COSMA_GEMM_BACKEND=onednn      # use oneDNN backend
export COSMA_CPU_MEMORY_ALIGNMENT=64
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact
export FI_PROVIDER=psm3
```

> **Note:** If built with only one backend (`COSMA_WITH_SFC_GEMM=ON` alone,
> or `COSMA_WITH_ONEDNN=ON` alone), the `COSMA_GEMM_BACKEND` variable is
> ignored and the available backend is always used.

---

### Step 7 — Run interactively (single node, quick test)

```bash
# On a compute node (e.g. via salloc)
salloc --nodes=1 --ntasks-per-node=2 --cpus-per-task=64 \
       --partition=emr --constraint="c1state" --exclusive --time=00:10:00

source /swtools/intel/2025.3/oneapi-vars.sh --force
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact
export FI_PROVIDER=psm3
export COSMA_CPU_MEMORY_ALIGNMENT=64
export LD_LIBRARY_PATH=$HOME/onednn_install/lib64:$LD_LIBRARY_PATH

cd /path/to/COSMA

# libxsmm backend, small problem
export COSMA_GEMM_BACKEND=libxsmm
mpirun -n 2 ./numactl_wrapper.sh ./build/tests/test_bf16_mpi 4096 4096 4096 3

# oneDNN backend, same problem
export COSMA_GEMM_BACKEND=onednn
mpirun -n 2 ./numactl_wrapper.sh ./build/tests/test_bf16_mpi 4096 4096 4096 3
```

---

### Step 8 — Run via Slurm (multi-node)

A ready-to-use sbatch script is included in the repo. Copy and edit it,
or use it directly with environment variable overrides.

#### 8a — Create the sbatch script

A template `run_bf16_test.sbatch` is provided. Here is a minimal
example you can adapt:

```bash
#!/bin/bash
#SBATCH --job-name=bf16-test
#SBATCH --partition=emr
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=build/bf16_%j.out
#SBATCH --error=build/bf16_%j.err
#SBATCH --exclusive
#SBATCH --constraint="c1state"

source /swtools/intel/2025.3/oneapi-vars.sh --force

export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact
export FI_PROVIDER=psm3
export COSMA_CPU_MEMORY_ALIGNMENT=64
export LD_LIBRARY_PATH=$HOME/onednn_install/lib64:$LD_LIBRARY_PATH

# Backend selection (override via --export=ALL,COSMA_GEMM_BACKEND=onednn)
export COSMA_GEMM_BACKEND=${COSMA_GEMM_BACKEND:-libxsmm}
export COSMA_BF16_RESHUFFLE_MODE=${COSMA_BF16_RESHUFFLE_MODE:-0}

# Problem size (override via --export=ALL,COSMA_M=32768,...)
M=${COSMA_M:-8192}
N=${COSMA_N:-8192}
K=${COSMA_K:-8192}
NREPS=${COSMA_NREPS:-5}

NODES=${SLURM_NNODES}
cd /path/to/COSMA

echo "Nodes: ${NODES}, Ranks: $((NODES*2)), Problem: ${M}x${N}x${K}, Backend: ${COSMA_GEMM_BACKEND}"

mpirun -n $((NODES * 2)) ./numactl_wrapper.sh ./build/tests/test_bf16_mpi ${M} ${N} ${K} ${NREPS}
```

#### 8b — Submit jobs

```bash
# 2-node run with libxsmm, 8k^3
sbatch --nodes=2 run_bf16_test.sbatch

# 2-node run with oneDNN, 8k^3
sbatch --nodes=2 --export=ALL,COSMA_GEMM_BACKEND=onednn run_bf16_test.sbatch

# 4-node run with libxsmm, 32k^3
sbatch --nodes=4 --export=ALL,COSMA_GEMM_BACKEND=libxsmm,COSMA_M=32768,COSMA_N=32768,COSMA_K=32768 run_bf16_test.sbatch

# 8-node scaling sweep, both backends
for NODES in 2 4 8; do
  sbatch --nodes=$NODES --export=ALL,COSMA_GEMM_BACKEND=libxsmm run_bf16_test.sbatch
  sbatch --nodes=$NODES --export=ALL,COSMA_GEMM_BACKEND=onednn  run_bf16_test.sbatch
done
```

> **Important:** Use `--export=ALL,VAR=val` (not shell `export VAR=val`
> before `sbatch`) to pass environment variables into the job. The `ALL`
> ensures the rest of the environment is inherited.

---

### Step 9 — Check results

```bash
# Check if jobs passed correctness
grep -E 'PASSED|FAILED' build/bf16_*.out

# View performance summary
grep -E 'Nodes|Problem|Backend|Mode|Wall time|Total GFLOP|compute-only|Check-norm' build/bf16_*.out

# Check for errors
cat build/bf16_*.err
```

**Expected output** (example, 2 nodes, 8k^3):

```
Nodes     : 2
Problem   : 8192 x 8192 x 8192
Backend   : libxsmm
  Check-norm    : 0.003106894732730351455646
  PASSED
  Check-norm    : 0.003576104779453027383840
  PASSED
Mode           : BLOCKED_COMM (K-outer A, leaf reshuffle)
Wall time/call : 0.0423407 s
Total GFLOP/s  : 25968.2
  compute-only   : 186856.8 GFLOP/s (agg), 46714.2 (per rank)
```

A check-norm ≤ 0.01 and `PASSED` means correctness is verified against
a float32 MKL reference GEMM.

---

### Step 10 — NUMA binding (numactl_wrapper.sh)

The provided `numactl_wrapper.sh` binds 2 MPI ranks per node to
separate sockets for optimal memory locality on dual-socket systems:

| Local rank | CPU cores | NUMA node |
|-----------|-----------|-----------|
| 0 | 0–63 | 0 |
| 1 | 64–127 | 1 |

This script is required for good performance — without it, ranks may
contend for the same memory controller.

Usage: always launch through the wrapper:

```bash
mpirun -n P ./numactl_wrapper.sh ./build/tests/test_bf16_mpi M N K [nreps]
```

---

### Quick Reference — Complete Workflow

```bash
# 1. Clone
git clone -b feature/bf16-blocked-comm https://github.com/egeor/COSMA.git
cd COSMA

# 2. Load toolchain (login node)
source /data/swtools/intel/2025.3/oneapi-vars.sh --force

# 3. Configure (both backends)
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_BLAS=MKL \
  -DCOSMA_SCALAPACK=OFF \
  -DCOSMA_WITH_SFC_GEMM=ON \
  -DCOSMA_WITH_ONEDNN=ON \
  -DONEDNN_PATH=$HOME/onednn_install

# 4. Build
make -j$(nproc) test_bf16_mpi

# 5. Submit (from repo root)
cd ..
sbatch --nodes=2 --export=ALL,COSMA_GEMM_BACKEND=libxsmm run_bf16_test.sbatch
sbatch --nodes=2 --export=ALL,COSMA_GEMM_BACKEND=onednn  run_bf16_test.sbatch

# 6. Check results
grep -E 'PASSED|FAILED|Total GFLOP' build/bf16_*.out
```

---

## Scaling Results

### 4 nodes / 8 ranks — 16384 × 8192 × 16384

| Reshuffle | Total GFLOP/s | pack_A (ms) | compute (ms) | MPI (ms) | Wall (ms) | Compute GFLOP/s/rank |
|-----------|--------------|-------------|-------------|---------|----------|---------------------|
| Mode 0 (leaf) | 34,339 | 0.8 | 14.0 | 114.0 | 128.1 | 39,367 |
| Mode 1 (post-ag) | 34,144 | 0.9 | 13.8 | 114.9 | 128.8 | 39,833 |

Strategy: `pm2, pk4` — no split_n step, so mode 1 falls back to leaf reshuffle
(the `a_reshuffled_` flag ensures correctness).

### 8 nodes / 16 ranks — 32768 × 8192 × 16384

| Reshuffle | Total GFLOP/s | pack_A (ms) | compute (ms) | MPI (ms) | Wall (ms) | Compute GFLOP/s/rank |
|-----------|--------------|-------------|-------------|---------|----------|---------------------|
| Mode 0 (leaf) | 62,776 | 0.9 | 14.1 | 126.9 | 140.1 | 39,024 |
| Mode 1 (post-ag) | 65,084 | 0.9 | 12.8 | 122.2 | 135.1 | 42,968 |

Strategy: `pm2, pn2, pk4` — has split_n, so mode 1 reshuffles after allgather.
Mode 1 shows +3.7% total GFLOP/s and +10.1% compute GFLOP/s per rank.

### 16 nodes / 32 ranks — 32768 × 16384 × 32768

| Reshuffle | Total GFLOP/s | pack_A (ms) | compute (ms) | MPI (ms) | Wall (ms) | Compute GFLOP/s/rank |
|-----------|--------------|-------------|-------------|---------|----------|---------------------|
| Mode 0 (leaf) | 152,058 | 2.4 | 20.9 | 210.1 | 231.4 | 52,537 |
| Mode 1 (post-ag) | 140,933 | 0.0 | 21.5 | 213.5 | 249.7 | 51,128 |

Strategy: `pm2, pn2, pk8` — has split_n. Mode 1 eliminates pack_A entirely.
Total dominated by MPI time (~210 ms); run-to-run variance ~5-7%.

### Key observations

- All configurations pass correctness checks
- Compute-only kernel rate: ~39k–52k GFLOP/s per rank depending on problem size
- Mode 1 benefit is most visible at 8 nodes (+10% local compute throughput)
- At 4 nodes mode 1 gracefully falls back to leaf reshuffle (no split_n in strategy)
- MPI communication dominates wall time at all scales (80–90%)
