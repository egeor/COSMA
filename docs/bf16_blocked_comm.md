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

## Running Experiments

### Prerequisites

```bash
# Intel toolchain
source /swtools/intel/mpi/latest/env/vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh --force

# Build
cd build
make -j$(nproc) cosma

# Compile test
icpx -std=c++17 -O2 -fopenmp -mavx512f -mavx512bw -mavx512bf16 \
  -DCOSMA_WITH_SFC_GEMM -DCOSMA_WITH_MKL_BLAS \
  -I../src -I../libs/sfc_ca_gemm -I_deps/libxsmm-src/include \
  -I_deps/costa-src/src -I_deps/costa-src/src/grid2grid \
  -I$MKLROOT/include -I$I_MPI_ROOT/include \
  ../tests/test_bf16_mpi.cpp \
  -Lsrc/cosma -lcosma -L_deps/costa-build/src/costa -lcosta \
  -L_deps/libxsmm-build -lxsmm -ldl \
  -L$MKLROOT/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
  -liomp5 -lpthread -lm \
  -L$I_MPI_ROOT/lib/release -L$I_MPI_ROOT/lib -lmpi -lmpicxx \
  -o test_bf16_mpi
```

### Test usage

```
mpirun -n P ./numactl_wrapper.sh ./build/test_bf16_mpi M N K [nreps]
```

The test automatically selects PREPACKED or BLOCKED_COMM based on the
strategy. It runs a single-rank correctness check, then a distributed
correctness check (float32 MKL reference), then a timed loop.

### Example: 16-node scaling with different modes

```bash
# Mode 0 (leaf reshuffle), no overlap
sbatch run_bf16_16n_prepacked.sbatch

# Mode 1 (post-allgather reshuffle), no overlap
COSMA_BF16_RESHUFFLE_MODE=1 sbatch --export=ALL run_bf16_16n_prepacked.sbatch

# Mode 0 with communication-computation overlap
COSMA_OVERLAP_COMM_AND_COMP=ON sbatch --export=ALL run_bf16_16n_prepacked.sbatch

# Mode 1 with overlap (best for hiding reshuffle cost)
COSMA_BF16_RESHUFFLE_MODE=1 COSMA_OVERLAP_COMM_AND_COMP=ON \
  sbatch --export=ALL run_bf16_16n_prepacked.sbatch
```

### Scaling sweep (4, 8, 16 nodes)

```bash
for NODES in 4 8 16; do
  sbatch run_bf16_scaling.sbatch $NODES
done
```

See `run_bf16_scaling.sbatch` for the parameterized script.

---

## Performance Results (32768 × 16384 × 32768, 16 nodes / 32 ranks)

| Reshuffle | Overlap | Total GFLOP/s | pack_A (ms) | compute (ms) | Local GFLOP/s/rank |
|-----------|---------|--------------|-------------|-------------|-------------------|
| Mode 0 (leaf) | OFF | 149,587 | 2.4 | 21.3 | 46,394 |
| Mode 1 (post-ag) | OFF | 139,608 | 0.0 | 21.5 | 51,140 |
| Mode 0 (leaf) | ON  | 147,851 | 2.3 | 21.2 | 46,766 |
| Mode 1 (post-ag) | ON  | 140,873 | 0.0 | 21.5 | 51,140 |

- Pure kernel rate: ~51,500 GFLOP/s per rank (~1.66 PFLOP/s aggregate)
- Mode 1 removes 2.3 ms reshuffle from compute path → +9.5% local throughput
- Total GFLOP/s dominated by MPI time (~215 ms); run-to-run variance ~5-7%
