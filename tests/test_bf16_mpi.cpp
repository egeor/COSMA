/**
 * Multi-MPI-rank BFloat16 GEMM benchmark using COSMA's decomposition algorithm.
 * The global M x N x K problem is decomposed across P ranks by COSMA strategy.
 * Each rank gets a sub-problem; local GEMM calls sfc_ca_gemm.
 *
 * Design: Packing happens ONCE before the timed loop; all timed iterations
 *         operate natively on packed tensors. Unpacking happens ONCE after
 *         the timed loop (only for correctness validation).
 *
 * Includes:
 *   - Correctness check (single-rank reference GEMM)
 *   - Distributed correctness check (validates C = A*B across all ranks)
 *   - Performance with timing breakdown (compute, MPI breakdown)
 *
 * Usage: mpirun -n P ./test_bf16_mpi M N K [nreps]
 */
#include <cosma/bfloat16.hpp>
#include <cosma/multiply.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>
#include <cosma/matrix.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/sfc_gemm_wrapper.hpp>
#include <cosma/blas.hpp>
#include <mkl.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using bf16 = cosma::bfloat16;

// ---------------------------------------------------------------------------
//  MPI call timing via PMPI wrappers
// ---------------------------------------------------------------------------
static double t_allgather = 0, t_reduce_scatter = 0, t_barrier = 0;
static int    n_allgather = 0, n_reduce_scatter = 0;

extern "C" {

int MPI_Allgather(const void *sb, int sc, MPI_Datatype st,
                  void *rb, int rc, MPI_Datatype rt, MPI_Comm c) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Allgather(sb, sc, st, rb, rc, rt, c);
    t_allgather += PMPI_Wtime() - t0;
    ++n_allgather;
    return r;
}

int MPI_Allgatherv(const void *sb, int sc, MPI_Datatype st,
                   void *rb, const int *rc, const int *d,
                   MPI_Datatype rt, MPI_Comm c) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Allgatherv(sb, sc, st, rb, rc, d, rt, c);
    t_allgather += PMPI_Wtime() - t0;
    ++n_allgather;
    return r;
}

int MPI_Reduce_scatter_block(const void *sb, void *rb, int rc,
                             MPI_Datatype dt, MPI_Op op, MPI_Comm c) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Reduce_scatter_block(sb, rb, rc, dt, op, c);
    t_reduce_scatter += PMPI_Wtime() - t0;
    ++n_reduce_scatter;
    return r;
}

int MPI_Reduce_scatter(const void *sb, void *rb, const int *rc,
                       MPI_Datatype dt, MPI_Op op, MPI_Comm c) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Reduce_scatter(sb, rb, rc, dt, op, c);
    t_reduce_scatter += PMPI_Wtime() - t0;
    ++n_reduce_scatter;
    return r;
}

int MPI_Barrier(MPI_Comm c) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Barrier(c);
    t_barrier += PMPI_Wtime() - t0;
    return r;
}

} // extern "C"

static void reset_mpi_timers() {
    t_allgather = t_reduce_scatter = t_barrier = 0;
    n_allgather = n_reduce_scatter = 0;
}

// Reference GEMM in float32 (col-major, no-trans)
static void ref_gemm_f32(int M, int N, int K,
                         const float *A, const float *B, float *C) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p)
                acc += A[p * M + i] * B[j * K + p];
            C[j * M + i] = acc;
        }
    }
}

// Run a single-rank correctness test using cosma::gemm<bf16> directly
static bool correctness_test(int rank) {
    const int M = 1024, N = 1024, K = 1024;
    if (rank == 0) {
        std::cout << "\n--- Correctness test: " << M << "x" << N << "x" << K
                  << " (single-rank gemm) ---\n";
    }
    if (rank != 0) return true; // only rank 0 runs this

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> A(M * K), B(K * N), C(M * N, bf16(0.0f));
    std::vector<float> Af(M * K), Bf(K * N), Cf(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        float v = dist(rng);
        A[i] = bf16(v);
        Af[i] = float(A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        float v = dist(rng);
        B[i] = bf16(v);
        Bf[i] = float(B[i]);
    }

    // Reset pack cache so the correctness test packs freshly
    cosma::get_sfc_gemm_cache().reset_pack_cache();

    cosma::gemm(M, N, K, bf16(1.0f), A.data(), M, B.data(), K,
                bf16(0.0f), C.data(), M);
    ref_gemm_f32(M, N, K, Af.data(), Bf.data(), Cf.data());

    double max_abs = 0;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float got = float(C[i]), ref = Cf[i];
        double ae = std::fabs(got - ref);
        if (ae > max_abs) max_abs = ae;
        sum_sq_err += ae * ae;
        sum_sq_ref += (double)ref * ref;
    }
    double check_norm = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;
    std::cout << "  Max abs err: " << max_abs
              << "  Check-norm: " << check_norm << "\n";
    bool ok = (check_norm < 0.01);
    std::cout << "  " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // --- Single-rank correctness test (rank 0 only) ---
    int correct = correctness_test(rank) ? 1 : 0;
    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!correct) {
        if (rank == 0) std::cerr << "Correctness check FAILED, aborting.\n";
        MPI_Finalize();
        return 1;
    }

    // --- Parse args ---
    int M = 16384, N = 16384, K = 16384, nreps = 3;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc > 4) nreps = std::atoi(argv[4]);

    cosma::Strategy strategy(M, N, K, nprocs);

    if (rank == 0) {
        std::cout << "\n--- Performance test ---\n";
        std::cout << "Global GEMM   : " << M << " x " << N << " x " << K << "\n";
        std::cout << "Ranks         : " << nprocs << "\n";
        std::cout << "Strategy      : " << strategy << "\n";
        std::cout << "Reps          : " << nreps << "\n" << std::flush;
    }

    auto ctx = cosma::make_context<bf16>();

    cosma::CosmaMatrix<bf16> A(ctx, 'A', strategy, rank);
    cosma::CosmaMatrix<bf16> B(ctx, 'B', strategy, rank);
    cosma::CosmaMatrix<bf16> C(ctx, 'C', strategy, rank);

    // Fill with deterministic random data (fixed seed per rank)
    {
        std::mt19937 rng(42 + rank);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < (size_t)A.matrix_size(); ++i)
            A.matrix_pointer()[i] = bf16(dist(rng));
        for (size_t i = 0; i < (size_t)B.matrix_size(); ++i)
            B.matrix_pointer()[i] = bf16(dist(rng));
    }

    bf16 alpha(1.0f), beta(0.0f);

    // --- Distributed correctness check ---
    // 1) Gather global A and B to rank 0 BEFORE multiply
    auto mpi_type = cosma::mpi_mapper<bf16>::getType();
    std::vector<bf16> all_A_buf, all_B_buf;
    if (rank == 0) {
        size_t total_A = 0, total_B = 0;
        for (int i = 0; i < strategy.P; ++i) {
            total_A += A.matrix_size(i);
            total_B += B.matrix_size(i);
        }
        all_A_buf.resize(total_A);
        all_B_buf.resize(total_B);
        std::memcpy(all_A_buf.data(), A.matrix_pointer(),
                    A.matrix_size() * sizeof(bf16));
        std::memcpy(all_B_buf.data(), B.matrix_pointer(),
                    B.matrix_size() * sizeof(bf16));
        size_t off_a = A.matrix_size(), off_b = B.matrix_size();
        for (int i = 1; i < strategy.P; ++i) {
            PMPI_Recv(all_A_buf.data() + off_a, A.matrix_size(i), mpi_type,
                      i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            off_a += A.matrix_size(i);
            PMPI_Recv(all_B_buf.data() + off_b, B.matrix_size(i), mpi_type,
                      i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            off_b += B.matrix_size(i);
        }
    } else if (rank < strategy.P) {
        PMPI_Ssend(A.matrix_pointer(), A.matrix_size(), mpi_type,
                   0, 100, MPI_COMM_WORLD);
        PMPI_Ssend(B.matrix_pointer(), B.matrix_size(), mpi_type,
                   0, 101, MPI_COMM_WORLD);
    }

    // Assemble global A and B in column-major on rank 0
    std::vector<float> globAf, globBf;
    if (rank == 0) {
        globAf.resize((size_t)M * K, 0.0f);
        globBf.resize((size_t)K * N, 0.0f);
        size_t off_a = 0;
        for (int i = 0; i < strategy.P; ++i) {
            int lsz = A.matrix_size(i);
            for (int j = 0; j < lsz; ++j) {
                auto [row, col] = A.global_coordinates(j, i);
                if (row >= 0 && col >= 0)
                    globAf[col * M + row] = float(all_A_buf[off_a + j]);
            }
            off_a += lsz;
        }
        size_t off_b = 0;
        for (int i = 0; i < strategy.P; ++i) {
            int lsz = B.matrix_size(i);
            for (int j = 0; j < lsz; ++j) {
                auto [row, col] = B.global_coordinates(j, i);
                if (row >= 0 && col >= 0)
                    globBf[col * K + row] = float(all_B_buf[off_b + j]);
            }
            off_b += lsz;
        }
        all_A_buf.clear(); all_A_buf.shrink_to_fit();
        all_B_buf.clear(); all_B_buf.shrink_to_fit();
    }

    // 2) Run COSMA multiply (warmup)
    cosma::get_sfc_gemm_cache().reset_pack_cache();
    cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, alpha, beta);

    // 3) Gather distributed C to rank 0
    std::vector<bf16> all_C_buf;
    if (rank == 0) {
        size_t total_C = 0;
        for (int i = 0; i < strategy.P; ++i)
            total_C += C.matrix_size(i);
        all_C_buf.resize(total_C);
        std::memcpy(all_C_buf.data(), C.matrix_pointer(),
                    C.matrix_size() * sizeof(bf16));
        size_t off_c = C.matrix_size();
        for (int i = 1; i < strategy.P; ++i) {
            PMPI_Recv(all_C_buf.data() + off_c, C.matrix_size(i), mpi_type,
                      i, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            off_c += C.matrix_size(i);
        }
    } else if (rank < strategy.P) {
        PMPI_Ssend(C.matrix_pointer(), C.matrix_size(), mpi_type,
                   0, 102, MPI_COMM_WORLD);
    }

    // 4) On rank 0: assemble global C, compute reference, compare
    if (rank == 0) {
        std::vector<float> globCf((size_t)M * N, 0.0f);
        size_t off_c = 0;
        for (int i = 0; i < strategy.P; ++i) {
            int lsz = C.matrix_size(i);
            for (int j = 0; j < lsz; ++j) {
                auto [row, col] = C.global_coordinates(j, i);
                if (row >= 0 && col >= 0)
                    globCf[col * M + row] = float(all_C_buf[off_c + j]);
            }
            off_c += lsz;
        }
        all_C_buf.clear(); all_C_buf.shrink_to_fit();

        // Reference: C_ref = A * B in float32 using MKL
        std::vector<float> refCf((size_t)M * N, 0.0f);
        std::cout << "\n--- Distributed correctness: " << M << "x" << N
                  << "x" << K << " (float32 ref via MKL) ---\n" << std::flush;
        {
            float one = 1.0f, zero = 0.0f;
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, one,
                        globAf.data(), M,
                        globBf.data(), K,
                        zero, refCf.data(), M);
        }
        globAf.clear(); globAf.shrink_to_fit();
        globBf.clear(); globBf.shrink_to_fit();

        double sum_sq_err = 0.0, sum_sq_ref = 0.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            double ae = std::fabs(globCf[i] - refCf[i]);
            if (ae > max_abs) max_abs = ae;
            sum_sq_err += ae * ae;
            sum_sq_ref += (double)refCf[i] * refCf[i];
        }
        double check_norm = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;
        std::printf("  Max abs err : %.6f\n", max_abs);
        std::printf("  Check-norm  : %.8f\n", check_norm);
        bool ok = (check_norm < 0.01);
        std::cout << "  " << (ok ? "PASSED" : "FAILED") << "\n" << std::flush;
    }
    PMPI_Barrier(MPI_COMM_WORLD);

    // --- Timed performance loop ---
    cosma::get_sfc_gemm_cache().reset_timers();
    reset_mpi_timers();

    PMPI_Barrier(MPI_COMM_WORLD);
    double wall0 = PMPI_Wtime();

    for (int rep = 0; rep < nreps; ++rep) {
        cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, alpha, beta);
    }

    PMPI_Barrier(MPI_COMM_WORLD);
    double wall1 = PMPI_Wtime();
    double wall_per_call = (wall1 - wall0) / nreps;

    // Gather per-phase timers
    auto &tm = cosma::get_sfc_gemm_cache().timers();
    double local_pack_a   = tm.pack_a   / nreps;
    double local_pack_b   = tm.pack_b   / nreps;
    double local_compute  = tm.compute  / nreps;
    double local_unpack_c = tm.unpack_c / nreps;
    double local_gemm_total = local_pack_a + local_pack_b + local_compute + local_unpack_c;

    // MPI breakdown per call
    double local_allgather    = t_allgather    / nreps;
    double local_reduce_scat  = t_reduce_scatter / nreps;
    double local_mpi_total    = local_allgather + local_reduce_scat;
    double local_other        = wall_per_call - local_gemm_total - local_mpi_total;

    // Reduce max across ranks
    double timings[9] = {local_pack_a, local_pack_b, local_compute,
                         local_unpack_c, local_gemm_total,
                         local_allgather, local_reduce_scat,
                         local_mpi_total, local_other};
    double max_t[9];
    PMPI_Reduce(timings, max_t, 9, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double total_gflops = 2.0 * (double)M * N * K * 1e-9;

    if (rank == 0) {
        std::cout << "\n=== COSMA bf16 multiply results ===\n";
        std::cout << "Global GEMM    : " << M << " x " << N << " x " << K << "\n";
        std::cout << "Total GFLOP    : " << total_gflops << "\n";
        std::cout << "Wall time/call : " << wall_per_call << " s\n";
        std::cout << "Total GFLOP/s  : " << total_gflops / wall_per_call << "\n";
        std::cout << "Per-rank avg   : " << total_gflops / wall_per_call / nprocs << " GFLOP/s\n";
        std::cout << "\n--- Timing breakdown (max across ranks, per call) ---\n";
        std::printf("  pack_A         : %.4f s\n", max_t[0]);
        std::printf("  pack_B         : %.4f s\n", max_t[1]);
        std::printf("  compute        : %.4f s\n", max_t[2]);
        std::printf("  unpack_C       : %.4f s\n", max_t[3]);
        std::printf("  gemm total     : %.4f s\n", max_t[4]);
        std::printf("  MPI_Allgather  : %.4f s  (%d calls)\n", max_t[5], n_allgather);
        std::printf("  MPI_ReduceScat : %.4f s  (%d calls)\n", max_t[6], n_reduce_scatter);
        std::printf("  MPI total      : %.4f s\n", max_t[7]);
        std::printf("  other/overhead : %.4f s\n", max_t[8]);
        double compute_gflops = total_gflops / max_t[2];
        std::printf("  compute-only   : %.1f GFLOP/s (agg), %.1f (per rank)\n",
                    compute_gflops, compute_gflops / nprocs);
        std::cout << "===================================\n";
    }

    MPI_Finalize();
    return 0;
}
