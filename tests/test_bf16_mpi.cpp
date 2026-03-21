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
#include <cosma/environment_variables.hpp>
#include <cosma/blas.hpp>
#include <mkl.h>
#include <mpi.h>
#include <libxsmm.h>

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

    // Convert bf16 result to float for libxsmm_matdiff
    std::vector<float> Ctest(M * N);
    for (int i = 0; i < M * N; ++i)
        Ctest[i] = float(C[i]);

    libxsmm_matdiff_info norms;
    libxsmm_matdiff_clear(&norms);
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, (libxsmm_blasint)(M * N), 1,
                    Cf.data(), Ctest.data(), 0, 0);
    std::printf("  L1 reference  : %.25g\n", norms.l1_ref);
    std::printf("  L1 test       : %.25g\n", norms.l1_tst);
    std::printf("  L2 abs.error  : %.24f\n", norms.l2_abs);
    std::printf("  L2 rel.error  : %.24f\n", norms.l2_rel);
    std::printf("  Linf abs.error: %.24f\n", norms.linf_abs);
    std::printf("  Linf rel.error: %.24f\n", norms.linf_rel);
    std::printf("  Check-norm    : %.24f\n", norms.normf_rel);
    bool ok = (norms.normf_rel < 0.01);
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
    if (cosma::get_overlap_comm_and_comp()) {
        strategy.enable_overlapping_comm_and_comp();
    }

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

    // --- Mode selection and packing (before correctness check) ---
    // The correctness check exercises the actual mode (prepacked or
    // blocked_comm) to validate that the kernel produces correct results.
    bool can_prepacked = true;
    for (size_t s = 0; s < strategy.n_steps(); ++s) {
        if (strategy.parallel_step(s) &&
            (strategy.split_m(s) || strategy.split_n(s))) {
            can_prepacked = false;
            break;
        }
    }
    if (rank == 0) {
        std::fprintf(stderr, "[info] can_prepacked=%d (strategy has %zu steps)\n",
                     (int)can_prepacked, strategy.n_steps());
        std::fflush(stderr);
    }

    const char *mode_label = "STANDARD (per-call pack/unpack)";

    // Local C dims for unpack after correctness multiply
    int local_m_C = 0, local_n_C = 0;
    if (rank < strategy.P) {
        auto& c_blocks = C.initial_layout();
        local_m_C = c_blocks[0].rows.end_ - c_blocks[0].rows.start_ + 1;
        local_n_C = c_blocks[0].cols.end_ - c_blocks[0].cols.start_ + 1;
    }

    if (can_prepacked) {
        mode_label = "PREPACKED";
        auto desc = cosma::get_sfc_gemm_cache().block_desc();
        int bm = desc.bm, bn = desc.bn, bk = desc.bk;

        int local_m_A = M;
        int local_k_A = (int)((size_t)A.matrix_size() / local_m_A);
        int local_n_B = N;
        int local_k_B = (int)((size_t)B.matrix_size() / local_n_B);
        size_t a_sz = (size_t)A.matrix_size();
        size_t b_sz = (size_t)B.matrix_size();

        // Pack A in-place (col-major -> VNNI blocked)
        {
            bf16 *a_ptr = A.matrix_pointer();
            int lbm = bm, lbk = bk;
            while (local_m_A % lbm != 0 && lbm > 1) --lbm;
            while (local_k_A % lbk != 0 && lbk > 1) --lbk;
            std::vector<bf16> a_tmp(a_sz);
            if (rank == 0)
                std::fprintf(stderr, "[pack] A: sz=%zu m=%d k=%d bm=%d bk=%d\n",
                    a_sz, local_m_A, local_k_A, lbm, lbk);
            cosma::pack_A_to_blocked_vnni(a_ptr, a_tmp.data(), local_m_A, local_k_A, lbm, lbk);
            std::memcpy(a_ptr, a_tmp.data(), a_sz * sizeof(bf16));
        }
        // Pack B in-place (col-major -> blocked)
        {
            bf16 *b_ptr = B.matrix_pointer();
            int lbk = bk, lbn = bn;
            while (local_k_B % lbk != 0 && lbk > 1) --lbk;
            while (local_n_B % lbn != 0 && lbn > 1) --lbn;
            std::vector<bf16> b_tmp(b_sz);
            if (rank == 0)
                std::fprintf(stderr, "[pack] B: sz=%zu k=%d n=%d bk=%d bn=%d\n",
                    b_sz, local_k_B, local_n_B, lbk, lbn);
            cosma::pack_B_to_blocked(b_ptr, b_tmp.data(), local_k_B, local_n_B, lbk, lbn);
            std::memcpy(b_ptr, b_tmp.data(), b_sz * sizeof(bf16));
        }
        cosma::get_sfc_gemm_cache().set_prepacked(true);
    } else {
        mode_label = "BLOCKED_COMM (K-outer A + reshuffle)";
        auto desc = cosma::get_sfc_gemm_cache().block_desc();
        int bm = desc.bm, bn = desc.bn, bk = desc.bk;

        auto& a_blocks = A.initial_layout();
        int local_m_A = a_blocks[0].rows.end_ - a_blocks[0].rows.start_ + 1;
        int local_k_A = a_blocks[0].cols.end_ - a_blocks[0].cols.start_ + 1;
        size_t a_sz = (size_t)A.matrix_size();

        auto& b_blocks = B.initial_layout();
        int local_k_B = b_blocks[0].rows.end_ - b_blocks[0].rows.start_ + 1;
        int local_n_B = b_blocks[0].cols.end_ - b_blocks[0].cols.start_ + 1;
        size_t b_sz = (size_t)B.matrix_size();

        // Pack A in-place to K-outer VNNI
        // Layout: [Kb][Mb][bk/2][bm][2]
        {
            bf16 *a_ptr = A.matrix_pointer();
            int lbm = bm, lbk_a = bk;
            while (local_m_A % lbm != 0 && lbm > 1) --lbm;
            while (local_k_A % lbk_a != 0 && lbk_a > 1) --lbk_a;
            std::vector<bf16> a_tmp(a_sz);
            if (rank == 0)
                std::fprintf(stderr, "[blocked_comm] pack A K-outer: sz=%zu m=%d k=%d bm=%d bk=%d\n",
                    a_sz, local_m_A, local_k_A, lbm, lbk_a);
            cosma::pack_A_to_blocked_vnni_Kouter(a_ptr, a_tmp.data(),
                                                  local_m_A, local_k_A, lbm, lbk_a);
            std::memcpy(a_ptr, a_tmp.data(), a_sz * sizeof(bf16));
        }
        // Pack B in-place to standard blocked
        {
            bf16 *b_ptr = B.matrix_pointer();
            int lbk = bk, lbn = bn;
            while (local_k_B % lbk != 0 && lbk > 1) --lbk;
            while (local_n_B % lbn != 0 && lbn > 1) --lbn;
            std::vector<bf16> b_tmp(b_sz);
            if (rank == 0)
                std::fprintf(stderr, "[blocked_comm] pack B: sz=%zu k=%d n=%d bk=%d bn=%d\n",
                    b_sz, local_k_B, local_n_B, lbk, lbn);
            cosma::pack_B_to_blocked(b_ptr, b_tmp.data(), local_k_B, local_n_B, lbk, lbn);
            std::memcpy(b_ptr, b_tmp.data(), b_sz * sizeof(bf16));
        }

        cosma::get_sfc_gemm_cache().set_blocked_comm(true);
        int rmode = cosma::get_sfc_gemm_cache().reshuffle_mode();
        if (rmode == 1)
            mode_label = "BLOCKED_COMM (K-outer A, reshuffle=post-allgather)";
        else
            mode_label = "BLOCKED_COMM (K-outer A, reshuffle=leaf-gemm)";
    }

    // 2) Run COSMA multiply (correctness — exercises actual mode)
    cosma::get_sfc_gemm_cache().reset_pack_cache();
    cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, alpha, beta);

    // Unpack C from blocked [Nb][Mb][bn][bm] to col-major for correctness
    if (rank < strategy.P && local_m_C > 0 && local_n_C > 0) {
        auto desc = cosma::get_sfc_gemm_cache().block_desc();
        int bm = desc.bm, bn = desc.bn;
        int lbm = bm, lbn = bn;
        while (local_m_C % lbm != 0 && lbm > 1) --lbm;
        while (local_n_C % lbn != 0 && lbn > 1) --lbn;
        size_t c_sz = (size_t)C.matrix_size();
        std::vector<bf16> c_tmp(c_sz);
        cosma::unpack_C_from_blocked(C.matrix_pointer(), c_tmp.data(),
                                     local_m_C, local_n_C, lbm, lbn);
        std::memcpy(C.matrix_pointer(), c_tmp.data(), c_sz * sizeof(bf16));
    }

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
        std::cout << "\n--- Distributed correctness (" << mode_label << "): "
                  << M << "x" << N << "x" << K
                  << " (float32 ref via MKL) ---\n" << std::flush;
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

        libxsmm_matdiff_info norms;
        libxsmm_matdiff_clear(&norms);
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32,
                        (libxsmm_blasint)((size_t)M * N), 1,
                        refCf.data(), globCf.data(), 0, 0);
        std::printf("  L1 reference  : %.25g\n", norms.l1_ref);
        std::printf("  L1 test       : %.25g\n", norms.l1_tst);
        std::printf("  L2 abs.error  : %.24f\n", norms.l2_abs);
        std::printf("  L2 rel.error  : %.24f\n", norms.l2_rel);
        std::printf("  Linf abs.error: %.24f\n", norms.linf_abs);
        std::printf("  Linf rel.error: %.24f\n", norms.linf_rel);
        std::printf("  Check-norm    : %.24f\n", norms.normf_rel);
        bool ok = (norms.normf_rel < 0.01);
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

    cosma::get_sfc_gemm_cache().set_prepacked(false);
    cosma::get_sfc_gemm_cache().set_blocked_comm(false);

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
        std::cout << "Mode           : " << mode_label << "\n";
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
