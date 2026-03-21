#include <cosma/blas.hpp>

// extern "C" {
#ifdef COSMA_WITH_MKL_BLAS
#include <mkl.h>
#endif

#ifdef COSMA_WITH_BLIS_BLAS
#include <blis.h>
#endif

#if defined(COSMA_WITH_BLAS) && !defined(COSMA_WITH_MKL_BLAS)
#include <cblas.h>
// this is for backward compatibility,
// in case CBLAS_LAYOUT is not defined
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
// }

// The file is not needed if GPU is used
//
#if defined(COSMA_WITH_MKL_BLAS) || defined(COSMA_WITH_BLIS_BLAS) || defined(COSMA_WITH_BLAS)
namespace cosma {
void gemm(const int M,
          const int N,
          const int K,
          const double alpha,
          const double *A,
          const int lda,
          const double *B,
          const int ldb,
          const double beta,
          double *C,
          const int ldc) {
    cblas_dgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const std::complex<double> alpha,
          const std::complex<double> *A,
          const int lda,
          const std::complex<double> *B,
          const int ldb,
          const std::complex<double> beta,
          std::complex<double> *C,
          const int ldc) {
    cblas_zgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                reinterpret_cast<const double*>(&alpha),
                reinterpret_cast<const double*>(A),
                lda,
                reinterpret_cast<const double*>(B),
                ldb,
                reinterpret_cast<const double*>(&beta),
                reinterpret_cast<double*>(C),
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const float alpha,
          const float *A,
          const int lda,
          const float *B,
          const int ldb,
          const float beta,
          float *C,
          const int ldc) {
    cblas_sgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const std::complex<float> alpha,
          const std::complex<float> *A,
          const int lda,
          const std::complex<float> *B,
          const int ldb,
          const std::complex<float> beta,
          std::complex<float> *C,
          const int ldc) {
    cblas_cgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                reinterpret_cast<const float*>(&alpha),
                reinterpret_cast<const float*>(A),
                lda,
                reinterpret_cast<const float*>(B),
                ldb,
                reinterpret_cast<const float*>(&beta),
                reinterpret_cast<float*>(C),
                ldc);
}

} // namespace cosma
#endif

// BFloat16 GEMM via sfc_ca_gemm — always available when COSMA_WITH_SFC_GEMM
#ifdef COSMA_WITH_SFC_GEMM
#include <cosma/sfc_gemm_wrapper.hpp>
#include <chrono>
#include <cstring>

namespace cosma {

void gemm(const int M,
          const int N,
          const int K,
          const bfloat16 alpha,
          const bfloat16 *A,
          const int lda,
          const bfloat16 *B,
          const int ldb,
          const bfloat16 beta,
          bfloat16 *C,
          const int ldc) {
    using clk = std::chrono::high_resolution_clock;

    auto &cache = get_sfc_gemm_cache();
    auto  desc  = cache.block_desc();
    auto &tm    = cache.timers();
    tm.calls++;

    int bm = desc.bm, bn = desc.bn, bk = desc.bk;
    while (M % bm != 0 && bm > 1) --bm;
    while (N % bn != 0 && bn > 1) --bn;
    while (K % bk != 0 && bk > 1) --bk;

    if (cache.is_prepacked()) {
        // --- Fully pre-packed mode: A, B, C all in blocked layout ---
        // A: VNNI [Mb][Kb][bk/2][bm][2], B: [Nb][Kb][bn][bk], C: [Nb][Mb][bn][bm]
        const float fb = float(beta);
        const size_t c_sz = (size_t)M * N;

        bfloat16 *C_save = nullptr;
        if (fb != 0.0f) {
            C_save = cache.scratch_C(c_sz);
            auto t0s = clk::now();
            std::memcpy(C_save, C, c_sz * sizeof(bfloat16));
            auto t1s = clk::now();
            tm.unpack_c += std::chrono::duration<double>(t1s - t0s).count();
        }

        auto t0 = clk::now();
        void *cfg = cache.get_config(M, N, K);
        run_sfc_gemm(cfg, const_cast<bfloat16 *>(A),
                     const_cast<bfloat16 *>(B), C);
        auto t1 = clk::now();
        tm.compute += std::chrono::duration<double>(t1 - t0).count();

        if (float(alpha) != 1.0f || fb != 0.0f) {
            const float fa = float(alpha);
            t0 = clk::now();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < c_sz; ++i) {
                float val = fa * float(C[i]);
                if (fb != 0.0f) val += fb * float(C_save[i]);
                C[i] = bfloat16(val);
            }
            t1 = clk::now();
            tm.unpack_c += std::chrono::duration<double>(t1 - t0).count();
        }
        return;
    }

    if (cache.is_blocked_comm()) {
        // --- Blocked-comm mode: all matrices in blocked layout ---
        // A: K-outer VNNI [Kb][Mb][bk/2][bm][2]  (MPI comm format)
        //    or M-outer if reshuffle_mode==1 (already reshuffled)
        // B: [Nb][Kb][bn][bk]                     (kernel native format)
        // C: [Nb][Mb][bn][bm]                     (kernel native format)
        const float fb = float(beta);
        const size_t a_sz = (size_t)M * K;
        const size_t c_sz = (size_t)M * N;

        // Determine the A pointer for the kernel (always M-outer, a_k_outer=0)
        bfloat16 *A_for_kernel;
        if (cache.is_a_reshuffled()) {
            // A already reshuffled to M-outer by multiply/overlap layer
            A_for_kernel = const_cast<bfloat16 *>(A);
        } else {
            // Mode 0: reshuffle A from K-outer to M-outer here
            A_for_kernel = cache.scratch_A(a_sz);
            auto t0 = clk::now();
            reshuffle_A_Kouter_to_Mouter(A, A_for_kernel, M, K, bm, bk);
            auto t1 = clk::now();
            tm.pack_a += std::chrono::duration<double>(t1 - t0).count();
        }

        // For beta != 0: save C_old (in blocked layout)
        bfloat16 *C_save = nullptr;
        if (fb != 0.0f) {
            C_save = cache.scratch_C(c_sz);
            auto t0s = clk::now();
            std::memcpy(C_save, C, c_sz * sizeof(bfloat16));
            auto t1s = clk::now();
            tm.unpack_c += std::chrono::duration<double>(t1s - t0s).count();
        }

        auto t0 = clk::now();
        void *cfg = cache.get_config(M, N, K);
        run_sfc_gemm(cfg, A_for_kernel,
                     const_cast<bfloat16 *>(B), C);
        auto t1 = clk::now();
        tm.compute += std::chrono::duration<double>(t1 - t0).count();

        // For beta != 0: accumulate C = alpha * C_new + beta * C_old
        if (float(alpha) != 1.0f || fb != 0.0f) {
            const float fa = float(alpha);
            t0 = clk::now();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < c_sz; ++i) {
                float val = fa * float(C[i]);
                if (fb != 0.0f) val += fb * float(C_save[i]);
                C[i] = bfloat16(val);
            }
            t1 = clk::now();
            tm.unpack_c += std::chrono::duration<double>(t1 - t0).count();
        }
        return;
    }

    // --- Standard mode: pack from column-major, compute, unpack ---
    bfloat16 *A_blk = cache.scratch_A((size_t)M * K);
    bfloat16 *B_blk = cache.scratch_B((size_t)K * N);
    bfloat16 *C_blk = cache.scratch_C((size_t)M * N);

    auto t0 = clk::now();
    pack_A_to_blocked_vnni(A, A_blk, M, K, bm, bk);
    auto t1 = clk::now();
    tm.pack_a += std::chrono::duration<double>(t1 - t0).count();

    t0 = clk::now();
    pack_B_to_blocked(B, B_blk, K, N, bk, bn);
    t1 = clk::now();
    tm.pack_b += std::chrono::duration<double>(t1 - t0).count();

    t0 = clk::now();
    void *cfg = cache.get_config(M, N, K);
    run_sfc_gemm(cfg, A_blk, B_blk, C_blk);
    t1 = clk::now();
    tm.compute += std::chrono::duration<double>(t1 - t0).count();

    t0 = clk::now();
    if (float(alpha) != 1.0f || float(beta) != 0.0f) {
        const float fa = float(alpha), fb = float(beta);
        const int Mb = M / bm;
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                int nb = j / bn, n2 = j % bn;
                int mb = i / bm, m2 = i % bm;
                int blk_idx = nb * (Mb * bn * bm) + mb * (bn * bm)
                            + n2 * bm + m2;
                float ab = float(C_blk[blk_idx]);
                float c_old = (fb != 0.0f) ? float(C[j * ldc + i]) : 0.0f;
                C[j * ldc + i] = bfloat16(fa * ab + fb * c_old);
            }
        }
    } else {
        unpack_C_from_blocked(C_blk, C, M, N, bm, bn);
    }
    t1 = clk::now();
    tm.unpack_c += std::chrono::duration<double>(t1 - t0).count();
}

} // namespace cosma
#endif
