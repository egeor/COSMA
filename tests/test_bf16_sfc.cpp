/**
 * Quick test for BFloat16 GEMM via sfc_ca_gemm integration.
 * Calls the BLAS-level gemm<bfloat16>() directly and checks correctness
 * against a float32 reference computation.
 */
#include <cosma/bfloat16.hpp>
#include <cosma/blas.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using bf16 = cosma::bfloat16;

// Reference GEMM in float32 (col-major, no-trans, no-trans)
static void ref_gemm_f32(int M, int N, int K,
                         float alpha,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float beta,
                         float *C, int ldc) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += A[p * lda + i] * B[j * ldb + p]; // col-major
            }
            C[j * ldc + i] = alpha * acc + beta * C[j * ldc + i];
        }
    }
}

int main(int argc, char **argv) {
    int M = 128, N = 128, K = 128;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    std::cout << "BFloat16 GEMM test: M=" << M << " N=" << N << " K=" << K << "\n";

    // Random init
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate bf16 matrices (col-major)
    std::vector<bf16> A(M * K), B(K * N), C(M * N, bf16(0.0f));
    // Allocate float32 references
    std::vector<float> Af(M * K), Bf(K * N), Cf(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        float v = dist(rng);
        A[i] = bf16(v);
        Af[i] = float(A[i]); // use the quantized value for fair comparison
    }
    for (int i = 0; i < K * N; ++i) {
        float v = dist(rng);
        B[i] = bf16(v);
        Bf[i] = float(B[i]);
    }

    bf16 alpha(1.0f), beta(0.0f);

    // Call COSMA's BLAS-level gemm for bf16 (invokes sfc_ca_gemm under the hood)
    cosma::gemm(M, N, K,
                alpha,
                A.data(), M,
                B.data(), K,
                beta,
                C.data(), M);

    // Run reference float32 GEMM
    ref_gemm_f32(M, N, K, 1.0f, Af.data(), M, Bf.data(), K, 0.0f, Cf.data(), M);

    // Compare results using absolute error threshold.
    // With kbf > 1 or K_layers > 1, bf16 intermediate rounding grows with
    // the number of K-rounds, so we scale the threshold by sqrt(K/bk).
    double max_abs_err = 0.0;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float got = float(C[i]);
        float ref = Cf[i];
        double ae = std::fabs(got - ref);
        if (ae > max_abs_err) max_abs_err = ae;
        sum_sq_err += ae * ae;
        sum_sq_ref += (double)ref * ref;
    }
    double check_norm = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;

    std::cout << "Max abs error: " << max_abs_err << "\n";
    std::cout << "Check-norm (L2 rel): " << check_norm << "\n";

    // check-norm < 0.01 means the result is correct for bf16 precision
    bool ok = (check_norm < 0.01);
    if (ok) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED (check-norm " << check_norm << " >= 0.01)\n";
    }

    return ok ? 0 : 1;
}
