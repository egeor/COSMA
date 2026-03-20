#pragma once
#include <complex>
#include <cosma/bfloat16.hpp>

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
          const int ldc);

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
          const int ldc);

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
          const int ldc);

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
          const int ldc);

// BFloat16 GEMM via sfc_ca_gemm (blocked layout, VNNI packing)
// Note: for bf16, the data is expected in COSMA's flat col-major layout.
// This function internally packs to blocked format, calls sfc_ca_gemm,
// and unpacks the result. Use gemm_blocked_bf16() to avoid packing.
#ifdef COSMA_WITH_SFC_GEMM
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
          const int ldc);
#endif
} // namespace cosma
