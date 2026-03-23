#pragma once

#include <complex>
#include <mpi.h>
#include <cosma/bfloat16.hpp>
#include <immintrin.h>

namespace cosma {

// Custom MPI reduction for bfloat16: element-wise floating-point sum.
// MPI_SUM on MPI_UINT16_T does integer addition, which is wrong for BF16.
// Uses AVX-512 to vectorize bf16<->fp32 conversion and addition.
__attribute__((target("avx512f,avx512bw,avx512bf16")))
inline void
bfloat16_sum_op(void *invec, void *inoutvec, int *len, MPI_Datatype * /*dt*/) {
    const int n = *len;
    auto *in_raw  = static_cast<const uint16_t *>(invec);
    auto *out_raw = static_cast<uint16_t *>(inoutvec);

    // Process 16 bf16 elements per AVX-512 iteration
    const int simd_width = 16;
    const int n_simd = n & ~(simd_width - 1); // round down to multiple of 16

    #pragma omp parallel for schedule(static) num_threads(16)
    for (int i = 0; i < n_simd; i += simd_width) {
        // Load 16 x uint16 bf16 values
        __m256i in_bf16  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in_raw + i));
        __m256i out_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(out_raw + i));
        // Convert bf16 -> fp32 by shifting left 16 bits
        __m512i in_i32  = _mm512_cvtepu16_epi32(in_bf16);
        __m512i out_i32 = _mm512_cvtepu16_epi32(out_bf16);
        __m512 in_f32  = _mm512_castsi512_ps(_mm512_slli_epi32(in_i32, 16));
        __m512 out_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(out_i32, 16));
        // Add in fp32
        __m512 sum_f32 = _mm512_add_ps(in_f32, out_f32);
        // Convert fp32 -> bf16 with round-to-nearest-even
        __m256i sum_bf16 = _mm512_cvtneps_pbh(sum_f32);
        // Store 16 x uint16 bf16 results
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_raw + i), sum_bf16);
    }

    // Scalar tail for remaining elements
    for (int i = n_simd; i < n; ++i) {
        float a = static_cast<float>(reinterpret_cast<const bfloat16 *>(in_raw)[i]);
        float b = static_cast<float>(reinterpret_cast<const bfloat16 *>(out_raw)[i]);
        reinterpret_cast<bfloat16 *>(out_raw)[i] = bfloat16(a + b);
    }
}

// Get or create the custom BF16 MPI_Op
inline MPI_Op get_bfloat16_sum_op() {
    static MPI_Op bf16_op = MPI_OP_NULL;
    if (bf16_op == MPI_OP_NULL) {
        MPI_Op_create(bfloat16_sum_op, /*commutative=*/1, &bf16_op);
    }
    return bf16_op;
}

/**
 * Maps a primitive numeric type to a MPI type and sum operation.
 *
 * @tparam Scalar the numeric type to be mapped
 */
template <typename Scalar>
struct mpi_mapper {
  static inline MPI_Datatype getType();
  static inline MPI_Op getSumOp();
};

template <>
inline MPI_Datatype mpi_mapper<double>::getType() {
  return MPI_DOUBLE;
}
template <>
inline MPI_Op mpi_mapper<double>::getSumOp() {
  return MPI_SUM;
}

template <>
inline MPI_Datatype mpi_mapper<float>::getType() {
  return MPI_FLOAT;
}
template <>
inline MPI_Op mpi_mapper<float>::getSumOp() {
  return MPI_SUM;
}

template <>
inline MPI_Datatype mpi_mapper<std::complex<float>>::getType() {
  return MPI_C_FLOAT_COMPLEX;
}
template <>
inline MPI_Op mpi_mapper<std::complex<float>>::getSumOp() {
  return MPI_SUM;
}

template <>
inline MPI_Datatype mpi_mapper<std::complex<double>>::getType() {
  return MPI_C_DOUBLE_COMPLEX;
}
template <>
inline MPI_Op mpi_mapper<std::complex<double>>::getSumOp() {
  return MPI_SUM;
}

template <>
inline MPI_Datatype mpi_mapper<bfloat16>::getType() {
  return MPI_UINT16_T;
}
template <>
inline MPI_Op mpi_mapper<bfloat16>::getSumOp() {
  return get_bfloat16_sum_op();
}

// Removes const qualifier
//
template <typename Scalar>
struct mpi_mapper<const Scalar> {
  static inline MPI_Datatype getType();
};

template <typename Scalar>
inline MPI_Datatype mpi_mapper<const Scalar>::getType() {
  return mpi_mapper<Scalar>::getType();
}

} // end namespace cosma
