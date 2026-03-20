#pragma once

#include <complex>
#include <mpi.h>
#include <cosma/bfloat16.hpp>

namespace cosma {

// Custom MPI reduction for bfloat16: element-wise floating-point sum.
// MPI_SUM on MPI_UINT16_T does integer addition, which is wrong for BF16.
inline void
bfloat16_sum_op(void *invec, void *inoutvec, int *len, MPI_Datatype * /*dt*/) {
    const auto *in  = static_cast<const bfloat16 *>(invec);
    auto       *out = static_cast<bfloat16 *>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        // Convert to FP32, add, convert back to BF16
        float sum = static_cast<float>(in[i]) + static_cast<float>(out[i]);
        out[i] = bfloat16(sum);
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
