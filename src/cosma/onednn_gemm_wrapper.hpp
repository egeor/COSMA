#pragma once

#include <cosma/bfloat16.hpp>
#include <map>
#include <memory>
#include <tuple>

// Forward-declare oneDNN types to avoid header dependency in the public header.
namespace dnnl {
class engine;
class stream;
class matmul;
class memory;
namespace impl { class primitive_desc_t; }
}

namespace cosma {

/// Cached oneDNN matmul primitive for a given (M, N, K) sub-problem.
/// Owns the dnnl::engine, stream, primitive, and memory descriptors.
///
/// Layout mapping (COSMA col-major → oneDNN row-major):
///   COSMA:  C(M×N) = A(M×K) * B(K×N)   column-major
///   oneDNN: C_T(N×M) = B_T(N×K) * A_T(K×M)   row-major
///
///   oneDNN "src"     = B^T  (N×K, plain bf16 = col-major B reinterpreted)
///   oneDNN "weights" = A^T  (K×M, blocked VNNI chosen by oneDNN)
///   oneDNN "dst"     = C^T  (N×M, plain bf16 = col-major C reinterpreted)
///
/// A (weights) is packed using the same pack_A_to_blocked_vnni routine as
/// sfc_ca_gemm: [Mb][Kb][bk/2][bm][2] with bm=32, bk=32.
/// No oneDNN reorder — identical weight layout for both backends.
/// B and C stay in flat column-major layout (no packing needed).
class onednn_gemm_cache {
public:
    onednn_gemm_cache();
    ~onednn_gemm_cache();

    onednn_gemm_cache(const onednn_gemm_cache &) = delete;
    onednn_gemm_cache &operator=(const onednn_gemm_cache &) = delete;

    /// Execute C = A*B in bf16 using oneDNN matmul.
    /// A, B, C are column-major bf16 pointers.
    /// A is reordered (packed) into oneDNN's optimal weights format on first
    /// call for each (M,N,K); subsequent calls with the same A pointer skip
    /// the reorder.
    void gemm(int M, int N, int K,
              const bfloat16 *A, const bfloat16 *B, bfloat16 *C);

    /// Pack A (col-major M×K) into oneDNN's optimal weights format.
    /// Returns a pointer to the packed buffer (owned by the cache).
    /// The packed buffer is reused across calls for the same (M,N,K).
    bfloat16 *pack_A(int M, int N, int K, const bfloat16 *A);

    /// Execute matmul using pre-packed A (weights) and flat B, C.
    void gemm_prepacked(int M, int N, int K,
                        const bfloat16 *A_packed,
                        const bfloat16 *B, bfloat16 *C);

    /// Reset cached A packing state (force re-pack on next call).
    void reset_pack_cache() { last_A_ptr_ = nullptr; }

private:
    struct matmul_entry;  // pimpl

    matmul_entry &get_entry(int M, int N, int K);

    using key_t = std::tuple<int, int, int>;
    std::map<key_t, std::unique_ptr<matmul_entry>> cache_;

    const bfloat16 *last_A_ptr_ = nullptr;
    size_t last_A_sz_ = 0;
};

/// Access the global singleton oneDNN GEMM cache.
onednn_gemm_cache &get_onednn_gemm_cache();

/// Query environment variable COSMA_GEMM_BACKEND.
/// Returns true if set to "onednn" (case-insensitive).
bool use_onednn_backend();

} // namespace cosma
