#pragma once

#include <cosma/bfloat16.hpp>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace cosma {

/// Block-size descriptor for sfc_ca_gemm blocked layouts.
struct blocked_layout_desc {
    int bm = 32;
    int bn = 32;
    int bk = 32;
    long kbf = 1;
    long K_layers = 1;
};

/// Accumulated timing breakdown for bf16 GEMM phases.
struct sfc_gemm_timers {
    double pack_a   = 0.0; // seconds in pack_A
    double pack_b   = 0.0; // seconds in pack_B
    double compute  = 0.0; // seconds in run_sfc_gemm
    double unpack_c = 0.0; // seconds in unpack_C / alpha-beta
    int    calls    = 0;   // number of gemm calls
};

/// Cache of JIT-compiled sfc_ca_gemm kernels keyed by (M, N, K).
/// One instance lives inside cosma_context<bfloat16> so that kernel
/// generation is hoisted out of the hot multiply path.
///
/// Also owns persistent scratch buffers for blocked-layout packing
/// so that allocations happen once and are reused across calls.
class sfc_gemm_cache {
public:
    sfc_gemm_cache();
    explicit sfc_gemm_cache(blocked_layout_desc desc);
    ~sfc_gemm_cache();

    sfc_gemm_cache(const sfc_gemm_cache &) = delete;
    sfc_gemm_cache &operator=(const sfc_gemm_cache &) = delete;

    /// Return (and lazily create) a gemm_config for the given (M, N, K).
    /// Returns an opaque pointer (gemm_config_t* from sfc_ca_gemm).
    void *get_config(int M, int N, int K);

    /// Pre-generate configs for all leaf sub-problem sizes that appear
    /// in the given COSMA strategy.  Call this from register_state().
    void pregenerate_for_strategy(int m, int n, int k,
                                  const std::vector<int> &divisors,
                                  const std::string &split_dimension,
                                  const std::string &step_type,
                                  int P);

    blocked_layout_desc block_desc() const { return desc_; }

    /// Get persistent scratch buffers, resizing only if needed.
    bfloat16 *scratch_A(size_t n);
    bfloat16 *scratch_B(size_t n);
    bfloat16 *scratch_C(size_t n);

    /// Timing accumulator — reset before a timed region.
    sfc_gemm_timers &timers() { return timers_; }
    void reset_timers() { timers_ = sfc_gemm_timers{}; }

    /// Pointer-based skip-packing: when the same source pointer+size is
    /// passed to gemm(), packing is skipped and the existing scratch buffer
    /// is reused.  reset_pack_cache() forces re-packing on the next call.
    const bfloat16 *last_A_ptr() const { return last_A_ptr_; }
    size_t          last_A_sz()  const { return last_A_sz_; }
    const bfloat16 *last_B_ptr() const { return last_B_ptr_; }
    size_t          last_B_sz()  const { return last_B_sz_; }
    void set_last_A(const bfloat16 *p, size_t n) { last_A_ptr_ = p; last_A_sz_ = n; }
    void set_last_B(const bfloat16 *p, size_t n) { last_B_ptr_ = p; last_B_sz_ = n; }
    void reset_pack_cache() { last_A_ptr_ = nullptr; last_B_ptr_ = nullptr; }

private:
    blocked_layout_desc desc_;
    using key_t = std::tuple<int, int, int>;
    std::map<key_t, void *> cache_;

    bfloat16 *buf_A_ = nullptr;
    size_t     cap_A_  = 0;
    bfloat16 *buf_B_ = nullptr;
    size_t     cap_B_  = 0;
    bfloat16 *buf_C_ = nullptr;
    size_t     cap_C_  = 0;
    sfc_gemm_timers timers_;

    const bfloat16 *last_A_ptr_ = nullptr;
    size_t          last_A_sz_  = 0;
    const bfloat16 *last_B_ptr_ = nullptr;
    size_t          last_B_sz_  = 0;
};

/// Pack a column-major M*K matrix (A) into blocked VNNI format
/// [Mb][Kb][bk/2][bm][2] for bf16.
/// Both src and dst must hold M*K bfloat16 elements.
void pack_A_to_blocked_vnni(const bfloat16 *src, bfloat16 *dst,
                            int M, int K, int bm, int bk);

/// Pack a column-major K*N matrix (B) into blocked format
/// [Nb][Kb][bn][bk].
void pack_B_to_blocked(const bfloat16 *src, bfloat16 *dst,
                       int K, int N, int bk, int bn);

/// Pack a column-major M*N matrix (C) into blocked format
/// [Nb][Mb][bn][bm].
void pack_C_to_blocked(const bfloat16 *src, bfloat16 *dst,
                       int M, int N, int bm, int bn);

/// Unpack blocked C [Nb][Mb][bn][bm] back to column-major M*N.
void unpack_C_from_blocked(const bfloat16 *src, bfloat16 *dst,
                           int M, int N, int bm, int bn);

/// Run the local bf16 GEMM via sfc_ca_gemm on already-blocked data.
///   A: blocked VNNI [Mb][Kb][bk/2][bm][2]  (M*K elements)
///   B: blocked [Nb][Kb][bn][bk]             (N*K elements)
///   C: blocked [Nb][Mb][bn][bm]             (M*N elements)
/// Computes  C = A * B  (alpha/beta handled externally for now).
/// config is an opaque gemm_config_t* obtained from sfc_gemm_cache::get_config().
void run_sfc_gemm(void *config, bfloat16 *A, bfloat16 *B, bfloat16 *C);

/// Access the global singleton cache used by gemm(). Useful for
/// querying/resetting timers and pre-allocating scratch.
sfc_gemm_cache &get_sfc_gemm_cache();

} // namespace cosma
