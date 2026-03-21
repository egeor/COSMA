// Pull in the sfc_ca_gemm library first, before our wrapper header.
// The SFC_CA_GEMM_INCLUDE_DIR cmake variable must point to the directory
// containing sfc_ca_gemm.hpp and sfc_ca_gemm.cpp.
//
// We need run_gemm<> which is a template defined in sfc_ca_gemm.cpp (not
// the header).  We include the .cpp directly but suppress main() and the
// benchmark driver via a guard macro.
//
// This must come before sfc_gemm_wrapper.hpp because the wrapper header
// forward-declares gemm_config_t, and sfc_ca_gemm.hpp defines it.
#define SFC_CA_GEMM_NO_MAIN
#include <sfc_ca_gemm.cpp>

#include <cosma/sfc_gemm_wrapper.hpp>
#include <cosma/environment_variables.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cosma {

// ---------------------------------------------------------------------------
//  sfc_gemm_cache
// ---------------------------------------------------------------------------

sfc_gemm_cache::sfc_gemm_cache() = default;

sfc_gemm_cache::sfc_gemm_cache(blocked_layout_desc desc) : desc_(desc) {}

sfc_gemm_cache::~sfc_gemm_cache() {
    for (auto &kv : cache_) {
        auto *cfg = static_cast<gemm_config_t *>(kv.second);
        // Free libxsmm-allocated internals before deleting the struct.
        if (cfg->gemm_scratch) {
            auto **arr = static_cast<libxsmm_bfloat16 **>(cfg->gemm_scratch);
            if (arr[0]) libxsmm_free(arr[0]); // global_scratch block
            libxsmm_free(arr);                // pointer array
            cfg->gemm_scratch = nullptr;
        }
        if (cfg->sfc_index_map) {
            libxsmm_free(cfg->sfc_index_map);
            cfg->sfc_index_map = nullptr;
        }
        if (cfg->scratch_B) {
            libxsmm_free(cfg->scratch_B);
            cfg->scratch_B = nullptr;
        }
        delete cfg;
    }
    if (buf_A_) { libxsmm_free(buf_A_); buf_A_ = nullptr; }
    if (buf_B_) { libxsmm_free(buf_B_); buf_B_ = nullptr; }
    if (buf_C_) { libxsmm_free(buf_C_); buf_C_ = nullptr; }
}

void *sfc_gemm_cache::get_config(int M, int N, int K) {
    // Always use a_k_outer=0 (M-outer): in blocked_comm mode the caller
    // reshuffles A from K-outer to M-outer before the kernel call.
    int ako = 0;
    key_t key{M, N, K, ako};
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    int bm = desc_.bm;
    int bn = desc_.bn;
    int bk = desc_.bk;

    // Clamp block sizes to matrix dimensions and ensure divisibility
    while (M % bm != 0 && bm > 1) --bm;
    while (N % bn != 0 && bn > 1) --bn;
    while (K % bk != 0 && bk > 1) --bk;

    long kbf = desc_.kbf;
    long K_layers = desc_.K_layers;

    gemm_config_t *cfg = setup_gemm_config<libxsmm_bfloat16>(
        M, N, K, bm, bn, bk, kbf, K_layers,
        /*m_step=*/1, /*n_step=*/1,
        /*unblocked_bc=*/0, /*use_nts=*/0,
        /*a_k_outer=*/ako);

    std::fprintf(stderr,
        "[sfc_ca_gemm] new config: M=%d N=%d K=%d bm=%d bn=%d bk=%d "
        "kbf=%ld K_layers=%ld brcount=%ld a_k_outer=%d\n",
        M, N, K, bm, bn, bk, kbf, K_layers, 
        static_cast<gemm_config_t *>(cfg)->brcount, ako);
    std::fflush(stderr);

    cache_[key] = cfg;
    return cfg;
}

void sfc_gemm_cache::pregenerate_for_strategy(
    int m, int n, int k,
    const std::vector<int> & /*divisors*/,
    const std::string & /*split_dimension*/,
    const std::string & /*step_type*/,
    int /*P*/) {
    // For simplicity, pre-generate for the full problem size.
    // A more refined version would walk the strategy tree and
    // generate configs for every leaf sub-problem size.
    (void)get_config(m, n, k);
}

static bfloat16 *alloc_aligned(size_t n) {
    void *p = libxsmm_aligned_malloc(n * sizeof(bfloat16), 64);
    // First-touch: parallel zero to place pages on local NUMA node
    bfloat16 *bp = static_cast<bfloat16 *>(p);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        bp[i].raw = 0;
    return bp;
}

bfloat16 *sfc_gemm_cache::scratch_A(size_t n) {
    if (cap_A_ < n) {
        if (buf_A_) libxsmm_free(buf_A_);
        buf_A_ = alloc_aligned(n);
        cap_A_ = n;
    }
    return buf_A_;
}

bfloat16 *sfc_gemm_cache::scratch_B(size_t n) {
    if (cap_B_ < n) {
        if (buf_B_) libxsmm_free(buf_B_);
        buf_B_ = alloc_aligned(n);
        cap_B_ = n;
    }
    return buf_B_;
}

bfloat16 *sfc_gemm_cache::scratch_C(size_t n) {
    if (cap_C_ < n) {
        if (buf_C_) libxsmm_free(buf_C_);
        buf_C_ = alloc_aligned(n);
        cap_C_ = n;
    }
    return buf_C_;
}

// ---------------------------------------------------------------------------
//  Packing helpers  (element-level reference; production should use TPP)
// ---------------------------------------------------------------------------

void pack_A_to_blocked_vnni(const bfloat16 *src, bfloat16 *dst,
                            int M, int K, int bm, int bk) {
    // Source: column-major M×K  (element (i,j) at src[j*M + i])
    // Dest:   [Mb][Kb][bk/2][bm][2]   (VNNI with factor 2 for bf16)
    const int Mb = M / bm;
    const int Kb = K / bk;
    const int vnni = 2;

    #pragma omp parallel for collapse(2)
    for (int mb = 0; mb < Mb; ++mb) {
        for (int kb = 0; kb < Kb; ++kb) {
            for (int k2 = 0; k2 < bk; ++k2) {
                for (int m2 = 0; m2 < bm; ++m2) {
                    int gi = mb * bm + m2;
                    int gj = kb * bk + k2;
                    // dst index: [mb][kb][k2/vnni][m2][k2%vnni]
                    int dst_idx = mb * (Kb * (bk / vnni) * bm * vnni)
                                + kb * ((bk / vnni) * bm * vnni)
                                + (k2 / vnni) * (bm * vnni)
                                + m2 * vnni
                                + (k2 % vnni);
                    dst[dst_idx].raw = src[gj * M + gi].raw;
                }
            }
        }
    }
}

void pack_A_to_blocked_vnni_Kouter(const bfloat16 *src, bfloat16 *dst,
                                   int M, int K, int bm, int bk) {
    // Source: column-major M×K  (element (i,j) at src[j*M + i])
    // Dest:   [Kb][Mb][bk/2][bm][2]   (K-outer VNNI, compatible with COSMA comm)
    //
    // K outermost makes allgather-expand-K = flat concat, K-split = pointer offset.
    const int Mb = M / bm;
    const int Kb = K / bk;
    const int vnni = 2;
    const int blk_elems = (bk / vnni) * bm * vnni;  // = bk * bm

    #pragma omp parallel for collapse(2)
    for (int kb = 0; kb < Kb; ++kb) {
        for (int mb = 0; mb < Mb; ++mb) {
            for (int k2 = 0; k2 < bk; ++k2) {
                for (int m2 = 0; m2 < bm; ++m2) {
                    int gi = mb * bm + m2;
                    int gj = kb * bk + k2;
                    // dst index: [kb][mb][k2/vnni][m2][k2%vnni]
                    int dst_idx = kb * (Mb * blk_elems)
                                + mb * blk_elems
                                + (k2 / vnni) * (bm * vnni)
                                + m2 * vnni
                                + (k2 % vnni);
                    dst[dst_idx].raw = src[gj * M + gi].raw;
                }
            }
        }
    }
}

void reshuffle_A_Kouter_to_Mouter(const bfloat16 *src, bfloat16 *dst,
                                  int M, int K, int bm, int bk) {
    // Source: K-outer [Kb][Mb][bk*bm]   (tile at (kb,mb) = offset kb*Mb + mb)
    // Dest:   M-outer [Mb][Kb][bk*bm]   (tile at (mb,kb) = offset mb*Kb + kb)
    // Each tile is bm*bk contiguous elements, copied intact.
    const int Mb = M / bm;
    const int Kb = K / bk;
    const int tile_elems = bm * bk;
    const size_t tile_bytes = (size_t)tile_elems * sizeof(bfloat16);

    #pragma omp parallel for collapse(2)
    for (int kb = 0; kb < Kb; ++kb) {
        for (int mb = 0; mb < Mb; ++mb) {
            const bfloat16 *s = src + ((size_t)kb * Mb + mb) * tile_elems;
            bfloat16       *d = dst + ((size_t)mb * Kb + kb) * tile_elems;
            std::memcpy(d, s, tile_bytes);
        }
    }
}

void pack_B_to_blocked(const bfloat16 *src, bfloat16 *dst,
                       int K, int N, int bk, int bn) {
    // Source: column-major K×N  (element (i,j) at src[j*K + i])
    // Dest:   [Nb][Kb][bn][bk]
    const int Nb = N / bn;
    const int Kb = K / bk;

    #pragma omp parallel for collapse(2)
    for (int nb = 0; nb < Nb; ++nb) {
        for (int kb = 0; kb < Kb; ++kb) {
            for (int n2 = 0; n2 < bn; ++n2) {
                for (int k2 = 0; k2 < bk; ++k2) {
                    int gi = kb * bk + k2;  // row in B (K dim)
                    int gj = nb * bn + n2;  // col in B (N dim)
                    int dst_idx = nb * (Kb * bn * bk)
                                + kb * (bn * bk)
                                + n2 * bk
                                + k2;
                    dst[dst_idx].raw = src[gj * K + gi].raw;
                }
            }
        }
    }
}

void pack_C_to_blocked(const bfloat16 *src, bfloat16 *dst,
                       int M, int N, int bm, int bn) {
    // Source: column-major M×N  (element (i,j) at src[j*M + i])
    // Dest:   [Nb][Mb][bn][bm]
    const int Nb = N / bn;
    const int Mb = M / bm;

    #pragma omp parallel for collapse(2)
    for (int nb = 0; nb < Nb; ++nb) {
        for (int mb = 0; mb < Mb; ++mb) {
            for (int n2 = 0; n2 < bn; ++n2) {
                for (int m2 = 0; m2 < bm; ++m2) {
                    int gi = mb * bm + m2;
                    int gj = nb * bn + n2;
                    int dst_idx = nb * (Mb * bn * bm)
                                + mb * (bn * bm)
                                + n2 * bm
                                + m2;
                    dst[dst_idx].raw = src[gj * M + gi].raw;
                }
            }
        }
    }
}

void unpack_C_from_blocked(const bfloat16 *src, bfloat16 *dst,
                           int M, int N, int bm, int bn) {
    // Source: [Nb][Mb][bn][bm]
    // Dest:   column-major M×N
    const int Nb = N / bn;
    const int Mb = M / bm;

    #pragma omp parallel for collapse(2)
    for (int nb = 0; nb < Nb; ++nb) {
        for (int mb = 0; mb < Mb; ++mb) {
            for (int n2 = 0; n2 < bn; ++n2) {
                for (int m2 = 0; m2 < bm; ++m2) {
                    int gi = mb * bm + m2;
                    int gj = nb * bn + n2;
                    int src_idx = nb * (Mb * bn * bm)
                                + mb * (bn * bm)
                                + n2 * bm
                                + m2;
                    dst[gj * M + gi].raw = src[src_idx].raw;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  run_sfc_gemm – thin wrapper around sfc_ca_gemm's run_gemm<>
// ---------------------------------------------------------------------------

void run_sfc_gemm(void *config_opaque,
                  bfloat16 *A, bfloat16 *B, bfloat16 *C) {
    static_assert(sizeof(bfloat16) == sizeof(libxsmm_bfloat16),
                  "bfloat16 size mismatch with libxsmm_bfloat16");

    auto *config = static_cast<gemm_config_t *>(config_opaque);
    run_gemm<libxsmm_bfloat16>(config,
                               reinterpret_cast<libxsmm_bfloat16 *>(A),
                               reinterpret_cast<libxsmm_bfloat16 *>(B),
                               reinterpret_cast<libxsmm_bfloat16 *>(C));
}

// Singleton accessor for the global sfc_gemm_cache used by gemm().
// Intentionally leaked (heap-allocated, never destructed) to avoid
// atexit ordering issues: libxsmm_finalize() frees all libxsmm-tracked
// allocations, so a static-local destructor running afterward would
// double-free those pointers.  The OS reclaims everything at exit.
static sfc_gemm_cache &singleton_cache() {
    static blocked_layout_desc desc{32, 32, 32, 2, 3};
    static auto *cache = []() {
        auto *c = new sfc_gemm_cache(desc);
        c->set_reshuffle_mode(get_bf16_reshuffle_mode());
        return c;
    }();
    return *cache;
}

sfc_gemm_cache &get_sfc_gemm_cache() {
    return singleton_cache();
}

} // namespace cosma
