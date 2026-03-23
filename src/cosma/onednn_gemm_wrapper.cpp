#include <cosma/onednn_gemm_wrapper.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "oneapi/dnnl/dnnl.hpp"

#ifdef COSMA_WITH_SFC_GEMM
#include <cosma/sfc_gemm_wrapper.hpp>
#endif

namespace cosma {

// ---------------------------------------------------------------------------
//  matmul_entry: one cached primitive per (M, N, K)
// ---------------------------------------------------------------------------

struct onednn_gemm_cache::matmul_entry {
    dnnl::engine  eng;
    dnnl::stream  strm;
    dnnl::matmul  prim;

    // Memory descriptors
    dnnl::memory::desc src_md;      // B^T plain (N×K bf16)
    dnnl::memory::desc weights_md;  // A^T VNNI blocked (K×M bf16)
    dnnl::memory::desc dst_md;      // C^T plain (N×M bf16)

    // Pre-allocated oneDNN memory objects (reuse across calls)
    dnnl::memory src_mem;
    dnnl::memory weights_mem;
    dnnl::memory dst_mem;

    // Packed A buffer (M-outer VNNI, final layout for oneDNN)
    std::vector<bfloat16> packed_A;
    // Scratch for K-outer intermediate
    std::vector<bfloat16> kouter_A;

    int M, N, K;
};

// ---------------------------------------------------------------------------
//  onednn_gemm_cache
// ---------------------------------------------------------------------------

onednn_gemm_cache::onednn_gemm_cache() = default;
onednn_gemm_cache::~onednn_gemm_cache() = default;

onednn_gemm_cache::matmul_entry &
onednn_gemm_cache::get_entry(int M, int N, int K) {
    key_t key{M, N, K};
    auto it = cache_.find(key);
    if (it != cache_.end())
        return *it->second;

    auto e = std::make_unique<matmul_entry>();
    e->M = M;
    e->N = N;
    e->K = K;

    e->eng  = dnnl::engine(dnnl::engine::kind::cpu, 0);
    e->strm = dnnl::stream(e->eng);

    // Layout mapping: COSMA col-major C(M×N) = A(M×K) * B(K×N)
    // becomes oneDNN row-major C_T(N×M) = B_T(N×K) * A_T(K×M)
    //
    // oneDNN "src"     = B^T: logical (N, K), plain bf16
    // oneDNN "weights" = A^T: logical (K, M), let oneDNN choose blocked
    // oneDNN "dst"     = C^T: logical (N, M), plain bf16
    //
    // Column-major M×K stored contiguously is the same memory layout as
    // row-major K×M.  But oneDNN's "ab" format_tag means row-major, i.e.,
    // logical dims (rows, cols) with stride = cols.
    //
    // Col-major A (M×K): stride between columns = M, stride between rows = 1
    //   = row-major with dims (K, M) and tag "ab" ← this is A^T
    //
    // Col-major B (K×N): stride between columns = K, stride between rows = 1
    //   = row-major with dims (N, K) and tag "ab" ← this is B^T

    // src = B^T (N×K), plain
    e->src_md = dnnl::memory::desc({N, K}, dnnl::memory::data_type::bf16,
                                   dnnl::memory::format_tag::ab);

    // weights = A^T (K×M), let oneDNN choose optimal (VNNI blocked)
    auto weights_any = dnnl::memory::desc({K, M}, dnnl::memory::data_type::bf16,
                                          dnnl::memory::format_tag::any);

    // dst = C^T (N×M), plain
    e->dst_md = dnnl::memory::desc({N, M}, dnnl::memory::data_type::bf16,
                                   dnnl::memory::format_tag::ab);

    // Create matmul primitive descriptor — oneDNN picks optimal weights layout
    auto matmul_pd = dnnl::matmul::primitive_desc(e->eng, e->src_md,
                                                   weights_any, e->dst_md);

    e->weights_md = matmul_pd.weights_desc();

    e->prim = dnnl::matmul(matmul_pd);

    // Pre-allocate memory objects
    e->src_mem     = dnnl::memory(e->src_md, e->eng);
    e->weights_mem = dnnl::memory(e->weights_md, e->eng);
    e->dst_mem     = dnnl::memory(e->dst_md, e->eng);

    // Allocate packed A buffers — sized for M*K elements
    size_t weights_bytes = e->weights_md.get_size();
    size_t a_elems = (size_t)M * K;
    e->packed_A.resize(std::max(a_elems, weights_bytes / sizeof(bfloat16)));
    e->kouter_A.resize(a_elems);

    // Print layout info for verification
    auto inner_blks = e->weights_md.get_inner_blks();
    auto inner_idxs = e->weights_md.get_inner_idxs();
    std::fprintf(stderr,
        "[onednn_gemm] new matmul: M=%d N=%d K=%d "
        "weights_size=%zu bytes, inner_blocks:",
        M, N, K, weights_bytes);
    for (size_t i = 0; i < inner_blks.size(); i++)
        std::fprintf(stderr, " %lld@dim%lld",
                     (long long)inner_blks[i], (long long)inner_idxs[i]);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);

    auto *raw_ptr = e.get();
    cache_[key] = std::move(e);
    return *raw_ptr;
}

bfloat16 *onednn_gemm_cache::pack_A(int M, int N, int K,
                                     const bfloat16 *A) {
    auto &e = get_entry(M, N, K);

    // Same pipeline as sfc_ca_gemm blocked_comm:
    // 1) Pack col-major A to K-outer VNNI: [Kb][Mb][bk/2][bm][2]
    // 2) Reshuffle K-outer → M-outer: [Mb][Kb][bk/2][bm][2]
    // This ensures identical packing cost for both backends.
    const int bm = 32, bk = 32;
#ifdef COSMA_WITH_SFC_GEMM
    // Step 1: col-major → K-outer VNNI
    pack_A_to_blocked_vnni_Kouter(A, e.kouter_A.data(), M, K, bm, bk);

    // Step 2: reshuffle [Kb][Mb][bk/2][bm][2] → [Mb][Kb][bk/2][bm][2]
    // Each inner block has bk/2 * bm * 2 = bk * bm elements.
    const int Mb = M / bm;
    const int Kb = K / bk;
    const size_t blk_sz = (size_t)bk * bm;  // 32*32 = 1024 elements per block
    const bfloat16 *src = e.kouter_A.data();
    bfloat16 *dst = e.packed_A.data();
    for (int kb = 0; kb < Kb; ++kb) {
        for (int mb = 0; mb < Mb; ++mb) {
            // K-outer source index: [kb][mb] → offset (kb * Mb + mb) * blk_sz
            // M-outer dest index:   [mb][kb] → offset (mb * Kb + kb) * blk_sz
            std::memcpy(dst + (size_t)(mb * Kb + kb) * blk_sz,
                        src + (size_t)(kb * Mb + mb) * blk_sz,
                        blk_sz * sizeof(bfloat16));
        }
    }
#else
    // Fallback: use oneDNN reorder (plain → blocked VNNI)
    auto plain_md = dnnl::memory::desc({K, M}, dnnl::memory::data_type::bf16,
                                        dnnl::memory::format_tag::ab);
    auto plain_mem = dnnl::memory(plain_md, e.eng,
                                  const_cast<void *>(static_cast<const void *>(A)));
    e.weights_mem.set_data_handle(e.packed_A.data());
    dnnl::reorder(plain_mem, e.weights_mem).execute(e.strm, plain_mem, e.weights_mem);
    e.strm.wait();
#endif

    e.weights_mem.set_data_handle(e.packed_A.data());
    return e.packed_A.data();
}

void onednn_gemm_cache::gemm_prepacked(int M, int N, int K,
                                        const bfloat16 *A_packed,
                                        const bfloat16 *B, bfloat16 *C) {
    auto &e = get_entry(M, N, K);

    // Point memory objects to user data
    // src = B (col-major K×N = row-major N×K = B^T)
    e.src_mem.set_data_handle(const_cast<void *>(static_cast<const void *>(B)));
    // weights = pre-packed A
    e.weights_mem.set_data_handle(const_cast<void *>(static_cast<const void *>(A_packed)));
    // dst = C (col-major M×N = row-major N×M = C^T)
    e.dst_mem.set_data_handle(static_cast<void *>(C));

    e.prim.execute(e.strm, {
        {DNNL_ARG_SRC,     e.src_mem},
        {DNNL_ARG_WEIGHTS, e.weights_mem},
        {DNNL_ARG_DST,     e.dst_mem}
    });
    e.strm.wait();
}

void onednn_gemm_cache::gemm(int M, int N, int K,
                              const bfloat16 *A, const bfloat16 *B,
                              bfloat16 *C) {
    // Pack A if pointer changed
    size_t a_sz = (size_t)M * K;
    if (A != last_A_ptr_ || a_sz != last_A_sz_) {
        pack_A(M, N, K, A);
        last_A_ptr_ = A;
        last_A_sz_ = a_sz;
    }

    auto &e = get_entry(M, N, K);
    gemm_prepacked(M, N, K, e.packed_A.data(), B, C);
}

const bfloat16 *onednn_gemm_cache::last_packed_A(int M, int N, int K) {
    auto &e = get_entry(M, N, K);
    return e.packed_A.data();
}

// ---------------------------------------------------------------------------
//  Singleton
// ---------------------------------------------------------------------------

static onednn_gemm_cache &singleton() {
    static auto *cache = new onednn_gemm_cache();
    return *cache;
}

onednn_gemm_cache &get_onednn_gemm_cache() {
    return singleton();
}

// ---------------------------------------------------------------------------
//  Backend selection
// ---------------------------------------------------------------------------

bool use_onednn_backend() {
    static int cached = -1;
    if (cached < 0) {
        const char *val = std::getenv("COSMA_GEMM_BACKEND");
        if (val) {
            std::string s(val);
            for (auto &c : s) c = std::tolower(c);
            cached = (s == "onednn" || s == "dnnl") ? 1 : 0;
        } else {
            cached = 0;
        }
        if (cached) {
            std::fprintf(stderr, "[cosma] Using oneDNN GEMM backend\n");
        }
    }
    return cached == 1;
}

} // namespace cosma
