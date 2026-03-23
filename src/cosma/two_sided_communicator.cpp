#include <cosma/interval.hpp>
#include <cosma/math_utils.hpp>
#include <cosma/matrix.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/bfloat16.hpp>
#include <cosma/profiler.hpp>
#include <cosma/strategy.hpp>
#include <cosma/two_sided_communicator.hpp>

#include <mpi.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstring>
#include <future>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <tuple>

namespace cosma {

namespace two_sided_communicator {
// two_sided_communicator() = default;
// two_sided_communicator(const Strategy* strategy, MPI_Comm comm):
//     communicator::communicator(strategy, comm) {}

/*
 * (first see the comment in communicator.hpp)
 * The idea is the following:
 *      - if only 1 block per rank should be communicated:
 *        don't allocate new space, just perform all-gather
 *
 *      - if more than 1 blocks per rank should be communicated:
 *        allocate new space and let the all-gather be performed
 *        on the level of all blocks per rank. After the communication,
 *        reshuffle the local data by putting first blocks from each rank
 * first, then all second blocks from each rank and so on.
 */
template <typename Scalar>
void copy(MPI_Comm comm,
          int rank,
          int div,
          Interval &P,
          Scalar *in,
          Scalar *out,
          Scalar *reshuffle_buffer,
          std::vector<std::vector<int>> &size_before,
          std::vector<int> &total_before,
          int total_after) {
    PE(multiply_communication_other);
    // int div = strategy_->divisor(step);
    // MPI_Comm subcomm = active_comm(step);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int relative_rank = rank - P.first();
    int local_size = total_before[relative_rank];

    int sum = 0;
    std::vector<int> total_size(div);
    std::vector<int> dspls(div);
    // int off = offset(P, div);

    std::vector<int> subgroup(div);
    bool same_size = true;

    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        int temp_size = total_before[target];
        dspls[i] = sum;
        sum += temp_size;
        total_size[i] = temp_size;
        same_size &= temp_size == local_size;
    }

    int n_blocks = size_before[relative_rank].size();
    Scalar *receive_pointer = n_blocks > 1 ? reshuffle_buffer : out;
    PL();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    PE(multiply_communication_copy);
    if (same_size) {
        MPI_Allgather(in,
                      local_size,
                      mpi_type,
                      receive_pointer,
                      local_size,
                      mpi_type,
                      comm);
    } else {
        MPI_Allgatherv(in,
                       local_size,
                       mpi_type,
                       receive_pointer,
                       total_size.data(),
                       dspls.data(),
                       mpi_type,
                       comm);
    }
    PL();

    PE(multiply_communication_other);
    if (n_blocks > 1) {
        int index = 0;
        std::vector<int> block_offset(div);
        // order all first sequential parts of all groups first and so on..
        for (int block = 0; block < n_blocks; block++) {
            for (int rank = 0; rank < div; rank++) {
                int target = P.locate_in_interval(div, rank, off);
                int dsp = dspls[rank] + block_offset[rank];
                int b_size = size_before[target][block];
                std::copy(reshuffle_buffer + dsp,
                          reshuffle_buffer + dsp + b_size,
                          out + index);
                index += b_size;
                block_offset[rank] += b_size;
            }
        }
    }
    PL();
#ifdef DEBUG
    std::cout << "Content of the copied matrix in rank " << rank
              << " is now: " << std::endl;
    for (int j = 0; j < sum; j++) {
        std::cout << out[j] << " , ";
    }
    std::cout << std::endl;

#endif
}

template <typename Scalar>
void reduce(MPI_Comm comm,
            int rank,
            int div,
            Interval &P,
            Scalar *LC, // expanded_matrix
            Scalar *C,  // original matrix
            Scalar *reshuffle_buffer,
            Scalar *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            Scalar beta) {
    PE(multiply_communication_other);
    // int div = strategy_->divisor(step);
    // MPI_Comm subcomm = active_comm(step);

    std::vector<int> subgroup(div);

    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);
    // int gp, off;
    // std::tie(gp, off) = group_and_offset(P, div);

    // reorder the elements as:
    // first all blocks that should be sent to rank 0 then all blocks for
    // rank 1 and so on...
    int n_blocks = c_expanded[off].size();
    std::vector<int> block_offset(n_blocks);
    Scalar *send_pointer = n_blocks > 1 ? reshuffle_buffer : LC;

    int sum = 0;
    for (int i = 0; i < n_blocks; ++i) {
        block_offset[i] = sum;
        sum += c_expanded[off][i];
    }

    std::vector<int> recvcnts(div);

    bool same_size = true;
    int index = 0;
    // go through the communication ring
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        recvcnts[i] = c_total_current[target];

        same_size = same_size && recvcnts[i] == recvcnts[0];

        if (n_blocks > 1) {
            for (int block = 0; block < n_blocks; ++block) {
                int b_offset = block_offset[block];
                int b_size = c_current[target][block];
                std::copy(LC + b_offset,
                          LC + b_offset + b_size,
                          reshuffle_buffer + index);
                index += b_size;
                block_offset[block] += b_size;
            }
        }
    }

    Scalar *receive_pointer = beta != Scalar{0} ? reduce_buffer : C;
    PL();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    PE(multiply_communication_reduce);

    auto mpi_sum = mpi_mapper<Scalar>::getSumOp();
    if (same_size) {
        MPI_Reduce_scatter_block(send_pointer,
                           receive_pointer,
                           recvcnts[0],
                           mpi_type,
                           mpi_sum,
                           comm);
    } else {
        MPI_Reduce_scatter(send_pointer,
                           receive_pointer,
                           recvcnts.data(),
                           mpi_type,
                           mpi_sum,
                           comm);
    }
    PL();

    PE(multiply_communication_other);
    if (beta != Scalar{0}) {
        // sum up receiving_buffer with C
        for (int el = 0; el < recvcnts[gp]; ++el) {
            C[el] = beta * C[el] + reduce_buffer[el];
        }
    }
    PL();
}

template void copy<float>(MPI_Comm comm,
                          int rank,
                          int div,
                          Interval &P,
                          float *in,
                          float *out,
                          float *reshuffle_buffer,
                          std::vector<std::vector<int>> &size_before,
                          std::vector<int> &total_before,
                          int total_after);

template void copy<double>(MPI_Comm comm,
                           int rank,
                           int div,
                           Interval &P,
                           double *in,
                           double *out,
                           double *reshuffle_buffer,
                           std::vector<std::vector<int>> &size_before,
                           std::vector<int> &total_before,
                           int total_after);

template void
copy<std::complex<float>>(MPI_Comm comm,
                          int rank,
                          int div,
                          Interval &P,
                          std::complex<float> *in,
                          std::complex<float> *out,
                          std::complex<float> *reshuffle_buffer,
                          std::vector<std::vector<int>> &size_before,
                          std::vector<int> &total_before,
                          int total_after);

template void
copy<std::complex<double>>(MPI_Comm comm,
                           int rank,
                           int div,
                           Interval &P,
                           std::complex<double> *in,
                           std::complex<double> *out,
                           std::complex<double> *reshuffle_buffer,
                           std::vector<std::vector<int>> &size_before,
                           std::vector<int> &total_before,
                           int total_after);

// bf16 copy is a full specialization above — no explicit instantiation needed

template void reduce<float>(MPI_Comm comm,
                            int rank,
                            int div,
                            Interval &P,
                            float *LC,
                            float *C,
                            float *reshuffle_buffer,
                            float *reduce_buffer,
                            std::vector<std::vector<int>> &c_current,
                            std::vector<int> &c_total_current,
                            std::vector<std::vector<int>> &c_expanded,
                            std::vector<int> &c_total_expanded,
                            float beta);

template void reduce<double>(MPI_Comm comm,
                             int rank,
                             int div,
                             Interval &P,
                             double *LC,
                             double *C,
                             double *reshuffle_buffer,
                             double *reduce_buffer,
                             std::vector<std::vector<int>> &c_current,
                             std::vector<int> &c_total_current,
                             std::vector<std::vector<int>> &c_expanded,
                             std::vector<int> &c_total_expanded,
                             double beta);

template void
reduce<std::complex<float>>(MPI_Comm comm,
                            int rank,
                            int div,
                            Interval &P,
                            std::complex<float> *LC,
                            std::complex<float> *C,
                            std::complex<float> *reshuffle_buffer,
                            std::complex<float> *reduce_buffer,
                            std::vector<std::vector<int>> &c_current,
                            std::vector<int> &c_total_current,
                            std::vector<std::vector<int>> &c_expanded,
                            std::vector<int> &c_total_expanded,
                            std::complex<float> beta);

template void
reduce<std::complex<double>>(MPI_Comm comm,
                             int rank,
                             int div,
                             Interval &P,
                             std::complex<double> *LC,
                             std::complex<double> *C,
                             std::complex<double> *reshuffle_buffer,
                             std::complex<double> *reduce_buffer,
                             std::vector<std::vector<int>> &c_current,
                             std::vector<int> &c_total_current,
                             std::vector<std::vector<int>> &c_expanded,
                             std::vector<int> &c_total_expanded,
                             std::complex<double> beta);

// ---------------------------------------------------------------------------
// BFloat16 optimized infrastructure:
// - AVX-512 vectorized bf16 reduction kernel
// - Persistent SHM window for intra-node reduce_scatter bypass
// - Aligned scratch buffers (no malloc on critical path)
// - Parallel memcpy for large transfers
// ---------------------------------------------------------------------------

// Aligned buffer cache — avoids malloc/free on every call.
// Grows as needed but never shrinks. Uses 64-byte aligned allocation.
struct aligned_int_cache {
    int *buf = nullptr;
    size_t capacity = 0;

    int *get(size_t n) {
        if (n <= capacity) return buf;
        if (buf) std::free(buf);
        capacity = n;
        buf = static_cast<int *>(std::aligned_alloc(64, capacity * sizeof(int)));
        return buf;
    }
};

static aligned_int_cache &get_block_offset_cache() {
    static auto *c = new aligned_int_cache();
    return *c;
}
static aligned_int_cache &get_recvcnts_cache() {
    static auto *c = new aligned_int_cache();
    return *c;
}
static aligned_int_cache &get_displs_cache() {
    static auto *c = new aligned_int_cache();
    return *c;
}

// Parallel memcpy using OpenMP — saturates memory bandwidth
static void parallel_memcpy(void *dst, const void *src, size_t nbytes) {
    char *d = static_cast<char *>(dst);
    const char *s = static_cast<const char *>(src);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nbytes; i += 4096) {
        size_t chunk = std::min(static_cast<size_t>(4096), nbytes - i);
        std::memcpy(d + i, s + i, chunk);
    }
}

__attribute__((target("avx512f,avx512bw,avx512bf16")))
static void shm_reduce_bf16_kernel(const uint16_t *src, uint16_t *dst, int count) {
    const int simd_width = 16;
    const int n_simd = count & ~(simd_width - 1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_simd; i += simd_width) {
        __m256i acc_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + i));
        __m256i src_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        __m512i acc_i32 = _mm512_cvtepu16_epi32(acc_bf16);
        __m512i src_i32 = _mm512_cvtepu16_epi32(src_bf16);
        __m512 acc_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(acc_i32, 16));
        __m512 src_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(src_i32, 16));
        __m512 sum_f32 = _mm512_add_ps(acc_f32, src_f32);
        __m256i sum_bf16 = _mm512_cvtneps_pbh(sum_f32);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), sum_bf16);
    }
    for (int i = n_simd; i < count; ++i) {
        float a = static_cast<float>(reinterpret_cast<cosma::bfloat16 *>(dst)[i]);
        float b = static_cast<float>(reinterpret_cast<const cosma::bfloat16 *>(src)[i]);
        reinterpret_cast<cosma::bfloat16 *>(dst)[i] = cosma::bfloat16(a + b);
    }
}

// Persistent SHM window cache — avoids allocate/free on every call.
struct shm_win_cache {
    MPI_Win win = MPI_WIN_NULL;
    cosma::bfloat16 *shm_base = nullptr;
    MPI_Aint capacity = 0; // in elements
    cosma::bfloat16 **rank_ptrs = nullptr;
    int cached_div = 0;
    int cached_comm_size = -1;

    void ensure(MPI_Comm comm, MPI_Aint need_elems, int div, int my_rank_in_group) {
        if (win != MPI_WIN_NULL && need_elems <= capacity && div == cached_div) return;
        if (win != MPI_WIN_NULL) MPI_Win_free(&win);
        capacity = need_elems;
        cached_div = div;
        MPI_Aint shm_bytes = capacity * static_cast<MPI_Aint>(sizeof(cosma::bfloat16));
        MPI_Win_allocate_shared(shm_bytes, sizeof(cosma::bfloat16),
                                MPI_INFO_NULL, comm, &shm_base, &win);
        if (rank_ptrs) std::free(rank_ptrs);
        rank_ptrs = static_cast<cosma::bfloat16 **>(
            std::aligned_alloc(64, div * sizeof(cosma::bfloat16 *)));
        for (int r = 0; r < div; ++r) {
            if (r == my_rank_in_group) {
                rank_ptrs[r] = shm_base;
            } else {
                MPI_Aint sz;
                int disp_unit;
                MPI_Win_shared_query(win, r, &sz, &disp_unit, &rank_ptrs[r]);
            }
        }
    }
};

static shm_win_cache &get_shm_cache() {
    static auto *cache = new shm_win_cache();
    return *cache;
}

// ---- copy<bfloat16> specialization: parallel reshuffle ----
template <>
void copy<cosma::bfloat16>(MPI_Comm comm,
          int rank,
          int div,
          Interval &P,
          cosma::bfloat16 *in,
          cosma::bfloat16 *out,
          cosma::bfloat16 *reshuffle_buffer,
          std::vector<std::vector<int>> &size_before,
          std::vector<int> &total_before,
          int total_after) {
    using Scalar = cosma::bfloat16;
    PE(multiply_communication_other);

    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int relative_rank = rank - P.first();
    int local_size = total_before[relative_rank];

    // Use cached aligned buffers instead of std::vector
    int *total_size = get_recvcnts_cache().get(div);
    int *dspls = get_displs_cache().get(div);

    int sum_total = 0;
    bool same_size = true;
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        int temp_size = total_before[target];
        dspls[i] = sum_total;
        sum_total += temp_size;
        total_size[i] = temp_size;
        same_size &= temp_size == local_size;
    }

    int n_blocks = size_before[relative_rank].size();
    Scalar *receive_pointer = n_blocks > 1 ? reshuffle_buffer : out;
    PL();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    PE(multiply_communication_copy);
    if (same_size) {
        MPI_Allgather(in, local_size, mpi_type,
                      receive_pointer, local_size, mpi_type, comm);
    } else {
        MPI_Allgatherv(in, local_size, mpi_type,
                       receive_pointer, total_size, dspls, mpi_type, comm);
    }
    PL();

    PE(multiply_communication_other);
    if (n_blocks > 1) {
        int *block_off = get_block_offset_cache().get(div);
        std::memset(block_off, 0, div * sizeof(int));
        int idx = 0;
        for (int block = 0; block < n_blocks; block++) {
            for (int r = 0; r < div; r++) {
                int target = P.locate_in_interval(div, r, off);
                int dsp = dspls[r] + block_off[r];
                int b_size = size_before[target][block];
                // Use parallel memcpy for large blocks
                if (b_size * static_cast<int>(sizeof(Scalar)) > 65536) {
                    parallel_memcpy(out + idx, reshuffle_buffer + dsp,
                                    b_size * sizeof(Scalar));
                } else {
                    std::memcpy(out + idx, reshuffle_buffer + dsp,
                                b_size * sizeof(Scalar));
                }
                idx += b_size;
                block_off[r] += b_size;
            }
        }
    }
    PL();
}

// ---- reduce<bfloat16> specialization ----
template <>
void
reduce<cosma::bfloat16>(MPI_Comm comm,
                        int rank,
                        int div,
                        Interval &P,
                        cosma::bfloat16 *LC,
                        cosma::bfloat16 *C,
                        cosma::bfloat16 *reshuffle_buffer,
                        cosma::bfloat16 *reduce_buffer,
                        std::vector<std::vector<int>> &c_current,
                        std::vector<int> &c_total_current,
                        std::vector<std::vector<int>> &c_expanded,
                        std::vector<int> &c_total_expanded,
                        cosma::bfloat16 beta) {
    using Scalar = cosma::bfloat16;
    PE(multiply_communication_other);

    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int n_blocks = c_expanded[off].size();
    int *block_offset = get_block_offset_cache().get(n_blocks);
    Scalar *send_pointer = n_blocks > 1 ? reshuffle_buffer : LC;

    int sum = 0;
    for (int i = 0; i < n_blocks; ++i) {
        block_offset[i] = sum;
        sum += c_expanded[off][i];
    }

    int *recvcnts = get_recvcnts_cache().get(div);
    int *displs = get_displs_cache().get(div);

    bool same_size = true;
    int index = 0;
    int total_send = 0;
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        recvcnts[i] = c_total_current[target];
        same_size = same_size && recvcnts[i] == recvcnts[0];
        displs[i] = total_send;
        total_send += recvcnts[i];

        if (n_blocks > 1) {
            for (int block = 0; block < n_blocks; ++block) {
                int b_offset = block_offset[block];
                int b_size = c_current[target][block];
                std::memcpy(reshuffle_buffer + index, LC + b_offset,
                            b_size * sizeof(Scalar));
                index += b_size;
                block_offset[block] += b_size;
            }
        }
    }

    Scalar *receive_pointer = beta != Scalar{0} ? reduce_buffer : C;
    PL();

    PE(multiply_communication_reduce);

    // Decide: SHM path (intra-node) or MPI path (inter-node).
    // Cache the per-node rank count once, then compare comm size.
    static int ranks_per_node = -1;
    if (ranks_per_node < 0) {
        MPI_Comm shm_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &shm_comm);
        MPI_Comm_size(shm_comm, &ranks_per_node);
        MPI_Comm_free(&shm_comm);
    }
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    bool use_shm = (comm_size <= ranks_per_node);

    if (use_shm) {
        // --- SHM reduce path: all ranks share memory ---
        auto &shm = get_shm_cache();
        shm.ensure(comm, total_send, div, gp);

        // Parallel copy send data into shared window
        parallel_memcpy(shm.shm_base, send_pointer,
                        static_cast<size_t>(total_send) * sizeof(Scalar));

        MPI_Barrier(comm);

        const int my_count = recvcnts[gp];
        const int my_displ = displs[gp];

        // Copy first rank's segment as initial accumulator
        parallel_memcpy(receive_pointer, shm.rank_ptrs[0] + my_displ,
                        static_cast<size_t>(my_count) * sizeof(Scalar));

        // Accumulate remaining ranks
        for (int r = 1; r < div; ++r) {
            shm_reduce_bf16_kernel(
                reinterpret_cast<const uint16_t *>(shm.rank_ptrs[r] + my_displ),
                reinterpret_cast<uint16_t *>(receive_pointer),
                my_count);
        }

        MPI_Barrier(comm);
    } else {
        // --- MPI reduce path: inter-node communication ---
        auto mpi_type = mpi_mapper<Scalar>::getType();
        auto mpi_sum = mpi_mapper<Scalar>::getSumOp();
        if (same_size) {
            MPI_Reduce_scatter_block(send_pointer, receive_pointer,
                                     recvcnts[0], mpi_type, mpi_sum, comm);
        } else {
            MPI_Reduce_scatter(send_pointer, receive_pointer,
                               recvcnts, mpi_type, mpi_sum, comm);
        }
    }

    PL();

    PE(multiply_communication_other);
    if (beta != Scalar{0}) {
        #pragma omp parallel for schedule(static)
        for (int el = 0; el < recvcnts[gp]; ++el) {
            C[el] = beta * C[el] + reduce_buffer[el];
        }
    }
    PL();
}

} // end namespace two_sided_communicator

} // namespace cosma
