#include <rocshmem/rocshmem.hpp>
#include "../util.h"

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
static constexpr int32_t kWarpSize = 64;

using namespace rocshmem;

namespace ll_kernels {

template <typename T>
__host__ __device__ T cell_div(T a, T b) {
  return (a + b - 1) / b;
}

// device warp function to copy the data from source to destination
template <typename T>
__device__ void warp_copy(T* dst, const T* src, size_t num_elems) {
  const int lane_id = threadIdx.x % kWarpSize;
  for (size_t i = lane_id; i < num_elems; i += kWarpSize) {
    dst[i] = src[i];
  }
  __threadfence();
}

// Dispatch kernel for low-latency deepEP
template <int kNumWarpsPerGroup, int kNumWarpGroups, typename T>
__global__ __launch_bounds__(kNumWarpsPerGroup * kNumWarpGroups * kWarpSize, 1)
void dispatch_kernel(void *packed_recv_x, int *packed_recv_src_info,
    int64_t *packed_recv_layout_range, int *packed_recv_count,
    int *global_atomic_counter, void *rdma_recv_x, int64_t *rdma_recv_count,
    void *rdma_x, const void *x, const int64_t *topk_idx,
    int *atomic_counter_per_expert, int *atomic_finish_counter_per_expert,
    int64_t *next_clean, int num_next_clean_int, int num_tokens, int hidden,
    int num_topk, int num_experts, int rank, int num_ranks) {
  // Kernel implementation goes here
  const int sm_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int warp_id = thread_id / kWarpSize;
  const int num_sms = static_cast<int>(gridDim.x);
  const int lane_id = thread_id % kWarpSize;
  constexpr int num_warps = kNumWarpsPerGroup * kNumWarpGroups;
  const int num_local_experts = num_experts / num_ranks;
  const int warp_group_id = warp_id / kNumWarpsPerGroup;
  const int sub_warp_id = warp_id % kNumWarpsPerGroup;
  const int responsible_expert_id = sm_id * kNumWarpGroups + warp_group_id;

  // size of each token in bytes
  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(T);
  // size of each message in bytes
  const size_t num_bytes_per_msg = sizeof(int4) + hidden_bytes;
  /**
   * TODO: Add assertions to check the following conditions:
   * _ASSERT(num_bytes_per_msg % sizeof(int4) == 0);
   */
  const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
  /**
   * TODO: Skip the sending phase based on the phase flag
   */
  if (thread_id == 0 && sm_id == 0) {
    printf("sm_id: %d, thread_id: %d, warp_id: %d, lane_id: %d, "
           "warp_group_id: %d, sub_warp_id: %d, responsible_expert_id: %d, "
           "num_sms: %d, num_warps: %d, hidden: %d, (bytes: %ld, num_int4: %ld), "
           "num_tokens: %d, num_topk: %d, num_experts: %d, rank: %d, num_ranks: %d\n",
           sm_id, thread_id, warp_id, lane_id, warp_group_id,
           sub_warp_id, responsible_expert_id, num_sms, num_warps,
           hidden, hidden_bytes, num_int4_per_msg, num_tokens, num_topk,
           num_experts, rank, num_ranks);
  }
  
  // Expert counts
  __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];
  if (warp_id < num_warps) {
    constexpr int num_threads = kNumWarpGroups * kNumWarpsPerGroup * kWarpSize;
    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      // Pointer to the token data
      // Dimensions: [num_tokens][hidden]
      const T* x_ptr = reinterpret_cast<const T*>(x) + token_idx * hidden;
      // Source symmetric heap buffer for RDMA write
      int* const rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx *
          num_bytes_per_msg);
      // Source data pointer to store the token data after the int4 header
      T* const rdma_x_vec = reinterpret_cast<T*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));

      // Each warp processes different top-k experts for the same token
      const int64_t dst_expert_idx = warp_id < num_topk ?
          static_cast<int64_t>(topk_idx[token_idx * num_topk + warp_id]) : -1;

      // thread 0 in the warp writes the source token index
      thread_id == 0 ? (*(rdma_x_src_idx) = token_idx) : 0;
      
      // #pragma unroll
      for (int i = thread_id; i < hidden; i += num_threads) {
        // Each thread in the thread block copies a portion of the token data
        rdma_x_vec[i] = x_ptr[i];
      }
      // Synchronize to ensure all threads have completed copying
      __syncthreads();
      // Only warps assigned to valid experts proceed
      if (dst_expert_idx >=0) {
        // Calculate the destination offset for RDMA write
        int slot_idx = lane_id == 0 ?
                       atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
                       : 0;
        // Broadcast the slot index to all threads in the warp
        slot_idx = __shfl(slot_idx, 0);
        const int dst_rank = static_cast<int>(dst_expert_idx / num_local_experts);
        const int dst_expert_local_idx =
            static_cast<int>(dst_expert_idx % num_local_experts);
        // Source ptr for rocSHMEM put
        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        // Destination ptr for rocSHMEM put
        // Dimensions: [num_experts][num_ranks][num_tokens]
        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                             dst_expert_local_idx * num_ranks * num_tokens *
                             num_bytes_per_msg + dst_rank * num_tokens *
                             num_bytes_per_msg + slot_idx * num_bytes_per_msg;

        if (dst_rank != rank) {
          // Remote RDMA write using rocSHMEM
          rocshmem_putmem_nbi(reinterpret_cast<void*>(dst_ptr),
              reinterpret_cast<void*>(src_ptr), num_bytes_per_msg, dst_rank);
        } else {
          // Local copy for same-rank communication
          warp_copy<T>(reinterpret_cast<T*>(dst_ptr),
              reinterpret_cast<T*>(src_ptr), hidden);
        }
      }
    }
  }
}

/**
 * Function to print rdma_x of dimensions [num_tokens][int4 + hidden]
 * for debugging purposes.
 */
template <typename T>
void print_rdma_x(void* rdma_x, int num_tokens, int hidden) {
  T* rdma_x_t = reinterpret_cast<T*>(rdma_x);
  for (int i = 0; i < num_tokens; i++) {
    std::cout << "Token " << i << ": ";
    int token_idx = *(reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(rdma_x) + i * (sizeof(int4) + hidden * sizeof(T))));
    std::cout << "(Index: " << token_idx << ") ";
    for (int j = 0; j < hidden; j++) {
      std::cout << rdma_x_t[i * (hidden + sizeof(int4)/sizeof(T)) + sizeof(int4)/sizeof(T) + j] << " ";
    }
    std::cout << std::endl;
  }
}

// Dispatch function to launch the kernel
template <typename T>
void dispatch(void *packed_recv_x, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* global_atomic_counter, void* rdma_recv_x, int64_t* rdma_recv_count,
    void* rdma_x, const void* x, const int64_t* topk_idx,
    int64_t* next_clean, int num_next_clean_int, int num_tokens, int hidden,
    int num_topk, int num_experts, int rank, int num_ranks,
    void* workspace) {
  // Kernel launch code goes here
  constexpr int kNumWarpsPerGroup = 8;
  constexpr int kNumWarpGroups = 2;

  constexpr int kNumMaxTopK = 9;
  // TODO: Add assertions to check that kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup
  ASSERT(kNumMaxTopK + 1 <= kNumWarpsPerGroup * kNumWarpGroups);

  const auto num_warps   = kNumWarpGroups * kNumWarpsPerGroup;
  const auto num_sms     = cell_div(num_experts, kNumWarpGroups);
  const auto num_threads = num_warps * kWarpSize;
  
  // TODO: Add assertions to check num_topk <= kNumMaxTopK
  ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  int* atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  int* atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;

  // TODO: Update the assert
  ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

  dim3 grid(num_sms);
  dim3 block(num_threads);

  std::cout << "Launching dispatch kernel with grid (" << grid.x << ", "
            << grid.y << ", " << grid.z << ") and block (" << block.x
            << ", " << block.y << ", " << block.z << ")\n"
            << ", num_warps: " << num_warps << ", num_sms: " << num_sms
            << ", num_threads: " << num_threads
            << ", kNumWarpsPerGroup: " << kNumWarpsPerGroup
            << ", kNumWarpGroups: " << kNumWarpGroups << std::endl;

  dispatch_kernel<kNumWarpsPerGroup, kNumWarpGroups, T><<<grid, block>>>(
      packed_recv_x, packed_recv_src_info, packed_recv_layout_range,
      packed_recv_count, global_atomic_counter, rdma_recv_x,
      rdma_recv_count, rdma_x, x, topk_idx, atomic_counter_per_expert,
      atomic_finish_counter_per_expert, next_clean, num_next_clean_int,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks);

  // Synchronize and check for errors
  CHECK_HIP(hipDeviceSynchronize());

  // Print rdma_x for debugging
  // print_rdma_x<T>(rdma_x, num_tokens, hidden);
}
}  // namespace ll_kernels