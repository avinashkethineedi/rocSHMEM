#include <rocshmem/rocshmem.hpp>
#include "../util.h"

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define FINISHED_SUM_TAG 1024
static constexpr int32_t kWarpSize = 64;

using namespace rocshmem;

namespace ll_kernels {

template <typename T>
__host__ __device__ T cell_div(T a, T b) {
  return (a + b - 1) / b;
}

// Warp synchronization function
__forceinline__ __device__ void warp_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Grid barrier function
__forceinline__ __device__ void grid_barrier(int* global_counter,
    int num_blocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(&global_counter[0], 1,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(global_counter,
                             __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) != num_blocks);
  }
  __syncthreads();
}

// Warp reduction sum function
__forceinline__ __device__ int warp_reduce_sum(int val) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
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
  const int wg_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int warp_id = thread_id / kWarpSize;
  const int num_wgs = static_cast<int>(gridDim.x);
  const int lane_id = thread_id % kWarpSize;
  constexpr int num_warps = kNumWarpsPerGroup * kNumWarpGroups;
  const int num_local_experts = num_experts / num_ranks;
  const int warp_group_id = warp_id / kNumWarpsPerGroup;
  const int sub_warp_id = warp_id % kNumWarpsPerGroup;
  const int responsible_expert_id = wg_id * kNumWarpGroups + warp_group_id;

  // size of each token in bytes
  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(T);
  // size of each message in bytes
  const size_t num_bytes_per_msg = sizeof(int) + hidden_bytes;
  /**
   * TODO: Add assertions to check the following conditions:
   * _ASSERT(num_bytes_per_msg % sizeof(int) == 0);
   */
  const size_t num_int_per_msg = num_bytes_per_msg / sizeof(int);
  /**
   * TODO: Skip the sending phase based on the phase flag
   */
  if (thread_id == 0 && wg_id == 0) {
    printf("num_wgs: %d, num_warps: %d, hidden: %d, (bytes: %ld, num_int: %ld), "
           "num_tokens: %d, num_topk: %d, num_experts: %d, rank: %d, "
           "num_ranks: %d, num_bytes_per_msg: %ld\n",
           num_wgs, num_warps, hidden, hidden_bytes, num_int_per_msg,
           num_tokens, num_topk, num_experts, rank, num_ranks, num_bytes_per_msg);
  }
  
  // Expert counts
  __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];
  if (warp_id < num_warps) {
    constexpr int num_threads = kNumWarpGroups * kNumWarpsPerGroup * kWarpSize;
    for (int token_idx = wg_id; token_idx < num_tokens; token_idx += num_wgs) {
      // Pointer to the token data
      // Dimensions: [num_tokens][hidden]
      const T* x_ptr = reinterpret_cast<const T*>(x) + token_idx * hidden;
      // Source symmetric heap buffer for RDMA write
      int* const rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx *
          num_bytes_per_msg);
      // Source data pointer to store the token data after the int header
      T* const rdma_x_vec = reinterpret_cast<T*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int));

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
                             num_bytes_per_msg + rank * num_tokens *
                             num_bytes_per_msg + slot_idx * num_bytes_per_msg;

        if (dst_rank != rank) {
          // Remote RDMA write using rocSHMEM
          rocshmem_putmem_nbi_wave(reinterpret_cast<void*>(dst_ptr),
              reinterpret_cast<void*>(src_ptr), num_bytes_per_msg, dst_rank);
        } else {
          // Local copy for same-rank communication
          warp_copy<T>(reinterpret_cast<T*>(dst_ptr),
              reinterpret_cast<T*>(src_ptr), hidden + sizeof(int)/sizeof(T));
        }

        warp_sync();
        // Increment local counter after ensuring PUTs are issued
        lane_id == 0 ?
          __hip_atomic_fetch_add(atomic_finish_counter_per_expert + dst_expert_idx, 1,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) : 0;
      }
    }
  }
  if (warp_id == num_warps - 1) {
    if (wg_id == 0) {
      // The first WG is also responsible for cleaning the next buffer
      for (int i = lane_id; i < num_next_clean_int; i += kWarpSize)
        next_clean[i] = 0;
      // Whis is this required ..?
      for (int i = lane_id; i < num_experts; i += kWarpSize) {
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i,
                               FINISHED_SUM_TAG, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    /**
     * Each work group is responsible for some destination experts, read `topk_idx` for them
     * and count the number of tokens sent to those experts
     */
    int expert_count[kNumWarpGroups] = {0};
    const int expert_begin_idx = wg_id * kNumWarpGroups;
    const int expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);

    // Per lane count
    for (int i = lane_id; i < num_tokens * num_topk; i += kWarpSize) {
      const int64_t idx = static_cast<int64_t>(topk_idx[i]);
      if (idx >= expert_begin_idx && idx < expert_end_idx) {
        expert_count[idx - expert_begin_idx]++;
      }
    }

    // Warp reduce
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      int sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i,
                               FINISHED_SUM_TAG - sum, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  // Synchronize work-group
  __syncthreads();

  // Notify each expert about the number of tokens sent to it
  if (responsible_expert_id < num_experts && sub_warp_id == 0 && lane_id == 0) {
    const int dst_rank = responsible_expert_id / num_local_experts;
    const int dst_expert_local_idx = responsible_expert_id % num_local_experts;
    const int num_tokens_sent =
        shared_num_tokens_sent_per_expert[responsible_expert_id -
                                          wg_id * kNumWarpGroups];

    // Wait until all tokens have been sent and counted
    while(__hip_atomic_load(atomic_finish_counter_per_expert + responsible_expert_id,
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) !=
          FINISHED_SUM_TAG * 2);

    if (dst_rank != rank) {
      rocshmem_long_atomic_add(
          rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
          -num_tokens_sent - 1, dst_rank);
    } else {
      // Local store for same-rank communication
      /**
       * Local store for same-rank communication
       * TODO: Does it require atomic store? each store is to a unique location
       */
      __hip_atomic_store(rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
                         -num_tokens_sent - 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }

    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_id] = 0;
    atomic_finish_counter_per_expert[responsible_expert_id] = 0;

    // Clean packed_recv_count
    if (dst_rank == 0)
      packed_recv_count[dst_expert_local_idx] = 0;
  }
  warp_sync();

  /**
   * TODO: Add SEND and RECV phase flags to control the synchronization
   */
  grid_barrier(global_atomic_counter, num_wgs);

  // Pack the data from rdma_recv_x to packed_recv_x
  if (responsible_expert_id < num_experts) {
    const int src_rank = responsible_expert_id / num_local_experts;
    const int local_expert_idx = responsible_expert_id % num_local_experts;
    /**
     * Pointer to the starting location of the local expert's data of it's
     * source rank in rdma_recv_x
     * Dimensions: [num_local_experts][num_ranks][num_tokens]
     */
    T* const rdma_recv_x_ptr = reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_tokens * num_bytes_per_msg +
        src_rank * num_tokens * num_bytes_per_msg);
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_x
     */
    T* const packed_recv_x_ptr = reinterpret_cast<T*>(packed_recv_x) +
        local_expert_idx * num_ranks * num_tokens * hidden;
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_src_info
     */
    int* const packed_recv_src_info_ptr = packed_recv_src_info +
        local_expert_idx * num_ranks * num_tokens;
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_layout_range
     */
    int64_t* const packed_recv_layout_range_ptr =
        packed_recv_layout_range + local_expert_idx * num_ranks;

    // Shared between sub-warps in warp groups
    // Why is this required..?
    __shared__ int shared_num_recv_tokens[kNumWarpGroups],
                   shared_recv_token_begin_idx[kNumWarpGroups];

    /**
     * Wait until tokens are received for the assigned local expert
     * from its source rank
     */
    int num_recv_tokens, recv_token_begin_idx;
    if (sub_warp_id == 0 && lane_id == 0) {
      while ((num_recv_tokens = __hip_atomic_load(
                  reinterpret_cast<int*>(rdma_recv_count + local_expert_idx *
                                         num_ranks + src_rank),
                  __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) == 0);
      num_recv_tokens = -num_recv_tokens - 1;
      /**
       * Once the number of received tokens is known, pack the data from
       * rdma_recv_x to packed_recv_x
       *
       * each local expert's data is stored contiguously for each rank in
       * packed_recv_x, but the each rank's starting location is not known until
       * the number of received tokens is known
       */
      recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx,
                                       num_recv_tokens);
      // Store the info in the shared buffer for other sub-warps in the warp group
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      // Pack the recv range info
      packed_recv_layout_range_ptr[src_rank] =
          (static_cast<int64_t>(recv_token_begin_idx) << 32) |
          static_cast<int64_t>(num_recv_tokens);
    }
    // Synchronize sub-warps in the warp group
    __syncthreads();
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];
    // Pack the received data from rdma_recv_x to packed_recv_x
    for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
      // Source token index in rdma_recv_x
      const int* const src_token_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_recv_x_ptr) +
          i * num_bytes_per_msg);
      // Write the source token index to packed_recv_src_info from lane 0
      if (lane_id == 0) {
        packed_recv_src_info_ptr[recv_token_begin_idx + i] = *src_token_idx;
      }
      warp_sync();
      // Source token data pointer in rdma_recv_x
      const T* const src_token_data = reinterpret_cast<T*>(
          reinterpret_cast<uint8_t*>(rdma_recv_x_ptr) +
          i * num_bytes_per_msg + sizeof(int));
      // Destination token data pointer in packed_recv_x
      T* const dst_token_data = packed_recv_x_ptr +
          (recv_token_begin_idx + i) * hidden;
      // Copy the token data
      warp_copy<T>(dst_token_data, src_token_data, hidden);
    }
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
  const auto num_wgs     = cell_div(num_experts, kNumWarpGroups);
  const auto num_threads = num_warps * kWarpSize;
  
  // TODO: Add assertions to check num_topk <= kNumMaxTopK
  ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  int* atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  int* atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;

  // TODO: Update the assert
  ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

  dim3 grid(num_wgs);
  dim3 block(num_threads);

  /**
   * Calculate the maximum number of co-resident work-groups per compute unit
   * based on the resource usage of the kernel
   */
  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu,
      dispatch_kernel<kNumWarpsPerGroup, kNumWarpGroups, T>,
      num_threads,
      0));
  // Get the number of compute units
  hipDeviceProp_t device_prop;
  CHECK_HIP(hipGetDeviceProperties(&device_prop, 0));
  const int num_cus = device_prop.multiProcessorCount;
  const int max_sustainable_wgs = max_co_resident_wgs_per_cu * num_cus;

  // printf for debugging
  std::cout << "Max co-resident WGs per CU: " << max_co_resident_wgs_per_cu
            << ", Num CUs: " << num_cus
            << ", Max sustainable WGs: " << max_sustainable_wgs << std::endl;

  std::cout << "Launching dispatch kernel with grid (" << grid.x << ", "
            << grid.y << ", " << grid.z << ") and block (" << block.x
            << ", " << block.y << ", " << block.z << ")\n"
            << ", num_warps: " << num_warps << ", num_wgs: " << num_wgs
            << ", num_threads: " << num_threads
            << ", kNumWarpsPerGroup: " << kNumWarpsPerGroup
            << ", kNumWarpGroups: " << kNumWarpGroups << std::endl;

  dispatch_kernel<kNumWarpsPerGroup, kNumWarpGroups, T><<<grid, block>>>(
      packed_recv_x, packed_recv_src_info, packed_recv_layout_range,
      packed_recv_count, global_atomic_counter, rdma_recv_x,
      rdma_recv_count, rdma_x, x, topk_idx, atomic_counter_per_expert,
      atomic_finish_counter_per_expert, next_clean, num_next_clean_int,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks);

}
}  // namespace ll_kernels