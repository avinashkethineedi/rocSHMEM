#include "../util.h"

/**
 * Low-latency buffer structure
 * - Separate dispatch and combine buffers
 * - Separate send and receive buffers
 * - Separate signaling buffers for dispatch and combine
 */
struct LLBuffer {
  // Number of signaling elements = number of experts
  int num_sig_elems {0};

  // Dispatch buffers
  void*    dispatch_send_buffer {nullptr};
  void*    dispatch_recv_buffer {nullptr};
  int64_t* dispatch_recv_count_buffer {nullptr};

  // Combine buffers
  void*    combine_send_buffer {nullptr};
  void*    combine_recv_buffer {nullptr};
  int64_t* combine_recv_flag_buffer {nullptr};

  std::pair<int64_t*, int> clean_meta() {
    ASSERT(dispatch_recv_count_buffer == combine_recv_flag_buffer);
    return {dispatch_recv_count_buffer, num_sig_elems};
  }
};

/**
 * Low-latency buffer layout
 * - Two sets of LLBuffer for double buffering
 * - Calculates total bytes required for allocation
 */
template <typename T>
struct LLBufferLayout {
  size_t total_bytes {0};
  LLBuffer buffers[2];

  template <typename out_ptr_t = void*,
            typename count_ptr_t = uint8_t*,
            typename in_ptr_t = void*>
  out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
      return reinterpret_cast<out_ptr_t>(
          reinterpret_cast<count_ptr_t>(ptr) + count);
  }

  LLBufferLayout(void* rdma_buffer, const int num_tokens, const int hidden,
      const int num_ranks, const int num_experts) {
    
    const int num_local_experts = num_experts / num_ranks;

    // Message sizes
    size_t num_bytes_per_dispatch_msg = sizeof(int4) + hidden * sizeof(T);
    size_t num_bytes_per_combine_msg  = sizeof(int4) + hidden * sizeof(T);

    // Send buffers sizes
    size_t dispatch_send_buffer_bytes = num_tokens *
                                        num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes  = num_experts *
                                        num_tokens *
                                        num_bytes_per_combine_msg;
    size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes,
                                        combine_send_buffer_bytes);

    // std::cout << "Dispatch send buffer bytes: "
    //           << dispatch_send_buffer_bytes
    //           << ", Combine send buffer bytes: "
    //           << combine_send_buffer_bytes
    //           << ", num bytes per dispatch msg: "
    //           << num_bytes_per_dispatch_msg
    //           << ", sizeof(T): "
    //           << sizeof(T)
    //           << ", num tokens: "
    //           << num_tokens
    //           << ", num experts: "
    //           << num_experts
    //           << ", hidden: "
    //           << hidden
    //           << ", send buffer bytes: "
    //           << send_buffer_bytes
    //           << ", sizeof(int4): "
    //           << sizeof(int4)
    //           << ", send_buffer_bytes % sizeof(int4): "
    //           << (send_buffer_bytes % sizeof(int4) == 0)
    //           << std::endl;

    ASSERT(send_buffer_bytes % sizeof(int4) == 0);
    total_bytes += send_buffer_bytes * 2;

    // receive buffers sizes
    size_t dispatch_recv_buffer_bytes = num_experts * num_tokens *
                                        num_bytes_per_dispatch_msg;
    size_t combine_recv_buffer_bytes  = num_experts * num_tokens *
                                        num_bytes_per_combine_msg;
    size_t recv_buffer_bytes = std::max(dispatch_recv_buffer_bytes,
                                        combine_recv_buffer_bytes);
    ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
    total_bytes += recv_buffer_bytes * 2;

    // Symmetric signaling buffers
    size_t signaling_buffer_bytes = num_experts * sizeof(int64_t);
    total_bytes += signaling_buffer_bytes * 2;

    // Assign pointers
    for (int i = 0; i < 2; ++ i) {
        buffers[i] = {
            num_experts,
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i),
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i)
        };
    }
  }
};

/**
 * Get RDMA size hint for low-latency buffers
 * - Used for allocating rocSHMEM symmetric memory buffer
 */
template <typename T>
size_t get_rmda_size_hint(int num_max_dispatch_tokens_per_rank, int hidden,
    int num_ranks, int num_experts) {
  LLBufferLayout<T> ll_buffer_layout(nullptr, num_max_dispatch_tokens_per_rank,
                      hidden, num_ranks, num_experts);
  return ll_buffer_layout.total_bytes;
}