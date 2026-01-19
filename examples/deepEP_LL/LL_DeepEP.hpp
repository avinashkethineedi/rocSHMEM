#include <mpi.h>

#include "LL_Data.hpp"
#include "LL_Buffers.hpp"
#include "LL_Kernels.hpp"

using namespace rocshmem;

template<typename T>
class LLDeepEP {
 private:
  int rank {0}, num_ranks {0};
  int device_id {0};

  // rocSHMEM buffer
  int64_t num_rdma_bytes {0};
  void*   rdma_buffer_ptr {nullptr};

  // Workspace (32 MiB)
  void* workspace {nullptr};

  // LL_DeepEP parameters
  const int num_tokens {0};
  const int hidden {0};
  const int num_topk {0};
  const int num_experts {0};

  // LL_DeepEP data
  LLData<T> ll_data;

  // LL Buffer index
  int ll_buffer_idx {0};

  // Dispatch buffers
  T*        packed_recv_x {nullptr};
  int*      packed_recv_src_info {nullptr};
  int64_t*  packed_recv_layout_range {nullptr};
  int*      packed_recv_count {nullptr};
  int*      global_atomic_counter {nullptr};

 public:
  LLDeepEP(int num_tokens_, int hidden_, int num_topk_, int num_experts_)
      : num_tokens(num_tokens_), hidden(hidden_),
        num_topk(num_topk_), num_experts(num_experts_),
        ll_data(num_tokens_, hidden_, num_topk_, num_experts_) {

    // Initialize rocSHMEM
    comm_init();

    // Get device info
    CHECK_HIP(hipGetDevice(&device_id));

    CHECK_HIP(hipExtMallocWithFlags(&workspace, NUM_WORKSPACE_BYTES,
              hipDeviceMallocUncached));
    CHECK_HIP(hipMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES));

    // Allocate rocSHMEM buffer
    num_rdma_bytes = get_rmda_size_hint<T>(
        num_tokens, hidden, num_ranks, num_experts);
    rdma_buffer_ptr = rocshmem_malloc(num_rdma_bytes);
    if (rdma_buffer_ptr == nullptr) {
      std::cerr << "Rank " << rank
                << ": Error in rocshmem_malloc. Aborting." << std::endl;
      comm_finalize();
      // Clean up other resources and exit
      exit(EXIT_FAILURE);
    }
    CHECK_HIP(hipMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes));

    // Allocate dispatch buffers
    allocate_dispatch_buffers();

    // Synchronize to ensure all allocations are done
    CHECK_HIP(hipDeviceSynchronize());
  }

  ~LLDeepEP() {
    CHECK_HIP(hipFree(workspace));
    CHECK_HIP(hipFree(packed_recv_x));
    CHECK_HIP(hipFree(packed_recv_src_info));
    CHECK_HIP(hipFree(packed_recv_layout_range));
    CHECK_HIP(hipFree(packed_recv_count));
    rocshmem_free(rdma_buffer_ptr);
    comm_finalize();
  }

  int get_rank() const {
    return rank;
  }

  int get_num_ranks() const {
    return num_ranks;
  }

  void ll_dispatch() { // hipStream_t stream : TODO: pass stream
    int num_local_experts {num_experts / num_ranks};

    // Buffer control
    LLBufferLayout<T> ll_layout(rdma_buffer_ptr, num_tokens,
                       hidden, num_ranks, num_experts);
    LLBuffer& buffer = ll_layout.buffers[ll_buffer_idx];
    LLBuffer& next_buffer = ll_layout.buffers[ll_buffer_idx ^= 1];

    // Launch dispatch kernel
    ll_kernels::dispatch<T>(packed_recv_x, packed_recv_src_info,
      packed_recv_layout_range, packed_recv_count, global_atomic_counter,
      buffer.dispatch_recv_buffer, buffer.dispatch_recv_count_buffer,
      buffer.dispatch_send_buffer, ll_data.X, ll_data.topk_idx,
      next_buffer.clean_meta().first, next_buffer.clean_meta().second,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks, workspace);
  }

 private:
  // Initialize rocSHMEM
  void comm_init() {

    // Initialize MPI
    int mpi_rank {0}, mpi_size {0};
    int ret {0};
    int provided {0};
    MPI_Init_thread (nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
      std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
    }
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

    // Initialize rocSHMEM with unique ID
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;
    if (mpi_rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << mpi_rank
        << ": Error in rocshmem_get_uniqueid. Aborting." << std::endl;
        MPI_Abort (MPI_COMM_WORLD, ret);
      }
    }
    MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    ret = rocshmem_set_attr_uniqueid_args(mpi_rank, mpi_size, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank
                << ": Error in rocshmem_set_attr_uniqueid_args. Aborting"
                << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank << ": Error in rocshmem_init_attr. Aborting."
                << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    rank      = rocshmem_my_pe();
    num_ranks = rocshmem_n_pes();
  }

  void comm_finalize() {
    rocshmem_finalize();
    MPI_Finalize();
  }

  // Allocate GPU buffers/tensors for dispatch
  void allocate_dispatch_buffers() {
    int num_local_experts = num_experts / num_ranks;
    size_t packed_recv_x_bytes =
        num_local_experts * num_ranks * num_tokens * hidden * sizeof(T);
    size_t packed_recv_src_info_bytes =
        num_local_experts * num_ranks * num_tokens * sizeof(int);
    size_t packed_recv_layout_range_bytes =
        num_local_experts * num_ranks * sizeof(int64_t);
    size_t packed_recv_count_bytes =
        num_local_experts * sizeof(int);

    CHECK_HIP(hipMalloc(&packed_recv_x, packed_recv_x_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_src_info, packed_recv_src_info_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_layout_range,
                        packed_recv_layout_range_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_count, packed_recv_count_bytes));
    CHECK_HIP(hipMalloc(&global_atomic_counter, sizeof(int)));
  }

};