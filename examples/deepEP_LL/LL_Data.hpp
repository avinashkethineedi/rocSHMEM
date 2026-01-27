#include "../util.h"
#include <random>
#include <vector>

template <typename T>
class LLData {
 public:

  // LL_DeepEP parameters
  const int num_tokens {0};
  const int hidden {0};
  const int num_topk {0};
  const int num_experts {0};

  // Input data
  T* X {nullptr};

  // Token to expert mapping
  int64_t *topk_idx {nullptr};

  // Number of tokens assigned to each expert
  std::vector<int> expert_token_count;

  enum class InitMode {
    Deterministic,
    Random
  };

 private:
  InitMode init_mode {InitMode::Deterministic};

 public:
  LLData(int num_tokens_, int hidden_, int num_topk_,
    int num_experts_, InitMode init_mode_ = InitMode::Deterministic)
      : num_tokens(num_tokens_), hidden(hidden_),
        num_topk(num_topk_), num_experts(num_experts_),
        init_mode(init_mode_), expert_token_count(num_experts, 0) {}

  ~LLData() {
    if (X) {
      CHECK_HIP(hipFree(X));
    }
    if (topk_idx) {
      CHECK_HIP(hipFree(topk_idx));
    }
  }

  // Function to generate input data
  void generate_data() {

    size_t x_size_bytes = num_tokens * hidden * sizeof(T);
    size_t topk_idx_size_bytes = num_tokens * num_topk * sizeof(int64_t);

    CHECK_HIP(hipMalloc(&X, x_size_bytes));
    CHECK_HIP(hipMalloc(&topk_idx, topk_idx_size_bytes));

    size_t x_size = num_tokens * hidden;

    // Launch kernel to fill input data (X)
    int threads_per_block = 1024;
    int blocks_per_grid = num_tokens; // one block per token

    int gpu_id {0};
    CHECK_HIP(hipGetDevice(&gpu_id));

    // print GPU id
    std::cout << "Generating data on GPU " << gpu_id << std::endl;

    data_kernel<<<blocks_per_grid, threads_per_block>>>(X, hidden, gpu_id);
    CHECK_HIP(hipDeviceSynchronize());

    switch (init_mode) {
      case InitMode::Deterministic:
        generate_topk_deterministic();
        break;
      case InitMode::Random:
        generate_topk_random();
        break;
    }

    // Debug prints
    print();
    // print_data();
  }

 private:
  void print() {
    std::cout << "LLData:" << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  hidden: " << hidden << std::endl;
    std::cout << "  num_experts: " << num_experts << std::endl;
    std::cout << "  num_topk: " << num_topk << std::endl;
    std::cout << "  init_mode: "
              << (init_mode == InitMode::Deterministic
                      ? "Deterministic"
                      : "Random")
              << std::endl;
  }

  void print_data() {
    size_t x_size = num_tokens * hidden;
    size_t topk_idx_size = num_tokens * num_topk;

    std::vector<T> h_X(x_size);
    std::vector<int64_t> h_topk_idx(topk_idx_size);

    CHECK_HIP(hipMemcpy(h_X.data(), X, x_size * sizeof(T),
              hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_topk_idx.data(), topk_idx,
              topk_idx_size * sizeof(int64_t), hipMemcpyDeviceToHost));

    std::cout << "Input Data (X):" << std::endl;
    for (size_t i = 0; i < num_tokens; i++) {
      std::cout << "Token " << i << ": ";
      for (int j = 0; j < hidden; j++) {
        std::cout << h_X[i * hidden + j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "Top-k Indices:" << std::endl;
    for (size_t i = 0; i < num_tokens; i++) {
      std::cout << "Token " << i << ": ";
      for (int k = 0; k < num_topk; k++) {
        std::cout << h_topk_idx[i * num_topk + k] << " ";
      }
      std::cout << std::endl;
    }

    // Print expert token counts
    std::cout << "Expert Token Counts:" << std::endl;
    for (int j = 0; j < num_experts; j++) {
      std::cout << "Expert " << j << ": " << expert_token_count[j] << " tokens" << std::endl;
    }
  }

  /**
   * GPU kernel to generate the input data
   * Each token's hidden vector is filled with the token index
   */
  __global__ static void data_kernel(T* X, int hidden, int rank) {
    int tkn_idx = blockIdx.x;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
      X[tkn_idx * hidden + h] = tkn_idx + 10 + rank * 1000;
    }
  }

  // Generate random top-k indices for each token
  void generate_topk_random() {
    size_t topk_idx_size = num_tokens * num_topk;
    std::vector<int64_t> h_topk_idx(topk_idx_size);
    std::vector<int> expert_indices(num_experts);

    /**
     * TODO: use simple rand() function and make sure unique experts are chosen
     * for each token
     */
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int j = 0; j < num_experts; j++) {
      expert_indices[j] = j;
    }

    for (size_t i = 0; i < num_tokens; i++) {
      std::shuffle(expert_indices.begin(), expert_indices.end(), gen);
      for (int k = 0; k < num_topk; k++) {
        h_topk_idx[i * num_topk + k] = expert_indices[k];
        expert_token_count[expert_indices[k]]++;
      }
    }
    /**
     * TODO: is the hipmemcpy required here? can topk_idx be accessed directly on host?
     */
    CHECK_HIP(hipMemcpy(topk_idx, h_topk_idx.data(),
                        topk_idx_size * sizeof(int64_t), hipMemcpyHostToDevice));
  }

  // Generate deterministic top-k indices for each token
  // Each token gets experts in round-robin fashion
  void generate_topk_deterministic() {
    size_t topk_idx_size = num_tokens * num_topk;
    std::vector<int64_t> h_topk_idx(topk_idx_size);
    for (size_t i = 0; i < num_tokens; i++) {
      for (int k = 0; k < num_topk; k++) {
        h_topk_idx[i * num_topk + k] = (i * num_topk + k) % num_experts;
        expert_token_count[h_topk_idx[i * num_topk + k]]++;
      }
    }
    /**
     * TODO: is the hipmemcpy required here? can topk_idx be accessed directly on host?
     */
    CHECK_HIP(hipMemcpy(topk_idx, h_topk_idx.data(),
                        topk_idx_size * sizeof(int64_t), hipMemcpyHostToDevice));
  }
};
