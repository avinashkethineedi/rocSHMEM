#include "LL_DeepEP.hpp"
#include <unistd.h>

using dtype = short;

int main (int argc, char **argv)
{
  int rank, num_ranks;

  // DeepEP_LL parameters with default values
  int num_tokens  = 128;
  int hidden      = 7168;
  int num_topk    = 8;
  int num_experts = 288;

  int num_iterations = 10;

  // parse command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "n:h:k:e:i:")) != -1) {
    switch (opt) {
      case 'n':
        num_tokens = atoi(optarg);
        break;
      case 'h':
        hidden = atoi(optarg);
        break;
      case 'k':
        num_topk = atoi(optarg);
        break;
      case 'e':
        num_experts = atoi(optarg);
        break;
      case 'i':
        num_iterations = atoi(optarg);
        break;
      case '?':
        if (optopt == 'n' || optopt == 'h' || optopt == 'k' ||
            optopt == 'e' || optopt == 'i') {
          std::cerr << "Option -" << static_cast<char>(optopt)
                    << " requires an argument." << std::endl;
        } else {
          std::cerr << "Unknown option -"
                  << static_cast<char>(optopt) << std::endl;
        }
        std::cerr << "Usage: " << argv[0]
                  << " [-n num_tokens] [-h hidden] [-k num_topk]"
                  << " [-e num_experts] [-i num_iterations]"
                  << std::endl;
        return EXIT_FAILURE;
    }
  }

  LLDeepEP<dtype> ll_deepep(num_tokens, hidden, num_topk, num_experts);

  rank = ll_deepep.get_rank();
  num_ranks = ll_deepep.get_num_ranks();

  std::cout << "rank: " << rank << ", n_RANKs: " << num_ranks << std::endl;

  // Run dispatch and combine multiple times
  for (int iter = 0; iter < num_iterations; iter++) {
      ll_deepep.ll_dispatch();
      ll_deepep.ll_combine();
  }

  return 0;
}