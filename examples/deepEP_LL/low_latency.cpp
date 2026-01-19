#include "LL_DeepEP.hpp"

using dtype = short;

int main (int argc, char **argv)
{
    int rank, num_ranks;

    // DeepEP_LL parameters
    // TODO: These parameters should be configurable via command-line arguments
    int num_tokens = 3;//128;
    int hidden = 8;//7168;
    int num_topk = 8;//8;
    int num_experts = 288;//288;

    LLDeepEP<dtype> ll_deepep(num_tokens, hidden, num_topk, num_experts);

    rank = ll_deepep.get_rank();
    num_ranks = ll_deepep.get_num_ranks();

    std::cout << "rank: " << rank << ", n_RANKs: " << num_ranks << std::endl;


    ll_deepep.ll_dispatch();

    return 0;
}