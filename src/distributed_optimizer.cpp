#include "distributed_optimizer.h"
#include <iostream>

using namespace std;

void DistributedOptimizer::average_gradients(
    const vector<torch::Tensor>& params,
    MPI_Comm comm)
{
    int world_size;
    MPI_Comm_size(comm, &world_size);

    // No-op for single-rank runs.
    if (world_size <= 1)
        return;

    // Guard: prevent autograd from tracking the gradient manipulation.
    torch::NoGradGuard no_grad;

    for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
        const auto& p = params[param_idx];

        // Skip parameters not involved in this backward pass.
        if (!p.grad().defined())
            continue;

        auto grad = p.grad();

        TORCH_CHECK(
            grad.scalar_type() == torch::kFloat,
            "DistributedOptimizer currently supports FP32 gradients only");

        // MPI_Allreduce needs a contiguous float buffer.
        // If grad is already contiguous, work aliases the same storage (no copy).
        // Otherwise, we work on a fresh copy and write back after averaging.
        const bool was_contiguous = grad.is_contiguous();
        torch::Tensor work = was_contiguous ? grad : grad.contiguous();

        // Sum gradients across all ranks, then divide by world_size to average.
        MPI_Allreduce(
            MPI_IN_PLACE,
            work.data_ptr<float>(),
            static_cast<int>(work.numel()),
            MPI_FLOAT,
            MPI_SUM,
            comm
        );

        work.div_(static_cast<float>(world_size));

        // Write averaged result back into the original tensor if it wasn't contiguous.
        if (!was_contiguous)
            grad.copy_(work);
    }
}
