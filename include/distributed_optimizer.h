#pragma once
#include <vector>
#include <mpi.h>
#include <torch/torch.h>

// E8 — Distributed Gradient Synchronization
//
// Averages gradients across all MPI ranks in `comm` so that every rank
// applies an identical parameter update after backward().
//
// Required call order per training step:
//
//   optimizer.zero_grad();
//   loss = compute_loss();
//   loss.backward();
//   DistributedOptimizer::average_gradients(model.parameters());  // <-- E8
//   optimizer.step();
//
// Constraints:
//   - FP32 gradients only.
//   - Gradients are averaged (SUM / world_size), not parameters.
//   - optimizer.step() is NOT called here — caller owns that.
//   - No-op when world_size == 1 (single-rank behavior unchanged).

class DistributedOptimizer {
public:
    // Average gradients of every param that has a defined gradient.
    // Params without a gradient (e.g. frozen layers) are skipped silently.
    static void average_gradients(
        const std::vector<torch::Tensor>& params,
        MPI_Comm comm = MPI_COMM_WORLD
    );
};
