#pragma once
#include <cstdint>
#include <utility>
#include <torch/torch.h>
#include "partition.h"

// E13: Local Backward Reconstruction Engine.
//
// Reconstructs parameter gradients for layer (target_layer-1) using ONLY
// locally available dependency paths. No new MPI operations are introduced.
//
// ─── Gradient paths covered ──────────────────────────────────────────────────
//   local src u ──edge──► local dst v  (same-rank edges, in ReverseCSR)
//   grad_aggr[v] ──► grad_h_src[u] ──► grad_pre_src[u] ──► grad_W, grad_b
//
// ─── Gradient paths NOT covered (require E14) ─────────────────────────────
//   remote u → local v : grad_aggr[v] known here, but h0[u] is on another rank
//   local u → remote v : u is local, but grad_aggr[v] lives on v's owning rank
//
// ─── Completeness ─────────────────────────────────────────────────────────
//   world_size == 1 : complete and correct (all dependencies are local)
//   world_size >  1 : partial; E14 will add remote contributions to the same
//                     grad_W / grad_b computation via returned tensors
//
// ─── E14 contract ─────────────────────────────────────────────────────────
//   E14 must deliver (per local vertex u) the sum of grad_aggr[v] for remote v
//   that u contributed to during target_layer's Gather. Those contributions are
//   added BEFORE run() is called (or run() is extended to accept extra grad_h).
//   The returned tensors from run() are the accumulation targets.
//
// ─── Prerequisites ────────────────────────────────────────────────────────
//   partition.layer_cache populated by E11 (superstep.cpp)
//   partition.reverse_graph built by E12   (loader.cpp)
//   partition.aggr_traces populated by E12 (mpi_exchange.cpp) — informational only here
struct LocalBackwardEngine {
    // Full reconstruction (Phase A + Phase C combined).
    // Equivalent to scatter_only() followed immediately by linear_phase().
    // Use this for world_size==1 where no E14 exchange is needed.
    // For world_size>1, decompose into scatter_only + exchange_into + linear_phase.
    static std::pair<torch::Tensor, torch::Tensor> run(
        const Partition&     partition,
        int32_t              target_layer,
        const torch::Tensor& grad_aggr   // [N, H] float32
    );

    // Phase A — Local scatter only.
    // Accumulates grad_aggr[v] into grad_h[u] for each ReverseCSR local edge u→v.
    // Applies the active_mask filter on dst vertices.
    // Does NOT apply the ReLU gate. Does NOT compute grad_W or grad_b.
    //
    // Returns grad_h[N, H] — complete for local paths, partial for world_size>1.
    // Pass to GradientExchange::exchange_into() before calling linear_phase().
    static torch::Tensor scatter_only(
        const Partition&     partition,
        int32_t              target_layer,
        const torch::Tensor& grad_aggr   // [N, H] float32
    );

    // Phase C — Linear backward.
    // Applies ReLU gate using LayerCache[src_layer].pre_activation,
    // then computes grad_W = grad_pre.T @ aggr and grad_b = grad_pre.sum(0).
    //
    // Call AFTER GradientExchange::exchange_into() has completed so that
    // grad_h contains both local and remote contributions.
    //
    // Returns: (grad_W[H, H], grad_b[H])
    static std::pair<torch::Tensor, torch::Tensor> linear_phase(
        const Partition&     partition,
        int32_t              src_layer,   // = target_layer - 1
        const torch::Tensor& grad_h       // [N, H] float32, COMPLETE
    );
};
