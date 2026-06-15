#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "partition.h"

// E14: Gradient Message Transport — rank-batched streaming exchange.
//
// After LocalBackwardEngine::scatter_only() fills grad_h with local contributions,
// GradientExchange::exchange_into() delivers remote contributions by routing
// grad_aggr[v] to every remote source vertex u that contributed to v during Gather.
//
// Transport protocol:
//   For each remote rank r (sequential):
//     1. Metadata Sendrecv: exchange n_pairs and n_idx counts
//     2. Chunk loop (≤ chunk_pairs_ pairs per iteration):
//        a. Chunk-meta Sendrecv: [n_sp, n_si] ↔ [n_rp, n_ri]
//        b. Float Sendrecv:      grad_aggr[v] payloads (n_sp × H floats each way)
//        c. Int Sendrecv:        combined [counts(n_sp) | dsts(n_si)] ↔ [counts(n_rp) | dsts(n_ri)]
//        d. Unpack into grad_h in-place
//
// Peak memory: 2 × chunk_pairs_ × hidden_dim × 4 bytes
// At DEFAULT_CHUNK_PAIRS=200000 and H=256: ~410 MB peak.
// This is well below the ~10 GB peak of the naive edge-based design.
//
// world_size == 1: exchange_into() is an unconditional no-op.
//
// Does NOT modify MPIExchange, E8 DistributedOptimizer, or
// the forward Gather→Compute→Scatter runtime.
class GradientExchange {
public:
    // Default chunk size. Peak memory = 2 × DEFAULT_CHUNK_PAIRS × H × sizeof(float).
    // Can be overridden via RunConfig::e14_chunk_pairs at construction time.
    static constexpr int32_t DEFAULT_CHUNK_PAIRS = 200'000;

    // Construct from rank/world_size/chunk_pairs in RunConfig.
    explicit GradientExchange(int32_t rank, int32_t world_size,
                              int32_t chunk_pairs = DEFAULT_CHUNK_PAIRS);

    // Rank-batched gradient exchange.
    //
    // Reads:
    //   AggregationTrace[target_layer].remote_contributors[v]  — who sent to v
    //   grad_aggr[v] for each local v with remote contributors  — their gradient
    //
    // Writes (in-place):
    //   grad_h[u] += grad_aggr[v]  for each cross-rank edge u→v
    //
    // grad_h is the buffer returned by LocalBackwardEngine::scatter_only().
    // After this call, grad_h contains COMPLETE contributions (local + remote).
    //
    // All ranks must call this simultaneously (collective).
    // Ranks with no remote contributors still participate with empty sends.
    void exchange_into(
        const Partition&     partition,
        int32_t              target_layer,
        const torch::Tensor& grad_aggr,   // [N, H] float32, read-only
        torch::Tensor&       grad_h       // [N, H] float32, in-place accumulation
    );

private:
    int32_t rank_;
    int32_t world_size_;
    int32_t chunk_pairs_;

    // Reusable buffers pre-allocated to chunk_pairs_ × H capacity.
    // Grown lazily on first call if H > initial estimate.
    // Freed when GradientExchange is destroyed.
    std::vector<float>   send_float_;
    std::vector<float>   recv_float_;
    std::vector<int32_t> send_int_;   // combined: [counts(n_sp) | global_dsts(n_si)]
    std::vector<int32_t> recv_int_;   // combined: [counts(n_rp) | global_dsts(n_ri)]

    // Build the grouped index structure for rank r:
    //   for each local v with ≥1 remote contributor on rank r
    //     v_locals   [i]   = local vertex ID
    //     u_offsets  [i]   = start of u entries for v in u_globals (CSR-style)
    //     u_globals  [...]  = global IDs of all u's on r that sent to each v
    //
    // u_offsets has size n_pairs+1 (standard CSR sentinel).
    // Source: AggregationTrace[target_layer].remote_contributors.
    void build_rank_groups(
        const Partition&       partition,
        int32_t                target_layer,
        int32_t                r,
        std::vector<int32_t>&  v_locals,
        std::vector<int32_t>&  u_offsets,
        std::vector<int32_t>&  u_globals
    ) const;
};
