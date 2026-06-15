#pragma once
#include <cstdint>
#include <vector>

// Maps global vertex IDs to (rank, local_id) using a range partition.
// 
// E9: Supports two partitioning strategies:
//   1. Vertex-balanced: Equal vertex counts per rank (legacy)
//   2. Edge-balanced:   Equal edge counts per rank (new)
//
// Range partition formula (vertex-balanced):
//   range_start[r] = (r * num_global) / world_size   (integer division)
//
// Edge-balanced formula:
//   Assign consecutive vertices to ranks until each rank accumulates
//   approximately target_edges_per_rank = total_edges / world_size.
//
// Both strategies maintain:
//   - Deterministic partitioning
//   - Contiguous ownership ranges
//   - O(log world_size) lookup complexity
//
// Immutable after init().

struct VertexMap {
    int32_t              num_global  = 0;
    int32_t              world_size  = 1;
    std::vector<int32_t> range_start;  // size: world_size + 1

    // E9: Initialize with vertex-balanced partitioning (legacy).
    void init(int32_t n_global, int32_t n_ranks);

    // E9: Initialize with edge-balanced partitioning.
    // Requires degree information for all vertices.
    // degrees[v] = out-degree of global vertex v (size: n_global).
    void init_edge_balanced(int32_t n_global, int32_t n_ranks,
                           const std::vector<int32_t>& degrees);

    // E9.1: Initialize with hybrid cost-based partitioning.
    // cost(v) = alpha * 1 + beta * degree(v)
    // Balances combined vertex and edge overhead across ranks.
    void init_hybrid_balanced(int32_t n_global, int32_t n_ranks,
                             const std::vector<int32_t>& degrees,
                             float alpha, float beta);

    // Which rank owns this global vertex ID?
    int32_t owner_rank(int32_t global_id) const;

    // Global vertex ID → local index on its owner rank.
    int32_t to_local(int32_t global_id) const;

    // Local index on a given rank → global vertex ID.
    int32_t to_global(int32_t local_id, int32_t rank) const;

    // Number of vertices owned by rank r.
    int32_t local_count(int32_t rank) const;
};
