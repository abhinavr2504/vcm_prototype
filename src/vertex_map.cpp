#include "vertex_map.h"
#include "assertions.h"
#include <algorithm>

using namespace std;

void VertexMap::init(int32_t n_global, int32_t n_ranks) {
    num_global = n_global;
    world_size = n_ranks;
    range_start.resize(n_ranks + 1);
    for (int32_t r = 0; r <= n_ranks; ++r)
        range_start[r] = static_cast<int32_t>(
            (static_cast<int64_t>(r) * n_global) / n_ranks);
}

// E9: Edge-balanced partitioning.
// Assigns consecutive vertices to ranks to balance edge counts.
// Algorithm:
//   1. Compute target_edges_per_rank = total_edges / world_size
//   2. For each rank, accumulate vertices until edge count >= target
//   3. Remainder vertices go to the last rank
// Handles skewed degree distributions robustly.
void VertexMap::init_edge_balanced(int32_t n_global, int32_t n_ranks,
                                  const vector<int32_t>& degrees) {
    ASSERT(static_cast<int32_t>(degrees.size()) == n_global,
           "init_edge_balanced: degrees array size mismatch");
    
    num_global = n_global;
    world_size = n_ranks;
    range_start.resize(n_ranks + 1);
    
    // Handle edge case: empty graph or single rank.
    if (n_global == 0 || n_ranks == 1) {
        for (int32_t r = 0; r <= n_ranks; ++r)
            range_start[r] = (r == 0) ? 0 : n_global;
        
        ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
               "init_edge_balanced: range_start size mismatch");
        ASSERT(range_start[0] == 0,
               "init_edge_balanced: range_start[0] != 0");
        ASSERT(range_start[world_size] == num_global,
               "init_edge_balanced: range_start[world_size] != num_global");
        return;
    }
    
    // Compute total edges.
    int64_t total_edges = 0;
    for (int32_t v = 0; v < n_global; ++v)
        total_edges += degrees[v];
    
    // Handle edge case: zero edges (all isolated vertices).
    if (total_edges == 0) {
        // Fall back to vertex-balanced partitioning.
        for (int32_t r = 0; r <= n_ranks; ++r)
            range_start[r] = static_cast<int32_t>(
                (static_cast<int64_t>(r) * n_global) / n_ranks);
        
        ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
               "init_edge_balanced: range_start size mismatch (zero edges)");
        ASSERT(range_start[0] == 0,
               "init_edge_balanced: range_start[0] != 0 (zero edges)");
        ASSERT(range_start[world_size] == num_global,
               "init_edge_balanced: range_start[world_size] != num_global (zero edges)");
        return;
    }
    
    // Target edges per rank (round up to avoid leaving last rank overloaded).
    const int64_t target_edges = (total_edges + n_ranks - 1) / n_ranks;
    
    // Assign vertices to ranks.
    range_start[0] = 0;
    int32_t current_rank = 0;
    int64_t accumulated_edges = 0;
    
    for (int32_t v = 0; v < n_global; ++v) {
        accumulated_edges += degrees[v];
        
        // Close current rank if:
        //   1. We've reached the target edge count
        //   2. There are more ranks to fill
        //   3. We're not at the last vertex (avoid empty trailing ranks)
        if (accumulated_edges >= target_edges && 
            current_rank < n_ranks - 1 && 
            v < n_global - 1) {
            current_rank++;
            range_start[current_rank] = v + 1;
            accumulated_edges = 0;
        }
    }
    
    // Fill any remaining unfilled rank boundaries with n_global.
    // This handles cases where skewed degree distribution causes
    // some ranks to never reach the target edge count.
    for (int32_t r = current_rank + 1; r <= n_ranks; ++r) {
        range_start[r] = n_global;
    }
    
    // Validation: ensure all invariants hold.
    ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
           "init_edge_balanced: range_start size mismatch");
    ASSERT(range_start[0] == 0,
           "init_edge_balanced: range_start[0] != 0");
    ASSERT(range_start[world_size] == num_global,
           "init_edge_balanced: range_start[world_size] != num_global");
    
    // Ensure monotonic increasing boundaries.
    for (int32_t r = 0; r < n_ranks; ++r) {
        ASSERT(range_start[r] <= range_start[r + 1],
               "init_edge_balanced: range_start not monotonic");
    }
}

// E9.1: Hybrid cost-based partitioning.
// Balances: cost(v) = alpha * 1 + beta * degree(v)
// Algorithm:
//   1. Compute cost for each vertex
//   2. Compute target_cost_per_rank = total_cost / world_size
//   3. Assign consecutive vertices until accumulated cost >= target
//   4. Handles skewed distributions robustly
void VertexMap::init_hybrid_balanced(int32_t n_global, int32_t n_ranks,
                                    const vector<int32_t>& degrees,
                                    float alpha, float beta) {
    ASSERT(static_cast<int32_t>(degrees.size()) == n_global,
           "init_hybrid_balanced: degrees array size mismatch");
    ASSERT(alpha >= 0.0f && beta >= 0.0f,
           "init_hybrid_balanced: alpha and beta must be non-negative");
    
    num_global = n_global;
    world_size = n_ranks;
    range_start.resize(n_ranks + 1);
    
    // Handle edge case: empty graph or single rank.
    if (n_global == 0 || n_ranks == 1) {
        for (int32_t r = 0; r <= n_ranks; ++r)
            range_start[r] = (r == 0) ? 0 : n_global;
        
        ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
               "init_hybrid_balanced: range_start size mismatch");
        ASSERT(range_start[0] == 0,
               "init_hybrid_balanced: range_start[0] != 0");
        ASSERT(range_start[world_size] == num_global,
               "init_hybrid_balanced: range_start[world_size] != num_global");
        return;
    }
    
    // Compute total cost.
    double total_cost = 0.0;
    for (int32_t v = 0; v < n_global; ++v) {
        total_cost += alpha + beta * degrees[v];
    }
    
    // Handle edge case: zero cost (shouldn't happen with alpha > 0).
    if (total_cost <= 0.0) {
        // Fall back to vertex-balanced partitioning.
        for (int32_t r = 0; r <= n_ranks; ++r)
            range_start[r] = static_cast<int32_t>(
                (static_cast<int64_t>(r) * n_global) / n_ranks);
        
        ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
               "init_hybrid_balanced: range_start size mismatch (zero cost)");
        ASSERT(range_start[0] == 0,
               "init_hybrid_balanced: range_start[0] != 0 (zero cost)");
        ASSERT(range_start[world_size] == num_global,
               "init_hybrid_balanced: range_start[world_size] != num_global (zero cost)");
        return;
    }
    
    // Target cost per rank.
    const double target_cost = total_cost / n_ranks;
    
    // Assign vertices to ranks.
    range_start[0] = 0;
    int32_t current_rank = 0;
    double accumulated_cost = 0.0;
    
    for (int32_t v = 0; v < n_global; ++v) {
        accumulated_cost += alpha + beta * degrees[v];
        
        // Close current rank if:
        //   1. We've reached the target cost
        //   2. There are more ranks to fill
        //   3. We're not at the last vertex (avoid empty trailing ranks)
        if (accumulated_cost >= target_cost && 
            current_rank < n_ranks - 1 && 
            v < n_global - 1) {
            current_rank++;
            range_start[current_rank] = v + 1;
            accumulated_cost = 0.0;
        }
    }
    
    // Fill any remaining unfilled rank boundaries with n_global.
    for (int32_t r = current_rank + 1; r <= n_ranks; ++r) {
        range_start[r] = n_global;
    }
    
    // Validation: ensure all invariants hold.
    ASSERT(range_start.size() == static_cast<size_t>(world_size + 1),
           "init_hybrid_balanced: range_start size mismatch");
    ASSERT(range_start[0] == 0,
           "init_hybrid_balanced: range_start[0] != 0");
    ASSERT(range_start[world_size] == num_global,
           "init_hybrid_balanced: range_start[world_size] != num_global");
    
    // Ensure monotonic increasing boundaries.
    for (int32_t r = 0; r < n_ranks; ++r) {
        ASSERT(range_start[r] <= range_start[r + 1],
               "init_hybrid_balanced: range_start not monotonic");
    }
}

// Returns the largest r such that range_start[r] <= global_id.
int32_t VertexMap::owner_rank(int32_t global_id) const {
    ASSERT(global_id >= 0 && global_id < num_global,
           "owner_rank: global_id out of range");

    // upper_bound gives the first element strictly greater than global_id.
    // Stepping back one position yields the owning rank's start index.
    auto it = std::upper_bound(range_start.begin(), range_start.end(), global_id);
    const int32_t rank = static_cast<int32_t>(it - range_start.begin()) - 1;

    ASSERT(rank >= 0 && rank < world_size,
           "owner_rank: computed rank out of range");

    return rank;
}

int32_t VertexMap::to_local(int32_t global_id) const {
    ASSERT(global_id >= 0 && global_id < num_global,
           "to_local: global_id out of range");

    const int32_t rank = owner_rank(global_id);
    const int32_t local_id = global_id - range_start[rank];

    ASSERT(local_id >= 0 && local_id < local_count(rank),
           "to_local: computed local_id out of range");

    return local_id;
}

int32_t VertexMap::to_global(int32_t local_id, int32_t rank) const {
    ASSERT(rank >= 0 && rank < world_size,
           "to_global: rank out of range");
    ASSERT(local_id >= 0 && local_id < local_count(rank),
           "to_global: local_id out of range for rank");

    return range_start[rank] + local_id;
}

int32_t VertexMap::local_count(int32_t rank) const {
    ASSERT(rank >= 0 && rank < world_size,
           "local_count: rank out of range");

    return range_start[rank + 1] - range_start[rank];
}
