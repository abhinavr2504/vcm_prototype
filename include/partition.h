#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "graph.h"
#include "vertex_state.h"
#include "frontier.h"
#include "inbox.h"
#include "outbox.h"
#include "vertex_map.h"
#include "mpi_exchange.h"
#include "config.h"

// E11: Forward-state cache for multi-layer backpropagation.
// One entry per executed layer, stored as partition.layer_cache[L].
// All tensors are clones (immutable snapshots) — not aliases.

// E12: Aggregation semantics descriptor — stored inside LayerCache.
// Captures the aggregation type used at this layer. SUM is current default.
// argmax_source is populated only for MAX aggregation; empty for SUM/MEAN.
struct AggregationMetadata {
    AggregationType type = AggregationType::SUM;

    // argmax_source[v] = global vertex ID of the source whose message produced
    // the maximum value at destination v during Gather.
    // Defined ONLY when type == MAX. Empty (undefined) tensor for SUM and MEAN.
    torch::Tensor argmax_source;  // shape: [N], int32; undefined unless MAX
};

struct LayerCache {
    torch::Tensor       aggregated;       // Aggregation result after Gather     [N, H]
    torch::Tensor       pre_activation;   // Linear output before ReLU (W@a+b)  [N, H]
    torch::Tensor       activated;        // Snapshot of hidden_next after ReLU  [N, H]
    torch::Tensor       active_mask;      // Active vertices at this layer (bool) [N]
    AggregationMetadata aggr_meta;        // E12: aggregation type + argmax src
};

// E12: Local reverse adjacency — built once at load time, immutable thereafter.
//
// For each local vertex v, gives the global src IDs of LOCAL in-neighbors
// (i.e., src vertices that live on THIS rank and have an edge pointing to v).
//
// row_ptr indexed by local dst ID. col_idx stores GLOBAL src IDs so that
// E14 gradient routing can derive the sender rank via vertex_map.owner_rank().
//
// Remote in-neighbors (src on another rank) are NOT here — they are captured
// per-layer at runtime in AggregationTrace.remote_contributors.
//
// Memory cost (OGBN-Products, 4 ranks, ~30% local edge fraction):
//   row_ptr: (75,001) × 4 bytes  ≈  300 KB
//   col_idx:  555,750 edges × 4  ≈  2.22 MB
//   Total:                       ≈  2.52 MB per rank
struct ReverseCSR {
    std::vector<int32_t> row_ptr;   // size: num_local_vertices + 1
    std::vector<int32_t> col_idx;   // global src IDs of local in-neighbors

    bool built() const { return !row_ptr.empty(); }

    // Number of local in-neighbors for local vertex v.
    int32_t in_degree(int32_t local_v) const {
        return row_ptr[local_v + 1] - row_ptr[local_v];
    }
};

// E12: Per-layer runtime dependency trace.
//
// Stores ONLY information that is permanently unavailable after Gather:
//   → The global IDs of REMOTE vertices (other ranks) that sent embeddings
//     to each local vertex during one layer's MPI exchange.
//
// NOT stored (all recoverable):
//   Local in-neighbors    → ReverseCSR + LayerCache.active_mask
//   Contribution count    → remote_contributors[v].size() + local active in-degree
//   Aggregation type      → compile-time constant (AggregationType in config.h)
//
// Future consumers:
//   E14 (Gradient Message Transport):
//     remote_contributors[v] → know which global src vertices receive gradient msgs
//   E15 (Distributed Backward Propagation):
//     traverse reversed dependency structure across ranks
//
// Memory cost (OGBN-Products, 4 ranks, ~70% remote fraction, 75K local verts):
//   vector metadata:  75,000 × 24 bytes        ≈  1.80 MB per layer
//   remote IDs:       75,000 × 17.3 × 4 bytes  ≈  5.19 MB per layer
//   Total per layer:                            ≈  6.99 MB per layer
//   For 2 layers:                               ≈ 14.00 MB per rank
struct AggregationTrace {
    // remote_contributors[local_v] = global vertex IDs from OTHER ranks that
    // sent embedding messages to local vertex v during this layer's Gather.
    //
    // Captured inside MPIExchange::wait_and_unpack() BEFORE recv_bufs_ is
    // freed and BEFORE inbox.flip() destroys the old inbox buffer.
    // This is the only window where remote sender identities are accessible.
    std::vector<std::vector<int32_t>> remote_contributors;  // [num_local][*]

    // Set to true inside wait_and_unpack() after population.
    // Always check valid before reading remote_contributors.
    bool valid = false;
};

// Owns the complete local state for one rank.
//
// num_vertices is the LOCAL vertex count for this rank.
// All buffers (hidden_curr, hidden_next, aggr_buf, vertex_state, inbox,
// frontier) are sized by num_vertices (local count), not the global total.
//
// CSRGraph::row_ptr is indexed by local vertex ID (0..num_vertices-1).
// CSRGraph::col_idx stores GLOBAL destination vertex IDs; MPIExchange uses
// owner_rank(dst) to classify each message as local or remote.
//
// Invariant I-05: hidden_curr is read-only during a superstep.
//                 hidden_next is write-only during a superstep.
//                 advance_hidden() swaps them at the superstep boundary only.
//
// Phase E1: hidden_curr, hidden_next, aggr_buf are now torch::Tensor.
//           Underlying storage remains contiguous float arrays.
//           Accessor methods (hcurr, hnext, vaggr) return raw float* for
//           compatibility with message serialization and existing scatter/gather code.

struct Partition {
    int32_t num_vertices = 0;  // local vertex count for this rank
    int32_t hidden_dim   = 0;
    int32_t rank         = 0;  // MPI rank that owns this partition

    VertexMap vertex_map;      // global↔local ID translation; immutable after init()

    CSRGraph               graph;
    std::vector<VertexState> vertex_state;  // indexed by local vertex ID

    // Phase E1: Tensor-backed embedding buffers.
    // Shape: [num_vertices, hidden_dim]
    // Storage: contiguous, CPU, float32
    torch::Tensor hidden_curr;
    torch::Tensor hidden_next;
    torch::Tensor aggr_buf;

    // Phase E3: snapshots of aggregation buffers and active masks per layer.
    std::vector<torch::Tensor> aggr_snapshots;
    std::vector<torch::Tensor> aggr_masks;

    // E11: Forward-state cache — one LayerCache per executed layer.
    std::vector<LayerCache> layer_cache;

    // E12: Local reverse adjacency for backward traversal.
    // Built once in loader.cpp immediately after the forward CSR is constructed.
    // Encodes only local-to-local edges (same-rank src → same-rank dst).
    // Remote in-neighbors are captured per-layer in aggr_traces below.
    ReverseCSR reverse_graph;

    // E12: Per-layer dependency traces capturing remote contributor IDs.
    // One AggregationTrace per executed layer, populated during forward pass.
    // Cleared and rebuilt from scratch each time Runtime::run() is called.
    std::vector<AggregationTrace> aggr_traces;

    FrontierSet frontier_curr;
    FrontierSet frontier_next;

    InboxBuffer  inbox;
    int32_t      num_threads = 1;
    std::vector<OutboxBuffer> thread_outboxes;  // [num_threads]; one per OpenMP thread

    // Communication abstraction.  Routes local messages directly to inbox;
    // transports remote messages via MPI (Phase C3).
    MPIExchange mpi_exchange;

    // n_local:     local vertex count (== vertex_map.local_count(r))
    // vm:          fully initialised VertexMap for the whole graph
    // r:           this rank's MPI rank
    // world_size:  total number of MPI ranks
    void init(int32_t n_local, int32_t h_dim, int32_t n_threads,
              CSRGraph g, VertexMap vm, int32_t r, int32_t world_size);

    // Phase E1: Accessor methods return raw float* for compatibility with
    // message serialization and existing scatter/gather code.
    float*       hcurr(int32_t v);
    const float* hcurr(int32_t v) const;
    float*       hnext(int32_t v);
    const float* hnext(int32_t v) const;
    float*       vaggr(int32_t v);

    void advance_hidden();    // swap hidden_curr <-> hidden_next
    void advance_frontier();  // frontier_curr <- frontier_next; clear frontier_next
};