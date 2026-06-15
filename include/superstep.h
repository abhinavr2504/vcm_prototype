#pragma once
#include <cstdint>
#include <vector>
#include "partition.h"
#include "compute.h"
#include "timing.h"
#include "gnn_layer.h"

// Executes one GNN superstep (one layer) on a Partition.
//
// Execution model: true Pregel message-passing.
//   Vertices NEVER directly read neighbour hidden states.
//   All aggregation is over inbox message payloads.
//
// Pipeline per superstep:
//   gather (inbox payloads) → aggregate → compute → scatter (hnext payloads)
//   → serial merge (outbox → inbox.write) → inbox.flip
//   → build next frontier from inbox state
//
// Phase 1: single thread, no OpenMP, no MPI.
// Phase 2: replace each per-vertex loop body with #pragma omp parallel for;
//          allocate per-thread OutboxBuffer; serial merge OR-folds them.

struct SuperstepExecutor {
    ComputeEngine compute;

    // Bootstrap scatter: reads hidden_curr of all frontier vertices, emits
    // messages to out-neighbours, merges into inbox, flips, and rebuilds
    // frontier_curr from message-arrival semantics.
    // Must be called once by Runtime::run() before the layer loop.
    // timing: optional pointer for Phase D timing instrumentation.
    void seed(Partition& p, RuntimeTiming* timing = nullptr);

    // Phase E2: accepts GNNLayer reference; W() and b() extracted internally.
    // layer_params: parameter owner for this superstep's layer.
    // layer: superstep index passed through for Phase A tracing (-1 = none).
    // timing: optional pointer for Phase D timing instrumentation.
    void run(Partition&      p,
             const GNNLayer& layer_params,
             int32_t         layer   = -1,
             RuntimeTiming*  timing  = nullptr);
};