#pragma once
#include <cstdint>
#include <vector>

// Pregel-style message: carries an embedding payload from src to dst.
// payload.size() == hidden_dim at all times.
//
// E12: src_rank is the MPI rank that originated this message.
//   Set by MPIExchange::wait_and_unpack() — NOT serialized on the wire.
//   -1 = unset (default, safe to ignore in single-rank runs).
//   Used in SuperstepExecutor::run() Gather phase to identify remote
//   contributors for AggregationTrace capture.
struct Message {
    int32_t            src;
    int32_t            dst;
    std::vector<float> payload;  // [hidden_dim]
    int32_t            src_rank = -1;  // E12: sender rank, not serialized
};
