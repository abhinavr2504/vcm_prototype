#pragma once
#include <cstdint>
#include <mpi.h>

// Per-rank runtime statistics for Phase D observability.
// Tracks message counts, byte counts, and superstep counts.
// Minimal overhead: simple integer counters incremented inline.
struct RuntimeStatistics {
    int64_t local_messages_sent     = 0;
    int64_t remote_messages_sent    = 0;
    int64_t local_messages_received = 0;
    int64_t remote_messages_received = 0;
    int64_t bytes_sent              = 0;
    int64_t bytes_received          = 0;
    int32_t supersteps              = 0;

    // Compute total messages sent/received.
    int64_t total_messages_sent() const {
        return local_messages_sent + remote_messages_sent;
    }

    int64_t total_messages_received() const {
        return local_messages_received + remote_messages_received;
    }

    // Compute remote message fraction (0.0 if no messages sent).
    double remote_fraction() const {
        const int64_t total = total_messages_sent();
        return (total > 0) ? static_cast<double>(remote_messages_sent) / total : 0.0;
    }
};

// Global statistics aggregated across all ranks.
struct GlobalStatistics {
    int64_t total_vertices        = 0;
    int64_t total_edges           = 0;
    int64_t total_messages        = 0;
    int64_t total_remote_messages = 0;
    int64_t total_bytes_sent      = 0;
    int64_t total_bytes_received  = 0;

    // Load balance metrics.
    int32_t min_vertices = 0;
    int32_t max_vertices = 0;
    double  avg_vertices = 0.0;
    int32_t min_edges    = 0;
    int32_t max_edges    = 0;
    double  avg_edges    = 0.0;

    // Imbalance ratio: max / avg (1.0 = perfect balance).
    double vertex_imbalance() const {
        return (avg_vertices > 0.0) ? max_vertices / avg_vertices : 1.0;
    }

    double edge_imbalance() const {
        return (avg_edges > 0.0) ? max_edges / avg_edges : 1.0;
    }

    // Remote message fraction across all ranks.
    double remote_fraction() const {
        return (total_messages > 0)
                   ? static_cast<double>(total_remote_messages) / total_messages
                   : 0.0;
    }
};

// Collect global statistics from all ranks using MPI collectives.
// local_vertices: this rank's vertex count.
// local_edges:    this rank's edge count.
// stats:          this rank's RuntimeStatistics.
// Returns GlobalStatistics aggregated across all ranks.
inline GlobalStatistics collect_global_statistics(
    int32_t local_vertices,
    int32_t local_edges,
    const RuntimeStatistics& stats,
    int32_t world_size)
{
    GlobalStatistics global;

    // Aggregate totals.
    int64_t local_total_messages = stats.total_messages_sent();
    MPI_Allreduce(&local_vertices,      &global.total_vertices, 1,
                  MPI_INT,   MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_edges,         &global.total_edges,    1,
                  MPI_INT,   MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_total_messages, &global.total_messages, 1,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stats.remote_messages_sent, &global.total_remote_messages, 1,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stats.bytes_sent,     &global.total_bytes_sent,     1,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stats.bytes_received, &global.total_bytes_received, 1,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Load balance: min/max/avg for vertices and edges.
    MPI_Allreduce(&local_vertices, &global.min_vertices, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_vertices, &global.max_vertices, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_edges,    &global.min_edges,    1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_edges,    &global.max_edges,    1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    global.avg_vertices = static_cast<double>(global.total_vertices) / world_size;
    global.avg_edges    = static_cast<double>(global.total_edges)    / world_size;

    return global;
}
