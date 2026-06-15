#pragma once
#include <vector>
#include <mpi.h>
#include "message.h"
#include "inbox.h"
#include "vertex_map.h"
#include "statistics.h"

// Forward declaration — full definition in partition.h.
// AggregationTrace is owned by Partition; MPIExchange writes into it during
// wait_and_unpack() via a raw pointer. No ownership is transferred.
struct AggregationTrace;

// Communication abstraction between scatter and the next gather.
//
// Phase C3 (real MPI transport — current):
//   exchange() fires MPI_Isend / MPI_Irecv for remote_pending_.
//   wait_and_unpack() calls MPI_Waitall, deserialises received payloads,
//   translates global dst → local dst, and writes them before flipping.
//
// Wire format per message (stride = 2 + hidden_dim floats):
//   [0]      : src  (int32_t bits stored in float slot via memcpy)
//   [1]      : dst  (int32_t bits stored in float slot via memcpy)
//   [2..2+H) : payload floats
//
// world_size == 1: exchange() is a no-op; only local_pending_ is delivered.
//
// Invariant I-01: inbox.write() and inbox.flip() are called only inside
//   wait_and_unpack(). No caller outside this class may touch inbox directly.

class MPIExchange {
public:
    // Must be called once before any enqueue().
    // vm:         copy of the VertexMap for the whole graph (immutable after init).
    // rank:       this process's MPI rank.
    // world_size: MPI_Comm_size of MPI_COMM_WORLD.
    // hidden_dim: payload width; used to compute the wire stride.
    void init(VertexMap vm, int32_t rank, int32_t world_size, int32_t hidden_dim);

    // Classify and stage a message.
    // m.dst must be a GLOBAL vertex ID.
    void enqueue(const Message& m);

    // Initiate communication.
    // world_size==1: no-op.
    // world_size> 1: MPI_Alltoall to exchange per-rank message counts,
    //                MPI_Irecv for each source rank that will send to us,
    //                MPI_Isend for each dest rank we are sending to.
    void exchange();

    // Block until all transfers complete, then deliver all pending messages
    // (remote received + local staged) to inbox and flip the double buffer.
    // Owns the only call sites for inbox.write() and inbox.flip() (I-01).
    //
    // E12: If trace != nullptr, captures remote contributor IDs into
    //   trace->remote_contributors before recv_bufs_ is freed.
    //   The capture window is: after deserialize() and before recv_bufs_.clear().
    //   After recv_bufs_.clear() remote src global IDs are permanently gone.
    //   Passing nullptr (default) skips capture — used by seed() bootstrap.
    void wait_and_unpack(InboxBuffer& inbox, AggregationTrace* trace = nullptr);

    // Access statistics for reporting.
    const RuntimeStatistics& statistics() const { return stats_; }

    // Increment superstep counter (called by Runtime after each layer).
    void increment_superstep() { stats_.supersteps++; }

private:
    VertexMap vertex_map_;
    int32_t   rank_        = 0;
    int32_t   world_size_  = 1;
    int32_t   hidden_dim_  = 0;
    bool      initialized_ = false;

    RuntimeStatistics stats_;  // Phase D: track message/byte counts

    std::vector<Message> local_pending_;
    std::vector<Message> remote_pending_;

    // State that lives between exchange() and wait_and_unpack().
    std::vector<std::vector<float>> send_bufs_;   // [world_size]; freed after Waitall
    std::vector<std::vector<float>> recv_bufs_;   // [world_size]; freed after delivery
    std::vector<MPI_Request>        requests_;

    // Wire stride in floats: 2 header slots + payload.
    int32_t stride() const { return 2 + hidden_dim_; }

    void    serialize  (const Message& m, float* buf) const;
    Message deserialize(const float*   buf)            const;
};