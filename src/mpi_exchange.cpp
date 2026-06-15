#include "mpi_exchange.h"
#include "partition.h"
#include "assertions.h"
#include <cstring>
#include <iostream>
#include <utility>

using namespace std;

void MPIExchange::init(VertexMap vm, int32_t rank,
                       int32_t world_size, int32_t hidden_dim) {
    vertex_map_  = move(vm);
    rank_        = rank;
    world_size_  = world_size;
    hidden_dim_  = hidden_dim;
    initialized_ = true;
}

void MPIExchange::serialize(const Message& m, float* buf) const {
    ASSERT(static_cast<int32_t>(m.payload.size()) == hidden_dim_,
           "serialize: payload size mismatch");

    memcpy(buf + 0, &m.src, sizeof(int32_t));
    memcpy(buf + 1, &m.dst, sizeof(int32_t));
    memcpy(buf + 2, m.payload.data(), hidden_dim_ * sizeof(float));
}

Message MPIExchange::deserialize(const float* buf) const {
    Message m;
    memcpy(&m.src, buf + 0, sizeof(int32_t));
    memcpy(&m.dst, buf + 1, sizeof(int32_t));
    m.payload.resize(hidden_dim_);
    memcpy(m.payload.data(), buf + 2, hidden_dim_ * sizeof(float));
    return m;
}

void MPIExchange::enqueue(const Message& m) {
    const int32_t owner = vertex_map_.owner_rank(m.dst);
    ASSERT(owner >= 0 && owner < world_size_,
           "enqueue: invalid owner_rank for dst");

    if (owner == rank_)
        local_pending_.push_back(m);
    else
        remote_pending_.push_back(m);
}

void MPIExchange::exchange() {
    if (world_size_ == 1) return;

    const int32_t S = stride();

    // Count how many messages go to each rank.
    vector<int32_t> send_counts(world_size_, 0);
    for (const Message& m : remote_pending_)
        send_counts[vertex_map_.owner_rank(m.dst)]++;

    // Learn how many messages each rank is sending us.
    vector<int32_t> recv_counts(world_size_, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    stats_.remote_messages_sent += static_cast<int64_t>(remote_pending_.size());

    // Pack outgoing messages into per-rank send buffers.
    send_bufs_.assign(world_size_, {});
    for (int32_t r = 0; r < world_size_; ++r)
        if (send_counts[r] > 0)
            send_bufs_[r].resize(send_counts[r] * S);

    vector<int32_t> pos(world_size_, 0);
    for (const Message& m : remote_pending_) {
        const int32_t r = vertex_map_.owner_rank(m.dst);
        serialize(m, send_bufs_[r].data() + pos[r] * S);
        pos[r]++;
    }

    for (int32_t r = 0; r < world_size_; ++r)
        if (r != rank_ && send_counts[r] > 0)
            stats_.bytes_sent += static_cast<int64_t>(send_counts[r] * S * sizeof(float));

    remote_pending_.clear();

    // Allocate receive buffers.
    recv_bufs_.assign(world_size_, {});
    for (int32_t r = 0; r < world_size_; ++r)
        if (recv_counts[r] > 0)
            recv_bufs_[r].resize(recv_counts[r] * S);

    for (int32_t r = 0; r < world_size_; ++r)
        if (r != rank_ && recv_counts[r] > 0)
            stats_.bytes_received += static_cast<int64_t>(recv_counts[r] * S * sizeof(float));

    // Post non-blocking receives first, then sends.
    requests_.clear();
    for (int32_t r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;
        if (recv_counts[r] > 0) {
            MPI_Request req;
            MPI_Irecv(recv_bufs_[r].data(), recv_counts[r] * S,
                      MPI_FLOAT, r, 0, MPI_COMM_WORLD, &req);
            requests_.push_back(req);
        }
    }
    for (int32_t r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;
        if (send_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_bufs_[r].data(), send_counts[r] * S,
                      MPI_FLOAT, r, 0, MPI_COMM_WORLD, &req);
            requests_.push_back(req);
        }
    }
}

void MPIExchange::wait_and_unpack(InboxBuffer& inbox, AggregationTrace* trace) {
    // Wait for all in-flight sends and receives to complete.
    if (!requests_.empty()) {
        MPI_Waitall(static_cast<int>(requests_.size()),
                    requests_.data(), MPI_STATUSES_IGNORE);
        requests_.clear();
    }

    const int32_t S = stride();
    int64_t remote_received = 0;

    // Unpack remote messages into the inbox.
    for (int32_t r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;
        const int32_t n = static_cast<int32_t>(recv_bufs_[r].size()) / S;
        remote_received += n;

        for (int32_t i = 0; i < n; ++i) {
            Message m = deserialize(recv_bufs_[r].data() + i * S);

            const int32_t local_dst = vertex_map_.to_local(m.dst);
            ASSERT(local_dst >= 0 && local_dst < vertex_map_.local_count(rank_),
                   "wait_and_unpack: invalid global->local translation");

            // E12: Capture the remote sender ID before recv_bufs_ is cleared.
            // This is the only point where the global source ID is available.
            // Local contributors are recoverable from ReverseCSR + active_mask
            // and are therefore not captured here.
            if (trace != nullptr) {
                trace->remote_contributors[local_dst].push_back(m.src);
            }

            m.dst = local_dst;
            inbox.write(m);
        }
    }

    stats_.remote_messages_received += remote_received;

    // Free receive buffers — global source IDs are now gone.
    recv_bufs_.clear();
    send_bufs_.clear();

    // Route local messages (same-rank) directly into the inbox.
    const int64_t local_count = static_cast<int64_t>(local_pending_.size());
    stats_.local_messages_sent     += local_count;
    stats_.local_messages_received += local_count;

    for (Message& m : local_pending_) {
        const int32_t local_dst = vertex_map_.to_local(m.dst);
        ASSERT(local_dst >= 0 && local_dst < vertex_map_.local_count(rank_),
               "wait_and_unpack: invalid local message dst translation");
        m.dst = local_dst;
        inbox.write(m);
    }
    local_pending_.clear();

    inbox.flip();

    // Mark the trace fully populated after all messages (local and remote) are processed.
    if (trace != nullptr) {
        trace->valid = true;
    }
}