#pragma once
#include <vector>
#include "message.h"

// Thread-local staging buffer written during scatter.
// Merged into InboxBuffer (local) or packed for MPI (remote) in serial phase.
// Phase 1: one OutboxBuffer per Partition (single thread).
// Phase 2: one per OpenMP thread; SuperstepExecutor passes array to pack_and_send.

struct OutboxBuffer {
    std::vector<Message> messages;

    void push(const Message& m);
    void clear();
};
