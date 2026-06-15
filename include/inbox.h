#pragma once
#include <cstdint>
#include <vector>
#include "message.h"

// Double-buffered inbox: write_idx buffer receives scatter messages;
// read_idx buffer is read in gather for remote messages (Phase 2 path).
//
// flip() ownership:
//   Phase 1 — called by SuperstepExecutor after serial merge.
//   Phase 2 — move flip() private; MPIExchange::wait_and_unpack calls it as
//              the final internal step. No public flip() should exist in Phase 2.

struct InboxBuffer {
    int32_t num_vertices = 0;
    std::vector<std::vector<Message>> bufs[2];  // bufs[0], bufs[1]
    int32_t read_idx  = 0;
    int32_t write_idx = 1;

    void init(int32_t n);

    void write(const Message& m);                          // serial phase only
    bool has(int32_t v)                            const;  // check read buffer
    const std::vector<Message>& get(int32_t v)     const;  // read buffer slot

    void flip();         // swaps read/write; clears new write buffer
    void clear_write();  // zero the write buffer
};
