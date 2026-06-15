#include "inbox.h"
#include <utility>

using namespace std;

void InboxBuffer::init(int32_t n) {
    num_vertices = n;
    bufs[0].assign(n, {});
    bufs[1].assign(n, {});
    read_idx  = 0;
    write_idx = 1;
}

void InboxBuffer::write(const Message& m) {
    bufs[write_idx][m.dst].push_back(m);
}

bool InboxBuffer::has(int32_t v) const {
    return !bufs[read_idx][v].empty();
}

const vector<Message>& InboxBuffer::get(int32_t v) const {
    return bufs[read_idx][v];
}

void InboxBuffer::clear_write() {
    for (auto& slot : bufs[write_idx]) slot.clear();
}

void InboxBuffer::flip() {
    std::swap(read_idx, write_idx);
    clear_write();  // prepare new write buffer for next superstep
}
