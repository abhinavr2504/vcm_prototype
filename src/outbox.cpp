#include "outbox.h"

void OutboxBuffer::push(const Message& m) {
    messages.push_back(m);
}

void OutboxBuffer::clear() {
    messages.clear();
}
