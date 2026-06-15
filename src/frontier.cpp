#include "frontier.h"
#include <algorithm>

using namespace std;

void FrontierSet::init(int32_t n) {
    num_vertices = n;
    bits.assign(n, 0);
}

void FrontierSet::set(int32_t v) {
    bits[v] = 1;
}

bool FrontierSet::test(int32_t v) const {
    return bits[v] != 0;
}

int32_t FrontierSet::popcount() const {
    int32_t count = 0;
    for (uint8_t b : bits) count += (b != 0);
    return count;
}

void FrontierSet::swap(FrontierSet& other) {
    bits.swap(other.bits);
    std::swap(num_vertices, other.num_vertices);
}

void FrontierSet::clear() {
    std::fill(bits.begin(), bits.end(), 0);
}
