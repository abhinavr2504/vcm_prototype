#pragma once
#include <cstdint>
#include <vector>

struct FrontierSet {
    int32_t num_vertices = 0;
    std::vector<uint8_t> bits;  // one byte per vertex

    void    init(int32_t n);
    void    set(int32_t v);           // single-caller: serial phase only
    bool    test(int32_t v) const;    // read-only; safe under parallel reads
    int32_t popcount()      const;
    void    swap(FrontierSet& other); // superstep boundary; single-threaded
    void    clear();
};
