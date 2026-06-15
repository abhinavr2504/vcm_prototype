#pragma once
#include <cstdint>
#include <vector>

struct CSRGraph {
    int32_t num_vertices = 0;
    int32_t num_edges    = 0;
    std::vector<int32_t> row_ptr;  // size: num_vertices + 1
    std::vector<int32_t> col_idx;  // size: num_edges
};
