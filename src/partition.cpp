#include "partition.h"
#include <algorithm>
#include <utility>

using namespace std;
/*Allocate and initialize all runtime state
for one MPI rank.(per-rank)*/

/*partition.cpp allocates and manages all per-rank runtime state—graph ownership, communication buffers, frontiers, and the three core GNN tensors (hidden_curr, hidden_next, and aggr_buf) that every forward and backward phase operates on*/
void Partition::init(int32_t n_local, int32_t h_dim, int32_t n_threads,
                     CSRGraph g, VertexMap vm, int32_t r, int32_t world_size) {
    num_vertices = n_local;
    hidden_dim   = h_dim;
    num_threads  = n_threads;
    rank         = r;
    graph        = std::move(g);// move CSRgraph into partition
    vertex_map   = std::move(vm); 

    vertex_state.assign(n_local, VertexState{});

    // Phase E1: Initialize tensors with zeros.
    // Shape: [num_vertices, hidden_dim]
    // Storage: contiguous, CPU, float32
    hidden_curr = torch::zeros({n_local, h_dim}, torch::kFloat32);
    hidden_next = torch::zeros({n_local, h_dim}, torch::kFloat32);
    aggr_buf    = torch::zeros({n_local, h_dim}, torch::kFloat32);

    frontier_curr.init(n_local);
    frontier_next.init(n_local);

    inbox.init(n_local);
    thread_outboxes.resize(num_threads);

    // Give MPIExchange its own copy of the VertexMap, this rank's ID,
    // the total world size, and hidden_dim (needed to compute the wire stride).
    mpi_exchange.init(vertex_map, r, world_size, h_dim);
}

float* Partition::hcurr(int32_t v) {
    return hidden_curr.data_ptr<float>() + v * hidden_dim;
}

const float* Partition::hcurr(int32_t v) const {
    return hidden_curr.data_ptr<float>() + v * hidden_dim;
}

float* Partition::hnext(int32_t v) {
    return hidden_next.data_ptr<float>() + v * hidden_dim;
}

const float* Partition::hnext(int32_t v) const {
    return hidden_next.data_ptr<float>() + v * hidden_dim;
}

float* Partition::vaggr(int32_t v) {
    return aggr_buf.data_ptr<float>() + v * hidden_dim;
}

void Partition::advance_hidden() {
    std::swap(hidden_curr, hidden_next);
}

void Partition::advance_frontier() {
    frontier_curr.swap(frontier_next);
    frontier_next.clear();
}