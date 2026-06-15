#include "loader.h"
#include "partition.h"
#include "vertex_map.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

// E12: Build the local Reverse CSR for partition p.
//
// Scans the forward CSR (local src → global dst) and accumulates in-edges
// where the destination vertex is ALSO owned by this rank (same-rank edges).
// Cross-rank edges are deliberately excluded: their contributor IDs appear
// only during MPI exchange and are captured live in AggregationTrace.
//
// Two-pass algorithm (count then fill) to avoid sorting or extra memory.
// col_idx stores GLOBAL src IDs (not local) so E14 gradient routing can
// derive the sender rank via vertex_map.owner_rank() without ambiguity.
//
// Complexity: O(local_edges)   Memory: O(local_in_edge_count)
static void build_reverse_csr(Partition& p) {
    const int32_t N    = p.num_vertices;
    const int32_t rank = p.rank;

    // Pass 1: count local in-degree for each local dst vertex.
    vector<int32_t> in_deg(N, 0);
    for (int32_t lu = 0; lu < N; ++lu) {
        for (int32_t e = p.graph.row_ptr[lu]; e < p.graph.row_ptr[lu + 1]; ++e) {
            const int32_t gdst = p.graph.col_idx[e];
            if (p.vertex_map.owner_rank(gdst) == rank) {
                in_deg[p.vertex_map.to_local(gdst)]++;
            }
        }
    }

    // Build row_ptr from in-degree counts (prefix sum).
    p.reverse_graph.row_ptr.resize(N + 1, 0);
    for (int32_t v = 0; v < N; ++v)
        p.reverse_graph.row_ptr[v + 1] = p.reverse_graph.row_ptr[v] + in_deg[v];

    const int32_t total_in_edges = p.reverse_graph.row_ptr[N];
    p.reverse_graph.col_idx.resize(total_in_edges);

    // Pass 2: fill col_idx with global src IDs.
    fill(in_deg.begin(), in_deg.end(), 0);
    for (int32_t lu = 0; lu < N; ++lu) {
        const int32_t gu = p.vertex_map.to_global(lu, rank);
        for (int32_t e = p.graph.row_ptr[lu]; e < p.graph.row_ptr[lu + 1]; ++e) {
            const int32_t gdst = p.graph.col_idx[e];
            if (p.vertex_map.owner_rank(gdst) == rank) {
                const int32_t ldst = p.vertex_map.to_local(gdst);
                const int32_t pos  = p.reverse_graph.row_ptr[ldst] + in_deg[ldst];
                p.reverse_graph.col_idx[pos] = gu;  // store GLOBAL src ID
                in_deg[ldst]++;
            }
        }
    }
}

Partition load_partition(const string& path, const RunConfig& cfg) {
    ifstream f(path);
    if (!f.is_open())
        throw runtime_error("Cannot open graph file: " + path);

    int32_t num_global_vertices = 0;
    int32_t num_edges_hint      = 0;
    string line;

    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        ss >> num_global_vertices >> num_edges_hint;
        break;
    }

    // Read the full edge list.  All ranks read the same file; each rank then
    // filters to only the edges it owns.
    vector<pair<int32_t, int32_t>> all_edges;
    all_edges.reserve(num_edges_hint);

    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        int32_t u, v;
        istringstream ss(line);
        if (ss >> u >> v) all_edges.push_back({u, v});
    }

    // E9: Compute vertex degrees for partitioning.
    vector<int32_t> degrees(num_global_vertices, 0);
    for (auto& [u, v] : all_edges) degrees[u]++;

    // Build the VertexMap using configured partitioning strategy.
    VertexMap vm;
    if (cfg.partition_mode == PartitionMode::EdgeBalanced) {
        vm.init_edge_balanced(num_global_vertices, cfg.world_size, degrees);

        if (cfg.rank == 0) {
            cout << "\nEdge-Balanced Partitioning:\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int64_t edge_count = 0;
                for (int32_t v = v_start; v < v_end; ++v)
                    edge_count += degrees[v];
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  edges=" << edge_count << "\n";
            }
            cout << "\n";
        }
    } else if (cfg.partition_mode == PartitionMode::HybridBalanced) {
        vm.init_hybrid_balanced(num_global_vertices, cfg.world_size, degrees,
                               cfg.partition_alpha, cfg.partition_beta);

        if (cfg.rank == 0) {
            cout << fixed << setprecision(1)
                 << "\nHybrid Cost-Based Partitioning\n (alpha=" << cfg.partition_alpha
                 << ", beta=" << cfg.partition_beta << "):\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int32_t vertex_count = v_end - v_start;
                int64_t edge_count = 0;
                double cost = 0.0;
                for (int32_t v = v_start; v < v_end; ++v) {
                    edge_count += degrees[v];
                    cost += cfg.partition_alpha + cfg.partition_beta * degrees[v];
                }
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  count=" << vertex_count
                     << "  edges=" << edge_count
                     << "  cost=" << setprecision(0) << cost << "\n";
            }
            cout << "\n";
        }
    } else {
        vm.init(num_global_vertices, cfg.world_size);

        if (cfg.rank == 0) {
            cout << "\nVertex-Balanced Partitioning:\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int64_t edge_count = 0;
                for (int32_t v = v_start; v < v_end; ++v)
                    edge_count += degrees[v];
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  edges=" << edge_count << "\n";
            }
            cout << "\n";
        }
    }

    const int32_t local_n = vm.local_count(cfg.rank);

    // Keep only edges whose source vertex is owned by this rank.
    // Row indices are local src IDs; col values are global dst IDs.
    vector<pair<int32_t, int32_t>> local_edges;  // (local_src, global_dst)
    local_edges.reserve(all_edges.size() / cfg.world_size + 1);

    for (auto& [gu, gv] : all_edges) {
        if (vm.owner_rank(gu) == cfg.rank)
            local_edges.push_back({vm.to_local(gu), gv});
    }

    // Build the local CSR.
    // row_ptr is indexed by local vertex ID (0..local_n-1).
    // col_idx stores GLOBAL destination IDs so MPIExchange can route messages.
    CSRGraph g;
    g.num_vertices = local_n;
    g.num_edges    = static_cast<int32_t>(local_edges.size());
    g.row_ptr.assign(local_n + 1, 0);
    g.col_idx.resize(g.num_edges);

    for (auto& [lu, gv] : local_edges) g.row_ptr[lu + 1]++;
    for (int32_t i = 1; i <= local_n; ++i) g.row_ptr[i] += g.row_ptr[i - 1];

    vector<int32_t> pos(g.row_ptr.begin(), g.row_ptr.begin() + local_n);
    for (auto& [lu, gv] : local_edges) g.col_idx[pos[lu]++] = gv;

    Partition p;
    p.init(local_n, cfg.hidden_dim, cfg.num_threads,
           move(g), move(vm), cfg.rank, cfg.world_size);

    // Initial hidden state: keyed on global vertex ID so values are consistent
    // regardless of how vertices are partitioned across ranks.
    for (int32_t lv = 0; lv < local_n; ++lv) {
        const int32_t gv  = p.vertex_map.to_global(lv, cfg.rank);
        const float   val = static_cast<float>(gv + 1)
                          / static_cast<float>(num_global_vertices);
        float* h = p.hcurr(lv);
        for (int32_t d = 0; d < cfg.hidden_dim; ++d) h[d] = val;
    }

    // Activate all local vertices in the initial frontier.
    for (int32_t lv = 0; lv < local_n; ++lv) {
        p.frontier_curr.set(lv);
        p.vertex_state[lv].active = true;
    }

    // E12: Build local Reverse CSR for future backward traversal.
    // Must be called after partition.init() and before any superstep.
    build_reverse_csr(p);

    return p;
}

Partition load_partition_from_dataset(const Dataset& dataset, const RunConfig& cfg) {
    const int32_t num_global_vertices = static_cast<int32_t>(dataset.num_nodes);

    // E9: Compute vertex degrees for partitioning.
    vector<int32_t> degrees(num_global_vertices, 0);
    for (size_t i = 0; i < dataset.edge_src.size(); ++i) {
        const int32_t gu = dataset.edge_src[i];
        degrees[gu]++;
    }

    VertexMap vm;
    if (cfg.partition_mode == PartitionMode::EdgeBalanced) {
        vm.init_edge_balanced(num_global_vertices, cfg.world_size, degrees);

        if (cfg.rank == 0) {
            cout << "\nEdge-Balanced Partitioning:\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int64_t edge_count = 0;
                for (int32_t v = v_start; v < v_end; ++v)
                    edge_count += degrees[v];
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  edges=" << edge_count << "\n";
            }
            cout << "\n";
        }
    } else if (cfg.partition_mode == PartitionMode::HybridBalanced) {
        vm.init_hybrid_balanced(num_global_vertices, cfg.world_size, degrees,
                               cfg.partition_alpha, cfg.partition_beta);

        if (cfg.rank == 0) {
            cout << fixed << setprecision(1)
                 << "\nHybrid Cost-Based Partitioning"
                 << " (alpha=" << cfg.partition_alpha
                 << ", beta=" << cfg.partition_beta << "):\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int32_t vertex_count = v_end - v_start;
                int64_t edge_count = 0;
                double cost = 0.0;
                for (int32_t v = v_start; v < v_end; ++v) {
                    edge_count += degrees[v];
                    cost += cfg.partition_alpha + cfg.partition_beta * degrees[v];
                }
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  count=" << vertex_count
                     << "  edges=" << edge_count
                     << "  cost=" << setprecision(0) << cost << "\n";
            }
            cout << "\n";
        }
    } else {
        vm.init(num_global_vertices, cfg.world_size);

        if (cfg.rank == 0) {
            cout << "\nVertex-Balanced Partitioning:\n";
            for (int32_t r = 0; r < cfg.world_size; ++r) {
                int32_t v_start = vm.range_start[r];
                int32_t v_end = vm.range_start[r + 1];
                int64_t edge_count = 0;
                for (int32_t v = v_start; v < v_end; ++v)
                    edge_count += degrees[v];
                cout << "  Rank" << r << ": vertices [" << v_start << ", " << v_end
                     << ")  edges=" << edge_count << "\n";
            }
            cout << "\n";
        }
    }

    const int32_t local_n = vm.local_count(cfg.rank);

    vector<pair<int32_t, int32_t>> local_edges;
    local_edges.reserve(dataset.edge_src.size() / cfg.world_size + 1);

    for (size_t i = 0; i < dataset.edge_src.size(); ++i) {
        const int32_t gu = dataset.edge_src[i];
        const int32_t gv = dataset.edge_dst[i];
        if (vm.owner_rank(gu) == cfg.rank)
            local_edges.push_back({vm.to_local(gu), gv});
    }

    CSRGraph g;
    g.num_vertices = local_n;
    g.num_edges    = static_cast<int32_t>(local_edges.size());
    g.row_ptr.assign(local_n + 1, 0);
    g.col_idx.resize(g.num_edges);

    for (auto& [lu, gv] : local_edges) g.row_ptr[lu + 1]++;
    for (int32_t i = 1; i <= local_n; ++i) g.row_ptr[i] += g.row_ptr[i - 1];

    vector<int32_t> pos(g.row_ptr.begin(), g.row_ptr.begin() + local_n);
    for (auto& [lu, gv] : local_edges) g.col_idx[pos[lu]++] = gv;

    Partition p;
    p.init(local_n, cfg.hidden_dim, cfg.num_threads,
           move(g), move(vm), cfg.rank, cfg.world_size);

    for (int32_t lv = 0; lv < local_n; ++lv) {
        p.frontier_curr.set(lv);
        p.vertex_state[lv].active = true;
    }

    // E12: Build local Reverse CSR for future backward traversal.
    build_reverse_csr(p);

    return p;
}