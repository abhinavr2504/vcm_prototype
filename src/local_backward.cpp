// local_backward.cpp
// E13: Local Backward Reconstruction
//
// Reconstructs the gradient w.r.t. Layer 0 parameters (grad_W0, grad_b0) given
// the gradient w.r.t. the last layer's aggregated input (grad_aggr).
//
// Three entry points:
//   scatter_only()  — scatter grad_aggr back through local edges → grad_h (partial)
//   linear_phase()  — apply ReLU gate + matmul → grad_W, grad_b
//   run()           — scatter_only() + linear_phase() (single-process convenience)

#include "local_backward.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

// ── Shared precondition check ────────────────────────────────────────────────

static void check_preconditions(const Partition& partition,
                                int32_t target_layer,
                                const torch::Tensor& grad_aggr)
{
    const int32_t src_layer = target_layer - 1;
    assert(src_layer >= 0 &&
           "E13: target_layer must be >= 1");
    assert(target_layer < static_cast<int32_t>(partition.layer_cache.size()) &&
           "E13: LayerCache not populated for target_layer");
    assert(src_layer < static_cast<int32_t>(partition.layer_cache.size()) &&
           "E13: LayerCache not populated for src_layer");
    assert(partition.reverse_graph.built() &&
           "E13: ReverseCSR not built — was load_partition called?");
    assert(grad_aggr.defined() &&
           "E13: grad_aggr is undefined — was retain_grad()/backward() called?");
    assert(grad_aggr.size(0) == partition.num_vertices &&
           grad_aggr.size(1) == partition.hidden_dim &&
           "E13: grad_aggr shape mismatch — expected [N, H]");
}

// ── scatter_only ─────────────────────────────────────────────────────────────
// Steps 1–3:
//   1. Build edge index tensors from the local ReverseCSR.
//   2. Gather grad_aggr[v] for each edge; mask out inactive destination vertices.
//   3. Scatter-add into grad_h: grad_h[u] += grad_aggr[v] for each edge u→v.
//
// Returns grad_h [N, H] containing only local-edge contributions.
// Remote contributions are added later by GradientExchange::exchange_into().

torch::Tensor LocalBackwardEngine::scatter_only(
    const Partition&     partition,
    int32_t              target_layer,
    const torch::Tensor& grad_aggr)
{
    check_preconditions(partition, target_layer, grad_aggr);

    const int32_t N = partition.num_vertices;
    const int32_t H = partition.hidden_dim;

    const LayerCache& dst_cache = partition.layer_cache[target_layer];
    const ReverseCSR& rcsr      = partition.reverse_graph;
    const int32_t total_in_edges = rcsr.row_ptr[N];

    if (total_in_edges == 0) {
        // No local-to-local edges. Return zeros so linear_phase() can still
        // run cleanly and produce defined (zero) gradients for W0/b0.
        return torch::zeros({N, H}, torch::kFloat32);
    }

    // Step 1: Build edge index tensors from the local ReverseCSR.
    //   v_vec[e] = local destination vertex
    //   u_vec[e] = local source vertex (converted from global ID)
    vector<int64_t> v_vec, u_vec;
    v_vec.reserve(total_in_edges);
    u_vec.reserve(total_in_edges);

    for (int32_t v = 0; v < N; ++v) {
        for (int32_t e = rcsr.row_ptr[v]; e < rcsr.row_ptr[v + 1]; ++e) {
            const int32_t g_src = rcsr.col_idx[e];
            const int32_t l_src = partition.vertex_map.to_local(g_src);
            assert(l_src >= 0 && "E13: ReverseCSR contains non-local source vertex");
            v_vec.push_back(static_cast<int64_t>(v));
            u_vec.push_back(static_cast<int64_t>(l_src));
        }
    }

    const int64_t E = static_cast<int64_t>(v_vec.size());
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor v_idx = torch::tensor(v_vec, idx_opts);  // [E]
    torch::Tensor u_idx = torch::tensor(u_vec, idx_opts);  // [E]

    // Step 2: Gather grad_aggr[v] for each edge.
    // For SUM aggregation: d(aggr[v])/d(h[u]) = 1, so grad_h[u] += grad_aggr[v].
    // Mask out edges where the destination vertex was not active this layer.
    torch::Tensor grad_per_edge =
        grad_aggr.detach().index_select(0, v_idx);  // [E, H]

    if (dst_cache.active_mask.defined()) {
        torch::Tensor edge_active = dst_cache.active_mask
                                        .index_select(0, v_idx)  // [E] bool
                                        .unsqueeze(1)             // [E, 1]
                                        .to(torch::kFloat32);
        grad_per_edge = grad_per_edge * edge_active;              // [E, H]
    }

    // Step 3: Scatter-add → grad_h[u] += grad_per_edge[e]
    torch::Tensor u_idx_exp =
        u_idx.unsqueeze(1).expand({E, H}).contiguous();  // [E, H]
    torch::Tensor grad_h = torch::zeros({N, H}, torch::kFloat32);
    grad_h.scatter_add_(0, u_idx_exp, grad_per_edge);

    return grad_h;  // [N, H] — local contributions only
}

// ── linear_phase ─────────────────────────────────────────────────────────────
// Steps 4–5:
//   4. Gate grad_h by the ReLU mask (pre_activation > 0) → grad_pre.
//   5. Compute parameter gradients:
//        grad_W = grad_pre.T @ aggr_src    [H, H]
//        grad_b = grad_pre.sum(0)          [H]

pair<torch::Tensor, torch::Tensor> LocalBackwardEngine::linear_phase(
    const Partition&     partition,
    int32_t              src_layer,
    const torch::Tensor& grad_h)
{
    const int32_t N = partition.num_vertices;
    const int32_t H = partition.hidden_dim;

    assert(src_layer >= 0 &&
           "E13 linear_phase: src_layer must be >= 0");
    assert(src_layer < static_cast<int32_t>(partition.layer_cache.size()) &&
           "E13 linear_phase: LayerCache not populated for src_layer");
    assert(grad_h.defined() && grad_h.size(0) == N && grad_h.size(1) == H &&
           "E13 linear_phase: grad_h must be defined [N, H]");

    const LayerCache& src_cache = partition.layer_cache[src_layer];

    // Step 4: Gate by the ReLU mask — zero out gradients where activation was off.
    assert(src_cache.pre_activation.defined() &&
           "E13 linear_phase: LayerCache[src_layer].pre_activation undefined");

    torch::Tensor relu_mask = src_cache.pre_activation
                                  .detach().gt(0.0f).to(torch::kFloat32);  // [N, H]
    torch::Tensor grad_pre  = grad_h * relu_mask;                           // [N, H]

    // Step 5: Compute parameter gradients via matmul.
    assert(src_cache.aggregated.defined() &&
           "E13 linear_phase: LayerCache[src_layer].aggregated undefined");

    torch::Tensor grad_W = torch::matmul(
        grad_pre.t(),                  // [H, N]
        src_cache.aggregated.detach()  // [N, H]
    );                                 // → [H, H]

    torch::Tensor grad_b = grad_pre.sum(0);  // [H]

    return {grad_W, grad_b};
}

// ── run ──────────────────────────────────────────────────────────────────────
// Convenience wrapper: scatter_only + linear_phase.
// Also prints a brief diagnostic summary.
// For distributed runs, use scatter_only + GradientExchange::exchange_into +
// linear_phase separately so remote contributions are included before Step 5.

pair<torch::Tensor, torch::Tensor> LocalBackwardEngine::run(
    const Partition&     partition,
    int32_t              target_layer,
    const torch::Tensor& grad_aggr)
{
    const int32_t src_layer = target_layer - 1;

    torch::Tensor grad_h = scatter_only(partition, target_layer, grad_aggr);
    auto [grad_W, grad_b] = linear_phase(partition, src_layer, grad_h);

    // Count uncovered remote paths for the diagnostic summary.
    const int32_t total_in_edges = partition.reverse_graph.row_ptr[partition.num_vertices];

    int64_t remote_paths_count = 0;
    if (target_layer < static_cast<int32_t>(partition.aggr_traces.size())) {
        const AggregationTrace& tr = partition.aggr_traces[target_layer];
        if (tr.valid) {
            for (const auto& vec : tr.remote_contributors)
                remote_paths_count += static_cast<int64_t>(vec.size());
        }
    }

    cout << "[E13] rank=" << partition.rank
         << " layer" << target_layer << "→" << src_layer << ":"
         << " local_edges=" << total_in_edges << " covered"
         << " | remote_contributors=" << remote_paths_count << " uncovered"
         << " | grad_W_norm=" << grad_W.norm().item<float>()
         << " | grad_b_norm=" << grad_b.norm().item<float>() << "\n";

    return {grad_W, grad_b};
}
