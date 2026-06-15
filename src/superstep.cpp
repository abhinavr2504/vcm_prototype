#include "superstep.h"
#include "assertions.h"
#include <iostream>
#include <omp.h>

using namespace std;

// ── Bootstrap scatter ─────────────────────────────────────────────────────────
// Seeds the inbox for layer 0 by scattering hidden_curr payloads to all
// out-neighbours of the initial frontier. Rebuilds frontier_curr from
// message-arrival semantics so that the layer loop starts correctly.
void SuperstepExecutor::seed(Partition& p, RuntimeTiming* timing) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;

    for (int32_t v = 0; v < N; ++v) {
        if (!p.frontier_curr.test(v)) continue;

        const float*  hv        = p.hcurr(v);
        const int32_t row_start = p.graph.row_ptr[v];
        const int32_t row_end   = p.graph.row_ptr[v + 1];

        for (int32_t e = row_start; e < row_end; ++e) {
            const int32_t u = p.graph.col_idx[e];
            Message m;
            m.src = p.vertex_map.to_global(v, p.rank);
            m.dst = u;
            m.payload.assign(hv, hv + H);
            p.thread_outboxes[0].push(m);
        }
    }

    assert_outbox_valid(p);

#ifdef GNN_PHASE_A
    {
        int32_t total_inbox = 0;
        for (int32_t _v = 0; _v < N; ++_v)
            total_inbox += (int32_t)p.inbox.bufs[p.inbox.write_idx][_v].size();
        cout << "[TRACE seed] outbox_msgs=" << p.thread_outboxes[0].messages.size()
             << "  inbox_msgs_before_flip=" << total_inbox << "\n";
    }
#endif

    for (int32_t t = 0; t < p.num_threads; ++t) {
        for (const Message& m : p.thread_outboxes[t].messages)
            p.mpi_exchange.enqueue(m);
        p.thread_outboxes[t].clear();
    }

    if (timing) timing->start_communication();
    p.mpi_exchange.exchange();
    p.mpi_exchange.wait_and_unpack(p.inbox);
    if (timing) timing->stop_communication();

    assert_inbox_valid(p);

    // Rebuild frontier_curr: a vertex is active if and only if it has inbox messages.
    p.frontier_curr.clear();
    for (int32_t v = 0; v < N; ++v) {
        const bool has_msgs      = p.inbox.has(v);
        p.vertex_state[v].active = has_msgs;
        if (has_msgs) p.frontier_curr.set(v);
    }

    assert_frontier_inbox_sync(p);
    assert_frontier_state_sync(p);

#ifdef GNN_PHASE_A
    cout << "[TRACE seed] frontier_active=" << p.frontier_curr.popcount() << "\n";
#endif
}

// ── Superstep ─────────────────────────────────────────────────────────────────
// One full Gather → Compute → Scatter cycle for the given layer.
// Timing hooks are injected via RuntimeTiming* for profiling.
void SuperstepExecutor::run(Partition&      p,
                            const GNNLayer& layer_params,
                            int32_t         layer,
                            RuntimeTiming*  timing) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;

    assert_frontier_next_empty(p);
    assert_frontier_state_sync(p);
    assert_frontier_inbox_sync(p);

    // Capture message statistics before this layer for the per-layer profile.
    int64_t stats_before_local_sent   = p.mpi_exchange.statistics().local_messages_sent;
    int64_t stats_before_remote_sent  = p.mpi_exchange.statistics().remote_messages_sent;
    int64_t stats_before_local_recv   = p.mpi_exchange.statistics().local_messages_received;
    int64_t stats_before_remote_recv  = p.mpi_exchange.statistics().remote_messages_received;
    int32_t active_vertices = p.frontier_curr.popcount();

#ifdef GNN_PHASE_A
    {
        int32_t inbox_occ = 0;
        for (int32_t _v = 0; _v < N; ++_v) inbox_occ += p.inbox.has(_v) ? 1 : 0;
        cout << "[TRACE L" << layer << " start] frontier_active=" << p.frontier_curr.popcount()
             << "  vertices_with_inbox=" << inbox_occ << "\n";
    }
#endif

    if (timing) timing->start_compute();

    // ── Gather: sum inbox payloads into the aggregation buffer ────────────────
    if (timing) {
        timing->start_gather();
        timing->start_aggregation();
    }

    #pragma omp parallel for schedule(dynamic) num_threads(p.num_threads)
    for (int32_t v = 0; v < N; ++v) {
        if (!p.frontier_curr.test(v)) continue;

        float* agg = p.vaggr(v);
        compute.zero_aggr(agg, H);

        for (const Message& m : p.inbox.get(v))
            compute.aggregate(m.payload.data(), agg, H);
    }

    if (timing) {
        timing->stop_aggregation(layer);
        timing->stop_gather(layer);
    }

#ifdef GNN_PHASE_A
    {
        int32_t gathered_verts = 0, gathered_msgs = 0;
        for (int32_t _v = 0; _v < N; ++_v) {
            if (!p.frontier_curr.test(_v)) continue;
            ++gathered_verts;
            gathered_msgs += (int32_t)p.inbox.get(_v).size();
        }
        cout << "[TRACE L" << layer << " gather] vertices=" << gathered_verts
             << "  messages=" << gathered_msgs << "\n";
    }
#endif

    // E11: Snapshot the aggregated buffer right after gather, before it is
    //      overwritten. Also record the aggregation type for the backward pass.
    LayerCache cache;
    cache.aggregated     = p.aggr_buf.clone();
    cache.aggr_meta.type = AggregationType::SUM;

    // E3: Save a snapshot and the active-vertex mask for the training loss.
    if (timing) timing->start_snapshot();
    {
        torch::Tensor snapshot = p.aggr_buf.clone();
        torch::Tensor mask = torch::zeros({N}, torch::kBool);
        bool* mask_ptr = mask.data_ptr<bool>();
        for (int32_t v = 0; v < N; ++v)
            mask_ptr[v] = p.frontier_curr.test(v);
        p.aggr_snapshots.push_back(snapshot);
        p.aggr_masks.push_back(mask);
    }
    if (timing) timing->stop_snapshot(layer);

    // ── Compute: linear transform + ReLU → hidden_next ───────────────────────
    // E11: Capture pre_activation (before ReLU) for the backward pass.
    //      Inactive vertices simply copy hcurr to hnext to keep buffers valid.
    torch::Tensor pre_activation = torch::empty({N, H}, torch::kFloat32);
    float* pre_act_ptr = pre_activation.data_ptr<float>();

    if (timing) {
        timing->start_layer_compute();
        timing->start_linear();
        timing->start_activation();
    }

    #pragma omp parallel for schedule(dynamic) num_threads(p.num_threads)
    for (int32_t v = 0; v < N; ++v) {
        if (!p.frontier_curr.test(v)) {
            // Inactive: pass through current hidden state unchanged.
            const float* src = p.hcurr(v);
            float*       dst = p.hnext(v);
            for (int32_t d = 0; d < H; ++d) dst[d] = src[d];
            for (int32_t d = 0; d < H; ++d) pre_act_ptr[v * H + d] = src[d];
            continue;
        }

        const float* aggr_v = p.vaggr(v);
        float*       out_v  = p.hnext(v);
        float*       pre_v  = pre_act_ptr + v * H;

        const float* W_ptr = layer_params.W().data_ptr<float>();
        const float* b_ptr = layer_params.b().data_ptr<float>();

        // Linear transform: out[d] = sum_h(W[d,h] * aggr[h]) + b[d]
        for (int32_t d = 0; d < H; ++d) {
            float sum = b_ptr[d];
            for (int32_t h = 0; h < H; ++h)
                sum += W_ptr[d * H + h] * aggr_v[h];
            pre_v[d] = sum;
        }

        // ReLU activation
        for (int32_t d = 0; d < H; ++d)
            out_v[d] = (pre_v[d] > 0.0f) ? pre_v[d] : 0.0f;
    }

    if (timing) {
        timing->stop_activation(layer);
        timing->stop_linear(layer);
        timing->stop_layer_compute(layer);
    }

    assert_no_nan_hnext(p);

    // E11: Store pre_activation and the post-activation hidden state.
    cache.pre_activation = pre_activation;
    cache.activated = p.hidden_next.clone();

    // ── Scatter: send hidden_next to each out-neighbour ───────────────────────
    if (timing) timing->start_scatter();

    #pragma omp parallel for schedule(dynamic) num_threads(p.num_threads)
    for (int32_t v = 0; v < N; ++v) {
        if (!p.frontier_curr.test(v)) continue;

        const int     tid       = omp_get_thread_num();
        const float*  hv        = p.hnext(v);
        const int32_t row_start = p.graph.row_ptr[v];
        const int32_t row_end   = p.graph.row_ptr[v + 1];

        for (int32_t e = row_start; e < row_end; ++e) {
            const int32_t u = p.graph.col_idx[e];
            Message m;
            m.src = p.vertex_map.to_global(v, p.rank);
            m.dst = u;
            m.payload.assign(hv, hv + H);
            p.thread_outboxes[tid].push(m);
        }
    }

    if (timing) timing->stop_scatter(layer);

    assert_outbox_valid(p);

#ifdef GNN_PHASE_A
    {
        int32_t scatter_verts = 0, scatter_msgs = 0;
        for (int32_t _v = 0; _v < N; ++_v)
            if (p.frontier_curr.test(_v)) ++scatter_verts;
        for (int32_t t = 0; t < p.num_threads; ++t)
            scatter_msgs += (int32_t)p.thread_outboxes[t].messages.size();
        cout << "[TRACE L" << layer << " scatter] vertices=" << scatter_verts
             << "  outbox_msgs=" << scatter_msgs << "\n";
    }
#endif

    if (timing) timing->stop_compute();

    // ── Route outbox messages through MPIExchange ─────────────────────────────
    if (timing) timing->start_pack();

    for (int32_t t = 0; t < p.num_threads; ++t) {
        for (const Message& m : p.thread_outboxes[t].messages)
            p.mpi_exchange.enqueue(m);
        p.thread_outboxes[t].clear();
    }

    if (timing) timing->stop_pack(layer);

    if (timing) {
        timing->start_communication();
        timing->start_mpi();
        timing->start_exchange();
    }

    p.mpi_exchange.exchange();

    if (timing) {
        timing->stop_exchange(layer);
        timing->start_unpack();
    }

    // E12: Prepare the trace for this layer. Each local vertex starts with
    //      an empty list of remote contributors; wait_and_unpack fills it in.
    AggregationTrace& trace = p.aggr_traces[layer];
    trace.valid = false;
    trace.remote_contributors.assign(p.num_vertices, {});

    p.mpi_exchange.wait_and_unpack(p.inbox, &trace);
    trace.valid = true;

    if (timing) {
        timing->stop_unpack(layer);
        timing->stop_mpi(layer);
        timing->stop_communication();
    }

    assert_inbox_valid(p);

#ifdef GNN_PHASE_A
    {
        int32_t inbox_msgs = 0;
        for (int32_t _v = 0; _v < N; ++_v)
            inbox_msgs += (int32_t)p.inbox.get(_v).size();
        cout << "[TRACE L" << layer << " flip] inbox_msgs=" << inbox_msgs
             << "  frontier_next_so_far=" << p.frontier_next.popcount() << "\n";
    }
#endif

    // ── Next frontier: active iff the vertex received at least one message ────
    for (int32_t v = 0; v < N; ++v) {
        const bool has_msgs = p.inbox.has(v);
        if (has_msgs) p.frontier_next.set(v);
        p.vertex_state[v].active = has_msgs;
    }

    // E11: Record which vertices were active this layer (needed for backward).
    {
        torch::Tensor mask = torch::zeros({N}, torch::kBool);
        bool* mask_ptr = mask.data_ptr<bool>();
        for (int32_t v = 0; v < N; ++v)
            mask_ptr[v] = p.frontier_curr.test(v);
        cache.active_mask = mask;
    }

    p.layer_cache.push_back(cache);

    // Record per-layer statistics for the timing profile.
    if (timing) {
        int64_t stats_after_local_sent   = p.mpi_exchange.statistics().local_messages_sent;
        int64_t stats_after_remote_sent  = p.mpi_exchange.statistics().remote_messages_sent;
        int64_t stats_after_local_recv   = p.mpi_exchange.statistics().local_messages_received;
        int64_t stats_after_remote_recv  = p.mpi_exchange.statistics().remote_messages_received;

        int64_t layer_msgs_sent   = (stats_after_local_sent   - stats_before_local_sent)  +
                                    (stats_after_remote_sent  - stats_before_remote_sent);
        int64_t layer_msgs_recv   = (stats_after_local_recv   - stats_before_local_recv)  +
                                    (stats_after_remote_recv  - stats_before_remote_recv);
        int64_t layer_remote_sent = stats_after_remote_sent - stats_before_remote_sent;
        int64_t layer_remote_recv = stats_after_remote_recv - stats_before_remote_recv;

        timing->record_layer_profile(layer, active_vertices, layer_msgs_sent, layer_msgs_recv,
                                    layer_remote_sent, layer_remote_recv);

        int64_t aggr_bytes    = static_cast<int64_t>(N) * H * sizeof(float);
        int64_t hidden_bytes  = static_cast<int64_t>(N) * H * sizeof(float);
        int64_t snapshot_bytes = aggr_bytes;
        timing->record_memory_stats(layer, aggr_bytes, hidden_bytes, hidden_bytes, snapshot_bytes);
    }

#ifdef GNN_PHASE_A
    for (int32_t _v = 0; _v < N; ++_v) {
        PHASE_A_ASSERT(p.frontier_next.test(_v) == p.vertex_state[_v].active,
            "frontier_next/vertex_state desync after frontier build");
    }
#endif
}