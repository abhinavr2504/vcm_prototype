#include "assertions.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

// ── Phase D Production Assertions ────────────────────────────────────────────

void runtime_assert_fail(const char* file, int line, const char* msg) {
    cerr << "[ASSERT FAIL] " << file << ":" << line << "  " << msg << "\n";
    abort();
}

// ── Phase A Assertions ───────────────────────────────────────────────────────

#ifdef GNN_PHASE_A

void phase_a_fail(const char* file, int line, const char* msg) {
    cerr << "[PHASE_A FAIL] " << file << ":" << line << "  " << msg << "\n";
    abort();
}

// ── assert_partition_sizes ───────────────────────────────────────────────────

void assert_partition_sizes(const Partition& p) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;
    PHASE_A_ASSERT(N > 0, "num_vertices must be > 0");
    PHASE_A_ASSERT(H > 0, "hidden_dim must be > 0");
    PHASE_A_ASSERT((int32_t)p.hidden_curr.size() == N * H,
        "hidden_curr size != num_vertices * hidden_dim");
    PHASE_A_ASSERT((int32_t)p.hidden_next.size() == N * H,
        "hidden_next size != num_vertices * hidden_dim");
    PHASE_A_ASSERT((int32_t)p.aggr_buf.size() == N * H,
        "aggr_buf size != num_vertices * hidden_dim");
    PHASE_A_ASSERT((int32_t)p.vertex_state.size() == N,
        "vertex_state size != num_vertices");
    PHASE_A_ASSERT(p.frontier_curr.num_vertices == N,
        "frontier_curr.num_vertices != num_vertices");
    PHASE_A_ASSERT(p.frontier_next.num_vertices == N,
        "frontier_next.num_vertices != num_vertices");
    PHASE_A_ASSERT((int32_t)p.frontier_curr.bits.size() == N,
        "frontier_curr.bits.size() != num_vertices");
    PHASE_A_ASSERT((int32_t)p.frontier_next.bits.size() == N,
        "frontier_next.bits.size() != num_vertices");
    PHASE_A_ASSERT(p.inbox.num_vertices == N,
        "inbox.num_vertices != num_vertices");
    PHASE_A_ASSERT((int32_t)p.inbox.bufs[0].size() == N,
        "inbox.bufs[0].size() != num_vertices");
    PHASE_A_ASSERT((int32_t)p.inbox.bufs[1].size() == N,
        "inbox.bufs[1].size() != num_vertices");
    PHASE_A_ASSERT(p.num_threads > 0,
        "num_threads must be > 0");
    PHASE_A_ASSERT((int32_t)p.thread_outboxes.size() == p.num_threads,
        "thread_outboxes.size() != num_threads");
}

// ── assert_frontier_next_empty ───────────────────────────────────────────────

void assert_frontier_next_empty(const Partition& p) {
    const int32_t count = p.frontier_next.popcount();
    if (count != 0) {
        cerr << "[PHASE_A FAIL] frontier_next must be empty at superstep entry:"
             << " popcount=" << count << "\n";
        abort();
    }
}

// ── assert_frontier_inbox_sync ───────────────────────────────────────────────

void assert_frontier_inbox_sync(const Partition& p) {
    for (int32_t v = 0; v < p.num_vertices; ++v) {
        const bool in_frontier = p.frontier_curr.test(v);
        const bool has_inbox   = p.inbox.has(v);
        if (in_frontier != has_inbox) {
            cerr << "[PHASE_A FAIL] frontier/inbox desync:"
                 << " v=" << v
                 << "  frontier_curr=" << in_frontier
                 << "  inbox.has=" << has_inbox << "\n";
            abort();
        }
    }
}

// ── assert_frontier_state_sync ───────────────────────────────────────────────

void assert_frontier_state_sync(const Partition& p) {
    for (int32_t v = 0; v < p.num_vertices; ++v) {
        const bool frontier_bit = p.frontier_curr.test(v);
        const bool state_active = p.vertex_state[v].active;
        if (frontier_bit != state_active) {
            cerr << "[PHASE_A FAIL] frontier/vertex_state desync:"
                 << " v=" << v
                 << "  frontier_curr=" << frontier_bit
                 << "  vertex_state.active=" << state_active << "\n";
            abort();
        }
    }
}

// ── assert_no_nan_hcurr ──────────────────────────────────────────────────────

void assert_no_nan_hcurr(const Partition& p) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;
    for (int32_t v = 0; v < N; ++v) {
        const float* h = p.hcurr(v);
        for (int32_t d = 0; d < H; ++d) {
            if (isnan(h[d]) || isinf(h[d])) {
                cerr << "[PHASE_A FAIL] NaN/Inf in hidden_curr:"
                     << " v=" << v << "  d=" << d << "  val=" << h[d] << "\n";
                abort();
            }
        }
    }
}

// ── assert_no_nan_hnext ──────────────────────────────────────────────────────

void assert_no_nan_hnext(const Partition& p) {
    // After the compute pass, ALL vertices write hidden_next
    // (active via linear + ReLU, inactive via hcurr copy). Check all.
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;
    for (int32_t v = 0; v < N; ++v) {
        const float* h = p.hnext(v);
        for (int32_t d = 0; d < H; ++d) {
            if (isnan(h[d]) || isinf(h[d])) {
                cerr << "[PHASE_A FAIL] NaN/Inf in hidden_next:"
                     << " v=" << v << "  d=" << d << "  val=" << h[d] << "\n";
                abort();
            }
        }
    }
}

// ── assert_outbox_valid ──────────────────────────────────────────────────────

void assert_outbox_valid(const Partition& p) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;
    for (int32_t t = 0; t < p.num_threads; ++t) {
        for (const Message& m : p.thread_outboxes[t].messages) {
            if (m.src < 0 || m.src >= N) {
                cerr << "[PHASE_A FAIL] outbox[" << t << "]: src=" << m.src
                     << " out of [0," << N << ")\n";
                abort();
            }
            if (m.dst < 0 || m.dst >= N) {
                cerr << "[PHASE_A FAIL] outbox[" << t << "]: dst=" << m.dst
                     << " out of [0," << N << ")\n";
                abort();
            }
            if ((int32_t)m.payload.size() != H) {
                cerr << "[PHASE_A FAIL] outbox[" << t << "]:"
                     << " payload.size()=" << m.payload.size()
                     << " != hidden_dim=" << H
                     << "  src=" << m.src << " dst=" << m.dst << "\n";
                abort();
            }
        }
    }
}

// ── assert_inbox_valid ───────────────────────────────────────────────────────

void assert_inbox_valid(const Partition& p) {
    const int32_t N = p.num_vertices;
    const int32_t H = p.hidden_dim;
    for (int32_t v = 0; v < N; ++v) {
        for (const Message& m : p.inbox.get(v)) {
            if (m.dst != v) {
                cerr << "[PHASE_A FAIL] inbox slot " << v
                     << " contains msg with dst=" << m.dst << "\n";
                abort();
            }
            if (m.src < 0 || m.src >= N) {
                cerr << "[PHASE_A FAIL] inbox: msg.src=" << m.src
                     << " out of [0," << N << ")\n";
                abort();
            }
            if ((int32_t)m.payload.size() != H) {
                cerr << "[PHASE_A FAIL] inbox: payload.size()=" << m.payload.size()
                     << " != hidden_dim=" << H
                     << "  src=" << m.src << " dst=" << m.dst << "\n";
                abort();
            }
        }
    }
}

#endif  // GNN_PHASE_A
