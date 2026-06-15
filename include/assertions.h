#pragma once
#include <cstdint>
#include "partition.h"

// ── Phase D Production Assertions ────────────────────────────────────────────
// Lightweight assertions for production runtime hardening.
// Always enabled. Use for critical invariants only (ownership, translation, etc).

void runtime_assert_fail(const char* file, int line, const char* msg);

#define ASSERT(cond, msg)                                        \
    do {                                                         \
        if (!(cond)) runtime_assert_fail(__FILE__, __LINE__, (msg)); \
    } while (0)

// ── Phase A Runtime Correctness Assertions ───────────────────────────────────
// Compile with -DGNN_PHASE_A to enable. All calls are no-ops otherwise.
// Do NOT ship these in a production build or a Phase D benchmark binary.

#ifdef GNN_PHASE_A

void phase_a_fail(const char* file, int line, const char* msg);

#define PHASE_A_ASSERT(cond, msg)                               \
    do {                                                         \
        if (!(cond)) phase_a_fail(__FILE__, __LINE__, (msg));    \
    } while (0)

// ── Structural ───────────────────────────────────────────────────────────────

// Call after load_partition(). Verifies buffer sizes and num_vertices fields.
void assert_partition_sizes(const Partition& p);

// Call at superstep entry and after seed(). Verifies frontier_next is empty.
// Catches: stale frontier_next not cleared by advance_frontier().
void assert_frontier_next_empty(const Partition& p);

// ── Invariant I-frt ──────────────────────────────────────────────────────────

// Call after seed() and at superstep entry.
// frontier_curr.test(v)  iff  inbox.has(v)  for all v.
// Catches: frontier not rebuilt from inbox; topology-based activation.
void assert_frontier_inbox_sync(const Partition& p);

// ── Invariant: vertex_state / frontier consistency ───────────────────────────

// Call after seed() and at superstep entry.
// vertex_state[v].active  iff  frontier_curr.test(v)  for all v.
// Catches: the two redundant active-state stores diverging.
void assert_frontier_state_sync(const Partition& p);

// ── Numerical ────────────────────────────────────────────────────────────────

// Call after advance_hidden() in Runtime::run().
// Checks hidden_curr for NaN and Inf across ALL vertices.
// Catches: NaN from bad weights, overflow, or uninitialized reads.
void assert_no_nan_hcurr(const Partition& p);

// Call after the compute loop inside SuperstepExecutor::run().
// Checks hidden_next for NaN and Inf across ALL vertices (all written after fix).
// Catches: NaN from linear_relu or from hcurr during the inactive-vertex copy.
void assert_no_nan_hnext(const Partition& p);

// ── Message validity ─────────────────────────────────────────────────────────

// Call after the scatter loop, before serial merge, in seed() and run().
// Checks: src in [0,N), dst in [0,N), payload.size() == hidden_dim.
// Catches: out-of-range destinations, wrong payload sizes.
void assert_outbox_valid(const Partition& p);

// Call after inbox.flip() in seed() and run().
// Checks: msg.dst == slot index, src in [0,N), payload.size() == hidden_dim.
// Catches: messages routed to wrong inbox slot; malformed messages after flip.
void assert_inbox_valid(const Partition& p);

#else   // GNN_PHASE_A not defined — all calls compile away

#define PHASE_A_ASSERT(cond, msg) ((void)0)
inline void assert_partition_sizes(const Partition&)      {}
inline void assert_frontier_next_empty(const Partition&)  {}
inline void assert_frontier_inbox_sync(const Partition&)  {}
inline void assert_frontier_state_sync(const Partition&)  {}
inline void assert_no_nan_hcurr(const Partition&)         {}
inline void assert_no_nan_hnext(const Partition&)         {}
inline void assert_outbox_valid(const Partition&)         {}
inline void assert_inbox_valid(const Partition&)          {}

#endif  // GNN_PHASE_A
