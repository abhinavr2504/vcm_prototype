// gradient_exchange.cpp
// E14: Gradient Message Transport — rank-batched streaming exchange.
//
// After the local backward pass (scatter_only), each rank knows the grad_aggr
// for its own vertices, but some remote vertices contributed to local
// aggregations. This module transports those remote gradient contributions
// back to the ranks that own the source vertices, completing grad_h before
// the linear_phase matmul.
//
// Protocol overview (per remote rank r):
//   Step 0: Exchange pair counts           (MPI tag 1000)
//   Step 1: Chunked chunk-meta exchange    (MPI tag 1001)
//   Step 2: Float payload exchange         (MPI tag 1002)
//   Step 3: Combined int buffer exchange   (MPI tag 1003)
//
// Memory bound: at most 2 × chunk_pairs_ × H × 4 bytes in flight at once.

#include "gradient_exchange.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <mpi.h>

using namespace std;

// ── Constructor ─────────────────────────────────────────────────────────────

GradientExchange::GradientExchange(int32_t rank, int32_t world_size, int32_t chunk_pairs)
    : rank_(rank), world_size_(world_size), chunk_pairs_(chunk_pairs)
{
    // Pre-allocate assuming hidden_dim ≈ 256; grows lazily if H is larger.
    const size_t float_cap = static_cast<size_t>(chunk_pairs_) * 256;
    const size_t int_cap   = static_cast<size_t>(chunk_pairs_) * 8;

    send_float_.reserve(float_cap);
    recv_float_.reserve(float_cap);
    send_int_.reserve(int_cap);
    recv_int_.reserve(int_cap);
}

// ── build_rank_groups ────────────────────────────────────────────────────────
// Scans aggr_traces[target_layer] and collects, for rank r:
//   v_locals  — local dst vertices that received at least one message from r
//   u_offsets — CSR offsets into u_globals (one per entry in v_locals)
//   u_globals — global source vertex IDs owned by r

void GradientExchange::build_rank_groups(
    const Partition&      partition,
    int32_t               target_layer,
    int32_t               r,
    vector<int32_t>& v_locals,
    vector<int32_t>& u_offsets,
    vector<int32_t>& u_globals) const
{
    v_locals.clear();
    u_offsets.clear();
    u_globals.clear();
    u_offsets.push_back(0);  // CSR sentinel

    if (target_layer >= static_cast<int32_t>(partition.aggr_traces.size())) return;
    const AggregationTrace& tr = partition.aggr_traces[target_layer];
    if (!tr.valid) return;

    const int32_t N = partition.num_vertices;
    for (int32_t v = 0; v < N; ++v) {
        const auto& contribs = tr.remote_contributors[v];
        int32_t added = 0;

        for (const int32_t u_g : contribs) {
            if (partition.vertex_map.owner_rank(u_g) == r) {
                u_globals.push_back(u_g);
                ++added;
            }
        }

        if (added > 0) {
            v_locals.push_back(v);
            u_offsets.push_back(static_cast<int32_t>(u_globals.size()));
        }
    }
    // Invariant: v_locals.size() == u_offsets.size() - 1
}

// ── exchange_into ────────────────────────────────────────────────────────────
// For each remote rank r, this rank sends: the grad_aggr[v] value for each
// local dst vertex that received a contribution from r, together with the list
// of global source IDs that should receive a copy of that gradient.
// On the receive side, each incoming payload is accumulated into grad_h[u_loc].

void GradientExchange::exchange_into(
    const Partition&     partition,
    int32_t              target_layer,
    const torch::Tensor& grad_aggr,
    torch::Tensor&       grad_h)
{
    if (world_size_ == 1) return;  // no remote contributors when running single-rank

    const int32_t N = partition.num_vertices;
    const int32_t H = partition.hidden_dim;

    // Grow buffers if H exceeds the initial reservation.
    const size_t needed = static_cast<size_t>(chunk_pairs_) * H;
    if (send_float_.capacity() < needed) send_float_.reserve(needed);
    if (recv_float_.capacity() < needed) recv_float_.reserve(needed);

    torch::Tensor ga_cont = grad_aggr.detach().contiguous();
    const float* ga = ga_cont.data_ptr<float>();

    assert(grad_h.is_contiguous() && "E14: grad_h must be contiguous");
    float* gh = grad_h.data_ptr<float>();

    int64_t total_remote_accum = 0;

    // Process one remote rank at a time so only one send/recv buffer pair
    // is live in memory at once.
    for (int32_t r = 0; r < world_size_; ++r) {
        if (r == rank_) continue;

        // Build the index of which vertices/sources are relevant for rank r.
        vector<int32_t> v_locals, u_offsets, u_globals;
        build_rank_groups(partition, target_layer, r, v_locals, u_offsets, u_globals);

        const int32_t n_send_pairs = static_cast<int32_t>(v_locals.size());

        // Step 0: Tell rank r how many pairs we will send, and learn theirs.
        int32_t n_recv_pairs = 0;
        MPI_Sendrecv(&n_send_pairs, 1, MPI_INT, r, 1000,
                     &n_recv_pairs, 1, MPI_INT, r, 1000,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Both sides iterate the same number of chunks.
        const auto chunks_needed = [&](int32_t n_pairs) -> int32_t {
            return (n_pairs == 0) ? 1 : (n_pairs + chunk_pairs_ - 1) / chunk_pairs_;
        };
        const int32_t n_iters = max(chunks_needed(n_send_pairs),
                                    chunks_needed(n_recv_pairs));

        for (int32_t ci = 0; ci < n_iters; ++ci) {

            // Range of pairs this chunk covers on the send side.
            const int32_t sp_start = ci * chunk_pairs_;
            const int32_t sp_end   = min(sp_start + chunk_pairs_, n_send_pairs);
            const int32_t n_sp     = max(0, sp_end - sp_start);

            const int32_t idx_start = (n_sp > 0) ? u_offsets[sp_start] : 0;
            const int32_t idx_end   = (n_sp > 0) ? u_offsets[sp_end]   : 0;
            const int32_t n_si      = idx_end - idx_start;  // number of dst IDs to send

            // Step 1: Swap chunk metadata: (n_sp, n_si) ↔ (n_rp, n_ri)
            int32_t scmeta[2] = { n_sp, n_si };
            int32_t rcmeta[2] = { 0,    0    };
            MPI_Sendrecv(scmeta, 2, MPI_INT, r, 1001,
                         rcmeta, 2, MPI_INT, r, 1001,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const int32_t n_rp = rcmeta[0];  // pairs to receive
            const int32_t n_ri = rcmeta[1];  // dst IDs to receive

            // Step 2: Exchange gradient payloads.
            // Each send pair contributes one grad_aggr[v] slice (H floats).
            send_float_.resize(static_cast<size_t>(n_sp) * H);
            for (int32_t p = 0; p < n_sp; ++p) {
                const int32_t v = v_locals[sp_start + p];
                const float* src = ga + static_cast<int64_t>(v) * H;
                float* dst = send_float_.data() + static_cast<size_t>(p) * H;
                copy(src, src + H, dst);
            }
            recv_float_.resize(static_cast<size_t>(n_rp) * H);

            MPI_Sendrecv(send_float_.data(), n_sp * H, MPI_FLOAT, r, 1002,
                         recv_float_.data(), n_rp * H, MPI_FLOAT, r, 1002,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Step 3: Exchange the integer buffer.
            // Layout: [per-pair dst-counts (n_sp ints) | dst global IDs (n_si ints)]
            const int32_t send_int_count = n_sp + n_si;
            const int32_t recv_int_count = n_rp + n_ri;
            send_int_.resize(static_cast<size_t>(send_int_count));
            recv_int_.resize(static_cast<size_t>(recv_int_count));

            for (int32_t p = 0; p < n_sp; ++p)
                send_int_[p] = u_offsets[sp_start + p + 1] - u_offsets[sp_start + p];

            copy(u_globals.begin() + idx_start,
                 u_globals.begin() + idx_end,
                 send_int_.begin() + n_sp);

            MPI_Sendrecv(send_int_.data(), send_int_count, MPI_INT, r, 1003,
                         recv_int_.data(), recv_int_count, MPI_INT, r, 1003,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Unpack: for each received (payload, dst_count, dst_ids) triple,
            // accumulate the payload into grad_h[local(u)] for each dst.
            const int32_t* recv_counts = recv_int_.data();
            const int32_t* recv_dsts   = recv_int_.data() + n_rp;
            int32_t dst_cursor = 0;

            for (int32_t p = 0; p < n_rp; ++p) {
                const int32_t n_dsts  = recv_counts[p];
                const float*  payload = recv_float_.data() + static_cast<size_t>(p) * H;

                for (int32_t d = 0; d < n_dsts; ++d) {
                    const int32_t u_g   = recv_dsts[dst_cursor + d];
                    const int32_t u_loc = partition.vertex_map.to_local(u_g);

                    if (u_loc < 0 || u_loc >= N) {
                        cerr << "[E14] rank=" << rank_ << " peer=" << r << ": received dst u_g=" << u_g << " u_loc=" << u_loc << " out of range N=" << N << "\n";
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    float* gh_u = gh + static_cast<int64_t>(u_loc) * H;
                    for (int32_t h = 0; h < H; ++h)
                        gh_u[h] += payload[h];

                    ++total_remote_accum;
                }
                dst_cursor += n_dsts;
            }
        }  // chunk loop
    }  // rank loop

    cout << "[E14] rank=" << partition.rank<< " layer" << target_layer<< ": remote_accumulations=" << total_remote_accum << "\n";
}
