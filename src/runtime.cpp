#include "runtime.h"
#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <mpi.h>
#include <torch/torch.h>

using namespace std;

void Runtime::init(RunConfig c, ModelConfig m, Partition p, const Dataset* dataset) {
    cfg       = c;
    model_cfg = m;
    partition = move(p);

    model.init(model_cfg);

    // If a real dataset is provided, copy its features into the local partition
    // vertices as their initial hidden state.
    if (dataset && dataset->features.defined()) {
        if (dataset->feature_dim != cfg.hidden_dim) {
            throw runtime_error("dataset feature_dim != cfg.hidden_dim");
        }

        const float* src_ptr = dataset->features.data_ptr<float>();
        const int32_t H = cfg.hidden_dim;
        for (int32_t lv = 0; lv < partition.num_vertices; ++lv) {
            const int32_t gv = partition.vertex_map.to_global(lv, cfg.rank);
            if (gv < 0 || gv >= dataset->num_nodes) continue;
            const float* src = src_ptr + gv * H;
            float* dst = partition.hcurr(lv);
            for (int32_t d = 0; d < H; ++d) dst[d] = src[d];
        }
    }
}

void Runtime::run() {
    partition.aggr_snapshots.clear();
    partition.aggr_masks.clear();
    partition.layer_cache.clear();

    // Pre-allocate one AggregationTrace slot per layer so superstep can
    // index directly by layer index.
    partition.aggr_traces.assign(cfg.num_layers, AggregationTrace{});

    timing.start_total();

    executor.seed(partition, &timing);

    for (int32_t layer = 0; layer < cfg.num_layers; ++layer) {
        const int32_t local_active = partition.frontier_curr.popcount();

        // All ranks must agree before any rank exits the loop.
        // MPI_Alltoall inside exchange() is a collective — divergent control
        // flow causes a permanent hang if any rank exits early.
        int32_t global_active = local_active;
        if (cfg.world_size > 1) {
            MPI_Allreduce(&local_active, &global_active,
                          1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        cout << "layer " << layer << "  active=" << local_active << "\n";

        if (global_active == 0) break;

        executor.run(partition, model.layer(layer), layer, &timing);

        partition.advance_hidden();
        partition.advance_frontier();
        partition.mpi_exchange.increment_superstep();
    }

    cout << "done  frontier=" << partition.frontier_curr.popcount() << "\n";

    timing.stop_total();
}

void Runtime::train_step(float lr) {
    if (partition.aggr_snapshots.empty()) {
        cout << "train_step: no snapshots available\n";
        return;
    }

    vector<torch::Tensor> params = model.parameters();
    for (auto& p : params) {
        if (p.grad().defined()) p.grad().zero_();
    }

    torch::Tensor total_loss = torch::zeros({}, torch::kFloat32);
    bool has_loss = false;

    const int32_t layer_count = min(
        cfg.num_layers,
        static_cast<int32_t>(partition.aggr_snapshots.size())
    );

    for (int32_t l = 0; l < layer_count; ++l) {
        torch::Tensor mask = partition.aggr_masks[l];
        torch::Tensor active_idx = torch::nonzero(mask).squeeze();
        if (active_idx.numel() == 0) continue;
        if (active_idx.dim() == 0) active_idx = active_idx.unsqueeze(0);

        torch::Tensor aggr_active =
            partition.aggr_snapshots[l].index_select(0, active_idx);

        torch::Tensor out = torch::relu(
            torch::matmul(aggr_active, model.layer(l).W().t()) + model.layer(l).b()
        );

        torch::Tensor loss = torch::mean(out * out);
        total_loss = total_loss + loss;
        has_loss = true;
    }

    if (!has_loss) {
        cout << "train_step: no active vertices\n";
        return;
    }

    total_loss.backward();

    {
        torch::NoGradGuard guard;
        for (auto& p : params) {
            if (!p.grad().defined()) continue;
            p.data().add_(-lr * p.grad());
        }
    }

    cout << fixed << setprecision(6)
         << "loss=" << total_loss.item<float>()
         << "  gradW=" << model.layer(0).W().grad().norm().item<float>()
         << "  gradB=" << model.layer(0).b().grad().norm().item<float>() << "\n";
}

void Runtime::print_summary() const {
    const RuntimeStatistics& stats = partition.mpi_exchange.statistics();

    const string sep = "========================================\n";

    cout << "\n" << sep
         << "RUNTIME SUMMARY — Rank " << cfg.rank << "/" << cfg.world_size << "\n"
         << sep
         << "Local vertices:           " << partition.num_vertices       << "\n"
         << "Local edges:              " << partition.graph.num_edges     << "\n"
         << "Local messages sent:      " << stats.local_messages_sent    << "\n"
         << "Local messages received:  " << stats.local_messages_received << "\n"
         << "Remote messages sent:     " << stats.remote_messages_sent   << "\n"
         << "Remote messages received: " << stats.remote_messages_received << "\n"
         << "Bytes sent:               " << stats.bytes_sent             << "\n"
         << "Bytes received:           " << stats.bytes_received         << "\n"
         << "Supersteps:               " << stats.supersteps             << "\n"
         << "\n"
         << fixed << setprecision(6)
         << "Timing:\n"
         << "  Total runtime:          " << timing.total_time         << " s\n"
         << "  Communication time:     " << timing.communication_time << " s\n"
         << "  Compute time:           " << timing.compute_time       << " s\n"
         << "\n";

    if (!timing.layer_timings.empty()) {
        cout << "Per-Layer Timing Breakdown:\n";
        for (size_t i = 0; i < timing.layer_timings.size(); ++i) {
            const LayerTiming& lt = timing.layer_timings[i];
            cout << "  Layer " << i << ":\n"
                 << "    Gather:  " << lt.gather_time  << " s\n"
                 << "    Compute: " << lt.compute_time << " s\n"
                 << "    Scatter: " << lt.scatter_time << " s\n"
                 << "    MPI:     " << lt.mpi_time     << " s\n";
        }
        cout << "  Totals across all layers:\n"
             << "    Gather:  " << timing.total_gather_time()         << " s\n"
             << "    Compute: " << timing.total_layer_compute_time()  << " s\n"
             << "    Scatter: " << timing.total_scatter_time()        << " s\n"
             << "    MPI:     " << timing.total_mpi_time()            << " s\n"
             << "\n";
    }

    if (!timing.layer_profiles.empty()) {
        cout << "Per-Layer Activity and Message Profiling:\n";
        for (size_t i = 0; i < timing.layer_profiles.size(); ++i) {
            const LayerProfile& lp = timing.layer_profiles[i];
            double active_fraction = (partition.num_vertices > 0)
                ? static_cast<double>(lp.active_vertices) / partition.num_vertices
                : 0.0;
            cout << "  Layer " << i << ":\n"
                 << "    Active vertices:     " << lp.active_vertices  << "\n"
                 << "    Active fraction:     " << setprecision(4) << active_fraction << "\n"
                 << "    Messages sent:       " << lp.messages_sent    << "\n"
                 << "    Messages received:   " << lp.messages_received << "\n"
                 << "    Remote sent:         " << lp.remote_sent      << "\n"
                 << "    Remote received:     " << lp.remote_received  << "\n";
        }
        cout << "  Totals across all layers:\n"
             << "    Active vertices:     " << timing.total_active_vertices()    << "\n"
             << "    Messages sent:       " << timing.total_messages_sent()      << "\n"
             << "    Messages received:   " << timing.total_messages_received()  << "\n"
             << "    Remote sent:         " << timing.total_remote_sent()        << "\n"
             << "    Remote received:     " << timing.total_remote_received()    << "\n"
             << "\n";
    }

    if (!timing.layer_compute_breakdown.empty() || !timing.layer_mpi_breakdown.empty()) {
        cout << "Detailed Runtime Cost Decomposition:\n";

        for (size_t i = 0; i < timing.layer_compute_breakdown.size(); ++i) {
            const LayerComputeBreakdown& lcb = timing.layer_compute_breakdown[i];
            const LayerMPIBreakdown& lmb = (i < timing.layer_mpi_breakdown.size())
                ? timing.layer_mpi_breakdown[i] : LayerMPIBreakdown{};
            const LayerMemoryStats& lms = (i < timing.layer_memory_stats.size())
                ? timing.layer_memory_stats[i] : LayerMemoryStats{};

            cout << setprecision(6)
                 << "  Layer " << i << ":\n"
                 << "    Compute Breakdown:\n"
                 << "      Aggregation:     " << lcb.aggregation_time  << " s\n"
                 << "      Linear:          " << lcb.linear_time       << " s\n"
                 << "      Activation:      " << lcb.activation_time   << " s\n"
                 << "      Hidden update:   " << lcb.hidden_update_time << " s\n"
                 << "      Snapshot:        " << lcb.snapshot_time     << " s\n"
                 << "    MPI Breakdown:\n"
                 << "      Pack:            " << lmb.pack_time         << " s\n"
                 << "      Exchange:        " << lmb.exchange_time     << " s\n"
                 << "      Unpack:          " << lmb.unpack_time       << " s\n"
                 << "    Memory (estimated):\n"
                 << "      aggr_buf:        " << lms.aggr_buf_bytes    << " bytes\n"
                 << "      hidden_curr:     " << lms.hidden_curr_bytes << " bytes\n"
                 << "      hidden_next:     " << lms.hidden_next_bytes << " bytes\n"
                 << "      snapshot:        " << lms.snapshot_bytes    << " bytes\n";
        }

        cout << "  Totals across all layers:\n"
             << "    Compute Breakdown:\n"
             << "      Aggregation:     " << timing.total_aggregation_time()  << " s\n"
             << "      Linear:          " << timing.total_linear_time()       << " s\n"
             << "      Activation:      " << timing.total_activation_time()   << " s\n"
             << "      Hidden update:   " << timing.total_hidden_update_time() << " s\n"
             << "      Snapshot:        " << timing.total_snapshot_time()     << " s\n"
             << "    MPI Breakdown:\n"
             << "      Pack:            " << timing.total_pack_time()         << " s\n"
             << "      Exchange:        " << timing.total_exchange_time()     << " s\n"
             << "      Unpack:          " << timing.total_unpack_time()       << " s\n"
             << "    Memory:\n"
             << "      Peak estimated:  " << timing.peak_memory_bytes()
             << " bytes (" << setprecision(2)
             << timing.peak_memory_bytes() / (1024.0 * 1024.0) << " MB)\n"
             << "\n";
    }

    GlobalStatistics global = collect_global_statistics(
        partition.num_vertices, partition.graph.num_edges, stats, cfg.world_size
    );

    if (cfg.rank == 0) {
        const char* partition_mode_str;
        switch (cfg.partition_mode) {
            case PartitionMode::VertexBalanced:  partition_mode_str = "Vertex-Balanced"; break;
            case PartitionMode::EdgeBalanced:    partition_mode_str = "Edge-Balanced";   break;
            case PartitionMode::HybridBalanced:  partition_mode_str = "Hybrid-Balanced"; break;
            default:                             partition_mode_str = "Unknown";          break;
        }

        cout << sep
             << "GLOBAL SUMMARY\n"
             << sep
             << "Partitioning mode:        " << partition_mode_str       << "\n"
             << "Total vertices:           " << global.total_vertices    << "\n"
             << "Total edges:              " << global.total_edges       << "\n"
             << "Total messages:           " << global.total_messages    << "\n"
             << "Total remote messages:    " << global.total_remote_messages << "\n"
             << "Total bytes sent:         " << global.total_bytes_sent  << "\n"
             << "Total bytes received:     " << global.total_bytes_received << "\n"
             << "\n"
             << setprecision(1)
             << "Load Balance:\n"
             << "  Vertices: min=" << global.min_vertices << "  max=" << global.max_vertices
             << "  avg=" << global.avg_vertices
             << "  imbalance=" << setprecision(3) << global.vertex_imbalance() << "\n"
             << "  Edges:    min=" << global.min_edges << "  max=" << global.max_edges
             << "  avg=" << setprecision(1) << global.avg_edges
             << "  imbalance=" << setprecision(3) << global.edge_imbalance() << "\n"
             << "\n"
             << "Communication:\n"
             << "  Remote fraction:        " << setprecision(4) << global.remote_fraction() << "\n"
             << sep;
    }
}