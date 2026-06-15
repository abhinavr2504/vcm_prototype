#pragma once
#include <mpi.h>
#include <vector>

// E10: Per-layer timing breakdown
struct LayerTiming {
    double gather_time  = 0.0;
    double compute_time = 0.0;
    double scatter_time = 0.0;
    double mpi_time     = 0.0;
};

// E10.1: Per-layer activity and message profiling
struct LayerProfile {
    int32_t active_vertices      = 0;
    int64_t messages_sent        = 0;
    int64_t messages_received    = 0;
    int64_t remote_sent          = 0;
    int64_t remote_received      = 0;
};

// E10.2: Detailed compute and MPI breakdown
struct LayerComputeBreakdown {
    double aggregation_time   = 0.0;
    double linear_time        = 0.0;
    double activation_time    = 0.0;
    double hidden_update_time = 0.0;
    double snapshot_time      = 0.0;
};

struct LayerMPIBreakdown {
    double pack_time     = 0.0;
    double exchange_time = 0.0;
    double unpack_time   = 0.0;
};

struct LayerMemoryStats {
    int64_t aggr_buf_bytes    = 0;
    int64_t hidden_curr_bytes = 0;
    int64_t hidden_next_bytes = 0;
    int64_t snapshot_bytes    = 0;
};

// Lightweight timing infrastructure using MPI_Wtime().
// Tracks total runtime, communication time, and compute time.
// E10: Extended with per-layer breakdown.
// E10.1: Extended with per-layer activity and message profiling.
// E10.2: Extended with detailed compute and MPI breakdown.
struct RuntimeTiming {
    double total_time        = 0.0;
    double communication_time = 0.0;
    double compute_time      = 0.0;

    // Start markers (not part of reported state).
    double total_start        = 0.0;
    double communication_start = 0.0;
    double compute_start      = 0.0;

    // E10: Per-layer timing breakdown
    std::vector<LayerTiming> layer_timings;
    double current_gather_start  = 0.0;
    double current_compute_start = 0.0;
    double current_scatter_start = 0.0;
    double current_mpi_start     = 0.0;

    // E10.1: Per-layer activity and message profiling
    std::vector<LayerProfile> layer_profiles;

    // E10.2: Detailed compute and MPI breakdown
    std::vector<LayerComputeBreakdown> layer_compute_breakdown;
    std::vector<LayerMPIBreakdown> layer_mpi_breakdown;
    std::vector<LayerMemoryStats> layer_memory_stats;
    
    double current_aggregation_start = 0.0;
    double current_linear_start      = 0.0;
    double current_activation_start  = 0.0;
    double current_hidden_update_start = 0.0;
    double current_snapshot_start    = 0.0;
    double current_pack_start        = 0.0;
    double current_exchange_start    = 0.0;
    double current_unpack_start      = 0.0;

    void start_total() {
        total_start = MPI_Wtime();
    }

    void stop_total() {
        total_time = MPI_Wtime() - total_start;
    }

    void start_communication() {
        communication_start = MPI_Wtime();
    }

    void stop_communication() {
        communication_time += MPI_Wtime() - communication_start;
    }

    void start_compute() {
        compute_start = MPI_Wtime();
    }

    void stop_compute() {
        compute_time += MPI_Wtime() - compute_start;
    }

    // E10: Fine-grained layer timing
    void init_layer(int32_t layer_idx) {
        if (static_cast<int32_t>(layer_timings.size()) <= layer_idx) {
            layer_timings.resize(layer_idx + 1);
        }
    }

    void start_gather() {
        current_gather_start = MPI_Wtime();
    }

    void stop_gather(int32_t layer_idx) {
        init_layer(layer_idx);
        layer_timings[layer_idx].gather_time += MPI_Wtime() - current_gather_start;
    }

    void start_layer_compute() {
        current_compute_start = MPI_Wtime();
    }

    void stop_layer_compute(int32_t layer_idx) {
        init_layer(layer_idx);
        layer_timings[layer_idx].compute_time += MPI_Wtime() - current_compute_start;
    }

    void start_scatter() {
        current_scatter_start = MPI_Wtime();
    }

    void stop_scatter(int32_t layer_idx) {
        init_layer(layer_idx);
        layer_timings[layer_idx].scatter_time += MPI_Wtime() - current_scatter_start;
    }

    void start_mpi() {
        current_mpi_start = MPI_Wtime();
    }

    void stop_mpi(int32_t layer_idx) {
        init_layer(layer_idx);
        layer_timings[layer_idx].mpi_time += MPI_Wtime() - current_mpi_start;
    }

    // E10: Compute totals across all layers
    double total_gather_time() const {
        double sum = 0.0;
        for (const auto& lt : layer_timings) sum += lt.gather_time;
        return sum;
    }

    double total_layer_compute_time() const {
        double sum = 0.0;
        for (const auto& lt : layer_timings) sum += lt.compute_time;
        return sum;
    }

    double total_scatter_time() const {
        double sum = 0.0;
        for (const auto& lt : layer_timings) sum += lt.scatter_time;
        return sum;
    }

    double total_mpi_time() const {
        double sum = 0.0;
        for (const auto& lt : layer_timings) sum += lt.mpi_time;
        return sum;
    }

    // E10.1: Per-layer activity and message profiling
    void init_profile(int32_t layer_idx) {
        if (static_cast<int32_t>(layer_profiles.size()) <= layer_idx) {
            layer_profiles.resize(layer_idx + 1);
        }
    }

    void record_layer_profile(int32_t layer_idx, int32_t active_verts,
                             int64_t msgs_sent, int64_t msgs_recv,
                             int64_t remote_s, int64_t remote_r) {
        init_profile(layer_idx);
        layer_profiles[layer_idx].active_vertices = active_verts;
        layer_profiles[layer_idx].messages_sent = msgs_sent;
        layer_profiles[layer_idx].messages_received = msgs_recv;
        layer_profiles[layer_idx].remote_sent = remote_s;
        layer_profiles[layer_idx].remote_received = remote_r;
    }

    // E10.1: Compute totals across all layers
    int64_t total_active_vertices() const {
        int64_t sum = 0;
        for (const auto& lp : layer_profiles) sum += lp.active_vertices;
        return sum;
    }

    int64_t total_messages_sent() const {
        int64_t sum = 0;
        for (const auto& lp : layer_profiles) sum += lp.messages_sent;
        return sum;
    }

    int64_t total_messages_received() const {
        int64_t sum = 0;
        for (const auto& lp : layer_profiles) sum += lp.messages_received;
        return sum;
    }

    int64_t total_remote_sent() const {
        int64_t sum = 0;
        for (const auto& lp : layer_profiles) sum += lp.remote_sent;
        return sum;
    }

    int64_t total_remote_received() const {
        int64_t sum = 0;
        for (const auto& lp : layer_profiles) sum += lp.remote_received;
        return sum;
    }

    // E10.2: Detailed compute breakdown
    void init_compute_breakdown(int32_t layer_idx) {
        if (static_cast<int32_t>(layer_compute_breakdown.size()) <= layer_idx) {
            layer_compute_breakdown.resize(layer_idx + 1);
        }
    }

    void start_aggregation() {
        current_aggregation_start = MPI_Wtime();
    }

    void stop_aggregation(int32_t layer_idx) {
        init_compute_breakdown(layer_idx);
        layer_compute_breakdown[layer_idx].aggregation_time += MPI_Wtime() - current_aggregation_start;
    }

    void start_linear() {
        current_linear_start = MPI_Wtime();
    }

    void stop_linear(int32_t layer_idx) {
        init_compute_breakdown(layer_idx);
        layer_compute_breakdown[layer_idx].linear_time += MPI_Wtime() - current_linear_start;
    }

    void start_activation() {
        current_activation_start = MPI_Wtime();
    }

    void stop_activation(int32_t layer_idx) {
        init_compute_breakdown(layer_idx);
        layer_compute_breakdown[layer_idx].activation_time += MPI_Wtime() - current_activation_start;
    }

    void start_hidden_update() {
        current_hidden_update_start = MPI_Wtime();
    }

    void stop_hidden_update(int32_t layer_idx) {
        init_compute_breakdown(layer_idx);
        layer_compute_breakdown[layer_idx].hidden_update_time += MPI_Wtime() - current_hidden_update_start;
    }

    void start_snapshot() {
        current_snapshot_start = MPI_Wtime();
    }

    void stop_snapshot(int32_t layer_idx) {
        init_compute_breakdown(layer_idx);
        layer_compute_breakdown[layer_idx].snapshot_time += MPI_Wtime() - current_snapshot_start;
    }

    // E10.2: Detailed MPI breakdown
    void init_mpi_breakdown(int32_t layer_idx) {
        if (static_cast<int32_t>(layer_mpi_breakdown.size()) <= layer_idx) {
            layer_mpi_breakdown.resize(layer_idx + 1);
        }
    }

    void start_pack() {
        current_pack_start = MPI_Wtime();
    }

    void stop_pack(int32_t layer_idx) {
        init_mpi_breakdown(layer_idx);
        layer_mpi_breakdown[layer_idx].pack_time += MPI_Wtime() - current_pack_start;
    }

    void start_exchange() {
        current_exchange_start = MPI_Wtime();
    }

    void stop_exchange(int32_t layer_idx) {
        init_mpi_breakdown(layer_idx);
        layer_mpi_breakdown[layer_idx].exchange_time += MPI_Wtime() - current_exchange_start;
    }

    void start_unpack() {
        current_unpack_start = MPI_Wtime();
    }

    void stop_unpack(int32_t layer_idx) {
        init_mpi_breakdown(layer_idx);
        layer_mpi_breakdown[layer_idx].unpack_time += MPI_Wtime() - current_unpack_start;
    }

    // E10.2: Memory statistics
    void init_memory_stats(int32_t layer_idx) {
        if (static_cast<int32_t>(layer_memory_stats.size()) <= layer_idx) {
            layer_memory_stats.resize(layer_idx + 1);
        }
    }

    void record_memory_stats(int32_t layer_idx, int64_t aggr_buf, int64_t hidden_curr,
                            int64_t hidden_next, int64_t snapshot) {
        init_memory_stats(layer_idx);
        layer_memory_stats[layer_idx].aggr_buf_bytes = aggr_buf;
        layer_memory_stats[layer_idx].hidden_curr_bytes = hidden_curr;
        layer_memory_stats[layer_idx].hidden_next_bytes = hidden_next;
        layer_memory_stats[layer_idx].snapshot_bytes = snapshot;
    }

    // E10.2: Compute totals
    double total_aggregation_time() const {
        double sum = 0.0;
        for (const auto& lcb : layer_compute_breakdown) sum += lcb.aggregation_time;
        return sum;
    }

    double total_linear_time() const {
        double sum = 0.0;
        for (const auto& lcb : layer_compute_breakdown) sum += lcb.linear_time;
        return sum;
    }

    double total_activation_time() const {
        double sum = 0.0;
        for (const auto& lcb : layer_compute_breakdown) sum += lcb.activation_time;
        return sum;
    }

    double total_hidden_update_time() const {
        double sum = 0.0;
        for (const auto& lcb : layer_compute_breakdown) sum += lcb.hidden_update_time;
        return sum;
    }

    double total_snapshot_time() const {
        double sum = 0.0;
        for (const auto& lcb : layer_compute_breakdown) sum += lcb.snapshot_time;
        return sum;
    }

    double total_pack_time() const {
        double sum = 0.0;
        for (const auto& lmb : layer_mpi_breakdown) sum += lmb.pack_time;
        return sum;
    }

    double total_exchange_time() const {
        double sum = 0.0;
        for (const auto& lmb : layer_mpi_breakdown) sum += lmb.exchange_time;
        return sum;
    }

    double total_unpack_time() const {
        double sum = 0.0;
        for (const auto& lmb : layer_mpi_breakdown) sum += lmb.unpack_time;
        return sum;
    }

    int64_t peak_memory_bytes() const {
        int64_t peak = 0;
        for (const auto& lms : layer_memory_stats) {
            int64_t layer_total = lms.aggr_buf_bytes + lms.hidden_curr_bytes +
                                  lms.hidden_next_bytes + lms.snapshot_bytes;
            if (layer_total > peak) peak = layer_total;
        }
        return peak;
    }
};



