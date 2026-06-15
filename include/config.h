#pragma once
#include <cstdint>
#include <string>

enum class OptimizerType {
    SGD,
    Adam
};

enum class LossType {
    MSE,
    CrossEntropy
};

enum class LayerType {
    GCN,
    GraphSAGE
};

// E9: Graph partitioning strategy
enum class PartitionMode {
    VertexBalanced,  // Equal vertex counts per rank (legacy)
    EdgeBalanced,    // Equal edge counts per rank (E9)
    HybridBalanced   // Equal hybrid cost per rank (E9.1)
};

// E12: Aggregation semantics descriptor.
// Stored at the config level — NOT per-vertex inside AggregationTrace —
// because it is a compile-time/config constant, fully recoverable later.
// Defined here so E13+ phases can query the aggregation type without
// looking it up from the runtime trace.
enum class AggregationType : uint8_t {
    SUM  = 0,  // current implementation: gradient flows to all contributors equally
    MEAN = 1,  // future: divide gradient by contributor count before routing
    MAX  = 2   // future: gradient flows only to argmax contributor
};

struct RunConfig {
    int32_t hidden_dim  = 4;
    int32_t num_layers  = 2;
    int32_t num_threads = 1;
    int32_t rank        = 0;
    int32_t world_size  = 1;
    PartitionMode partition_mode = PartitionMode::VertexBalanced;  // E9: default to vertex-balanced

    // E9.1: Hybrid cost model parameters
    // cost(vertex) = alpha * 1 + beta * degree(vertex)
    float partition_alpha = 10.0f;  // vertex overhead weight
    float partition_beta  = 1.0f;   // edge overhead weight

    // E14: Gradient exchange chunk size.
    // Sets the maximum number of (v, remote_rank) pairs processed per
    // MPI_Sendrecv call inside GradientExchange::exchange_into().
    // Peak memory during E14 ≈ 2 × e14_chunk_pairs × hidden_dim × 4 bytes.
    // At e14_chunk_pairs=200000 and hidden_dim=256: ~410 MB peak.
    // Override via config or a future CMake define (E14_CHUNK_PAIRS).
    int32_t e14_chunk_pairs = 200'000;
};

struct ModelConfig {
    int32_t hidden_dim  = 4;
    int32_t num_layers  = 2;
    LayerType layer_type = LayerType::GCN;
    int32_t num_classes = 0;
};

struct DatasetConfig {
    std::string root = "/home/hadoop/g2n2/tmp/dataset/ogbn_products";
    int32_t max_nodes = 300000;
    double  dataset_fraction = 0.0;
};

struct TrainingConfig {
    int32_t epochs = 1;
    int32_t patience = 10;
    float   learning_rate = 0.01f;
    OptimizerType optimizer_type = OptimizerType::SGD;
    LossType      loss_type      = LossType::MSE;  // Mean Squared Error (MSE)
    std::string best_checkpoint_path = "save_best.pt";
};
