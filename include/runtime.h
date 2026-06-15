#pragma once
#include <cstdint>
#include "config.h"
#include "partition.h"
#include "superstep.h"
#include "statistics.h"
#include "timing.h"
#include "gnn_model.h"

struct Dataset;

struct Runtime {
    RunConfig         cfg;
    ModelConfig       model_cfg;
    Partition         partition;
    SuperstepExecutor executor;
    GNNModel          model;      // Phase E2: owns all layer parameters

    RuntimeTiming timing;

    void init(RunConfig c, ModelConfig m, Partition p, const Dataset* dataset = nullptr);
    void run();
    void train_step(float lr);

    // Phase D: print distributed runtime summary.
    void print_summary() const;
};