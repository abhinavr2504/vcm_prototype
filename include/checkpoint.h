#pragma once
#include <string>
#include "gnn_model.h"

// Checkpoint persistence for model parameters only.
// Optimizer state is intentionally not saved.
struct Checkpoint {
    static void save_checkpoint(const GNNModel& model, const std::string& path);
    static void load_checkpoint(GNNModel& model, const std::string& path);
};
