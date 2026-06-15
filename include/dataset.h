#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "config.h"

struct Dataset {
    int64_t num_nodes = 0;
    int64_t num_edges = 0;
    int32_t feature_dim = 0;
    int32_t num_classes = 0;

    torch::Tensor features;
    torch::Tensor labels;

    std::vector<int32_t> edge_src;
    std::vector<int32_t> edge_dst;

    std::vector<uint8_t> train_mask;
    std::vector<uint8_t> val_mask;
    std::vector<uint8_t> test_mask;
};

Dataset load_ogbn_products(const DatasetConfig& cfg);
