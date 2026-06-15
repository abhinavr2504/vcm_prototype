#pragma once
#include <vector>
#include <cstdint>
#include <torch/torch.h>
#include "gnn_layer.h"

// Owns a stack of GNNLayer objects — one per GNN layer.
// One GNNModel per Runtime instance.

class GNNModel {
public:
    GNNModel() = default;

    void init(const ModelConfig& cfg);

    const GNNLayer& layer(int32_t index) const { return layers_[index]; }
    GNNLayer&       layer(int32_t index)       { return layers_[index]; }

    bool classifier_initialized() const { return classifier_initialized_; }
    const torch::Tensor& classifier_W() const { return classifier_W_; }
    const torch::Tensor& classifier_b() const { return classifier_b_; }
    torch::Tensor& classifier_W_mut() { return classifier_W_; }
    torch::Tensor& classifier_b_mut() { return classifier_b_; }

    int32_t num_layers() const { return static_cast<int32_t>(layers_.size()); }

    std::vector<torch::Tensor> parameters();
    std::vector<torch::Tensor> parameters() const;

private:
    std::vector<GNNLayer> layers_;
    torch::Tensor classifier_W_;
    torch::Tensor classifier_b_;
    bool classifier_initialized_ = false;
};