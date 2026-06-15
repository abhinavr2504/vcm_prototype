#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include "config.h"

// Owns the parameters for a single GNN layer.
// No compute logic lives here — all forward math stays in ComputeEngine.
//
// Initialization:
//   W = torch::eye(hidden_dim)
//   b = torch::zeros({hidden_dim})

class GNNLayer {
public:
    GNNLayer() = default;

    void init(int32_t hidden_dim, LayerType type);

    const torch::Tensor& W() const { return W_; }
    const torch::Tensor& b() const { return b_; }

    torch::Tensor& W_mut() { return W_; }
    torch::Tensor& b_mut() { return b_; }

    LayerType layer_type() const { return layer_type_; }

    std::vector<torch::Tensor> parameters() { return {W_, b_}; }
    std::vector<torch::Tensor> parameters() const { return {W_, b_}; }

private:
    torch::Tensor W_;
    torch::Tensor b_;
    LayerType layer_type_ = LayerType::GCN;
};