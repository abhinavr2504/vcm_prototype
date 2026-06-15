#include "gnn_layer.h"

void GNNLayer::init(int32_t hidden_dim, LayerType type) {
    W_ = torch::eye(hidden_dim, torch::kFloat32);
    b_ = torch::zeros({hidden_dim}, torch::kFloat32);
    W_.requires_grad_(true);
    b_.requires_grad_(true);
    layer_type_ = type;
}