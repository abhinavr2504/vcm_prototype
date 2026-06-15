#include "gnn_model.h"
#include <iostream>
#include <mpi.h>

using namespace std;

void GNNModel::init(const ModelConfig& cfg) {
    layers_.resize(cfg.num_layers);
    for (int32_t l = 0; l < cfg.num_layers; ++l)
        layers_[l].init(cfg.hidden_dim, cfg.layer_type);

    classifier_initialized_ = false;
    if (cfg.num_classes > 0) {
        // Small random weights break symmetry so the classifier differentiates
        // between classes from the first step. Bias stays at zero (standard).
        classifier_W_ = torch::randn({cfg.num_classes, cfg.hidden_dim}, torch::kFloat32) * 0.01f;
        classifier_b_ = torch::zeros({cfg.num_classes}, torch::kFloat32);

        // Broadcast from rank 0 so all ranks start with identical weights.
        MPI_Bcast(classifier_W_.data_ptr<float>(), classifier_W_.numel(),
                  MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(classifier_b_.data_ptr<float>(), classifier_b_.numel(),
                  MPI_FLOAT, 0, MPI_COMM_WORLD);

        classifier_W_.requires_grad_(true);
        classifier_b_.requires_grad_(true);
        classifier_initialized_ = true;
    }
}

vector<torch::Tensor> GNNModel::parameters() {
    vector<torch::Tensor> params;
    params.reserve(layers_.size() * 2);
    for (auto& layer : layers_) {
        vector<torch::Tensor> layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    if (classifier_initialized_) {
        params.push_back(classifier_W_);
        params.push_back(classifier_b_);
    }
    return params;
}

vector<torch::Tensor> GNNModel::parameters() const {
    vector<torch::Tensor> params;
    params.reserve(layers_.size() * 2);
    for (const auto& layer : layers_) {
        vector<torch::Tensor> layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    if (classifier_initialized_) {
        params.push_back(classifier_W_);
        params.push_back(classifier_b_);
    }
    return params;
}
