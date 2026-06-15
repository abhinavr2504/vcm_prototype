#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "config.h"
#include "runtime.h"
#include "dataset.h"
#include "local_backward.h"    // E13
#include "gradient_exchange.h" // E14

struct TrainingMetrics {
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;
};

class Trainer {
public:
    Trainer(Runtime& runtime, TrainingConfig config, const Dataset* dataset = nullptr);

    TrainingMetrics train();

private:
    Runtime&          runtime_;
    TrainingConfig    config_;
    const Dataset*    dataset_ = nullptr;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    GradientExchange  gradient_exchange_;  // E14: rank-batched gradient transport

    void init_optimizer();
    torch::Tensor compute_loss();

    // E13: Loss reconstruction using LayerCache[last_layer].aggregated with
    // requires_grad_(true). After calling loss.backward(), out_aggr_ref.grad()
    // contains ∂L/∂aggr[last_layer] which is then passed to LocalBackwardEngine.
    // Falls back to null Tensor if LayerCache is not populated.
    torch::Tensor compute_loss_e13(torch::Tensor& out_aggr_ref);

    TrainingMetrics compute_metrics(bool use_train_mask, bool use_val_mask);
    std::pair<torch::Tensor, torch::Tensor> build_indices_and_labels(
        const std::vector<uint8_t>& mask,
        const torch::Tensor& active_mask,
        int32_t& out_count) const;
    void reset_partition_for_epoch();
};
