#include "trainer.h"
#include "checkpoint.h"
#include "distributed_optimizer.h"
#include "local_backward.h"
#include "gradient_exchange.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

using namespace std;

namespace {

float accuracy_from_logits(const torch::Tensor& logits, const torch::Tensor& labels) {
    if (logits.numel() == 0) return 0.0f;
    torch::Tensor pred = logits.argmax(1);
    torch::Tensor correct = pred.eq(labels).to(torch::kFloat32);
    return correct.mean().item<float>();
}

// Weighted MPI all-reduce so every rank ends up with the global-average
// loss and accuracy across all partitions. Ranks that hold no labelled
// nodes pass local_count = 0; their zero-weight entry is still required
// so the collective doesn't hang.
void reduce_metric_pair(float& loss, float& acc, int32_t local_count) {
    float send[3] = {
        loss * static_cast<float>(local_count),
        acc  * static_cast<float>(local_count),
        static_cast<float>(local_count)
    };
    float recv[3] = {0.f, 0.f, 0.f};
    MPI_Allreduce(send, recv, 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (recv[2] > 0.f) {
        loss = recv[0] / recv[2];
        acc  = recv[1] / recv[2];
    } else {
        loss = 0.f;
        acc  = 0.f;
    }
}

} // namespace

Trainer::Trainer(Runtime& runtime, TrainingConfig config, const Dataset* dataset)
    : runtime_(runtime)
    , config_(config)
    , dataset_(dataset)
    , gradient_exchange_(runtime.cfg.rank,
                         runtime.cfg.world_size,
                         runtime.cfg.e14_chunk_pairs)
{
    init_optimizer();
}

void Trainer::init_optimizer() {
    vector<torch::Tensor> params = runtime_.model.parameters();

    if (config_.optimizer_type == OptimizerType::Adam) {
        auto options = torch::optim::AdamOptions(config_.learning_rate);
        optimizer_ = make_unique<torch::optim::Adam>(params, options);
    } else {
        auto options = torch::optim::SGDOptions(config_.learning_rate);
        optimizer_ = make_unique<torch::optim::SGD>(params, options);
    }
}

void Trainer::reset_partition_for_epoch() {
    Partition& p = runtime_.partition;
    const int32_t N = p.num_vertices;

    for (int32_t b = 0; b < 2; ++b) {
        for (auto& slot : p.inbox.bufs[b]) slot.clear();
    }
    p.inbox.read_idx = 0;
    p.inbox.write_idx = 1;

    p.frontier_curr.clear();
    p.frontier_next.clear();
    for (int32_t v = 0; v < N; ++v) {
        p.frontier_curr.set(v);
        p.vertex_state[v].active = true;
    }

    for (auto& outbox : p.thread_outboxes) outbox.clear();
}

pair<torch::Tensor, torch::Tensor> Trainer::build_indices_and_labels(
    const vector<uint8_t>& mask,
    const torch::Tensor& active_mask,
    int32_t& out_count) const
{
    const int32_t N = runtime_.partition.num_vertices;
    const bool* active_ptr = active_mask.data_ptr<bool>();

    int32_t count = 0;
    for (int32_t lv = 0; lv < N; ++lv) {
        if (!active_ptr[lv]) continue;
        const int32_t gv = runtime_.partition.vertex_map.to_global(lv, runtime_.cfg.rank);
        if (gv < 0 || gv >= static_cast<int32_t>(mask.size())) continue;
        if (mask[gv]) count++;
    }

    out_count = count;
    if (count == 0) return {torch::Tensor(), torch::Tensor()};

    torch::Tensor local_idx  = torch::empty({count}, torch::kInt64);
    torch::Tensor global_idx = torch::empty({count}, torch::kInt64);
    int64_t* local_ptr  = local_idx.data_ptr<int64_t>();
    int64_t* global_ptr = global_idx.data_ptr<int64_t>();

    int32_t pos = 0;
    for (int32_t lv = 0; lv < N; ++lv) {
        if (!active_ptr[lv]) continue;
        const int32_t gv = runtime_.partition.vertex_map.to_global(lv, runtime_.cfg.rank);
        if (gv < 0 || gv >= static_cast<int32_t>(mask.size())) continue;
        if (!mask[gv]) continue;
        local_ptr[pos]  = lv;
        global_ptr[pos] = gv;
        pos++;
    }

    torch::Tensor labels = dataset_->labels.index_select(0, global_idx);
    return {local_idx, labels};
}

TrainingMetrics Trainer::compute_metrics(bool use_train_mask, bool use_val_mask) {
    TrainingMetrics metrics;
    Partition& p = runtime_.partition;

    if (!dataset_ || !runtime_.model.classifier_initialized()) return metrics;

    const int32_t layer_idx = min(
        runtime_.cfg.num_layers - 1,
        static_cast<int32_t>(p.aggr_snapshots.size()) - 1
    );

    if (layer_idx < 0) return metrics;

    torch::Tensor active_mask = p.aggr_masks[layer_idx];

    // Compute logits for a given set of local vertex indices.
    auto compute_logits = [&](const torch::Tensor& local_idx) -> torch::Tensor {
        torch::Tensor aggr_active = p.aggr_snapshots[layer_idx].index_select(0, local_idx);
        torch::Tensor hidden = torch::relu(
            torch::matmul(aggr_active, runtime_.model.layer(layer_idx).W().t())
            + runtime_.model.layer(layer_idx).b()
        );
        return torch::matmul(hidden, runtime_.model.classifier_W().t())
               + runtime_.model.classifier_b();
    };

    if (use_train_mask) {
        int32_t count = 0;
        auto [local_idx, labels] = build_indices_and_labels(dataset_->train_mask, active_mask, count);
        if (count > 0) {
            torch::Tensor logits = compute_logits(local_idx);
            metrics.train_loss     = torch::nn::functional::cross_entropy(logits, labels).item<float>();
            metrics.train_accuracy = accuracy_from_logits(logits, labels);
        }
        reduce_metric_pair(metrics.train_loss, metrics.train_accuracy, count);
    }

    if (use_val_mask) {
        int32_t count = 0;
        auto [local_idx, labels] = build_indices_and_labels(dataset_->val_mask, active_mask, count);
        if (count > 0) {
            torch::Tensor logits = compute_logits(local_idx);
            metrics.val_loss     = torch::nn::functional::cross_entropy(logits, labels).item<float>();
            metrics.val_accuracy = accuracy_from_logits(logits, labels);
        }
        reduce_metric_pair(metrics.val_loss, metrics.val_accuracy, count);
    }

    return metrics;
}

torch::Tensor Trainer::compute_loss() {
    if (!dataset_) return torch::Tensor();
    if (!runtime_.model.classifier_initialized()) return torch::Tensor();

    Partition& p = runtime_.partition;
    const int32_t layer_idx = min(
        runtime_.cfg.num_layers - 1,
        static_cast<int32_t>(p.aggr_snapshots.size()) - 1
    );
    if (layer_idx < 0) return torch::Tensor();

    torch::Tensor active_mask = p.aggr_masks[layer_idx];

    int32_t count = 0;
    auto [local_idx, labels] = build_indices_and_labels(dataset_->train_mask, active_mask, count);

    if (count == 0) {
        // Ranks with no training samples must produce gradients for the same
        // parameters as the normal forward pass to prevent MPI collective divergence.
        torch::Tensor zero_loss = torch::zeros({1}, torch::kFloat32).squeeze();
        zero_loss = zero_loss + runtime_.model.layer(layer_idx).W().sum() * 0.0f;
        zero_loss = zero_loss + runtime_.model.layer(layer_idx).b().sum() * 0.0f;
        zero_loss = zero_loss + runtime_.model.classifier_W().sum() * 0.0f;
        zero_loss = zero_loss + runtime_.model.classifier_b().sum() * 0.0f;
        return zero_loss;
    }

    torch::Tensor aggr_active = p.aggr_snapshots[layer_idx].index_select(0, local_idx);
    torch::Tensor hidden = torch::relu(
        torch::matmul(aggr_active, runtime_.model.layer(layer_idx).W().t())
        + runtime_.model.layer(layer_idx).b()
    );
    torch::Tensor logits = torch::matmul(hidden, runtime_.model.classifier_W().t())
                          + runtime_.model.classifier_b();
    return torch::nn::functional::cross_entropy(logits, labels);
}

// E13: Reconstruct loss from LayerCache[last_layer].aggregated.
//
// aggr_last is created with requires_grad_(true) + retain_grad() so that
// after loss.backward(), aggr_last.grad() contains dL/d_aggr[last_layer].
// This gradient is then passed to LocalBackwardEngine::run() to propagate
// back through local graph topology to Layer0 parameters.
//
// count==0 path: aggr_last is touched in the zero_loss computation graph so
//   retain_grad() still populates aggr_last.grad() (as zero) after backward.
//   LocalBackwardEngine receives zero grad and returns zero grad_W0/b0,
//   ensuring all ranks call DistributedOptimizer with defined gradients.
torch::Tensor Trainer::compute_loss_e13(torch::Tensor& out_aggr_ref) {
    if (!dataset_) return torch::Tensor();
    if (!runtime_.model.classifier_initialized()) return torch::Tensor();

    Partition& p = runtime_.partition;
    const int32_t last_layer = static_cast<int32_t>(p.layer_cache.size()) - 1;
    if (last_layer < 0) return torch::Tensor();
    if (!p.layer_cache[last_layer].aggregated.defined()) return torch::Tensor();

    // Build the aggr leaf with requires_grad so grad flows back to it.
    torch::Tensor aggr_last = p.layer_cache[last_layer].aggregated
                                  .detach()
                                  .requires_grad_(true);
    aggr_last.retain_grad();
    out_aggr_ref = aggr_last;

    torch::Tensor active_mask_lc = p.layer_cache[last_layer].active_mask;

    int32_t count = 0;
    auto [local_idx, labels] = build_indices_and_labels(
        dataset_->train_mask, active_mask_lc, count);

    if (count == 0) {
        // Touch aggr_last and all relevant parameters so their grads are
        // defined (zero) after backward, preventing MPI collective divergence.
        torch::Tensor zero_loss =
            aggr_last.sum() * 0.0f
            + runtime_.model.layer(last_layer).W().sum()    * 0.0f
            + runtime_.model.layer(last_layer).b().sum()    * 0.0f
            + runtime_.model.classifier_W().sum()           * 0.0f
            + runtime_.model.classifier_b().sum()           * 0.0f;
        return zero_loss.squeeze();
    }

    torch::Tensor aggr_active = aggr_last.index_select(0, local_idx);
    torch::Tensor hidden = torch::relu(
        torch::matmul(aggr_active, runtime_.model.layer(last_layer).W().t())
        + runtime_.model.layer(last_layer).b()
    );
    torch::Tensor logits =
        torch::matmul(hidden, runtime_.model.classifier_W().t())
        + runtime_.model.classifier_b();

    return torch::nn::functional::cross_entropy(logits, labels);
}

TrainingMetrics Trainer::train() {
    TrainingMetrics metrics;

    reset_partition_for_epoch();
    runtime_.run();

    float best_val_accuracy = -1.0f;
    int32_t best_epoch = 0;
    int32_t epochs_no_improve = 0;
    int32_t epochs_run = 0;

    const bool is_rank0 = (runtime_.cfg.rank == 0);

    for (int32_t epoch = 1; epoch <= config_.epochs; ++epoch) {
        epochs_run = epoch;

        optimizer_->zero_grad();

        // E13: Use LayerCache reconstruction when available.
        // If the forward cache (E11) and reverse graph (E12) are populated,
        // use compute_loss_e13 so grad_aggr[last_layer] is accessible after
        // backward. Otherwise fall back to the E3 snapshot path.
        const int32_t num_layers = runtime_.cfg.num_layers;
        Partition& p = runtime_.partition;
        const bool e13_ready =
            !p.layer_cache.empty() &&
            static_cast<int32_t>(p.layer_cache.size()) >= num_layers &&
            p.reverse_graph.built() &&
            num_layers >= 2;

        torch::Tensor aggr_last_ref;
        torch::Tensor loss;

        if (e13_ready) {
            loss = compute_loss_e13(aggr_last_ref);
        } else {
            loss = compute_loss();
        }

        loss.backward();

        // E13 + E14: Backward reconstruction through local graph topology.
        // Phase A (scatter_only):  local ReverseCSR edges → grad_h (partial)
        // Phase B (exchange_into): MPI rank-batched exchange → grad_h complete
        // Phase C (linear_phase):  ReLU gate + matmul → grad_W0, grad_b0
        //
        // world_size==1: exchange_into is a no-op; result equals E13-only output.
        if (e13_ready
            && aggr_last_ref.defined()
            && aggr_last_ref.grad().defined()) {

            const int32_t target_layer = num_layers - 1;

            // Phase A: scatter local in-edges → partial grad_h
            torch::Tensor grad_h = LocalBackwardEngine::scatter_only(
                p, target_layer, aggr_last_ref.grad());

            // Phase B: MPI exchange adds remote contributions to grad_h
            gradient_exchange_.exchange_into(
                p, target_layer, aggr_last_ref.grad(), grad_h);

            // Phase C: gate by ReLU mask, matmul → grad_W0, grad_b0
            auto [grad_W0, grad_b0] = LocalBackwardEngine::linear_phase(
                p, target_layer - 1, grad_h);

            {
                torch::NoGradGuard ng;
                auto& W0 = runtime_.model.layer(0).W();
                auto& b0 = runtime_.model.layer(0).b();
                W0.mutable_grad() = grad_W0.clone();
                b0.mutable_grad() = grad_b0.clone();
            }
        }

        // E8: Average gradients across all ranks before the optimizer step.
        DistributedOptimizer::average_gradients(runtime_.model.parameters());

        optimizer_->step();

        metrics = compute_metrics(true, false);

        if (is_rank0) {
            cout << fixed << setprecision(6)
                 << "Epoch " << epoch << "\n"
                 << "Train Loss: "     << metrics.train_loss     << "\n"
                 << "Train Accuracy: " << metrics.train_accuracy << "\n";
        }
        {
            torch::NoGradGuard guard;
            TrainingMetrics val_metrics = compute_metrics(false, true);
            metrics.val_loss = val_metrics.val_loss;
            metrics.val_accuracy = val_metrics.val_accuracy;
        }
        if (is_rank0) {
            cout << "Validation Loss: "     << metrics.val_loss     << "\n"
                 << "Validation Accuracy: " << metrics.val_accuracy << "\n";
        }

        // metrics.val_accuracy is a global average, so early stopping and
        // checkpoint decisions are identical on every rank.
        if (metrics.val_accuracy > best_val_accuracy) {
            best_val_accuracy = metrics.val_accuracy;
            best_epoch = epoch;
            epochs_no_improve = 0;
            MPI_Barrier(MPI_COMM_WORLD);
        } else {
            epochs_no_improve++;
            if (epochs_no_improve >= config_.patience) {
                if (is_rank0) {
                    cout << "Early stopping at epoch " << epoch << "\n";
                }
                break;
            }
        }
    }

    if (best_epoch > 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        torch::NoGradGuard guard;

        Partition& p = runtime_.partition;
        const int32_t layer_idx = min(
            runtime_.cfg.num_layers - 1,
            static_cast<int32_t>(p.aggr_snapshots.size()) - 1
        );

        float test_accuracy = 0.0f;
        {
            torch::Tensor active_mask = p.aggr_masks[layer_idx];
            int32_t count = 0;
            auto [local_idx, labels] = build_indices_and_labels(dataset_->test_mask, active_mask, count);
            if (count > 0) {
                torch::Tensor aggr_active = p.aggr_snapshots[layer_idx].index_select(0, local_idx);
                torch::Tensor hidden = torch::relu(
                    torch::matmul(aggr_active, runtime_.model.layer(layer_idx).W().t())
                    + runtime_.model.layer(layer_idx).b()
                );
                torch::Tensor logits = torch::matmul(hidden, runtime_.model.classifier_W().t())
                                      + runtime_.model.classifier_b();
                test_accuracy = accuracy_from_logits(logits, labels);
            }
            float dummy_loss = 0.f;
            reduce_metric_pair(dummy_loss, test_accuracy, count);
        }

        if (is_rank0) {
            cout << fixed << setprecision(6)
                 << "Best Epoch: "               << best_epoch        << "\n"
                 << "Best Validation Accuracy: " << best_val_accuracy  << "\n"
                 << "Test Accuracy: "            << test_accuracy      << "\n"
                 << "Total Epochs Run: "         << epochs_run         << "\n";
        }
    }

    return metrics;
}
