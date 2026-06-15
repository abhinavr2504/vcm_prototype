#pragma once
#include <cstdint>
#include <torch/torch.h>

// Tensor math backend. Called exclusively from SuperstepExecutor.
// No execution control, no scheduling, no allocation.
//
// Phase E1: Migrated to LibTorch tensor operations.
//           Forward pass only. No autograd, no backward, no gradients.

struct ComputeEngine {
    // Zero the aggregation buffer for one vertex.
    void zero_aggr(float* aggr, int32_t hidden_dim) const;

    // Accumulate src_hidden into aggr in-place (mean-aggregation is handled
    // by the caller dividing by degree if needed; here we just sum).
    void aggregate(const float* src_hidden, float* aggr, int32_t hidden_dim) const;

    // Phase E1: Tensor-based linear + ReLU.
    // aggr_tensor: [hidden_dim] input aggregation
    // W: [hidden_dim, hidden_dim] weight matrix
    // b: [hidden_dim] bias vector
    // out_tensor: [hidden_dim] output (pre-allocated)
    // Computes: out = relu(W @ aggr + b)
    void linear_relu_tensor(const torch::Tensor& aggr_tensor,
                            const torch::Tensor& W,
                            const torch::Tensor& b,
                            torch::Tensor&       out_tensor) const;

    // Phase E1: OpenMP-safe tensor-based linear + ReLU using raw pointers.
    // Creates zero-copy tensor views via torch::from_blob (thread-safe).
    // aggr: [hidden_dim] input aggregation (raw pointer)
    // W: [hidden_dim, hidden_dim] weight matrix
    // b: [hidden_dim] bias vector
    // out: [hidden_dim] output (raw pointer, pre-allocated)
    // Computes: out = relu(W @ aggr + b)
    void linear_relu_tensor_ptr(const float*         aggr,
                                const torch::Tensor& W,
                                const torch::Tensor& b,
                                float*               out,
                                int32_t              hidden_dim) const;

    // Legacy raw-pointer interface (kept for compatibility during migration).
    // out[i] = relu( sum_j W[i*hidden_dim+j] * aggr[j] + b[i] )
    // W: [hidden_dim x hidden_dim] row-major
    // aggr, b, out: [hidden_dim]
    // out is pre-allocated; never allocated here.
    void linear_relu(const float* aggr,
                     const float* W,
                     const float* b,
                     float*       out,
                     int32_t      hidden_dim) const;
};
