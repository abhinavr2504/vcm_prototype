#include "compute.h"
#include <cstring>  // for std::memcpy

void ComputeEngine::zero_aggr(float* aggr, int32_t hidden_dim) const {
    for (int32_t d = 0; d < hidden_dim; ++d) aggr[d] = 0.0f;
}

void ComputeEngine::aggregate(const float* src_hidden, float* aggr, int32_t hidden_dim) const {
    for (int32_t d = 0; d < hidden_dim; ++d) aggr[d] += src_hidden[d];
}

// Phase E1: Tensor-based linear + ReLU using LibTorch operations.
// No autograd, no backward, no gradients.
void ComputeEngine::linear_relu_tensor(const torch::Tensor& aggr_tensor,
                                       const torch::Tensor& W,
                                       const torch::Tensor& b,
                                       torch::Tensor&       out_tensor) const {
    // Compute: out = relu(W @ aggr + b)
    // aggr_tensor: [hidden_dim]
    // W: [hidden_dim, hidden_dim]
    // b: [hidden_dim]
    // out_tensor: [hidden_dim]
    
    // Linear: W @ aggr + b
    // torch::matmul(W, aggr_tensor) computes matrix-vector product
    torch::Tensor linear_out = torch::matmul(W, aggr_tensor) + b;
    
    // ReLU: max(0, x)
    out_tensor.copy_(torch::relu(linear_out));
}

// Phase E1: OpenMP-safe version using raw pointers.
// WORKAROUND: torch::tensor(std::vector) and torch::from_blob() crash in this environment.
// Use torch::empty() + element-wise copy instead.
void ComputeEngine::linear_relu_tensor_ptr(const float*         aggr,
                                           const torch::Tensor& W,
                                           const torch::Tensor& b,
                                           float*               out,
                                           int32_t              hidden_dim) const {
    // Create empty tensor and copy data element-wise.
    // Avoids torch::tensor(std::vector) and torch::from_blob() which crash.
    torch::Tensor aggr_tensor = torch::empty({hidden_dim}, torch::kFloat32);
    for (int32_t i = 0; i < hidden_dim; ++i) {
        aggr_tensor[i] = aggr[i];
    }
    
    // Compute: result = relu(W @ aggr + b)
    torch::Tensor result = torch::relu(torch::matmul(W, aggr_tensor) + b);
    
    // Copy result back to output buffer.
    for (int32_t i = 0; i < hidden_dim; ++i) {
        out[i] = result[i].item<float>();
    }
}

// Legacy raw-pointer interface (kept for compatibility).
void ComputeEngine::linear_relu(const float* aggr,
                                const float* W,
                                const float* b,
                                float*       out,
                                int32_t      hidden_dim) const {
    for (int32_t i = 0; i < hidden_dim; ++i) {
        float sum = b[i];
        for (int32_t j = 0; j < hidden_dim; ++j) {
            sum += W[i * hidden_dim + j] * aggr[j];
        }
        out[i] = (sum > 0.0f) ? sum : 0.0f;
    }
}
