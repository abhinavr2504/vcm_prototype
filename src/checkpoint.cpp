#include "checkpoint.h"
#include <stdexcept>

using namespace std;

void Checkpoint::save_checkpoint(const GNNModel& model, const string& path) {
    vector<torch::Tensor> params = model.parameters();
    torch::save(params, path);
}

void Checkpoint::load_checkpoint(GNNModel& model, const string& path) {
    vector<torch::Tensor> params;
    torch::load(params, path);

    const int32_t expected = model.classifier_initialized()
                                 ? (model.num_layers() * 2 + 2)
                                 : (model.num_layers() * 2);
    if (static_cast<int32_t>(params.size()) != expected) {
        throw runtime_error("checkpoint parameter count mismatch");
    }

    int32_t idx = 0;
    {
        torch::NoGradGuard guard;
        for (int32_t l = 0; l < model.num_layers(); ++l) {
            model.layer(l).W_mut().copy_(params[idx++]);
            model.layer(l).b_mut().copy_(params[idx++]);
        }
        if (model.classifier_initialized()) {
            model.classifier_W_mut().copy_(params[idx++]);
            model.classifier_b_mut().copy_(params[idx++]);
        }
    }
}
