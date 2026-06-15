#include "dataset.h"
#include <zlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace {

bool is_numeric_line(const string& line) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size()) return false;
    const char c = line[i];
    return (c == '-' || c == '+' || (c >= '0' && c <= '9'));
}

bool parse_int64_token(const char*& p, int64_t& out) {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\n' || *p == '\r') return false;
    char* end = nullptr;
    const long long val = strtoll(p, &end, 10);
    if (end == p) return false;
    out = static_cast<int64_t>(val);
    p = end;
    if (*p == ',') ++p;
    return true;
}

bool parse_float_token(const char*& p, float& out) {
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\n' || *p == '\r') return false;
    char* end = nullptr;
    const float val = strtof(p, &end);
    if (end == p) return false;
    out = val;
    p = end;
    if (*p == ',') ++p;
    return true;
}

template <typename Fn>
void read_gz_lines(const string& path, Fn&& fn) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) throw runtime_error("Cannot open: " + path);

    const int kBufSize = 1 << 20;
    char buffer[kBufSize];
    string line;

    while (gzgets(file, buffer, kBufSize)) {
        line.append(buffer);
        if (!line.empty() && line.back() == '\n') {
            fn(line);
            line.clear();
        }
    }
    if (!line.empty()) fn(line);

    gzclose(file);
}

int64_t count_numeric_rows(const string& path) {
    int64_t count = 0;
    read_gz_lines(path, [&](const string& line) {
        if (!is_numeric_line(line)) return;
        count++;
    });
    return count;
}

}  // namespace

Dataset load_ogbn_products(const DatasetConfig& cfg) {
    Dataset dataset;

    const string root = cfg.root;
    const string feat_path = root + "/raw/node-feat.csv.gz";
    const string label_path = root + "/raw/node-label.csv.gz";
    const string edge_path = root + "/raw/edge.csv.gz";
    const string train_path = root + "/split/sales_ranking/train.csv.gz";
    const string val_path = root + "/split/sales_ranking/valid.csv.gz";
    const string test_path = root + "/split/sales_ranking/test.csv.gz";

    int64_t total_nodes = count_numeric_rows(feat_path);
    if (total_nodes <= 0) {
        throw runtime_error("node-feat.csv.gz has no data rows");
    }

    int64_t effective_nodes = total_nodes;
    if (cfg.dataset_fraction > 0.0 && cfg.dataset_fraction < 1.0) {
        effective_nodes = max<int64_t>(1, static_cast<int64_t>(total_nodes * cfg.dataset_fraction));
    }
    if (cfg.max_nodes > 0) {
        effective_nodes = min<int64_t>(effective_nodes, cfg.max_nodes);
    }

    dataset.num_nodes = effective_nodes;
    dataset.num_edges = 0;
    dataset.train_mask.assign(effective_nodes, 0);
    dataset.val_mask.assign(effective_nodes, 0);
    dataset.test_mask.assign(effective_nodes, 0);

    int32_t feature_dim = 0;
    torch::Tensor features;
    float* features_ptr = nullptr;

    int64_t row_idx = 0;
    read_gz_lines(feat_path, [&](const string& line) {
        if (!is_numeric_line(line)) return;
        if (row_idx >= effective_nodes) return;

        const char* p = line.c_str();
        vector<float> row;
        float val = 0.0f;
        while (parse_float_token(p, val)) row.push_back(val);

        if (feature_dim == 0) {
            feature_dim = static_cast<int32_t>(row.size());
            dataset.feature_dim = feature_dim;
            features = torch::zeros({effective_nodes, feature_dim}, torch::kFloat32);
            features_ptr = features.data_ptr<float>();
        }

        if (static_cast<int32_t>(row.size()) != feature_dim) {
            throw runtime_error("feature dimension mismatch in node-feat.csv.gz");
        }

        float* dst = features_ptr + row_idx * feature_dim;
        for (int32_t i = 0; i < feature_dim; ++i) dst[i] = row[i];
        row_idx++;
    });

    if (!features.defined()) {
        throw runtime_error("failed to load node features");
    }

    dataset.features = features;

    torch::Tensor labels = torch::zeros({effective_nodes}, torch::kInt64);
    int64_t* labels_ptr = labels.data_ptr<int64_t>();
    int64_t max_label = -1;

    int64_t label_idx = 0;
    read_gz_lines(label_path, [&](const string& line) {
        if (!is_numeric_line(line)) return;
        if (label_idx >= effective_nodes) return;

        const char* p = line.c_str();
        int64_t label = -1;
        if (!parse_int64_token(p, label)) return;

        labels_ptr[label_idx] = label;
        if (label > max_label) max_label = label;
        label_idx++;
    });

    dataset.labels = labels;
    dataset.num_classes = (max_label >= 0) ? static_cast<int32_t>(max_label + 1) : 0;

    auto load_mask = [&](const string& path, vector<uint8_t>& mask) {
        read_gz_lines(path, [&](const string& line) {
            if (!is_numeric_line(line)) return;
            const char* p = line.c_str();
            int64_t node_id = -1;
            if (!parse_int64_token(p, node_id)) return;
            if (node_id < 0 || node_id >= effective_nodes) return;
            mask[node_id] = 1;
        });
    };

    load_mask(train_path, dataset.train_mask);
    load_mask(val_path, dataset.val_mask);
    load_mask(test_path, dataset.test_mask);

    dataset.edge_src.clear();
    dataset.edge_dst.clear();
    dataset.edge_src.reserve(1024);
    dataset.edge_dst.reserve(1024);

    int64_t edge_count = 0;
    read_gz_lines(edge_path, [&](const string& line) {
        if (!is_numeric_line(line)) return;
        const char* p = line.c_str();
        int64_t src = -1;
        int64_t dst = -1;
        if (!parse_int64_token(p, src)) return;
        if (!parse_int64_token(p, dst)) return;
        if (src < 0 || dst < 0) return;
        if (src >= effective_nodes || dst >= effective_nodes) return;

        dataset.edge_src.push_back(static_cast<int32_t>(src));
        dataset.edge_dst.push_back(static_cast<int32_t>(dst));
        edge_count++;
    });

    dataset.num_edges = edge_count;

    return dataset;
}
