#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <mpi.h>
#include "config.h"
#include "dataset.h"
#include "loader.h"
#include "runtime.h"
#include "trainer.h"
#include "checkpoint.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const char* graph_path = "graphs/toy.graph";
    const char* save_path = nullptr;
    const char* load_path = nullptr;
    const char* dataset_name = nullptr;
    const char* dataset_root = nullptr;
    const char* partition_mode = nullptr;
    float partition_alpha = 10.0f;
    float partition_beta = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--dataset-root" && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "--load" && i + 1 < argc) {
            load_path = argv[++i];
        } else if (arg == "--partition" && i + 1 < argc) {
            partition_mode = argv[++i];
        } else if (arg == "--partition-alpha" && i + 1 < argc) {
            partition_alpha = atof(argv[++i]);
        } else if (arg == "--partition-beta" && i + 1 < argc) {
            partition_beta = atof(argv[++i]);
        } else if (arg.rfind("--", 0) != 0) {
            graph_path = argv[i];
        }
    }

    RunConfig cfg;
    cfg.hidden_dim = 4;
    cfg.num_layers = 2;
    cfg.partition_alpha = partition_alpha;
    cfg.partition_beta = partition_beta;

    if (partition_mode) {
        string mode_str = partition_mode;
        if (mode_str == "vertex") {
            cfg.partition_mode = PartitionMode::VertexBalanced;
        } else if (mode_str == "edge") {
            cfg.partition_mode = PartitionMode::EdgeBalanced;
        } else if (mode_str == "hybrid") {
            cfg.partition_mode = PartitionMode::HybridBalanced;
        } else {
            cerr << "Invalid --partition mode: " << partition_mode
                 << " (use 'vertex', 'edge', or 'hybrid')" << endl;
            MPI_Finalize();
            return 1;
        }
    }

    ModelConfig mcfg;
    mcfg.hidden_dim = cfg.hidden_dim;
    mcfg.num_layers = cfg.num_layers;
    mcfg.layer_type = LayerType::GCN;
    mcfg.num_classes = 0;

    {
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        cfg.rank = static_cast<int32_t>(mpi_rank);
        cfg.world_size = static_cast<int32_t>(mpi_size);
    }

    // Read OMP_NUM_THREADS so that thread_outboxes.size() always matches
    // the actual OpenMP thread count. Invariant: num_threads >= 1.
    const char* omp_env = getenv("OMP_NUM_THREADS");
    cfg.num_threads = (omp_env != nullptr) ? max(1, atoi(omp_env)) : 1;

    const bool is_rank0 = (cfg.rank == 0);

    int ret = 0;

    try {
        if (dataset_name && string(dataset_name) == "ogbn-products") {
            DatasetConfig dcfg;
            Dataset dataset = load_ogbn_products(dcfg);

            if (is_rank0) {
                int64_t train_count = 0;
                int64_t val_count = 0;
                int64_t test_count = 0;
                for (size_t i = 0; i < dataset.train_mask.size(); ++i) {
                    train_count += dataset.train_mask[i] ? 1 : 0;
                    val_count   += dataset.val_mask[i]   ? 1 : 0;
                    test_count  += dataset.test_mask[i]  ? 1 : 0;
                }

                cout << "Dataset loaded\n"
                     << "Nodes: "    << dataset.num_nodes << "\n"
                     << "Edges: "    << dataset.num_edges << "\n"
                     << "Features: " << dataset.num_nodes << " x " << dataset.feature_dim << "\n"
                     << "Classes: "  << dataset.num_classes << "\n"
                     << "Train nodes: "      << train_count << "\n"
                     << "Validation nodes: " << val_count   << "\n"
                     << "Test nodes: "       << test_count  << "\n";

                const int64_t label_print = min<int64_t>(dataset.num_nodes, 5);
                cout << "Labels: ";
                for (int64_t i = 0; i < label_print; ++i) {
                    const int64_t label = dataset.labels[i].item<int64_t>();
                    cout << label;
                    if (i + 1 < label_print) cout << ", ";
                }
                cout << "\nFeature dim: " << dataset.feature_dim << "\n";
            }

            MPI_Finalize();
            return 0;
        }

        Dataset dataset;
        bool use_dataset = false;
        if (dataset_root) {
            DatasetConfig dcfg;
            dcfg.root = dataset_root;
            dataset = load_ogbn_products(dcfg);
            use_dataset = true;

            cfg.hidden_dim = dataset.feature_dim;
            mcfg.hidden_dim = cfg.hidden_dim;
            mcfg.num_classes = dataset.num_classes;

            if (is_rank0) {
                cout << "Dataset nodes: "     << dataset.num_nodes  << "\n"
                     << "Dataset edges: "     << dataset.num_edges  << "\n"
                     << "Feature dimension: " << dataset.feature_dim << "\n";
            }
        }

        Partition p = use_dataset
                          ? load_partition_from_dataset(dataset, cfg)
                          : load_partition(graph_path, cfg);

        cout << "[rank " << cfg.rank << "/" << cfg.world_size << "]"
             << "  local_vertices=" << p.num_vertices
             << "  local_edges="    << p.graph.num_edges
             << "  hidden_dim="     << cfg.hidden_dim
             << "  layers="         << cfg.num_layers << "\n";

        Runtime rt;
        rt.init(cfg, mcfg, move(p), use_dataset ? &dataset : nullptr);

        if (load_path) {
            Checkpoint::load_checkpoint(rt.model, load_path);
        }

        TrainingConfig tcfg;
        tcfg.epochs = 2;
        tcfg.patience = 10;
        tcfg.learning_rate = 0.01f;
        tcfg.optimizer_type = OptimizerType::SGD;
        tcfg.loss_type = use_dataset ? LossType::CrossEntropy : LossType::MSE;

        Trainer trainer(rt, tcfg, use_dataset ? &dataset : nullptr);
        trainer.train();

        if (save_path) {
            Checkpoint::save_checkpoint(rt.model, save_path);
        }

        rt.print_summary();

        // Print per-vertex hidden states only for small toy graphs.
        if (!use_dataset) {
            cout << "\n[rank " << cfg.rank << "] hidden states after "
                 << cfg.num_layers << " layers:\n";

            for (int32_t lv = 0; lv < rt.partition.num_vertices; ++lv) {
                const int32_t gv = rt.partition.vertex_map.to_global(lv, cfg.rank);
                cout << "  v" << gv << ": [";
                const float* h = rt.partition.hcurr(lv);
                for (int32_t d = 0; d < cfg.hidden_dim; ++d) {
                    cout << fixed << setprecision(4) << h[d];
                    if (d < cfg.hidden_dim - 1) cout << ", ";
                }
                cout << "]\n";
            }
        }
    } catch (const exception& e) {
        cerr << "[rank " << cfg.rank << "] error: " << e.what() << "\n";
        ret = 1;
    }

    MPI_Finalize();
    return ret;
}
