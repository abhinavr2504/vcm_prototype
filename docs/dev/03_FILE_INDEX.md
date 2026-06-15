# 03 — File Index

> See also: [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | [13_CODE_NAVIGATION_GUIDE.md](13_CODE_NAVIGATION_GUIDE.md)

For every source file: its purpose, key classes/functions, and who depends on it.

---

## Source Files (`src/`)

---

### `src/main.cpp`
**Purpose:** Program entry point. Parses CLI, initializes MPI, orchestrates startup, hands control to Trainer.

**Key functions:**
- `main(argc, argv)` — CLI parsing, creates `RunConfig`, `ModelConfig`, `Dataset`, `Partition`, `Runtime`, `Trainer`

**Dependencies:**
- `config.h`, `dataset.h`, `loader.h`, `runtime.h`, `trainer.h`, `checkpoint.h`

**Used by:** Nothing (entry point)

**Notes:** Contains three documented bug-fix patterns:
- BUG 1: zero-rank loss causing MPI hang (handled in trainer.cpp)
- BUG 2: all ranks printing dataset summary (fixed: rank 0 only)
- BUG 3: printing 2.4M vertex hidden states for real dataset (suppressed by `use_dataset` flag)

---

### `src/dataset.cpp`
**Purpose:** Loads OGBN-Products from compressed CSV files (`.csv.gz`). All data loaded by every rank; no MPI broadcast.

**Key functions:**
- `load_ogbn_products(cfg)` — reads node features, labels, edges, train/val/test split masks from gzipped CSVs

**Dependencies:** `dataset.h`, `config.h`, zlib (gzopen/gzgets)

**Used by:** `main.cpp`

**Notes:**
- Features: `node-feat.csv.gz` — one row per node, 100 float columns
- Labels: `node-label.csv.gz` — integer class per node
- Edges: `edge.csv.gz` — (src, dst) pairs
- Split masks: `split/sales_ranking/train|valid|test.csv.gz`

---

### `src/loader.cpp`
**Purpose:** Two responsibilities: (1) build the per-rank `Partition` from dataset or raw graph file; (2) build the `ReverseCSR` (E12).

**Key functions:**
- `load_partition(path, cfg)` — load from raw `.graph` file (toy graphs)
- `load_partition_from_dataset(dataset, cfg)` — load from `Dataset` struct (OGBN-Products)
- `build_reverse_csr(partition)` — static local function, builds local in-edge CSR

**Dependencies:** `loader.h`, `partition.h`, `vertex_map.h`

**Used by:** `main.cpp`

**Notes:**
- Partition strategy (vertex/edge/hybrid balanced) is selected here
- `build_reverse_csr` is called automatically at end of loading, before any superstep
- Forward CSR stores GLOBAL dst IDs; ReverseCSR stores GLOBAL src IDs — this is intentional for routing

---

### `src/vertex_map.cpp`
**Purpose:** Maps global vertex IDs to (rank, local_id) pairs. Contains all three partitioning strategies.

**Key functions:**
- `init(n_global, n_ranks)` — vertex-balanced: ceil(N/world_size) per rank
- `init_edge_balanced(n_global, n_ranks, degrees)` — E9: greedy edge-count balance
- `init_hybrid_balanced(n_global, n_ranks, degrees, alpha, beta)` — E9.1: weighted cost model
- `owner_rank(global_id)` — binary search on `range_start[]`
- `to_local(global_id)` — `global_id - range_start[owner_rank]`
- `to_global(local_id, rank)` — `range_start[rank] + local_id`
- `local_count(rank)` — `range_start[rank+1] - range_start[rank]`

**Dependencies:** `vertex_map.h`, `assertions.h`

**Used by:** `loader.cpp`, `mpi_exchange.cpp`, `gradient_exchange.cpp`, `local_backward.cpp`

**Debug instrumentation:** `owner_rank()` contains a static debug counter that prints the first 50 calls with the `range_start` boundaries. This is E14 validation instrumentation, should be removed when done.

---

### `src/partition.cpp`
**Purpose:** Allocates and initializes all per-rank runtime state. Provides pointer accessors.

**Key functions:**
- `Partition::init(n, h, threads, graph, vm, rank, world_size)` — allocates tensors, frontiers, inbox, exchange
- `Partition::hcurr(v)` → `float*` pointer into `hidden_curr`
- `Partition::hnext(v)` → `float*` pointer into `hidden_next`
- `Partition::vaggr(v)` → `float*` pointer into `aggr_buf`
- `Partition::advance_hidden()` — `std::swap(hidden_curr, hidden_next)`
- `Partition::advance_frontier()` — swap and clear frontier bitsets

**Dependencies:** `partition.h`

**Used by:** `loader.cpp`, `runtime.cpp`, `superstep.cpp`, `trainer.cpp`

---

### `src/runtime.cpp`
**Purpose:** Orchestrates the forward pass layer loop. Also contains the legacy `train_step()` (pre-E13) and `print_summary()`.

**Key functions:**
- `Runtime::init(cfg, mcfg, partition, dataset)` — sets up model, copies features
- `Runtime::run()` — main layer loop: seed → superstep×L → advance
- `Runtime::train_step(lr)` — legacy L2-loss training (not used in current training path)
- `Runtime::print_summary()` — prints per-rank and global statistics, timing, E10/E11/E12 validation

**Dependencies:** `runtime.h`, `superstep.h`, `dataset.h`, MPI

**Used by:** `main.cpp`, `trainer.cpp`

**Debug instrumentation:**
- `[FORWARD CACHE]` — prints per-layer aggregation norm and global norm via MPI_Allreduce after each layer
- `[E12] Dependency Preservation Validation` — prints ReverseCSR stats and AggregationTrace counts after run
- `[rank %d] E12 layer=%zu remote_total=%lld` — per-rank remote contributor count

---

### `src/superstep.cpp`
**Purpose:** Implements one GNN superstep (Gather → Compute → Scatter → Exchange). Also captures E11 and E12 state.

**Key functions:**
- `SuperstepExecutor::seed(partition, timing)` — bootstrap scatter of initial features
- `SuperstepExecutor::run(partition, layer_params, layer, timing)` — one full superstep

**Dependencies:** `superstep.h`, `compute.h`, `partition.h`, `assertions.h`, OpenMP, MPI

**Used by:** `runtime.cpp`

**Debug instrumentation:**
- `[SRC CHECK]` — prints global ID for each message src during scatter and seed (very verbose)
- `[FORWARD CACHE]` and `[GLOBAL CACHE]` — aggregation norms per layer with MPI_Allreduce
- `[E12 DEBUG]`, `[E12 RESET]`, `[E12 AFTER ASSIGN]`, `[E12 END LAYER]` — AggregationTrace lifecycle

---

### `src/compute.cpp`
**Purpose:** Pure arithmetic operations: aggregation, linear transform, ReLU. No autograd, no MPI.

**Key functions:**
- `ComputeEngine::zero_aggr(aggr, H)` — zero out aggregation buffer
- `ComputeEngine::aggregate(src, aggr, H)` — `aggr += src` (SUM aggregation)
- `ComputeEngine::linear_relu_tensor(aggr, W, b, out)` — tensor-based linear+ReLU
- `ComputeEngine::linear_relu(aggr, W, b, out, H)` — raw-pointer version (legacy)

**Dependencies:** `compute.h`, torch

**Used by:** `superstep.cpp`

**Notes:** The compute loop in `superstep.cpp` actually bypasses these functions for the per-vertex linear transform, using raw pointer loops directly. These functions are legacy interfaces kept for compatibility.

---

### `src/mpi_exchange.cpp`
**Purpose:** Implements the forward message exchange protocol. Handles serialization, MPI send/receive, deserialization, inbox delivery. Also captures E12 `AggregationTrace` during unpack.

**Key functions:**
- `MPIExchange::init(vm, rank, world_size, H)` — store config
- `MPIExchange::enqueue(m)` — classify as local or remote
- `MPIExchange::exchange()` — `MPI_Alltoall` counts, `MPI_Isend`/`Irecv` data
- `MPIExchange::wait_and_unpack(inbox, trace)` — `MPI_Waitall`, deliver messages, capture E12 trace

**Wire format:** `[src_int32][dst_int32][payload_float×H]` stored as a flat float array (stride = H+2)

**Dependencies:** `mpi_exchange.h`, `partition.h`, `assertions.h`, MPI

**Used by:** `superstep.cpp`

**Debug instrumentation:**
- `[E12 STORE]` — prints first 20 captured remote contributors with src global ID
- `[E12 DEBUG] captured=%lld` — milestone counter every 1M captures

---

### `src/gnn_model.cpp`
**Purpose:** Owns and initializes all trainable parameters.

**Key functions:**
- `GNNModel::init(cfg)` — initializes N GNNLayer objects + classifier head
- `GNNModel::layer(i)` → `GNNLayer&` — access layer i's W and b
- `GNNModel::parameters()` → `vector<Tensor>` — all trainable parameters for optimizer
- `GNNModel::classifier_W()`, `classifier_b()` — classification head

**Dependencies:** `gnn_model.h`, `gnn_layer.h`

**Used by:** `runtime.cpp`, `trainer.cpp`, `distributed_optimizer.cpp`

---

### `src/gnn_layer.cpp`
**Purpose:** Single GNN layer parameter container.

**Key functions:**
- `GNNLayer::init(H)` — `W = eye(H)`, `b = zeros(H)` with `requires_grad=true`
- `GNNLayer::W()`, `GNNLayer::b()` — access tensors

**Dependencies:** `gnn_layer.h`

**Used by:** `gnn_model.cpp`

---

### `src/trainer.cpp`
**Purpose:** The training loop and all backward-pass logic (E8, E13, E14). Most actively modified file in the current investigation.

**Key functions:**
- `Trainer::Trainer(rt, tcfg, dataset)` — constructor, creates optimizer, moves to GPU if available
- `Trainer::train()` — epoch loop
- `Trainer::compute_loss_e13(epoch, verbose)` — loss reconstruction using E11 cache
- `Trainer::compute_accuracy(mask, verbose)` — evaluation
- `Trainer::print_epoch_summary(...)` — progress reporting

**Dependencies:** `trainer.h`, `runtime.h`, `local_backward.h`, `gradient_exchange.h`, `distributed_optimizer.h`, torch, MPI

**Used by:** `main.cpp`

**Debug instrumentation (active phase):**
- `[GLOBAL_AGGR_SUM]` — globally summed aggregation norm (pre-loss)
- `[GLOBAL_LABEL_SUM]` — global label sum (should be dataset-consistent across runs)
- `[GLOBAL_LOGITS_SUM]` — globally summed logit norm (diverges NP=1 vs NP=4)
- `[GLOBAL_GRAD_AGGR]` — globally summed gradient of aggregation (diverges)
- `[E14 BUILD]` — rank-group construction per rank during GradientExchange
- `[E14 ROUTE]` — individual gradient routing decisions
- `[LOSS DEBUG]` — loss value per epoch

---

### `src/local_backward.cpp`
**Purpose:** E13 — reconstructs gradients for Layer 0 using local graph structure and cached forward state.

**Key functions:**
- `LocalBackwardEngine::scatter_only(partition, target_layer, grad_aggr)` → `Tensor [N,H]`
- `LocalBackwardEngine::linear_phase(partition, src_layer, grad_h)` → `{grad_W, grad_b}`
- `LocalBackwardEngine::run(partition, target_layer, grad_aggr)` → `{grad_W, grad_b}`

**Dependencies:** `local_backward.h`, `partition.h`, torch

**Used by:** `trainer.cpp`

**Debug instrumentation:**
- `[E13 LOCAL]` — norm of grad_h after scatter_only
- `[E13 INPUT]` — norms of aggregated, pre_activation, grad_h, grad_pre before linear_phase
- `[E13 MATMUL]` — norm of grad_W after matmul

---

### `src/gradient_exchange.cpp`
**Purpose:** E14 — rank-batched cross-rank gradient transport (transpose of forward exchange).

**Key functions:**
- `GradientExchange::GradientExchange(rank, world_size, chunk_pairs)` — constructor
- `GradientExchange::exchange_into(partition, target_layer, grad_aggr, grad_h)` — main exchange
- `GradientExchange::build_rank_groups(partition, layer, r, v_locals, u_offsets, u_globals)` — index builder

**Dependencies:** `gradient_exchange.h`, `partition.h`, torch, MPI

**Used by:** `trainer.cpp`

---

### `src/distributed_optimizer.cpp`
**Purpose:** E8 — gradient averaging across all ranks using `MPI_Allreduce`.

**Key functions:**
- `DistributedOptimizer::average_gradients()` — allreduce all gradients, divide by world_size

**Dependencies:** `distributed_optimizer.h`, torch, MPI

**Used by:** `trainer.cpp`

---

### `src/checkpoint.cpp`
**Purpose:** Save/load model weights to/from disk.

**Key functions:**
- `Checkpoint::save_checkpoint(model, path)` — serialize to file
- `Checkpoint::load_checkpoint(model, path)` — deserialize from file

**Dependencies:** `checkpoint.h`, `gnn_model.h`, torch

**Used by:** `main.cpp`

---

### `src/frontier.cpp`, `src/inbox.cpp`, `src/outbox.cpp`, `src/graph.cpp`

These implement simple data structures. See header files for interface details.

---

## Header Files (`include/`)

| Header | Purpose |
|---|---|
| `config.h` | `RunConfig`, `ModelConfig`, `DatasetConfig`, `TrainingConfig`, all enums |
| `partition.h` | `Partition`, `LayerCache`, `AggregationTrace`, `ReverseCSR`, `AggregationMetadata` |
| `vertex_map.h` | `VertexMap` — global↔local ID translation |
| `graph.h` | `CSRGraph` — `row_ptr`, `col_idx`, `num_vertices`, `num_edges` |
| `message.h` | `Message` — `src`, `dst`, `payload` |
| `inbox.h` | `InboxBuffer` — double-buffered message store |
| `outbox.h` | `OutboxBuffer` — per-thread message accumulator |
| `frontier.h` | `FrontierSet` — bitset of active vertex IDs |
| `vertex_state.h` | `VertexState` — `active` flag |
| `mpi_exchange.h` | `MPIExchange` — forward message exchange layer |
| `gradient_exchange.h` | `GradientExchange` — E14 backward gradient delivery |
| `local_backward.h` | `LocalBackwardEngine` — E13 gradient reconstruction |
| `distributed_optimizer.h` | `DistributedOptimizer` — E8 gradient averaging |
| `compute.h` | `ComputeEngine` — aggregation, linear, ReLU |
| `gnn_model.h` | `GNNModel` — parameter container |
| `gnn_layer.h` | `GNNLayer` — single layer W, b |
| `runtime.h` | `Runtime` — top-level execution object |
| `superstep.h` | `SuperstepExecutor` — Gather→Compute→Scatter |
| `trainer.h` | `Trainer` — training loop |
| `loader.h` | `load_partition`, `load_partition_from_dataset` |
| `dataset.h` | `Dataset`, `load_ogbn_products` |
| `checkpoint.h` | `Checkpoint` — save/load model weights |
| `assertions.h` | `ASSERT()`, `assert_*` validation helpers |
| `timing.h` | `RuntimeTiming`, `LayerTiming`, `LayerProfile`, `LayerComputeBreakdown`, `LayerMPIBreakdown` |
| `statistics.h` | `RuntimeStatistics`, `GlobalStatistics` |

---

## Cross-References

- Data structure deep-dive → [04_DATA_STRUCTURES.md](04_DATA_STRUCTURES.md)
- Code navigation guide → [13_CODE_NAVIGATION_GUIDE.md](13_CODE_NAVIGATION_GUIDE.md)
