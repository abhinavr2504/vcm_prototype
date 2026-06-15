# 04 — Data Structures

> See also: [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | [06_FORWARD_PASS.md](06_FORWARD_PASS.md) | [07_BACKWARD_PASS.md](07_BACKWARD_PASS.md)

---

## Overview

The system's data is organized around two philosophies:
1. **Graph-level data** (global, distributed) — stored in `CSRGraph` and `VertexMap`
2. **Per-rank data** (local, owned by `Partition`) — stored in tensors and standard vectors

The table below shows the primary structures and which file defines them.

| Structure | Defined in | Lives in |
|---|---|---|
| `Partition` | `partition.h` | `Runtime` (one per rank) |
| `CSRGraph` | `graph.h` | `Partition` |
| `ReverseCSR` | `partition.h` | `Partition` |
| `VertexMap` | `vertex_map.h` | `Partition` and `MPIExchange` |
| `LayerCache` | `partition.h` | `Partition::layer_cache[]` |
| `AggregationTrace` | `partition.h` | `Partition::aggr_traces[]` |
| `AggregationMetadata` | `partition.h` | `LayerCache::aggr_meta` |
| `Message` | `message.h` | `Outbox`, `Inbox`, `MPIExchange` |
| `InboxBuffer` | `inbox.h` | `Partition` |
| `OutboxBuffer` | `outbox.h` | `Partition::thread_outboxes[]` |
| `FrontierSet` | `frontier.h` | `Partition` |
| `Runtime` | `runtime.h` | `main::rt` |
| `GNNModel` | `gnn_model.h` | `Runtime` |
| `GNNLayer` | `gnn_layer.h` | `GNNModel::layers_[]` |
| `Dataset` | `dataset.h` | `main` (const after load) |
| `Trainer` | `trainer.h` | `main::trainer` |
| `RunConfig` | `config.h` | `main`, passed everywhere |

---

## Partition

**Defined in:** `include/partition.h`  
**Implemented in:** `src/partition.cpp`  
**Role:** The complete local execution state for one MPI rank. Everything a rank knows about the graph and its own computation is stored here.

### Fields

```cpp
struct Partition {
    // Identity
    int32_t rank;
    int32_t num_vertices;    // count of local vertices
    int32_t hidden_dim;      // H — embedding dimension
    int32_t num_threads;     // OpenMP thread count

    // Topology
    CSRGraph        graph;          // local outgoing edges (local_src → global_dst)
    ReverseCSR      reverse_graph;  // local incoming edges (global_src) [E12]
    VertexMap       vertex_map;     // global↔local ID translation

    // Core GNN tensors (LibTorch, CPU float32)
    torch::Tensor   hidden_curr;    // [N, H] — current layer embeddings
    torch::Tensor   hidden_next;    // [N, H] — next layer result (write-only during superstep)
    torch::Tensor   aggr_buf;       // [N, H] — aggregation accumulator

    // Communication
    InboxBuffer              inbox;
    std::vector<OutboxBuffer> thread_outboxes;  // [num_threads]
    MPIExchange              mpi_exchange;

    // Activity tracking
    FrontierSet              frontier_curr;  // active vertices this layer
    FrontierSet              frontier_next;  // active vertices next layer
    std::vector<VertexState> vertex_state;   // per-vertex active flag

    // Legacy snapshots (pre-E11, kept for compatibility)
    std::vector<torch::Tensor> aggr_snapshots;
    std::vector<torch::Tensor> aggr_masks;

    // E11: Forward-state cache
    std::vector<LayerCache>       layer_cache;   // one per executed layer

    // E12: Dependency capture
    std::vector<AggregationTrace> aggr_traces;   // one per layer slot
};
```

### Lifecycle

1. **Created** by `loader.cpp::load_partition_from_dataset()` after reading the dataset
2. **Initialized** by `Partition::init()` — allocates tensors and communication structures
3. **Populated** at startup by `Runtime::init()` — copies dataset features into `hidden_curr`
4. **Modified** each epoch by `Runtime::run()` — supersteps update `hidden_curr`/`hidden_next`
5. **Extended** each run by `superstep.cpp` — appends to `layer_cache` and fills `aggr_traces`
6. **Read** by `trainer.cpp` for backward reconstruction

### Key Invariants

- `hidden_curr` is **read-only** during a superstep; only `hidden_next` is written
- After each layer: `advance_hidden()` swaps them
- `layer_cache` is cleared at the start of each epoch (in `Runtime::run()`)
- `aggr_traces` is pre-allocated to `num_layers` slots at the start of each epoch and filled in-place

### Pointer Accessors

```cpp
float*       hcurr(int32_t local_v);   // hidden_curr + v*H
float*       hnext(int32_t local_v);   // hidden_next + v*H
float*       vaggr(int32_t local_v);   // aggr_buf + v*H
const float* hcurr(int32_t local_v) const;
const float* hnext(int32_t local_v) const;
```

These are used heavily in performance-critical loops to avoid tensor indexing overhead.

---

## VertexMap

**Defined in:** `include/vertex_map.h`  
**Implemented in:** `src/vertex_map.cpp`  
**Role:** The global vertex ID space is divided into contiguous ranges, one range per rank. VertexMap encodes these ranges and provides O(log world_size) lookup.

### Fields

```cpp
struct VertexMap {
    int32_t num_global;              // total vertex count
    int32_t world_size;
    std::vector<int32_t> range_start;  // [world_size+1]; range_start[r..r+1) = rank r's vertices
};
```

### Key formulas

```
owner_rank(gv): binary search upper_bound on range_start → rank index
to_local(gv):   gv - range_start[owner_rank(gv)]
to_global(lv, rank): range_start[rank] + lv
local_count(rank):   range_start[rank+1] - range_start[rank]
```

### Invariants

- `range_start[0] == 0` always
- `range_start[world_size] == num_global` always
- `range_start` is monotonically non-decreasing
- VertexMap is **immutable** after initialization — never modified after `loader.cpp` runs

### Copies

VertexMap is value-copied into:
- `Partition::vertex_map` — used for local ID translation
- `MPIExchange::vertex_map_` — used for message routing during exchange

---

## CSRGraph (Forward)

**Defined in:** `include/graph.h`  
**Role:** Local adjacency list in Compressed Sparse Row format. Stores only outgoing edges from locally-owned vertices.

```cpp
struct CSRGraph {
    int32_t num_vertices;      // local vertex count
    int32_t num_edges;         // local edge count

    std::vector<int32_t> row_ptr;   // [num_vertices+1]; row_ptr[v..v+1) = edge range for v
    std::vector<int32_t> col_idx;   // [num_edges]; global destination vertex IDs
};
```

**Important:** `col_idx` stores **global** destination IDs, not local ones. This allows `MPIExchange` to route messages to the correct rank without additional translation.

**Traversal pattern (Scatter phase):**
```cpp
for e in [row_ptr[v], row_ptr[v+1]):
    global_dst = col_idx[e]
    // send Message from local v to global_dst
```

---

## ReverseCSR

**Defined in:** `include/partition.h`  
**Built in:** `src/loader.cpp::build_reverse_csr()`  
**Role:** Transpose of the forward CSR restricted to same-rank edges. Used by E13 (`scatter_only`) to propagate gradients from destinations back to sources.

```cpp
struct ReverseCSR {
    std::vector<int32_t> row_ptr;   // [num_local_vertices+1]; row_ptr[v..v+1) = in-edges for v
    std::vector<int32_t> col_idx;   // global src IDs of local in-neighbors

    bool built() const { return !row_ptr.empty(); }
    int32_t in_degree(int32_t local_v) const;
};
```

**Important:** `col_idx` stores **global** src IDs. This allows E14 gradient routing to call `owner_rank(src)` without ambiguity.

**What it does NOT include:** Cross-rank in-edges where the source is on another rank. Those are captured live by `AggregationTrace` during MPI exchange.

**Traversal pattern (E13 backward):**
```cpp
for e in [rcsr.row_ptr[v], rcsr.row_ptr[v+1]):
    global_src = rcsr.col_idx[e]
    local_src = vertex_map.to_local(global_src)
    grad_h[local_src] += grad_aggr[v]   // backward SUM aggregation
```

---

## LayerCache

**Defined in:** `include/partition.h`  
**Populated in:** `src/superstep.cpp::run()`  
**Role:** Snapshot of all forward-pass intermediate values for one layer. Required by E13 for local backward reconstruction.

```cpp
struct LayerCache {
    torch::Tensor aggregated;      // [N, H] — clone of aggr_buf after Gather
    torch::Tensor pre_activation;  // [N, H] — linear output before ReLU
    torch::Tensor activated;       // [N, H] — snapshot of hidden_next after ReLU
    torch::Tensor active_mask;     // [N] bool — which vertices were in frontier_curr
    AggregationMetadata aggr_meta; // E12 — aggregation type (SUM/MEAN/MAX)
};
```

**Why each field exists:**
- `aggregated` → needed by `linear_phase()`: `grad_W = grad_pre.T @ aggregated`
- `pre_activation` → needed by `linear_phase()`: `relu_mask = (pre > 0)`
- `activated` → needed by E15/E16 (future multi-layer backward)
- `active_mask` → needed by `scatter_only()`: only active (frontier) vertices contributed
- `aggr_meta.type` → needed to know the backward formula (SUM=1:1, MEAN=1/k, MAX=argmax)

**Critical property:** All tensors are `.clone()`s — they are completely detached from the runtime's computation buffers. `hidden_next` is reused and swapped every layer; if you store a reference instead of a clone, the value will change.

**Lifecycle:**
- Cleared: `Runtime::run()` at epoch start
- Populated: `superstep.cpp::run()` after Gather (aggregated), Compute (pre_activation, activated), and frontier build (active_mask)
- Read: `trainer.cpp::compute_loss_e13()`, `LocalBackwardEngine::scatter_only()`, `linear_phase()`

---

## AggregationTrace

**Defined in:** `include/partition.h`  
**Populated in:** `src/mpi_exchange.cpp::wait_and_unpack()`  
**Role:** Records which remote vertices sent messages to each local vertex during Gather. This is the routing table for E14 gradient transport.

```cpp
struct AggregationTrace {
    std::vector<std::vector<int32_t>> remote_contributors;
    // remote_contributors[local_v] = list of GLOBAL vertex IDs from other ranks
    //                                that sent to local_v during this layer

    bool valid = false;  // true after wait_and_unpack completes
};
```

**What it stores:** Only the **global src ID** of each remote contributor. Not the payload values (those are in `aggregated`). Not the timing. Not local contributors (those are recoverable from ReverseCSR).

**Capture window:** Inside `wait_and_unpack()`, the MPI receive buffer is available. `m.src` is the global ID of the remote sender. After `recv_bufs_.clear()` this information is permanently gone. The capture happens exactly here and only here.

**Memory cost (OGBN-Products, 4 ranks):**
- Per layer: ~7 MB per rank (vector metadata + ID storage)
- 2 layers: ~14 MB per rank — negligible

**Lifecycle:**
- Pre-allocated: `Runtime::run()` assigns `num_layers` empty AggregationTrace slots
- Populated: During `wait_and_unpack()`, `trace->valid` set to true
- Read: `GradientExchange::build_rank_groups()` uses it as the routing table

---

## Message

**Defined in:** `include/message.h`

```cpp
struct Message {
    int32_t src;                 // GLOBAL vertex ID of sender
    int32_t dst;                 // GLOBAL vertex ID of receiver (before delivery)
                                 // LOCAL vertex ID of receiver (after delivery to Inbox)
    std::vector<float> payload;  // [H] embedding vector
};
```

**Wire format** (in MPIExchange):
```
[src as float bytes][dst as float bytes][payload[0]..payload[H-1]]
total: (2 + H) floats per message
```

Int32 values are packed into float slots via `memcpy` (safe: same bit width).

---

## Runtime

**Defined in:** `include/runtime.h`  
**Implemented in:** `src/runtime.cpp`

```cpp
struct Runtime {
    RunConfig   cfg;          // rank, world_size, hidden_dim, num_layers, etc.
    ModelConfig model_cfg;
    GNNModel    model;        // all trainable parameters
    Partition   partition;    // local graph + execution state
    RuntimeTiming timing;     // E10 profiling

    SuperstepExecutor executor;  // stateless Gather→Compute→Scatter
};
```

**Lifecycle:** Created in `main.cpp`, initialized via `rt.init()`, then handed to `Trainer`. The `Trainer` holds a reference to `rt` and calls `rt.run()` at the start of each epoch.

---

## GNNLayer

```cpp
struct GNNLayer {
    torch::Tensor W_;  // [H, H] requires_grad=true; initialized to eye(H)
    torch::Tensor b_;  // [H]   requires_grad=true; initialized to zeros(H)

    torch::Tensor& W();
    torch::Tensor& b();
};
```

**Initialization:** Identity matrix for W ensures that at epoch 0, the forward pass is a no-op numerically (h_new = relu(h_old @ I + 0) = relu(h_old)). This is deliberate for debugging early startup behavior.

---

## Dataset

```cpp
struct Dataset {
    int64_t num_nodes;
    int64_t num_edges;
    int32_t feature_dim;    // 100 for OGBN-Products
    int32_t num_classes;    // 47 for OGBN-Products

    torch::Tensor features;  // [N, feature_dim] float32
    torch::Tensor labels;    // [N] int64

    std::vector<int32_t> edge_src;
    std::vector<int32_t> edge_dst;

    std::vector<uint8_t> train_mask;   // 1 if this node is in train set
    std::vector<uint8_t> val_mask;
    std::vector<uint8_t> test_mask;
};
```

**Ownership:** The `Dataset` object is created in `main.cpp` and passed by pointer to `Runtime::init()` (for feature initialization) and `Trainer` (for label/mask access). It is never modified after loading.

---

## Trainer

```cpp
class Trainer {
    Runtime&                 rt_;
    TrainingConfig           tcfg_;
    const Dataset*           dataset_;
    DistributedOptimizer     optimizer_;
    GradientExchange         grad_exchange_;  // E14

    float best_val_accuracy_;
    int   epochs_no_improve_;
};
```

The Trainer does not own the model or partition — it only holds a reference to `Runtime`. All gradient computations produce LibTorch tensors that are assigned to model parameter `.grad()` fields, then averaged and applied by the optimizer.

---

## Cross-References

- How tensors are used in the forward pass → [06_FORWARD_PASS.md](06_FORWARD_PASS.md)
- How LayerCache and AggregationTrace are used in backward → [07_BACKWARD_PASS.md](07_BACKWARD_PASS.md)
- How VertexMap determines message routing → [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)
