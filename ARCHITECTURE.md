# ARCHITECTURE.md
## Distributed Full-Batch GNN Training Prototype (G2N2-Inspired)



---

## Table of Contents

1. [System Motivation](#1-system-motivation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [E1–E10: Foundation Phases](#3-e1e10-foundation-phases)
4. [GNN Forward Pass Deep Dive](#4-gnn-forward-pass-deep-dive)
5. [Comparison to G2N2](#5-comparison-to-g2n2)
6. [E11: Forward-State Cache](#6-e11-forward-state-cache)
7. [E12: Dependency Capture](#7-e12-dependency-capture)
8. [ReverseCSR](#8-reversecsr)
9. [E13: Local Backward Reconstruction](#9-e13-local-backward-reconstruction)
10. [E14: Gradient Message Transport](#10-e14-gradient-message-transport)
11. [Current Blocker](#11-current-blocker)
12. [E15 and E16: Roadmap](#12-e15-and-e16-roadmap)
13. [File-by-File Architecture Reference](#13-file-by-file-architecture-reference)
14. [Current Status Summary](#14-current-status-summary)

---

## 1. System Motivation

### 1.1 What Are Graph Neural Networks?

A Graph Neural Network (GNN) is a model that operates on graph-structured data — data where relationships between entities (edges) matter as much as the entities themselves (vertices).

In a social network, for example, a person's interests are influenced not only by their own history but also by those of their friends and friends-of-friends. GNNs capture this by propagating information along edges, layer by layer.

Mathematically, the GNN update rule for vertex `v` at layer `l` is:

```
aggregated[v]  =  SUM{ h[u]  :  u → v is an edge }
h_new[v]       =  ReLU( W × aggregated[v] + b )
```

This is the *message passing* paradigm: vertices receive, aggregate, and transform.

### 1.2 Why Distributing a GNN Is Hard

Production-scale graphs — such as the **OGBN-Products** dataset used in this project (2.4 million nodes, 61 million edges) — cannot fit in the memory of a single machine. The graph must be split across multiple processes in a cluster.

However, the very nature of GNNs makes distribution non-trivial:

**The Boundary Problem:**  
When the graph is split across processes, edges cross partition boundaries. Vertex `u` on Process 0 may have an edge to vertex `v` on Process 1. For `v` to aggregate correctly, it must receive `h[u]` from Process 0 before it can compute its new embedding.

This inter-process communication is unavoidable and grows rapidly with the number of partitions.

**Intuitive example:**

```
Process 0 owns:          Process 1 owns:
  A → B → C               D → E → F

  A → D (CROSS-PARTITION EDGE)
```

When computing the embedding of `D` at layer 1, Process 1 must receive the hidden state `h[A]` from Process 0 before it can proceed. This requires network communication — not a local memory operation.

This is the fundamental structural challenge of distributed graph computation.

### 1.3 Why Full-Batch Training Is Especially Challenging

Most modern GNN frameworks sidestep the distributed communication problem by using **mini-batch training with neighbor sampling**: instead of updating all vertices using all edges, they randomly sample a small neighborhood per vertex per step, discarding infrequent edges.

This dramatically reduces communication but changes the training semantics: the model never sees the full graph topology in a single gradient step.

**Full-batch training** uses the entire graph in every training step. This makes training semantically cleaner — gradients are computed over the entire graph — but it forces every edge to be processed every epoch, making communication costs unavoidable.

The goal of this project is to implement *exactly* this: full-batch distributed training where the entire graph contributes to every gradient update.

### 1.4 Why Communication Becomes Expensive

Consider OGBN-Products partitioned across 4 MPI ranks:

- Each rank owns ~75,000 vertices
- Each vertex has an average in-degree of ~18
- However, ~70% of those neighbors may live on other ranks (the **remote fraction**)

This means that for each of the ~75,000 local vertices, approximately 12–13 incoming messages per vertex cross rank boundaries every layer, every epoch. With hidden dimension `H = 256` floats and 4-byte floats:

```
Per layer, per rank, per epoch:
  remote_messages ≈ 75,000 × 12 = 900,000 messages
  bytes           ≈ 900,000 × 256 × 4 = ~922 MB of data
```

This is approximately 1 GB of data moved *just for the forward pass* of a single layer on a single rank in a single epoch. At 4 ranks and 2 layers, this is ~8 GB of cross-network data per epoch.

E10 profiling confirmed: **MPI communication dominates total runtime**, often totalling more than 60% of wall-clock time. Compute is secondary.

### 1.5 Why Autograd Graphs Become Large

PyTorch's automatic differentiation (autograd) works by building a *computation graph* as the forward pass executes. Every tensor operation recorded with `requires_grad=True` creates a node in this graph. At backward time, PyTorch traverses this graph in reverse to compute gradients.

For a 2-layer GNN on a graph with 2.4M vertices and dimension 256:

```
Forward computation graph contains:
  Layer 0: N × H × H operations (matmul) = 2.4M × 256 × 256 = ~157B ops
  Layer 1: same
  Classifier: N × H × num_classes
```

The autograd graph therefore contains hundreds of millions of nodes, each holding a reference to its forward tensors. For truly distributed execution, this graph must somehow span multiple processes — a capability that stock PyTorch does not natively provide in a way that respects graph topology.

### 1.6 Why G2N2 Exists

**G2N2** (Graph-to-N2 Network, from the research paper this project draws from) addresses the distributed backward problem by:

1. Building **computation trees** — explicit data structures capturing which vertices contributed to each output vertex at each layer
2. Storing **activations** at every layer across all vertices
3. **Replaying the backward pass** over the computation tree rather than relying on PyTorch's built-in autograd to span machines

G2N2's key insight is: *you cannot backward-propagate through a distributed runtime using stock autograd, because message-passing operations break the gradient tape.* You must explicitly capture enough execution structure to reconstruct backward traversal yourself.

This project is a C++/MPI/LibTorch prototype that implements the same insight from first principles.

---

## 2. High-Level Architecture

### 2.1 Layered Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATASET LAYER                           │
│  ogbn_products/  →  features [N, 100]                          │
│                  →  labels   [N]                                │
│                  →  split masks (train/val/test)                │
└────────────────────────────┬────────────────────────────────────┘
                             │  loader.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PARTITIONING LAYER                         │
│  CSRGraph  +  VertexMap  +  PartitionMode                      │
│                                                                 │
│  Strategy: Vertex-Balanced | Edge-Balanced | Hybrid             │
│  Output: local vertex set, local edge set, cross-rank routing   │
└────────────────────────────┬────────────────────────────────────┘
                             │  partition.cpp, vertex_map.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RUNTIME LAYER                            │
│  Partition object: hidden buffers, frontiers, inbox, outbox     │
│  GNNModel: parameter ownership (W[l], b[l], classifier)        │
│  MPIExchange: cross-rank message transport                      │
└────────────────────────────┬────────────────────────────────────┘
                             │  runtime.cpp, superstep.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD EXECUTION LAYER                      │
│  For each layer l:                                              │
│    Gather:    aggregate incoming messages → aggr_buf            │
│    Compute:   linear + ReLU → hidden_next                       │
│    Scatter:   emit hidden_next to out-neighbors                 │
│    Exchange:  MPI_Alltoall to deliver remote messages           │
└────────────────────────────┬────────────────────────────────────┘
                             │  superstep.cpp, mpi_exchange.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STATE PRESERVATION LAYER                      │
│  LayerCache[l]:                                                 │
│    aggregated    [N, H]  — clone of aggr_buf after Gather      │
│    pre_activation[N, H]  — linear output before ReLU           │
│    activated     [N, H]  — snapshot after ReLU                 │
│    active_mask   [N]     — which vertices participated          │
│  AggregationTrace[l]:                                           │
│    remote_contributors[v] — global IDs of remote senders        │
│  ReverseCSR:                                                    │
│    row_ptr, col_idx — local in-edges (built once at load)       │
└────────────────────────────┬────────────────────────────────────┘
                             │  partition.h, superstep.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BACKWARD RECONSTRUCTION LAYER                  │
│                                                                 │
│  Phase A (E13 local):  scatter_only()                           │
│    grad_aggr → ReverseCSR → grad_h_local [N, H]               │
│                                                                 │
│  Phase B (E14 remote): exchange_into()                          │
│    grad_aggr → AggregationTrace → MPI → grad_h += remote       │
│                                                                 │
│  Phase C (E13 linear): linear_phase()                           │
│    grad_h → ReLU mask → grad_W, grad_b                         │
└────────────────────────────┬────────────────────────────────────┘
                             │  local_backward.cpp, gradient_exchange.cpp
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION LAYER                          │
│  DistributedOptimizer::average_gradients()                      │
│    MPI_Allreduce on all parameter gradients                     │
│  optimizer_→step()                                              │
│    Adam / SGD parameter update                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Explanations

**Dataset Layer:** Ingests OGBN-Products from CSV files and constructs node features, labels, and train/val/test split masks. All data is loaded globally by Rank 0 then distributed to per-rank partitions.

**Partitioning Layer:** Divides the global graph into per-rank subgraphs. Assigns every vertex an owning rank. Builds the VertexMap, which translates global vertex IDs to (rank, local_id) pairs. This layer runs once at startup.

**Runtime Layer:** The central stateful object of the system. `Partition` owns all per-rank buffers: hidden state tensors, aggregation buffers, frontiers, and message queues. `GNNModel` holds all trainable parameters.

**Forward Execution Layer:** The Gather → Compute → Scatter loop. Executes one superstep per GNN layer per epoch. This is the only place computation happens. This layer is **non-differentiable by design** — message payloads are stored in `std::vector<float>`, which PyTorch autograd cannot track.

**State Preservation Layer:** After E11 and E12, the forward pass stores enough information to reconstruct backward execution. This is the project's most critical architectural innovation: deliberately preserving execution state that a distributed backward pass will need.

**Backward Reconstruction Layer:** After E13 and E14, manually reconstructs gradients for earlier GNN layers using the preserved forward state and cross-rank gradient messages. This layer replaces what PyTorch autograd would ordinarily do automatically — but cannot, because of the non-differentiable forward runtime.

**Optimization Layer:** Averages gradients across all ranks using MPI_Allreduce, then applies the optimizer step. After this, all ranks hold identical model parameters.

---

## 3. E1–E10: Foundation Phases

### 3.1 E1 — Graph Loader Foundation

**Objective:** Establish the minimal infrastructure to load a graph, assign vertices to ranks, and execute a single message-passing superstep.

**Design rationale:** Before any GNN logic could be built, the runtime needed a representation of the graph that supported distributed execution. This required defining vertex ownership as a first-class concept from day one, rather than retrofitting it later.

**Core data structures introduced:**
- `CSRGraph`: compressed sparse row adjacency list with `row_ptr` and `col_idx` arrays. Each row `v` stores global destination vertex IDs for all of `v`'s outgoing edges.
- `VertexMap`: maps between global vertex IDs and (rank, local_id) pairs. Principal query operations: `owner_rank(gv)`, `to_local(gv)`, `to_global(lv, rank)`.
- `Partition`: central per-rank object owning the local CSRGraph, VertexMap, frontiers, hidden buffers, inbox, outbox, and exchange layer.
- `Message`: simple struct with `src` (global), `dst` (global), and `payload` (float vector).

**Runtime flow:**
```
1. Load entire graph on each rank
2. Assign vertices to ranks using VertexMap
3. Each rank retains only local vertex/edge records
4. Initialize frontiers: all local vertices active
5. Execute seed scatter: emit initial hidden states to all out-neighbors
6. MPI exchange delivers cross-rank messages
```

**Source files:** `loader.cpp`, `vertex_map.cpp`, `partition.cpp`, `graph.h`, `message.h`, `frontier.h`

---

### 3.2 E2 — CSR Runtime Representation

**Objective:** Formalize the CSR graph structure and introduce `GNNModel` as the authoritative owner of all trainable parameters.

**Design rationale:** During early development, model parameters were scattered across the runtime. This made gradient management impossible. E2 introduced `GNNModel` as a clean container for `GNNLayer` objects (`W`, `b`) and the classifier head (`classifier_W`, `classifier_b`). All optimizer-facing parameter lists flow through `model.parameters()`.

**Core data structures:**
- `GNNModel`: contains an ordered list of `GNNLayer` objects and a classifier head.
- `GNNLayer`: owns weight matrix `W` [H, H] and bias `b` [H] as `torch::Tensor` with `requires_grad=true`.

**Key discovery:** Making model parameters `torch::Tensor` with gradient tracking was necessary for optimizer integration but did NOT solve distributed backward propagation — those are two entirely separate problems.

**Source files:** `gnn_model.cpp`, `gnn_model.h`, `gnn_layer.cpp`, `gnn_layer.h`

---

### 3.3 E3 — Distributed Runtime Framework

**Objective:** Establish the Gather → Compute → Scatter execution loop as the permanent architectural invariant; validate snapshot-based training.

**Critical discovery:** Message payloads (`Message.payload`) are stored as `std::vector<float>`. PyTorch autograd cannot track gradients through raw float vectors. This means the runtime forward pass is **non-differentiable by construction** — any gradient computation must happen in a separate reconstruction step.

**Snapshot-Rebuild validation:** To confirm that training was at all possible, E3 introduced the first training approach:
1. During Gather, snapshot the aggregation buffer: `aggr_snapshot[l] = aggr_buf.clone()`
2. After forward pass, reconstruct `out = relu(snapshot @ W.T + b)` as a differentiable tensor operation
3. Compute loss and backward through this reconstructed computation only
4. Result: only the last layer's W and b received gradients; Layer 0 received none

This was intentional — it validated the training plumbing without solving multi-layer backprop.

**Source files:** `runtime.cpp`, `superstep.cpp`, `compute.cpp`, `trainer.cpp`

---

### 3.4 E4–E7 — Runtime Maturation

**E4 — Trainer Infrastructure:** Added `Trainer` class encapsulating the epoch loop, loss computation, optimizer management (Adam/SGD), metric tracking (train/val/test accuracy), and checkpoint save/load.

**E5 — Dataset Integration:** Integrated the OGBN-Products dataset. Features loaded from CSV, assigned as initial hidden states `hidden_curr ← dataset.features[gv]`. This transformed the system from a toy-graph prototype to a real-world benchmark system.

**E6 — Supervised Training:** Added cross-entropy loss with train-mask filtering. Vertices are only included in the loss if they appear in the train split. Enabled full training/validation/test pipeline.

**E7 — Evaluation Pipeline:** Added best-checkpoint tracking keyed on validation accuracy. Test accuracy computed only at the best epoch. This completed the standard ML training-evaluation loop.

**Important debugging landmark (BUG 1):** A class of subtle bugs was found where ranks with zero labeled vertices returned `loss.requires_grad = False`, causing `MPI_Allreduce` collective hangs because some ranks skipped the call. Fix: always produce a differentiable zero loss by touching the relevant parameters: `zero_loss = aggr.sum() * 0.0f + W.sum() * 0.0f + ...`. This pattern repeats throughout the codebase wherever a rank might have no labelled samples.

**Source files:** `trainer.cpp`, `trainer.h`, `dataset.cpp`, `checkpoint.cpp`

---

### 3.5 E8 — Distributed Gradient Synchronization

**Objective:** Convert N locally-diverging models into one logically shared distributed model.

**Problem before E8:** Each rank trained its own independent model. Although forward propagation was distributed (messages crossed ranks), backward propagation and parameter updates were entirely local. After 10 epochs, each rank's weights had evolved independently, learning only from its own partition's labeled vertices.

**Solution:** After `loss.backward()` but before `optimizer.step()`, each parameter's gradient is averaged across all ranks using `MPI_Allreduce`. Since every rank applies the identical averaged gradient, every optimizer step is mathematically equivalent. All ranks remain synchronized.

```cpp
// DistributedOptimizer::average_gradients()
for each parameter p in model.parameters():
    MPI_Allreduce(p.grad(), MPI_SUM, MPI_COMM_WORLD)
    p.grad() /= world_size
```

**E8 discovery (the most important early discovery):** After E8, it was observed that Layer 0's W and b still received undefined (null) gradients — even though E8 was implemented correctly. This initially appeared to be a bug in E8. Investigation revealed it was not. The true cause: training reconstruction only used `aggr_snapshot[last_layer]`, which did not include Layer 0 in its autograd graph. E8 was correct; the problem was structural, not communicational.

**Source files:** `distributed_optimizer.cpp`, `distributed_optimizer.h`, `trainer.cpp`

---

### 3.6 E9 — Edge-Balanced Partitioning

**Objective:** Replace vertex-balanced partitioning, which caused severe compute imbalance due to degree heterogeneity in power-law graphs.

**Problem:** OGBN-Products has a power-law degree distribution. Splitting vertices evenly means some ranks get many high-degree vertices (many edges) while others get low-degree vertices. Observed in practice:

```
Rank 0: 3.2M local edges (high-degree hub vertices)
Rank 3: 180K local edges (low-degree periphery vertices)
Compute imbalance ratio: ~18×
```

**Solution:** Assign vertices greedily such that cumulative edge count is balanced across ranks. A rank fills until its edge count reaches `total_edges / world_size`, then moves to the next rank.

**E9 discovery:** Edge balancing dramatically increased communication. In vertex-balanced mode, locality of high-degree hubs meant many edges were local. In edge-balanced mode, high-degree hubs were split across ranks, generating far more cross-partition edges. Observed remote fraction increasing from ~30% to ~70%.

**Key lesson:** *Balancing compute alone is insufficient. Communication dominates distributed GNN execution.*

**Source files:** `partition.cpp`, `loader.cpp`

---

### 3.7 E9.1 — Hybrid Partitioning

**Objective:** Balance compute and communication simultaneously using a weighted objective function.

**Design:**
```
cost(assignment) = alpha × vertex_imbalance + beta × edge_imbalance
```

With `alpha >> beta` (e.g., alpha=100, beta=1), the hybrid scheme avoids the communication explosion of pure edge balancing while still improving compute balance over pure vertex balancing. This is the current preferred partitioning strategy.

**Configuration:** Controlled by `PartitionMode::HybridBalanced` and tunable `alpha`/`beta` parameters in `config.h`.

**Source files:** `partition.cpp`, `config.h`

---

### 3.8 E10 — Runtime Profiling

**Objective:** Establish evidence-based performance analysis to avoid premature optimization.

**E10 core additions:**
- `RuntimeTiming`: wall-clock timers for total, communication, compute, and per-layer phases
- Per-layer breakdown: Gather time, Compute time, Scatter time, MPI time
- `LayerProfile`: active vertex counts, messages sent/received, remote vs. local message counts

**E10.1 — Activity Profiling:** Tracks how many vertices are active per layer. On large graphs with frontier propagation, later layers may activate fewer vertices than earlier layers (some vertices receive no messages).

**E10.2 — Cost Decomposition:** Breaks MPI time into pack, exchange, and unpack phases. Key finding: the `MPI_Alltoall` exchange itself (coordinated inter-node data movement) dominates over pack/unpack preprocessing.

**Major finding from E10:** On OGBN-Products with 4 ranks, communication consistently accounted for >60% of total runtime. This finding directly influenced all subsequent design decisions: the backward pass must also be communication-efficient.

**Source files:** `timing.h`, `statistics.h`, `runtime.cpp`, `superstep.cpp`

---

## 4. GNN Forward Pass Deep Dive

### 4.1 Overview

The forward pass executes `num_layers` supersteps. Each superstep is Gather → Compute → Scatter, followed by MPI exchange to deliver messages for the next superstep.

### 4.2 Bootstrap (Seed Phase)

Before the layer loop begins, the runtime performs a **seed scatter**: each vertex emits its initial feature vector (`hidden_curr`, initialized from dataset features) to all out-neighbors. This seeds the inbox for layer 0.

```
Seed phase (superstep.cpp::seed()):
  For each vertex v in frontier_curr:
    For each out-neighbor u (via CSRGraph):
      m.src = global_id(v)
      m.dst = global_id(u)
      m.payload = h_curr[v]   ← raw float copy, NOT a tensor
      outbox.push(m)
  MPIExchange.exchange()      ← delivers remote messages
  Inbox now contains layer-0 inputs
```

After the seed phase, `frontier_curr` is rebuilt to contain only vertices that received at least one message (i.e., vertices with in-degree ≥ 1).

### 4.3 Gather Phase

```cpp
// superstep.cpp — Gather:
for each active vertex v in frontier_curr:
    aggr_buf[v] = sum of all payloads in inbox[v]
```

This implements **SUM aggregation** explicitly. No LibTorch operation is used here — the loop operates on raw float pointers to avoid autograd overhead. This is intentional: the aggregation buffer is read and written manually, never tracked by PyTorch.

After Gather, `aggr_buf` is cloned into `LayerCache[l].aggregated` — a LibTorch tensor snapshot used by the backward reconstruction phases.

### 4.4 Compute Phase

```cpp
// superstep.cpp — Compute:
for each active vertex v:
    pre_activation[v] = W @ aggr_buf[v] + b   ← raw loop (E11: captured)
    hidden_next[v]    = ReLU(pre_activation[v])
```

The compute phase uses the GNN layer's `W` and `b` tensors, but extracts their raw pointers (`W.data_ptr<float>()`) for the loop. This is again deliberately outside autograd. E11 captures `pre_activation` (for the ReLU backward) and `activated` (snapshot of `hidden_next`) for later use.

**LibTorch usage here:**
- `torch::Tensor::data_ptr<float>()` — extracts the underlying float array for fast raw computation
- `p.hidden_next.clone()` — creates a detached snapshot of the post-ReLU embedding

### 4.5 Scatter Phase

```cpp
// superstep.cpp — Scatter:
for each active vertex v:
    for each out-neighbor u (via CSRGraph):
        m.src = global_id(v)
        m.dst = global_id(u)
        m.payload = hidden_next[v]   ← again, raw float copy
        thread_outboxes[tid].push(m)
```

Messages are pushed into thread-local outboxes (indexed by OpenMP thread ID), then merged and enqueued into `MPIExchange`.

### 4.6 MPI Exchange Phase

```cpp
// mpi_exchange.cpp — Exchange:
MPIExchange.enqueue(messages from all outboxes)
MPIExchange.exchange()         ← MPI_Alltoall coordinated transfer
MPIExchange.wait_and_unpack(inbox, &trace)
                               ← E12: capture remote_contributors during unpack
```

The exchange uses MPI_Alltoall with variable-length buffers. Each rank sends batched messages to every other rank: local messages are routed directly into the inbox (no network overhead), remote messages go through MPI.

**E12 capture window:** Inside `wait_and_unpack()`, as remote messages are unpacked from the MPI receive buffer, the sender's global ID is recorded into `AggregationTrace[l].remote_contributors[v]`. This is the **only** moment this information is available — before the receive buffer is freed and before the inbox is flipped.

### 4.7 LibTorch Tensor Operations in Training

After the forward pass completes, the `Trainer` reconstructs a differentiable computation graph using LibTorch:

```cpp
// trainer.cpp — compute_loss_e13():
aggr_last = LayerCache[last_layer].aggregated.detach().requires_grad_(true)
aggr_last.retain_grad()         // ← preserve grad even though it's a leaf

hidden = torch::relu(
    torch::matmul(aggr_last, model.layer(last_layer).W().t())
    + model.layer(last_layer).b()
)                               // [N, H]

logits = torch::matmul(hidden, model.classifier_W().t())
       + model.classifier_b()   // [N, num_classes]

loss = torch::nn::functional::cross_entropy(logits, labels)

loss.backward()
// Now: aggr_last.grad() = ∂L/∂aggr[last_layer]  -- shape [N, H]
//      model.layer(last).W.grad() = defined
//      model.classifier_W.grad() = defined
```

**Key LibTorch operations used:**
- `torch::matmul(A, B)` — matrix multiplication, differentiable, records autograd node
- `torch::relu(x)` — element-wise ReLU, records backward mask
- `torch::nn::functional::cross_entropy(logits, labels)` — numerically stable CE loss
- `.detach().requires_grad_(true)` — detaches from any previous computation graph, promotes to leaf requiring gradients
- `.retain_grad()` — instructs autograd to retain the gradient even for non-leaf tensors
- `loss.backward()` — traverses the autograd graph from loss backward to all `requires_grad` tensors

---

## 5. Comparison to G2N2

### 5.1 Subsystem-Level Comparison

| Subsystem | G2N2 | This Prototype | Status |
|---|---|---|---|
| **Forward execution** | Distributed, partition-based, Pregel BSP | Distributed, MPI-based, Pregel BSP | **Equivalent** |
| **Message passing** | Partition-to-partition batched exchange | MPI_Alltoall per-layer | **Equivalent** |
| **Vertex ownership model** | Worker/partition ownership | VertexMap rank assignment | **Equivalent** |
| **Aggregation semantics** | SUM/MEAN/MAX supported | SUM only (E12 captures type metadata) | **Partially Equivalent** |
| **Activation preservation** | Full forward-state storage per layer | LayerCache per layer (E11) | **Equivalent** |
| **Dependency capture** | Computation trees preserve full execution DAG | AggregationTrace captures remote contributors | **Partially Equivalent** |
| **Local backward reconstruction** | Traverses computation tree layer by layer | scatter_only + linear_phase over ReverseCSR | **Equivalent** |
| **Cross-partition gradient transport** | Gradient messages along reversed dependencies | E14 rank-batched GradientExchange | **Equivalent** |
| **Full distributed backward** | Yes — computation tree drives backward | E15 not yet implemented | **Missing** |
| **Global computation graph** | Full computation-tree execution mode | E16 not yet implemented | **Missing** |
| **Mini-batch sampling** | Yes, neighbor sampling supported | Not implemented | **Missing** |
| **Multi-relation graphs** | Paper supports heterogeneous | Homogeneous only | **Missing** |

### 5.2 Backward propagation — The Central Gap

The most important architectural difference is in **how gradients propagate across layers**:

**G2N2 Backward:**
```
Loss
 ↓  (autograd through classification head)
grad_aggr[Layer_N]
 ↓  (computation tree tells us which vertices contributed to Layer_N)
grad_h[Layer_N-1 vertices]
 ↓  (computation tree at Layer_N-1)
grad_h[Layer_N-2 vertices]
 ...
grad_h[Layer_0 vertices]
 ↓
grad_W[0], grad_b[0]
```

**Current Prototype Backward (after E13/E14):**
```
Loss
 ↓  (autograd through classification head + last GNN layer)
grad_aggr[last_layer]
 ↓  Phase A: scatter_only via ReverseCSR (local edges only)
     Phase B: exchange_into via GradientExchange (remote edges via MPI)
grad_h[src_layer vertices] = grad_h_local + grad_h_remote
 ↓  Phase C: linear_phase (ReLU gate + matmul)
grad_W[0], grad_b[0]
```

The current prototype is equivalent to G2N2 for a *2-layer GNN* where only the boundary between Layer 1 and Layer 0 requires gradient transport. For deeper networks (3+ layers), the prototype does not yet propagate gradients through all intermediate layers.

### 5.3 Communication Philosophy Comparison

| Aspect | G2N2 | This Prototype |
|---|---|---|
| Forward unit | (src_partition, dst_partition) batch | MPI_Alltoall per layer |
| Backward unit | Gradient messages per computation tree edge | GradBatch: (v, dst_rank) → one gradient vector |
| Routing table | Computation tree | AggregationTrace |
| Deduplication | Partition-level | Per (local_v, remote_rank) pair |
| Transport | VCM-native | MPI_Sendrecv (rank-sequential) |

The E14 rank-batched design achieves **6× lower communication volume** compared to a naive edge-per-message approach, precisely matching G2N2's partition-level batching philosophy.

---

## 6. E11: Forward-State Cache

### 6.1 Why Activations Must Be Preserved

Consider the training reconstruction problem. After the forward pass completes, the runtime has consumed and discarded all the intermediate data it used to compute the final output. Specifically:

- `aggr_buf` is overwritten each superstep
- `hidden_next` is swapped into `hidden_curr` and used immediately
- Inbox messages are consumed and freed

Without any preservation, the only information available for training is the final hidden state `hidden_curr` — which contains no information about how earlier layers contributed.

**E11's answer:** Before any data is discarded, clone the critical tensors into a persistent `LayerCache` per layer.

### 6.2 What LayerCache Stores

```cpp
struct LayerCache {
    torch::Tensor aggregated;       // clone of aggr_buf after Gather     [N, H]
    torch::Tensor pre_activation;   // linear output before ReLU          [N, H]
    torch::Tensor activated;        // snapshot of hidden_next after ReLU [N, H]
    torch::Tensor active_mask;      // which vertices were active (bool)   [N]
    AggregationMetadata aggr_meta;  // E12: aggregation type (SUM/MEAN/MAX)
};
```

- **`aggregated`:** Required by E13's `linear_phase()` to compute `grad_W = grad_pre.T @ aggregated`
- **`pre_activation`:** Required by E13 to apply the ReLU backward gate: `grad_pre = grad_h * (pre_activation > 0)`
- **`active_mask`:** Required to filter out inactive vertices during backward (they should not receive gradient updates)
- **`activated`:** Available for future computation tree reconstruction

### 6.3 How LayerCache Differs From Autograd

PyTorch's autograd implicitly preserves all intermediate tensors while a computation is connected through the gradient graph. When you call `loss.backward()`, autograd follows the chain rule by traversing stored tensor references.

LayerCache is **not** an autograd mechanism. The tensors it stores are:
1. **Detached** from the forward autograd graph (`clone()` produces a new tensor not in any graph)
2. **Stored explicitly** by the developer, not recorded by PyTorch
3. **Used manually** in the backward reconstruction code via raw data pointers and LibTorch matmuls

This is precisely what makes it suitable for distributed execution: autograd cannot cross MPI boundaries, but explicit tensor storage can.

**Example:** At the time `LayerCache[0].aggregated` is created, the forward pass has already performed:
```
h[v] = 3.5, 1.2, -0.8, ... (from dataset features)
aggr_buf[w] += h[v]         (v is neighbor of w)
LayerCache[0].aggregated = aggr_buf.clone()
```
The `clone()` captures the numerical values of the aggregated result. It does NOT record "where these values came from" in a way autograd can traverse. That dependency capture is the job of E12's AggregationTrace.

### 6.4 The E11 Discovery

E11 was designed with the assumption that storing all activations would be sufficient to reconstruct a multi-layer training graph. This assumption was proven false:

```
cache[1].aggregated = sum of h[u] for u → w   (already a snapshot)
cache[0].activated  = h[u] for u              (also a snapshot)
```

Attempting to connect `cache[0].activated` into the computation graph of `cache[1].aggregated` fails because there is no PyTorch operation that created `cache[1].aggregated` from `cache[0].activated`. The aggregation happened in the raw C++ Gather loop, completely outside autograd.

Therefore `cache[1].aggregated.grad()` cannot flow back to `cache[0].activated` automatically. The dependency information has been lost. This realization drove the E12 redesign.

---

## 7. E12: Dependency Capture

### 7.1 The Fundamental Problem

After E11, the project had all forward state. But it still could not answer:

**"Which specific vertices contributed to vertex `v`'s aggregation at layer `l`?"**

- For **local contributors** (same rank): this is recoverable from the CSR graph. Vertices on this rank with an edge to `v` must have contributed (if they were active).
- For **remote contributors** (other ranks): this information is available *only* during `wait_and_unpack()`, when the MPI receive buffer contains tagged sender IDs. Once the receive buffer is freed, this information is **permanently lost**.

E12 ensures this information is captured at the only moment it exists.

### 7.2 AggregationTrace

```cpp
struct AggregationTrace {
    // remote_contributors[local_v] = list of global vertex IDs from OTHER ranks
    // that sent embedding messages to local vertex v during this layer's Gather.
    std::vector<std::vector<int32_t>> remote_contributors;  // [num_local][*]
    bool valid = false;
};
```

**Example with a small graph:**

```
Partition 0 owns: A (local=0), B (local=1)
Partition 1 owns: C (local=0), D (local=1)

Edges: A→C, B→C, D→B  (cross-partition edges)

During Layer 0 forward, Partition 1's Gather:
  C receives messages from: A (rank 0, global=0), B (rank 0, global=1)
    → trace.remote_contributors[0] = [0, 1]   (C's local_id=0)
  D receives nothing from remote
    → trace.remote_contributors[1] = []

During Layer 0 forward, Partition 0's Gather:
  B receives message from: D (rank 1, global=3)
    → trace.remote_contributors[1] = [3]   (B's local_id=1)
  A receives nothing from remote
    → trace.remote_contributors[0] = []
```

### 7.3 What AggregationTrace Does NOT Store

The design deliberately avoids redundancy:

| Information | Stored in AggregationTrace? | Why not? |
|---|---|---|
| Local contributor IDs | No | Recoverable from CSR graph (ReverseCSR) |
| Message payload values | No | Available in LayerCache.aggregated |
| Contribution counts | No | `remote_contributors[v].size()` computes it |
| Aggregation type | No | Stored in `LayerCache.aggr_meta.type` |

**Only** the remote contributor global IDs are stored — the one piece of information that cannot be recovered after unpack.

### 7.4 Memory Cost

On OGBN-Products (4 ranks, ~75K local vertices, ~70% remote fraction):

```
For 2 layers:
  vector metadata:   75,000 × 24 bytes     ≈  1.80 MB per layer
  remote IDs:        75,000 × 17.3 × 4     ≈  5.19 MB per layer
  Total:                                   ≈  14.0 MB per rank (2 layers)
```

This is negligible compared to the partition's hidden buffers (75K × 256 × 4 × 2 = ~150 MB).

### 7.5 Validated Results (E12)

OGBN-Products, 4 ranks, 2 layers:

```
Layer 0: 3,386,795 remote contributors captured
Layer 1: 3,330,713 remote contributors captured
Sanity check: remote_contributors_total == remote_msgs_received → YES
```

E12 is the **first major step toward G2N2**: it captures the dependency structure that makes distributed backward traversal possible.

---

## 8. ReverseCSR

### 8.1 Why ReverseCSR Exists

The forward graph stores **outgoing** edges: for each local vertex `v`, the CSR gives `v`'s outgoing neighbors `u` (i.e., `v → u`).

The backward pass requires traversal in the **reverse direction**: for each local vertex `u`, find all local vertices `v` such that `v → u` is an edge (i.e., v's whose output contributed to `u`'s aggregation during the forward pass).

This is not a standard operation on the forward CSR. It requires constructing the **transpose graph**.

### 8.2 Forward vs. Reverse Traversal

```
Forward graph (CSRGraph):
  row_ptr[v], row_ptr[v+1]  → range of col_idx for src v
  col_idx[e]                → global dst of edge e

Forward use:
  Scatter: for v in frontier:
               for u in col_idx[row_ptr[v]..row_ptr[v+1]]:
                   send h[v] → h[u]

Reverse graph (ReverseCSR):
  row_ptr[v], row_ptr[v+1]  → range of col_idx for dst v
  col_idx[e]                → GLOBAL ID of src u (same rank only)

Backward use (E13 scatter_only):
  for v in all local vertices:
      for u in rcsr.col_idx[rcsr.row_ptr[v]..rcsr.row_ptr[v+1]]:
          grad_h[to_local(u)] += grad_aggr[v]   ← SUM gradient
```

### 8.3 Small Graph Example

```
Graph (4 vertices, all local to rank 0):
  0 → 1 → 3
  0 → 2 → 3
  2 → 1

Forward CSR (col_idx stores out-neighbors):
  row_ptr: [0, 2, 3, 5, 5]
  col_idx: [1, 2,  3,  1, 3]
           (0's nbrs)(1's)(2's)

Reverse CSR (col_idx stores in-neighbors):
  Vertex 0: no incoming edges
  Vertex 1: incoming from 0, 2
  Vertex 2: incoming from 0
  Vertex 3: incoming from 1, 2

  row_ptr: [0, 0, 2, 3, 5]
  col_idx: [0, 2,  0,  1, 2]
           (1's in)(2's)(3's)
```

### 8.4 ReverseCSR Properties

- **Built once at load time** (`loader.cpp`, immediately after the forward CSR is constructed)
- **Immutable** — the topology does not change across epochs
- **Local-only** — only encodes same-rank edges. Cross-rank backward dependencies are handled by AggregationTrace + GradientExchange (E14)
- **Memory cost:** For OGBN-Products with 4 ranks and ~30% local edge fraction: approximately 2.52 MB per rank

```cpp
struct ReverseCSR {
    std::vector<int32_t> row_ptr;   // size: num_local_vertices + 1
    std::vector<int32_t> col_idx;   // global src IDs of local in-neighbors
    bool built() const { return !row_ptr.empty(); }
    int32_t in_degree(int32_t local_v) const {
        return row_ptr[local_v + 1] - row_ptr[local_v];
    }
};
```

The `col_idx` stores **global** src IDs (not local) so that E14's gradient routing can derive sender ranks via `vertex_map.owner_rank()`.

---

## 9. E13: Local Backward Reconstruction

### 9.1 The Problem E13 Solves

Before E13, backward propagation ended at the last GNN layer:

```
Training reconstruction (pre-E13):
  aggr_snapshot[last_layer] → Layer(last) → Classifier → Loss
                                                            ↓ backward
  Only Layer(last).W and Layer(last).b receive gradients.
  Layer(0).W and Layer(0).b: grad = undefined.
```

The issue is structural: Layer 0's parameters (W0, b0) never appear in the autograd graph because the path `aggr[L0] → Layer0 → scatter → MPI → gather → aggr[L1]` goes through raw C++ code that autograd cannot track.

E13's solution: **manually reconstruct** the portion of the backward pass that autograd cannot handle, using the preserved forward state (E11) and the dependency structure (E12).

### 9.2 E13 Algorithm

E13 operates on the boundary between the last GNN layer and the one before it. Given `grad_aggr[last_layer]` (obtained from `aggr_last.grad()` after `loss.backward()`):

**Phase A — scatter_only (local edge contribution):**

```
For each local edge v → u (via ReverseCSR):
    grad_h_local[u] += grad_aggr[v] * active_mask[v]

        │
        ▼ [scatter_add implementation]
v_idx = [dst vertex  for each local in-edge]
u_idx = [src vertex  for each local in-edge]
grad_per_edge = grad_aggr[v_idx] * active_mask[v_idx]  // [E, H]
grad_h_local = scatter_add(destination=u_idx, source=grad_per_edge, size=N)
```

This implements the backward of SUM aggregation: `∂(sum_u h[u]) / ∂h[u] = 1`, so each source vertex accumulates the gradient of its destination's aggregation.

**Phase C — linear_phase (parameter gradients):**

```
After Phase B (E14 remote exchange, see §10):
    grad_h = grad_h_local + grad_h_remote

ReLU backward gate:
    relu_mask = (LayerCache[src_layer].pre_activation > 0)  // [N, H]
    grad_pre  = grad_h * relu_mask                          // [N, H]

Parameter gradients:
    grad_W = grad_pre.T @ LayerCache[src_layer].aggregated  // [H, H]
    grad_b = grad_pre.sum(0)                                // [H]
```

### 9.3 Validation Results (E13)

After E13, on all MPI ranks:
```
layer[0].W grad.defined = 1  ✓
layer[0].b grad.defined = 1  ✓
layer[1].W grad.defined = 1  ✓
layer[1].b grad.defined = 1  ✓
classifier.W grad.defined = 1  ✓
classifier.b grad.defined = 1  ✓
```

**This was the first time Layer 0 ever received gradients in the project's history.**

### 9.4 Comparison to G2N2

| Aspect | G2N2 | E13 |
|---|---|---|
| Mechanism | Computation tree backward traversal | ReverseCSR scatter_add |
| Dependency source | Computation tree edges | Local ReverseCSR |
| Remote edges | Included in computation tree | Handled separately (E14) |
| Multi-layer recursion | Full depth | Single boundary (L1 → L0) only until E15 |
| Correctness | Full | Correct for local edges; E14 adds remote |

---

## 10. E14: Gradient Message Transport

### 10.1 The Distributed Backward Problem

After E13 Phase A, each rank holds `grad_h_local[u]` for all local vertices `u` — but this only accounts for **local in-edges**. The remote in-edges (where `u`'s forward contribution was aggregated by a vertex on another rank) have not yet been accounted for.

**Example:**

```
Forward:
  Rank 0 owns vertex A (local=0)
  Rank 1 owns vertex B (local=0)
  Edge: A → B  (Rank 0 sent h[A] to Rank 1 during forward)

Backward:
  Rank 1 has: grad_aggr[B] = ∂L/∂aggr[B]  (from autograd)
  Rank 0 needs: grad_h[A] += grad_aggr[B]   (since d(aggr[B])/d(h[A]) = 1 for SUM)

  But grad_aggr[B] lives on Rank 1     ← Must be sent to Rank 0
  And grad_h[A] must be updated on Rank 0 ← Remote accumulation
```

E14 is exactly this: transporting `grad_aggr[v]` from the rank that computed it (the destination rank in forward) to the rank that needs it (the source rank in forward).

### 10.2 Mathematical Basis

For SUM aggregation:
```
aggr[v] = Σ_u h[u]    for all u ∈ in-neighbors(v)

∂aggr[v]/∂h[u] = 1

Therefore:
grad_h[u] = Σ_{v: u sent to v} grad_aggr[v]
```

For local edges, E13 already handles this accumulation. For remote edges where `v` is on a different rank than `u`, the gradient value `grad_aggr[v]` must be transported from rank R_v to rank R_u.

E14 is the **transpose** of the forward exchange. Forward sent `h[u]` along edges u→v. E14 sends `grad_aggr[v]` along the same edges in reverse: v→u.

### 10.3 Rank-Batched Transport Design

A naive design would send one message per (v, u) pair — but this is wasteful. If three vertices u1, u2, u3 on rank R_u all contributed to the same local v, the gradient vector `grad_aggr[v]` would be sent three times to R_u.

**The rank-batched design** (adopted after explicit architectural review):

```
For each local vertex v:
    For each remote rank r:
        contributors_r = { u ∈ AggregationTrace[l].remote_contributors[v]
                          : vertex_map.owner_rank(u) == r }
        if contributors_r is non-empty:
            send GradBatch {
                payload = grad_aggr[v]   ← transmitted ONCE
                dsts    = contributors_r  ← list of u's on rank r
            } to rank r
```

**On the receiving rank r:**
```
For each received GradBatch b:
    For each u_global in b.dsts:
        u_local = vertex_map.to_local(u_global)
        grad_h[u_local] += b.payload        ← fan-out: same vector, multiple destinations
```

### 10.4 Communication Volume Comparison

For OGBN-Products (4 ranks, H=256):

| Transport scheme | Gradient payload bytes | Index bytes | Total |
|---|---|---|---|
| Edge-per-message (naive) | 10.5M × 1024 = **10.7 GB** | 42 MB | **10.75 GB** |
| Rank-batched (implemented) | 1.75M × 1024 = **1.79 GB** | 42 MB | **1.83 GB** |

The rank-batched design achieves **~6× reduction** in communication. This matches the forward pass volume in order of magnitude and preserves the VCM philosophy: communication happens at the (source_rank, destination_rank) granularity, not per-edge.

### 10.5 GradientExchange Protocol

```cpp
class GradientExchange {
    // Operates per remote rank, sequentially.
    // At most one (send, recv) buffer pair in memory at a time.

    void build_rank_groups(partition, target_layer, r,
                           v_locals, u_offsets, u_globals);
    // Builds the (v, rank_r, dst_list) index for all local vertices v
    // that received from rank r during forward.

    void exchange_into(partition, target_layer, grad_aggr, grad_h);
    // For each remote rank r:
    //   1. Exchange pair counts (MPI_Sendrecv, tag 1000)
    //   2. Chunked exchange of float payloads (tag 1002)
    //   3. Chunked exchange of dst global IDs (tag 1003)
    //   4. Unpack: for each received batch, accumulate into grad_h[to_local(dst)]
};
```

The exchange is **sequential per remote rank** (not Alltoall) to bound peak memory. At most `chunk_pairs_ × H × 4` bytes are in flight at any time.

### 10.6 AggregationTrace as E14 Routing Table

The AggregationTrace is the natural routing table for E14:

```
AggregationTrace[L].remote_contributors[v]
    = list of global source IDs that sent to v during forward layer L

E14 routing:
    for each v:
        for each u in AggregationTrace[L].remote_contributors[v]:
            target_rank = vertex_map.owner_rank(u)
            enqueue GradBatch(payload=grad_aggr[v], dst=u) for target_rank
```

No new data structure is required. AggregationTrace, designed in E12, serves perfectly as the E14 routing table.

### 10.7 Validation Status

The following properties have been verified in the implemented system:

| Property | Verification method | Result |
|---|---|---|
| Remote contributor IDs match forward sender IDs | `remote_contributors_total == remote_msgs_received` | ✓ Verified |
| Routing: each u reaches the correct owner rank | owner_rank(u_global) matches actual receiving rank | ✓ Verified |
| Transport: payloads arrive with correct content | MPI_Sendrecv round-trip test | ✓ Verified |
| Accumulation: grad_h increases after E14 | `||grad_h_after|| > ||grad_h_before||` | ✓ Verified |
| NP=1 baseline unchanged | world_size==1 is a no-op | ✓ Verified |

---

## 11. Current Blocker

### 11.1 Statement of the Blocker

**Gradient transport operates correctly. However, NP=1 and NP=4 training results are not yet numerically equivalent.**

Specifically: given the same model initialization, same dataset, and same number of training epochs, the gradient magnitudes and training metrics diverge between single-rank and 4-rank runs.

### 11.2 Why This Matters

Two separate concepts must be distinguished:

**Communication correctness (verified):** The right gradient values are sent to the right ranks. The AggregationTrace captures the correct remote contributors. MPI transport delivers the correct payloads. Accumulation targets the correct local vertex indices.

**Mathematical correctness (not yet verified):** The total gradient `grad_W0` computed across the distributed system should equal the `grad_W0` that autograd would compute on a single-rank run of the same data.

### 11.3 Simple Example of the Distinction

Consider a minimal graph: 4 vertices, 4 edges, 2 ranks.

```
Rank 0 owns: vertex A (local=0), vertex B (local=1)
Rank 1 owns: vertex C (local=0), vertex D (local=1)

Edges: A→C, B→C, B→D, C→A

Layer 0 aggregation:
  aggr[C] = h[A] + h[B]           ← C receives from A, B (remote from Rank 0's perspective)
  aggr[A] = h[C]                  ← A receives from C (remote from Rank 1's perspective)
```

Single rank (NP=1):
```
All 4 vertices on one rank.
autograd computes the full gradient correctly, traversing all dependencies.
grad_W0 = sum of all contributions from both layers.
```

4 ranks (NP=4):
```
Rank 1 computes: grad_aggr[C] = ∂L/∂aggr[C]
E13:             grad_h_local[C's local in-neighbors from Rank 1] — none (A, B are remote)
E14:             sends grad_aggr[C] to Rank 0 for both A and B
Rank 0 receives: grad_h_remote[A] += grad_aggr[C]
                 grad_h_remote[B] += grad_aggr[C]
linear_phase:    grad_W0 += grad_pre.T @ aggr_src_layer

Question: Is  grad_W0 (NP=4)  ==  grad_W0 (NP=1)?
```

This equality should hold by the chain rule of calculus. **Current investigation aims to identify why it does not numerically hold.**

### 11.4 Hypothesized Causes

The gap may arise from one or more of the following:

1. **Incorrect phase boundary:** The `linear_phase()` function applies the ReLU gate and computes `grad_W` using only the *local* portion of `LayerCache[src_layer].aggregated`. In a distributed setting, the aggregated values at the source layer were computed from both local and remote contributions. If the local aggregated tensor holds a different subset of contributions per rank, the overall sum of `grad_W` across ranks will be incorrect.

2. **Missing multi-layer recursion:** For a 2-layer GNN, E13/E14 currently only propagates gradients from Layer 1 back to Layer 0. For deeper networks or when Layer 0 itself needs to propagate further back, additional rounds of gradient transport are required.

3. **Accumulation asymmetry:** The forward scatter sums contributions symmetrically. The backward scatter must also sum all contributions, including those from remote sources not yet included in the local `grad_h` before `linear_phase` is called.

4. **Loss-weighted gradient reduction mismatch:** The E8 gradient averaging (`MPI_Allreduce / world_size`) may interact incorrectly with the manually assigned E13/E14 gradients on W0/b0.

### 11.5 Practical Impact

Until mathematical equivalence is achieved, the distributed prototype cannot claim to reproduce the same training dynamics as single-node full-batch training. This is the primary remaining correctness blocker before E15 can be designed.

---

## 12. E15 and E16: Roadmap

### 12.1 E15 — Distributed Backward Pass

**Goal:** Extend backward propagation beyond the single Layer1 → Layer0 boundary to support arbitrary-depth networks with full distributed correctness.

**What E15 must do:**

```
Current (post-E14):
  Loss → Layer_N (autograd) → grad_aggr[N] → E13+E14 → grad_W[0], grad_b[0]
  (Only one backward step: N → N-1 → Layer 0)

E15 target:
  Loss → Layer_N (autograd) → grad_aggr[N] → E13+E14 → grad_h[N-1]
       → E15 iterates: grad_aggr[N-1] from grad_h[N-1] → E13+E14 → grad_h[N-2]
       → ... → grad_h[0]
  Each iteration: ReverseCSR scatter + GradientExchange + linear_phase
```

**E15 interface:**

```cpp
// Proposed:
void DistributedBackward::run_all_layers(
    Partition& partition,
    GNNModel& model,
    GradientExchange& grad_exchange,
    const torch::Tensor& grad_aggr_last
)
{
    // Layer N → N-1:
    grad_h = scatter_only(partition, N, grad_aggr_last)
    grad_exchange.exchange_into(partition, N, grad_aggr_last, grad_h)
    [grad_W[N-1], grad_b[N-1]] = linear_phase(partition, N-1, grad_h)
    grad_aggr_prev = reconstruct_aggr_grad(partition, N-1, grad_h)

    // Layer N-1 → N-2:
    // repeat...
}
```

The key challenge in E15 is reconstructing `grad_aggr[l-1]` from `grad_h[l-1]`. For SUM aggregation:
```
h[v] = ReLU(W @ aggr[v] + b)
grad_aggr[v] = W.T @ (grad_h[v] * relu_mask[v])
```
This is a purely local operation that does not require MPI. E15 can therefore iterate through layers entirely using local operations between GradientExchange calls.

**E15 also resolves the current blocker:** once the full multi-layer backward is implemented correctly and validated against NP=1 results, gradient equivalence becomes verifiable.

### 12.2 E16 — Global Computation Graph Reconstruction

**Goal:** Implement the full G2N2 computation-tree training mode, where the backward pass is driven by an explicitly constructed global dependency graph rather than by the layer-by-layer iteration of E15.

**What E16 must do:**

1. **Build a global computation graph:** For each target labeled vertex `t`, trace back through all layers to identify which vertices contributed (directly or transitively) to `t`'s prediction. This is the **computation tree** rooted at `t`.

2. **Store computation trees:** The tree must be explicitly represented as a distributed data structure. Vertices on different ranks are tree nodes; edges are the GNN message-passing edges.

3. **Execute backward over the computation tree:** Gradient messages flow from the loss (at the root) backward through the tree edges to the leaf vertices at Layer 0.

**Key difference from E15:**

| E15 | E16 |
|---|---|
| Propagates gradients through all layers sequentially | Propagates gradients along specific computation tree paths |
| Every vertex participates in backward, including those that did not contribute to any labeled prediction | Only vertices in computation trees participate in backward |
| Simpler but may compute unnecessary gradients | More precise but requires more complex data structures |
| Closer to current implementation | Closer to G2N2's canonical design |

E16 would make the prototype the closest equivalent to G2N2's published architecture. It enables selective backward computation (critical for efficiency at scale) and would allow the runtime to prove end-to-end equivalence with G2N2's training protocol.

---

## 13. File-by-File Architecture Reference

### 13.1 Complete File Table

| File | Purpose | Architecture Layer | Phase Introduced |
|---|---|---|---|
| `src/main.cpp` | Entry point; MPI init, config parsing, runtime construction | System | E1 |
| `src/loader.cpp` | Dataset ingestion, CSR construction, ReverseCSR build, partition loading | Dataset / Partitioning | E1, E5, E12 |
| `include/loader.h` | Loader interface | Dataset | E1 |
| `src/dataset.cpp` | Feature/label loading from CSV; train/val/test mask construction | Dataset | E5 |
| `include/dataset.h` | Dataset struct: features tensor, labels tensor, split masks | Dataset | E5 |
| `src/partition.cpp` | Partition struct init; vertex-balanced, edge-balanced, hybrid partitioning | Partitioning | E1, E9, E9.1 |
| `include/partition.h` | Partition struct: all per-rank state; LayerCache, ReverseCSR, AggregationTrace | Partitioning / Runtime | E1, E11, E12 |
| `src/vertex_map.cpp` | Global↔local ID translation; owner_rank queries | Partitioning | E1 |
| `include/vertex_map.h` | VertexMap interface | Partitioning | E1 |
| `src/runtime.cpp` | Layer loop orchestration; forward cache/trace management; profiling | Runtime | E3, E10, E11, E12 |
| `include/runtime.h` | Runtime struct: RunConfig, Partition ownership, GNNModel | Runtime | E3 |
| `src/superstep.cpp` | Gather → Compute → Scatter; E11 cache capture; E12 trace capture | Forward Execution | E1, E11, E12 |
| `include/superstep.h` | SuperstepExecutor interface | Forward Execution | E1 |
| `src/compute.cpp` | ComputeEngine: zero_aggr, aggregate (SUM), linear_relu | Forward Execution | E1 |
| `include/compute.h` | ComputeEngine interface; aggregation primitives | Forward Execution | E1 |
| `src/mpi_exchange.cpp` | MPI_Alltoall-based message exchange; remote contributor capture (E12) | Communication | E3, E12 |
| `include/mpi_exchange.h` | MPIExchange class; statistics tracking | Communication | E3 |
| `src/gnn_model.cpp` | Parameter initialization; layer access; parameter list for optimizer | Model | E2 |
| `include/gnn_model.h` | GNNModel: layer registry, classifier head | Model | E2 |
| `src/gnn_layer.cpp` | Layer weight/bias initialization | Model | E2 |
| `include/gnn_layer.h` | GNNLayer: W tensor, b tensor | Model | E2 |
| `src/trainer.cpp` | Training loop; E13/E14 backward orchestration; metrics; early stopping | Training | E4, E8, E13, E14 |
| `include/trainer.h` | Trainer interface; TrainingConfig; TrainingMetrics | Training | E4 |
| `src/distributed_optimizer.cpp` | MPI_Allreduce gradient averaging | Training | E8 |
| `include/distributed_optimizer.h` | DistributedOptimizer::average_gradients() | Training | E8 |
| `src/local_backward.cpp` | E13: scatter_only, linear_phase, run | Backward | E13 |
| `include/local_backward.h` | LocalBackwardEngine interface | Backward | E13 |
| `src/gradient_exchange.cpp` | E14: rank-batched gradient transport | Backward | E14 |
| `include/gradient_exchange.h` | GradientExchange class; chunk protocol | Backward | E14 |
| `src/assertions.cpp` | Runtime invariant checks (frontier/inbox sync, NaN detection) | Debug | E1 |
| `include/assertions.h` | Assertion macros; phase-conditioned debug checks | Debug | E1 |
| `include/timing.h` | RuntimeTiming: total, communication, compute; per-layer breakdown | Profiling | E10 |
| `include/statistics.h` | RuntimeStatistics; GlobalStatistics; load balance metrics | Profiling | E10 |
| `include/config.h` | RunConfig, ModelConfig, PartitionMode, AggregationType enums | Configuration | E1 |
| `include/message.h` | Message struct: src, dst, payload | Communication | E1 |
| `include/inbox.h` | InboxBuffer: double-buffered message store, indexed by local vertex | Communication | E1 |
| `include/outbox.h` | OutboxBuffer: per-thread message accumulator | Communication | E1 |
| `include/frontier.h` | FrontierSet: bitset of active vertex IDs | Runtime | E1 |
| `include/graph.h` | CSRGraph: row_ptr, col_idx, num_vertices, num_edges | Graph | E1 |
| `include/vertex_state.h` | VertexState: active flag per vertex | Runtime | E1 |
| `src/checkpoint.cpp` | Model save/load to disk | Training | E4 |
| `include/checkpoint.h` | Checkpoint interface | Training | E4 |
| `src/frontier.cpp` | FrontierSet implementation | Runtime | E1 |
| `src/inbox.cpp` | InboxBuffer flip, write, read operations | Communication | E1 |
| `src/outbox.cpp` | OutboxBuffer push | Communication | E1 |
| `src/partition_e12.cpp` | AggregationTrace construction helper | Backward | E12 |
| `src/graph.cpp` | CSRGraph construction | Graph | E1 |

### 13.2 Dependency Diagram

```
main.cpp
  ├── loader.cpp          → dataset.cpp, partition.cpp, vertex_map.cpp
  │     └── (builds ReverseCSR for E12)
  ├── runtime.cpp         → superstep.cpp, mpi_exchange.cpp, gnn_model.cpp
  │     └── (orchestrates Gather → Compute → Scatter per layer)
  └── trainer.cpp         → local_backward.cpp, gradient_exchange.cpp
        │                    distributed_optimizer.cpp
        ├── compute_loss_e13()  → uses LayerCache (E11)
        ├── loss.backward()     → LibTorch autograd
        ├── scatter_only()      → uses ReverseCSR + LayerCache (E12, E13)
        ├── exchange_into()     → uses AggregationTrace + MPI (E12, E14)
        ├── linear_phase()      → uses LayerCache.aggregated, pre_activation (E11, E13)
        └── average_gradients() → MPI_Allreduce (E8)
```

---

## 14. Current Status Summary

### 14.1 Phase Completion Table

| Phase | Name | Status | Notes |
|---|---|---|---|
| **E1** | Graph Loader Foundation | ✅ **Complete** | CSR, VertexMap, Partition, Message established |
| **E2** | CSR Runtime Representation | ✅ **Complete** | GNNModel parameter ownership |
| **E3** | Distributed Runtime Framework | ✅ **Complete** | Gather → Compute → Scatter invariant |
| **E4** | Gather Phase | ✅ **Complete** | Trainer, optimizer, checkpointing |
| **E5** | Aggregate Phase | ✅ **Complete** | OGBN-Products dataset integration |
| **E6** | Compute Phase | ✅ **Complete** | Supervised training with CE loss |
| **E7** | Scatter Phase | ✅ **Complete** | Evaluation pipeline with best-model tracking |
| **E8** | Distributed Gradient Synchronization | ✅ **Complete** | MPI_Allreduce gradient averaging |
| **E9** | Edge-Balanced Partitioning | ✅ **Complete** | Near-perfect edge balance |
| **E9.1** | Hybrid Partitioning | ✅ **Complete** | Current preferred strategy |
| **E10** | Runtime Profiling | ✅ **Complete** | Per-layer timing, activity, cost decomposition |
| **E11** | Forward-State Cache | ✅ **Complete** | LayerCache per layer validated |
| **E12** | Computation Graph Capture | ✅ **Complete** | AggregationTrace + ReverseCSR validated |
| **E13** | Local Backward Reconstruction | ✅ **Complete** | Layer 0 gradients now defined on all ranks |
| **E14** | Gradient Message Transport | ⚠️ **Implemented, Validation Pending** | Transport verified; NP=1 ≠ NP=4 numerically |
| **E15** | Distributed Backward Pass | 🔲 **Not Started** | Requires E14 mathematical correctness first |
| **E16** | Global Computation Graph Reconstruction | 🔲 **Not Started** | Requires E15 |

### 14.2 What Has Been Demonstrated

- ✅ Full-batch distributed GNN forward propagation across multiple MPI ranks
- ✅ Message-passing via MPI_Alltoall with correct aggregation semantics
- ✅ Distributed parameter synchronization (E8) — single logical model across ranks
- ✅ Runtime profiling with evidence that communication dominates (E10)
- ✅ Three partitioning strategies (vertex/edge/hybrid balanced)
- ✅ OGBN-Products training with supervised cross-entropy loss
- ✅ Forward-state preservation via LayerCache (E11)
- ✅ Dependency capture via AggregationTrace + ReverseCSR (E12)
- ✅ Layer 0 gradients propagated for the first time (E13)
- ✅ Cross-rank gradient transport (E14) with verified routing and accumulation

### 14.3 What Remains

- ⚠️ **Immediate:** Resolve NP=1 vs NP=4 gradient equivalence (E14 mathematical correctness)
- 🔲 **E15:** Multi-layer distributed backward (generalizing E13/E14 to arbitrary depth)
- 🔲 **E16:** Global computation graph reconstruction (G2N2 analogue)
- 🔲 **Mini-batch support:** Neighbor sampling for scalability
- 🔲 **MAX/MEAN aggregation backward:** Currently only SUM is supported
- 🔲 **Distributed checkpointing:** Current checkpoints are rank-local
- 🔲 **Heterogeneous graph support:** Currently homogeneous only

### 14.4 Relationship to a Complete G2N2 Prototype

```
G2N2 Capability Coverage (Post-E14)

Distributed Forward:          ████████████████████ 100%
Activation Preservation:      ████████████████████ 100%
Dependency Capture:           ████████████████░░░░  80%  (SUM only; not full comp. tree)
Local Backward:               ████████████████░░░░  80%  (2-layer; needs multi-layer)
Gradient Transport:           ████████████░░░░░░░░  60%  (transport correct; math pending)
Full Distributed Backward:    ░░░░░░░░░░░░░░░░░░░░   0%  (E15 not started)
Computation-Tree Training:    ░░░░░░░░░░░░░░░░░░░░   0%  (E16 not started)

Overall G2N2 Equivalence:     approximately 55–65%
```

---

## Appendix: Key Design Invariants

The following invariants must be preserved by all future work:

| Invariant | Rule |
|---|---|
| **I-Exec-01** | The runtime execute model is always Gather → Compute → Scatter. Never change this. |
| **I-Exec-02** | `hidden_curr` is read-only during a superstep. `hidden_next` is write-only. `advance_hidden()` swaps at superstep boundary only. |
| **I-Frontier** | A vertex is active if and only if its inbox is non-empty. |
| **I-MPI-01** | All MPI collectives (MPI_Allreduce, MPI_Alltoall, MPI_Sendrecv with symmetric tags) must be called by all ranks simultaneously. |
| **I-Autograd** | Runtime execution is non-differentiable by design. Do not try to add `requires_grad` to message payloads or `aggr_buf`. |
| **I-E12** | AggregationTrace may only be read in E14+. It is captured during forward and must not be modified afterward. |
| **I-E13** | `scatter_only` must complete before `exchange_into`. `exchange_into` must complete before `linear_phase`. |
| **I-E8** | `average_gradients()` must be called after all gradient computations (autograd + E13/E14) and before `optimizer.step()`. |
| **I-VertexMap** | `vertex_map` is immutable after initialization. All rank/local lookups go through `vertex_map.owner_rank()` and `vertex_map.to_local()`. |

---

*End of ARCHITECTURE.md*
