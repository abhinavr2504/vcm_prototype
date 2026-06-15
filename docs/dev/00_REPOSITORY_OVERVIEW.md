# 00 — Repository Overview


##  Project 

This is a **distributed full-batch Graph Neural Network (GNN) training runtime** written from scratch in:

- **C++17** — core runtime, graph logic, data structures
- **MPI** (Message Passing Interface) — inter-process communication between cluster nodes
- **LibTorch** (PyTorch C++ API) — tensor math, trainable parameters, and loss computation
- **OpenMP** — shared-memory parallelism within a single node

The system is designed to train GNNs where the entire graph is distributed across multiple processes (MPI ranks), each process owning a subset of vertices and edges. This is called **full-batch distributed GNN training**.

The project takes strong architectural inspiration from the published **G2N2** system (a distributed GNN framework built on the Apache Giraph/Pregel BSP model), but re-implements the ideas from first principles for research and deep understanding.

---

## The Problem Being Solved

Training a GNN on a large graph faces two compounding challenges:

### Challenge 1 — Memory
A graph like **OGBN-Products** has 2.4 million nodes and 61 million edges. Storing all node embeddings and intermediate activations for training on a single GPU is infeasible.

### Challenge 2 — Communication
GNNs propagate information along edges. In distributed training, many edges cross partition boundaries — a node on Rank 0 may have a neighbor on Rank 3. Computing the GNN correctly requires message-passing across process boundaries over the network.

The naive approach (partition, compute locally, done) is incorrect. Correct distributed GNN training requires:
1. Routing messages across ranks during the forward pass
2. Routing **gradients** back along those same paths during the backward pass
3. Synchronizing model parameters after every gradient step

### Why Existing Frameworks Are Insufficient for This Research
Existing solutions like PyTorch Geometric (PyG), DGL/DistDGL, and DeepSpeed hide their communication mechanisms behind abstractions. This project intentionally **does not use any of those** — it builds everything explicitly so the execution mechanics are observable and modifiable.

---

## Why G2N2 Inspired This Work

The G2N2 paper demonstrated that:
1. GNN forward propagation can be expressed as a series of Pregel-style **supersteps** (Gather→Compute→Scatter)
2. The backward pass can be executed along **computation trees** — explicit data structures that record which vertices contributed to each output
3. This enables true distributed gradient computation without relying on PyTorch's built-in autograd to span machines (which it cannot do across MPI processes)

This project works toward the same goal: replacing stock autograd with manually-reconstructed backward passes that respect the graph topology and MPI boundaries.

---

## Current Status (as of June 2026)

The project has completed **E1 through E14** of its phased development roadmap.

| Capability | Status |
|---|---|
| Distributed forward pass (MPI) | ✅ Complete |
| OGBN-Products dataset loading | ✅ Complete |
| Three partitioning strategies | ✅ Complete |
| Distributed gradient sync (E8) | ✅ Complete |
| Runtime profiling (E10) | ✅ Complete |
| Forward-state cache (E11) | ✅ Complete |
| Dependency capture (E12) | ✅ Complete |
| Local backward reconstruction (E13) | ✅ Complete |
| Cross-rank gradient transport (E14) | ⚠️ Implemented, numerical equivalence pending |
| Full distributed backward pass (E15) | 🔲 Not started |
| Computation-tree training (E16) | 🔲 Not started |

**Current blocker:** E14 gradient transport is mechanically correct (routing verified, accumulation verified) but NP=1 and NP=4 produce numerically divergent gradients. Identifying the cause is the active investigation.

---

## Architectural Evolution — E1 to E16

The project evolved through a structured sequence of experiments (E-phases). Each phase added one capability and validated it before moving to the next.

| Phase | Name | Key Addition |
|---|---|---|
| **E1** | Graph Loader Foundation | CSR graph, VertexMap, Partition, Message, frontier-based execution |
| **E2** | CSR Runtime Representation | GNNModel parameter ownership |
| **E3** | Distributed Runtime Framework | Gather→Compute→Scatter loop, snapshot-based training validation |
| **E4** | Gather Phase | Trainer, Adam/SGD, epoch loop, checkpoint |
| **E5** | Aggregate Phase | OGBN-Products dataset integration |
| **E6** | Compute Phase | Cross-entropy loss, train/val/test masks |
| **E7** | Scatter Phase | Best-checkpoint tracking, evaluation pipeline |
| **E8** | Distributed Gradient Synchronization | MPI_Allreduce on parameter gradients |
| **E9** | Edge-Balanced Partitioning | Greedy edge-count-balanced partition assignment |
| **E9.1** | Hybrid Partitioning | Weighted cost `alpha × verts + beta × edges` |
| **E10** | Runtime Profiling | Per-layer timing, activity profiling, cost decomposition |
| **E11** | Forward-State Cache | `LayerCache` per layer: aggregated, pre_activation, activated, mask |
| **E12** | Computation Graph Capture | `AggregationTrace` (remote contributors) + `ReverseCSR` |
| **E13** | Local Backward Reconstruction | scatter_only + linear_phase; Layer 0 gradients now defined |
| **E14** | Gradient Message Transport | Rank-batched `GradientExchange`; cross-rank gradient delivery |
| **E15** | Distributed Backward Pass | *(Not started)* Multi-layer distributed backward |
| **E16** | Global Computation Graph Reconstruction | *(Not started)* G2N2-style computation-tree training |

---

## Repository Locations

```
vcm_prototype/
├── CMakeLists.txt          # Build system
├── ARCHITECTURE.md         # Architecture reference (pre-E14 era)
├── src/                    # All implementation files (.cpp)
├── include/                # All header files (.h)
├── docs/                   # Project documentation (research-style)
│   ├── dev/                # ← THIS DOCUMENTATION SYSTEM (you are here)
│   └── *.md                # Historical design docs and appendices
└── graphs/                 # Small toy graphs for testing
```

---

## Reading Guide for New Researchers

Read in this order:
1. **This file** (00_REPOSITORY_OVERVIEW.md) — orient yourself
2. **01_ARCHITECTURE.md** — understand the system structure
3. **02_EXECUTION_FLOW.md** — understand how code runs
4. **04_DATA_STRUCTURES.md** — understand the key objects
5. **06_FORWARD_PASS.md** and **07_BACKWARD_PASS.md** — understand GNN computation
6. **08_E_PHASE_EVOLUTION.md** — understand the history and why decisions were made
7. **11_CURRENT_STATUS_AND_BLOCKERS.md** — understand where we are today

---

## Cross-References

- Architecture details → [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
- Execution walkthrough → [02_EXECUTION_FLOW.md](02_EXECUTION_FLOW.md)
- File index → [03_FILE_INDEX.md](03_FILE_INDEX.md)
- Data structures → [04_DATA_STRUCTURES.md](04_DATA_STRUCTURES.md)
- MPI communication → [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)
- Forward pass → [06_FORWARD_PASS.md](06_FORWARD_PASS.md)
- Backward pass → [07_BACKWARD_PASS.md](07_BACKWARD_PASS.md)
- Phase history → [08_E_PHASE_EVOLUTION.md](08_E_PHASE_EVOLUTION.md)
- Debugging guide → [09_VALIDATION_AND_DEBUGGING.md](09_VALIDATION_AND_DEBUGGING.md)
- G2N2 comparison → [10_G2N2_COMPARISON.md](10_G2N2_COMPARISON.md)
- Current blockers → [11_CURRENT_STATUS_AND_BLOCKERS.md](11_CURRENT_STATUS_AND_BLOCKERS.md)
- Developer guide → [12_DEVELOPER_GUIDE.md](12_DEVELOPER_GUIDE.md)
- Code navigation → [13_CODE_NAVIGATION_GUIDE.md](13_CODE_NAVIGATION_GUIDE.md)
- Glossary → [14_GLOSSARY.md](14_GLOSSARY.md)
