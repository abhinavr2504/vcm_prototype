# 07 — Backward Pass

> See also: [06_FORWARD_PASS.md](06_FORWARD_PASS.md) | [04_DATA_STRUCTURES.md](04_DATA_STRUCTURES.md) | [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)

---

## The Core Problem

After the forward pass completes, we have:
```
logits[v] = classifier_W × ReLU(W¹ × ReLU(W⁰ × aggr⁰[v] + b⁰) + b¹) + classifier_b
```

We want `∂L/∂W⁰`, `∂L/∂b⁰`, `∂L/∂W¹`, `∂L/∂b¹`, `∂L/∂classifier_W`, `∂L/∂classifier_b`.

Stock PyTorch autograd can compute gradients for **last layer** and classifier parameters. But it cannot compute gradients for W⁰ and b⁰ because the path from logits back to W⁰ goes through:

```
logits → h¹ → aggr¹ → h⁰ → aggr⁰ → W⁰, b⁰
```

And the step `h⁰ → aggr¹` is the forward pass **Gather**, which uses raw C++ loops invisible to autograd.

The backward pass is implemented in four phases across three E-milestones:

| Phase | Name | E-phase | File |
|---|---|---|---|
| **½** | Loss reconstruction | E11 | `trainer.cpp::compute_loss_e13()` |
| **½** | Autograd backward (last layer) | PyTorch | `loss.backward()` |
| **A** | Local scatter | E13 | `local_backward.cpp::scatter_only()` |
| **B** | Remote gradient delivery | E14 | `gradient_exchange.cpp::exchange_into()` |
| **C** | Linear backward | E13 | `local_backward.cpp::linear_phase()` |

---

## Phase ½ — Loss Reconstruction (`compute_loss_e13`)

**Location:** `trainer.cpp`

The key insight is to "re-enter" the computation graph at the last layer's aggregation buffer, which was saved in `LayerCache`. We detach this value from any previous history, promote it to a leaf requiring gradients, and rebuild the forward graph from there.

```cpp
int32_t last_layer = num_layers - 1;

// Detach from any previous computation and make a new gradient-tracked leaf
torch::Tensor aggr_last = partition.layer_cache[last_layer]
                              .aggregated          // saved during Gather
                              .detach()            // sever any old gradient chain
                              .requires_grad_(true); // new leaf
aggr_last.retain_grad();        // preserve grad even though it's a leaf

// Rebuild the differentiable graph MANUALLY
torch::Tensor h_last = torch::relu(
    torch::matmul(aggr_last, model.layer(last_layer).W().t())
    + model.layer(last_layer).b()
);           // [N, H] — differentiable through W¹, b¹

torch::Tensor logits = torch::matmul(h_last, model.classifier_W().t())
                     + model.classifier_b();  // [N, num_classes]

// Filter to train mask; compute cross-entropy loss
torch::Tensor loss = cross_entropy_loss(logits, labels, train_mask);
```

After this, PyTorch's autograd graph connects: `loss → logits → h_last → aggr_last → (W¹, b¹, classifier_W, classifier_b)`.

**Note:** `aggr_last` is a leaf. Its gradient, `∂L/∂aggr_last`, is what E13 and E14 will use. **Layer 0 parameters are NOT yet in this graph.**

---

## ½ Continued — `loss.backward()`

```
loss.backward()
```

After this call:
- `model.layer(last_layer).W().grad()` = defined ✓
- `model.layer(last_layer).b().grad()` = defined ✓
- `model.classifier_W().grad()` = defined ✓
- `model.classifier_b().grad()` = defined ✓
- `aggr_last.grad()` = `∂L/∂aggr[last_layer]` ← shape [N, H] ✓
- `model.layer(0).W().grad()` = **undefined** ✗ (not in this graph)

The gradient `aggr_last.grad()` is what drives E13 and E14.

---

## Phase A — Local Scatter (`LocalBackwardEngine::scatter_only`)

**Location:** `local_backward.cpp`  
**Purpose:** Propagate `∂L/∂aggr[v]` backward through the SUM aggregation to all local source vertices.

### Mathematics

For SUM aggregation:
```
aggr[v] = Σ_{u: u→v} h[u]

∂aggr[v]/∂h[u] = 1  for each contributing u

∂L/∂h[u] += Σ_{v: u→v} ∂L/∂aggr[v]
```

### Implementation

```cpp
// Build edge index from ReverseCSR
for (int32_t v = 0; v < N; ++v) {
    for (int32_t e = rcsr.row_ptr[v]; e < rcsr.row_ptr[v+1]; ++e) {
        int32_t g_src = rcsr.col_idx[e];
        int32_t l_src = vertex_map.to_local(g_src);
        v_vec.push_back(v);       // dst vertex
        u_vec.push_back(l_src);   // src vertex (local ID)
    }
}

// grad_per_edge[e] = grad_aggr[v_vec[e]] * active_mask[v_vec[e]]
torch::Tensor grad_per_edge =
    grad_aggr.detach().index_select(0, v_idx)          // [E, H]
    * active_mask.index_select(0, v_idx).unsqueeze(1); // filter inactive

// scatter-add: grad_h[u] += grad_per_edge[e] for all edges e touching u
grad_h.scatter_add_(0, u_idx_expanded, grad_per_edge);
```

**Result:** `grad_h[N, H]` — local contributions only. Remote edge contributions are missing.

**Why active_mask is applied:**
- `active_mask[v]` = whether vertex v was in frontier_curr during this layer
- If v was inactive, it did not aggregate and therefore its `aggr[v]` is not meaningful
- The gradient of an inactive vertex's aggregation should not flow backward

---

## Phase B — Remote Gradient Delivery (`GradientExchange::exchange_into`)

**Location:** `gradient_exchange.cpp`  
**Purpose:** Deliver `∂L/∂aggr[v]` from the rank that holds v (destination during forward) to the rank that holds u (source during forward), for all cross-rank edges u→v.

### The Cross-Rank Problem

```
Forward:  Rank 0 (vertex u) → [MPI] → Rank 1 (vertex v)
              h[u] was sent to v during Gather

Backward: Rank 1 holds grad_aggr[v] = ∂L/∂aggr[v]
          Rank 0 needs grad_h[u]    += ∂L/∂aggr[v]  (since d(aggr[v])/d(h[u]) = 1)

          → Rank 1 must send grad_aggr[v] to Rank 0
```

E14 routing uses `AggregationTrace[last_layer].remote_contributors`:
```
AggregationTrace[v] = [u1, u2, u3, ...]  (global IDs of remote senders to v)
```

### Build Rank Groups

```cpp
build_rank_groups(partition, target_layer, rank_r,
                  v_locals,   // [i] = local v that has contributors from rank_r
                  u_offsets,  // [i..i+1) = range in u_globals for v_locals[i]
                  u_globals); // [*] = global IDs of u's on rank_r that sent to v
```

### Send Side (what this rank sends to rank r)

```cpp
// For each (v, rank_r) pair:
//   payload = grad_aggr[v_local]   (H floats, sent ONCE regardless of how many u's)
//   dsts    = u_offsets + u_globals (CSR-style list of global u IDs on rank_r)

MPI_Sendrecv(send_float_, send_int_,   // payloads + index (chunk)
             recv_float_, recv_int_);  // receive from rank_r simultaneously
```

### Receive Side (what this rank receives from rank r)

```cpp
// Unpack received payloads:
for (each received (payload [H], dst_list)):
    for each global_u in dst_list:
        local_u = vertex_map.to_local(global_u)
        grad_h[local_u] += payload   // fan-out: same vector, multiple destinations
```

**Key insight:** `payload` (= `grad_aggr[v]` from the sending rank) is applied to **each** u in the dst_list independently. This is correct because:
```
∂L/∂h[u1] += ∂L/∂aggr[v]
∂L/∂h[u2] += ∂L/∂aggr[v]    (same v, different u's)
```

After `exchange_into()`, `grad_h` contains both local and remote contributions — it is complete.

---

## Phase C — Linear Backward (`LocalBackwardEngine::linear_phase`)

**Location:** `local_backward.cpp`  
**Purpose:** Using the complete `grad_h`, compute `grad_W` and `grad_b` for Layer 0.

### Mathematics

```
h[v] = ReLU(W⁰ × aggr⁰[v] + b⁰)
pre[v] = W⁰ × aggr⁰[v] + b⁰

∂L/∂pre[v] = ∂L/∂h[v] * I(pre[v] > 0)    ← ReLU backward (0 if ReLU was gated)

∂L/∂W⁰ = Σ_v  ∂L/∂pre[v]ᵀ × aggr⁰[v]ᵀ   = grad_pre.T @ aggr (matrix, [H, H])
∂L/∂b⁰ = Σ_v  ∂L/∂pre[v]                  = grad_pre.sum(0)   (vector, [H])
```

### Implementation

```cpp
// ReLU backward gate
torch::Tensor relu_mask = LayerCache[0].pre_activation
                              .detach().gt(0.0f).to(kFloat32);  // [N, H]
torch::Tensor grad_pre = grad_h * relu_mask;                    // [N, H]

// Parameter gradients
torch::Tensor grad_W = torch::matmul(
    grad_pre.t(),                          // [H, N]
    LayerCache[0].aggregated.detach()      // [N, H]
);                                         // → [H, H]

torch::Tensor grad_b = grad_pre.sum(0);    // [H]
```

These are assigned back to the model parameter grads in `trainer.cpp`:
```cpp
model.layer(0).W().mutable_grad() = grad_W;   // or allreduced version
model.layer(0).b().mutable_grad() = grad_b;
```

---

## Full Backward Data Flow Diagram

```
loss (scalar)
    │ loss.backward()
    ▼
grad_aggr_last [N, H] = ∂L/∂aggr[layer_last]
    │
    ├── Phase A: scatter_only (local edges via ReverseCSR)
    │       for each local edge u→v:
    │         grad_h_local[u] += grad_aggr_last[v] * active_mask[v]
    │
    ├── Phase B: exchange_into (remote edges via MPI)
    │       AggregationTrace[last] → routing table
    │       MPI_Sendrecv: send grad_aggr[v] to owner(u) for all remote u→v
    │       receive: grad_h_local[u] += received payload
    │
    │   grad_h = grad_h_local (now complete: local + remote)
    │
    └── Phase C: linear_phase
            relu_mask = (LayerCache[0].pre_activation > 0)
            grad_pre  = grad_h × relu_mask
            grad_W0   = grad_pre.T @ LayerCache[0].aggregated
            grad_b0   = grad_pre.sum(0)
                              │
                              ▼
                    model.layer(0).W.grad = grad_W0
                    model.layer(0).b.grad = grad_b0

    ─────────────────────────────────────────────────────────
    E8: MPI_Allreduce on ALL gradients (W0, b0, W1, b1, clf.W, clf.b)
    optimizer.step()
```

---

## What Is Not Yet Implemented

For a **3-layer or deeper** GNN, the backward pass would require another round of:
1. Computing `grad_aggr[layer 1]` from `grad_h[layer 1]`
2. Running E13 scatter + E14 exchange + E13 linear_phase for layer 1 → layer 0

This is **E15 — Distributed Backward Pass** and is not yet implemented. Currently, the system only reconstructs the Layer1→Layer0 boundary.

For computation-tree based training (where gradients are routed along specific dependency paths rather than all edges), see the future **E16** phase.

---

## Cross-References

- Forward context → [06_FORWARD_PASS.md](06_FORWARD_PASS.md)
- AggregationTrace that drives E14 → [04_DATA_STRUCTURES.md](04_DATA_STRUCTURES.md#aggregationtrace)
- MPI transport protocol for E14 → [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)
- Current status and blockers → [11_CURRENT_STATUS_AND_BLOCKERS.md](11_CURRENT_STATUS_AND_BLOCKERS.md)
