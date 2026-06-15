# 06 — Forward Pass

> See also: [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | [02_EXECUTION_FLOW.md](02_EXECUTION_FLOW.md) | [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)

---

## What the Forward Pass Does

The forward pass computes the final embedding for each local vertex through `num_layers` supersteps of GNN computation. At the end, each rank holds the final `hidden_curr[N, H]` where `hidden_curr[v]` is the embedding of local vertex `v` after all GNN layers.

Mathematically, for an L-layer GCN:
```
h⁰[v]             = dataset.features[global(v)]          (initialization)
aggr^l[v]          = Σ_{u: u→v} h^(l-1)[u]               (Gather)
pre^l[v]           = W^l × aggr^l[v] + b^l                (linear transform)
h^l[v]             = ReLU(pre^l[v])                        (activation)

logits[v]          = classifier_W × h^L[v] + classifier_b
```

---

## Seed Phase — Initialization

**Location:** `superstep.cpp::SuperstepExecutor::seed()`  
**When:** Once per epoch, before the layer loop

```
Purpose: Seed the inbox with layer-0 inputs.

Each vertex v in frontier_curr (all vertices at start):
  For each out-neighbor u (via CSRGraph):
    m.src = global_id(v)
    m.dst = global_id(u)
    m.payload = hidden_curr[v]    ← initial feature vector
    thread_outboxes[0].push(m)

MPIExchange.enqueue(all messages)
MPIExchange.exchange()
MPIExchange.wait_and_unpack(inbox, nullptr)
   ↑ trace=nullptr here — seed layer not traced (no AggregationTrace)

After unpack:
  frontier_curr rebuilt = {v : inbox has messages for v}
  vertex_state[v].active = (inbox has messages)
```

After seed, only vertices that have **incoming edges** are active. Isolated vertices (no in-neighbors) drop out of the frontier permanently.

---

## Layer Loop

**Location:** `runtime.cpp::Runtime::run()`

```cpp
for (int32_t layer = 0; layer < cfg.num_layers; ++layer) {
    int32_t local_active  = partition.frontier_curr.popcount();
    int32_t global_active = 0;
    MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_active == 0) break;   // all done

    executor.run(partition, model.layer(layer), layer, &timing);

    partition.advance_hidden();      // swap hidden_curr ↔ hidden_next
    partition.advance_frontier();    // swap frontier_curr ↔ frontier_next, clear next
    partition.mpi_exchange.increment_superstep();
}
```

The `MPI_Allreduce` before each layer forces all ranks to agree on whether to continue. This is a global barrier — if any rank has active vertices, all ranks execute the layer.

---

## Superstep Detail

**Location:** `superstep.cpp::SuperstepExecutor::run()`

### Step 1 — Gather (Aggregation)

```cpp
// OpenMP parallel over active vertices
for (int32_t v = 0; v < N; ++v) {
    if (!frontier_curr.test(v)) continue;

    float* agg = partition.vaggr(v);     // raw pointer into aggr_buf[v×H]
    compute.zero_aggr(agg, H);           // zero out [H] floats

    for (const Message& m : inbox.get(v))
        compute.aggregate(m.payload.data(), agg, H);  // agg += payload (SUM)
}
```

This is **pure raw float arithmetic** — no autograd, no tensor operations. The result lives in `aggr_buf`, which is a LibTorch tensor but accessed only through its raw pointer.

**E11 cache capture (immediately after Gather):**
```cpp
LayerCache cache;
cache.aggregated = partition.aggr_buf.clone();   // ← full [N, H] tensor copy
cache.aggr_meta.type = AggregationType::SUM;
```

### Step 2 — Compute (Linear + ReLU)

```cpp
// OpenMP parallel over all vertices
for (int32_t v = 0; v < N; ++v) {
    if (!frontier_curr.test(v)) {
        // inactive: copy hidden_curr → hidden_next unchanged
        copy(hcurr(v), hcurr(v)+H, hnext(v));
        copy(hcurr(v), hcurr(v)+H, pre_act_ptr + v*H);
        continue;
    }

    const float* aggr_v = vaggr(v);
    float*       out_v  = hnext(v);
    float*       pre_v  = pre_act_ptr + v*H;

    // Linear: pre[v] = W @ aggr[v] + b
    const float* W_ptr = W.data_ptr<float>();  // [H, H]
    const float* b_ptr = b.data_ptr<float>();  // [H]
    for (int32_t d = 0; d < H; ++d) {
        float sum = b_ptr[d];
        for (int32_t h = 0; h < H; ++h)
            sum += W_ptr[d*H + h] * aggr_v[h];
        pre_v[d] = sum;                // ← capture pre-activation (E11)
    }

    // ReLU: h_next[v] = max(0, pre[v])
    for (int32_t d = 0; d < H; ++d)
        out_v[d] = (pre_v[d] > 0.0f) ? pre_v[d] : 0.0f;
}
```

**E11 cache capture (after Compute):**
```cpp
cache.pre_activation = pre_activation;                        // [N, H] tensor
cache.activated      = partition.hidden_next.clone();         // [N, H] snapshot
```

### Step 3 — Scatter (Message Emission)

```cpp
// OpenMP parallel over active vertices
for (int32_t v = 0; v < N; ++v) {
    if (!frontier_curr.test(v)) continue;

    const int tid = omp_get_thread_num();
    const float* hv = hnext(v);

    for (int32_t e = graph.row_ptr[v]; e < graph.row_ptr[v+1]; ++e) {
        const int32_t global_u = graph.col_idx[e];   // GLOBAL dst
        Message m;
        m.src     = vertex_map.to_global(v, rank);
        m.dst     = global_u;
        m.payload.assign(hv, hv + H);                // copy H floats
        thread_outboxes[tid].push(m);
    }
}
```

### Step 4 — MPI Exchange

```cpp
// Merge thread outboxes into MPIExchange
for (int t = 0; t < num_threads; ++t)
    for (const Message& m : thread_outboxes[t])
        mpi_exchange.enqueue(m);
    thread_outboxes[t].clear();

// Non-blocking send/receive
mpi_exchange.exchange();      // MPI_Alltoall + Isend/Irecv

// Block, deliver, E12 capture
AggregationTrace& trace = aggr_traces[layer];
trace.valid = false;
trace.remote_contributors.assign(num_vertices, {});

mpi_exchange.wait_and_unpack(inbox, &trace);   // ← E12 capture happens here

trace.valid = true;
```

### Step 5 — Frontier Update and Cache Storage

```cpp
// Build frontier_next from inbox availability
for (int32_t v = 0; v < N; ++v) {
    bool has_msgs = inbox.has(v);
    if (has_msgs) frontier_next.set(v);
    vertex_state[v].active = has_msgs;
}

// E11: capture active_mask from frontier_curr (not frontier_next)
cache.active_mask = tensor([frontier_curr.test(0), ..., frontier_curr.test(N-1)]);

partition.layer_cache.push_back(cache);
```

---

## Tensor Shapes Summary

| Buffer | Shape | Type | Direction |
|---|---|---|---|
| `hidden_curr` | `[N, H]` | `float32` | Read (Scatter source) |
| `hidden_next` | `[N, H]` | `float32` | Write (Compute output) |
| `aggr_buf` | `[N, H]` | `float32` | Write→Read (Gather result) |
| `LayerCache.aggregated` | `[N, H]` | `float32` | Stored (clone of aggr_buf) |
| `LayerCache.pre_activation` | `[N, H]` | `float32` | Stored (before ReLU) |
| `LayerCache.activated` | `[N, H]` | `float32` | Stored (after ReLU) |
| `LayerCache.active_mask` | `[N]` | `bool` | Stored (frontier_curr snapshot) |
| `Message.payload` | `[H]` | `float32` | In-flight (per message) |

All tensors are float32, CPU-bound, row-major contiguous.

---

## Why This is Non-Differentiable

PyTorch autograd builds a computation graph by recording operations on tensors with `requires_grad=True`. For the autograd graph to span the aggregation step, the following would need to be a tensor operation:

```
aggr[v] = sum of payloads from inbox[v]
```

But inbox payloads are `std::vector<float>` copies of floats that were extracted from `hidden_next` via `data_ptr<float>()`. The `data_ptr()` call extracts raw memory without recording any autograd operation. The `std::memcpy` in message serialization/deserialization is invisible to autograd.

Therefore, `aggr_buf` after the Gather phase is a tensor that holds correct numerical values, but has **no autograd parent**. Autograd cannot traverse from `aggr_buf` back to `hidden_next[layer-1]`.

This is why E11, E12, E13, and E14 exist: they reconstruct the necessary backward paths manually.

---

## Cross-References

- What happens at backward → [07_BACKWARD_PASS.md](07_BACKWARD_PASS.md)
- How MPI exchange is implemented → [05_MPI_COMMUNICATION.md](05_MPI_COMMUNICATION.md)
- What E11 cache is used for → [07_BACKWARD_PASS.md](07_BACKWARD_PASS.md)
