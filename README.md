# GNN Runtime — Phase 1 (complete)

Lightweight Pregel-style distributed GNN runtime. C++17 / OpenMP / MPI.

## Phase 1 scope

- Single process, single thread
- True Pregel message-passing semantics
- Vertices never read neighbour hidden states directly
- Aggregation over inbox message payloads only
- No MPI, no OpenMP, no libtorch, no training

## Fixes applied (end of Phase 1)

**Ping-pong embedding corruption** (`superstep.cpp` — compute pass in `run()`).
`advance_hidden()` is an unconditional full-buffer `std::swap`. Any inactive
vertex that skipped the compute pass left its `hidden_next` slot unwritten;
after the swap that stale or zero slot became `hidden_curr`, corrupting the
vertex's embedding for all subsequent layers. Fix: inactive vertices now copy
`hcurr → hnext` in the compute loop so every slot is written before the swap.
Silent on `toy.graph` (all vertices active every layer); exposed by any graph
with a sparse frontier such as `chain.graph`.

**`seed()` direct frontier bit write** (`superstep.cpp` — frontier rebuild in `seed()`).
The frontier rebuild in `seed()` wrote `frontier_curr.bits[v]` directly and
re-assigned `frontier_curr.num_vertices`, bypassing `FrontierSet::set()` and
duplicating a field already set by `Partition::init()`. Fix: replaced with
`frontier_curr.clear()` followed by `frontier_curr.set(v)`, consistent with
every other frontier write in the codebase.

## Build

```bash
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude \
  src/graph.cpp src/frontier.cpp src/inbox.cpp src/outbox.cpp \
  src/compute.cpp src/partition.cpp src/loader.cpp \
  src/superstep.cpp src/runtime.cpp src/main.cpp \
  -o gnn_runtime
```

Or with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./gnn_runtime graphs/toy.graph
```

Graph file format:

```
# comment
<num_vertices> <num_edges>
<src> <dst>
...
```

## File map

```
include/
  config.h          RunConfig
  graph.h           CSRGraph  (row_ptr, col_idx)
  vertex_state.h    VertexState (active flag)
  message.h         Message POD (src, dst, payload: vector<float>)
  frontier.h        FrontierSet (set/test/swap/popcount/clear)
  inbox.h           InboxBuffer (double-buffered per-vertex, flip private in Phase 2)
  outbox.h          OutboxBuffer (scatter staging; thread-local in Phase 2)
  compute.h         ComputeEngine (zero_aggr, aggregate, linear_relu)
  partition.h       Partition (owns all local state; flat embedding buffers)
  loader.h          load_partition()
  superstep.h       SuperstepExecutor::seed() + run()
  runtime.h         Runtime (layer loop, hidden/frontier advance)

src/
  *.cpp             implementations

graphs/
  toy.graph         5-vertex directed graph
```

## Execution model

### Bootstrap (once, before layer loop)

```
seed():
  for each active vertex v:
    emit Message{src=v, dst=u, payload=hcurr(v)} for each out-neighbour u
  serial merge: outbox → inbox.write()
  outbox.clear()
  inbox.flip()
  rebuild frontier_curr: active iff inbox non-empty
```

### Per superstep (one per GNN layer)

```
1. gather:   for each active v: zero aggr(v); for each msg in inbox(v): aggregate(msg.payload) → aggr(v)
2. compute:  for each v:
               active   → linear_relu(aggr(v), W, b) → hnext(v)
               inactive → copy hcurr(v) → hnext(v)   [preserves embedding across advance_hidden()]
3. scatter:  for each active v: emit Message{src=v, dst=u, payload=hnext(v)} for each out-neighbour u
4. merge:    outbox → inbox.write(); outbox.clear()
5. flip:     inbox.flip()   [moves to MPIExchange::wait_and_unpack in Phase 2]
6. frontier: frontier_next.set(v) for all v where inbox non-empty
7. advance:  Runtime calls advance_hidden(); advance_frontier()
```

## Invariants

| ID    | Invariant |
|-------|-----------|
| I-01  | inbox.write() called only in serial merge phase |
| I-02  | frontier_next.set() called only in serial merge phase |
| I-05  | hidden_curr read-only during superstep; advance_hidden() at boundary only |
| I-07  | aggr zeroed per superstep via zero_aggr(); never allocated at runtime |
| I-08  | ComputeEngine never called from Runtime or MPIExchange |
| I-msg | Vertices never directly read neighbour hidden states; all aggregation over inbox payloads |
| I-frt | Frontier activation derived exclusively from inbox non-empty; never from graph topology directly |
| I-swap | Every hidden_next slot is written in the compute pass (active: linear_relu; inactive: hcurr copy) before advance_hidden() swaps the buffers |
