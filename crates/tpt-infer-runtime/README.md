# tpt-infer-runtime

**Zero-allocation, bare-metal inference execution engine.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-runtime` walks a `tpt_infer_graph::ComputationGraph` in topological order,
dispatches each node onto a `tpt_infer_ops::Backend`, and stores every intermediate
activation in a caller-provided `tpt_infer_core::BumpArena`. The inference hot path
performs **no heap allocation**: topological-sort scratch space, activation buffers, and
every operator's output all live in the arena; only the final marked-output tensors are
copied into owned `TensorVec` values after execution finishes.

## Architecture

1. **Topological order** — Kahn's algorithm running entirely in arena scratch space
   (indegree array, CSR adjacency, ring-buffer queue); a cycle yields
   `RuntimeError::CycleDetected`.
2. **Input binding** — every `Operator::Input` node resolves to a named `Initializer`
   (weight) or the next supplied runtime input slice.
3. **Activation layout** — each non-input node gets a disjoint range of one contiguous
   `f32` activation buffer allocated with a single `BumpArena::alloc_slice` call.
4. **Dispatch** — nodes execute in order; kernel calls go through the `Backend` HAL
   where it has coverage (`matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`,
   `sigmoid`, `gelu`) and through this crate's own [`kernels`] module otherwise
   (broadcasting binary ops, pooling, batch norm, transpose, concat, data movement).
5. **Materialization** — marked outputs are copied out of the arena into owned
   `TensorVec` values — the only heap traffic in the whole call, and it happens after
   the hot path.

## Usage

`no_std` kernels, always available:

```rust
use tpt_infer_runtime::{argmax, kernels::{binary, BinaryOp}};

let a = [1.0f32, -2.0, 3.0];
let b = [10.0f32; 3];
let mut out = [0.0f32; 3];
binary(BinaryOp::Add, &a, &[3], &b, &[3], &mut out, &[3]).unwrap();
assert_eq!(out, [11.0, 8.0, 13.0]);
assert_eq!(argmax(&a), 2);
```

Full graph execution (feature `std`):

```rust,ignore
use tpt_infer_core::BumpArena;
use tpt_infer_ops::dispatch::select_backend;
use tpt_infer_runtime::{execute_graph, required_arena_bytes};

let graph = /* a ComputationGraph with marked outputs */;
let input = [0.0f32; 1 * 3 * 224 * 224];
let mut mem = vec![0u8; required_arena_bytes(&graph)];
let mut arena = BumpArena::new(&mut mem);
let backend = select_backend();
let outputs = execute_graph(&graph, &[&input], &mut arena, &backend).unwrap();
// outputs[i] is a TensorVec<f32> with the shape declared by the graph.
```

`execute` is a convenience wrapper over `execute_graph` for the common single-input,
single-output case; `required_arena_bytes` sizes the backing buffer up front.

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `alloc` | off | `tpt-infer-core/alloc` (heap-backed tensors in `core`) |
| `std` | off | `alloc` plus `tpt-infer-graph/std` and `tpt-infer-ops/std`; unlocks `execute`/`execute_graph`/`required_arena_bytes` |

With no features enabled, only `kernels`, `argmax`, and `RuntimeError` are available —
enough to build a custom plan-based executor on bare metal without the graph IR.

## `no_std`

`#![cfg_attr(not(any(test, feature = "alloc")), no_std)]`. `cargo build -p
tpt-infer-runtime --target thumbv7m-none-eabi` (default features, i.e. no `alloc`/`std`)
verifies the bare-metal build. On bare-metal targets the graph IR is unavailable, so
graph execution currently requires `std`; the kernels themselves are `no_std` and ready
for a future plan-based entry point that doesn't need `tpt-infer-graph`.

## Relationship to the rest of the workspace

Depends on `tpt-infer-core` (arena/tensor types), `tpt-infer-graph` (the IR it walks),
and `tpt-infer-ops` (the `Backend` it dispatches onto). It intentionally does **not**
depend on `tpt-infer-compile` yet — see that crate's changelog for status — so this
crate executes `ComputationGraph` IR directly rather than a `CompiledModel`. See the
[workspace README](../../README.md) for how the interpreted and AOT-compiled execution
paths compare.
