# tpt-infer-graph

**Compile-time-checked computation graph builder with const-generic shape validation.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-graph` offers two complementary ways to describe a model, sitting between
`tpt-infer-core` (tensors/shapes) and the crates that consume a built graph
(`tpt-infer-onnx`, `tpt-infer-compile`, `tpt-infer-runtime`, `tpt-infer-quantize`):

- **`GraphBuilder`** — a type-state builder whose type parameters track tensor shapes,
  so an incompatible operation (e.g. a `[1, 784] x [10, 10]` matmul) is a *compile
  error*, not a runtime panic.
- **`ComputationGraph`** — a dynamic graph (rank up to `tpt_infer_core::MAX_RANK`) of
  `Node`/`Edge`/`Initializer` values with `topological_sort()`, which is what the ONNX
  loader, AOT compiler, and runtime actually execute against.

## Compile-time shape checking

```rust
use tpt_infer_graph::{GraphBuilder, Sh};

// Input tensor of shape [1, 784].
let mut b = GraphBuilder::<Sh<1, 784>>::input("x");

// [1, 784] x [784, 128] -> [1, 128], verified by the compiler.
let w1 = b.add_input::<Sh<784, 128>>("w1");
let mut b = b.matmul(w1).relu();

// [1, 128] x [128, 10] -> [1, 10].
let w2 = b.add_input::<Sh<128, 10>>("w2");
let b = b.matmul(w2).softmax(-1);

let graph = b.into_graph();
assert_eq!(graph.nodes().len(), 7);
```

`Sh<M, K>` is a rank-2 compile-time shape marker (dense-layer shapes); `MatMulShape`,
`AddShape`, `FlattenShape`, and `TransposeShape` are the trait-level shape-arithmetic
rules the builder's methods are generic over. Shapes that involve division or runtime
attribute values (e.g. `Conv2d` output size, rank-4 `Flatten`) can't be expressed as
stable const-generic arithmetic and are only checked once lowered into a
`ComputationGraph`.

## Dynamic graphs

```rust
use tpt_infer_graph::{ComputationGraph, Node, Operator};

let mut g = ComputationGraph::new();
let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 784]).unwrap()).unwrap();
let y = g.add_node(Node::new(1, Operator::Relu, vec![x], &[1, 784]).unwrap()).unwrap();
g.mark_output(y).unwrap();
assert_eq!(g.topological_sort().unwrap(), vec![0, 1]);
```

`Operator` is a `#[non_exhaustive]` enum mirroring common ONNX opset-17 operators
(`MatMul`, `Conv2d`, `Add`/`Sub`/`Mul`/`Div`, `Relu`, `Sigmoid`, `Gelu`, `Softmax`,
`Reshape`, `Flatten`, `Transpose`, `Concat`, `MaxPool2d`, `AveragePool2d`, `BatchNorm`,
and `Custom(name)` for anything unmapped).

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `std` | yes | `alloc`-backed `GraphBuilder`/`ComputationGraph`/`Operator` modules (via `tpt-infer-core/alloc`) |

Without `std`, the crate compiles but exposes none of the graph-building API — it is a
placeholder for a future no-alloc IR representation.

## `no_std`

`#![no_std]` when built without the `std` feature. In practice the graph-building types
require `alloc` today, so `std` (the default) is needed to use this crate's public API.

## Relationship to the rest of the workspace

Depends on `tpt-infer-core` for `Shape`/`MAX_RANK`. Consumed by `tpt-infer-onnx` (builds
a `ComputationGraph` from a parsed model), `tpt-infer-compile` (AOT codegen walks the
graph in topological order), `tpt-infer-runtime` (interpreted execution), and
`tpt-infer-quantize` (PTQ walks the graph to find quantizable weights). See the
[workspace README](../../README.md) for the full picture.
