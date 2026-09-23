# tpt-infer-compile

**Ahead-of-time compiler that translates computation graphs into optimized Rust code.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

Rather than interpreting a `tpt_infer_graph::ComputationGraph` node-by-node at runtime
(that's `tpt-infer-runtime`'s job), `tpt-infer-compile` walks the graph once, ahead of
time, and emits self-contained Rust source for a `pub fn execute(input: &[f32]) ->
Vec<f32>` function. Handing that source to `rustc`/LLVM lets the compiler apply
model-specific optimizations (loop unrolling, constant folding, register allocation)
that a generic interpreter loop cannot.

## Usage

```rust
use tpt_infer_graph::{ComputationGraph, Node, Operator};
use tpt_infer_compile::aot_compile;

let mut g = ComputationGraph::new();
let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap()).unwrap();
let y = g.add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap()).unwrap();
g.mark_output(y).unwrap();

let model = aot_compile(&g).unwrap();
assert!(model.source().contains("pub fn execute"));
assert_eq!(model.input_shape(), &[1, 4]);
```

`CompiledModel::source()` is *not* automatically compiled or linked — write it to a
`.rs` file and invoke `rustc` yourself (see the crate's `aot_roundtrip` integration test
for the full write-then-compile-then-run workflow). `CompiledModel` also exposes
`input_shape()`, `output_shape()`, `node_count()`, and `folded_node_count()`.

## What's actually supported

`aot_compile` currently accepts graphs with:

- **Exactly one** non-initializer (`Operator::Input`) runtime input, since the
  generated signature is fixed as `fn execute(input: &[f32]) -> Vec<f32>`.
- **Exactly one** marked output (via `ComputationGraph::mark_output`/`infer_outputs`).
- Only the operators `MatMul`, `Add`/`Sub`/`Mul`/`Div` (same-shape operands), `Relu`,
  `Sigmoid`, and `Reshape`/`Flatten`. Anything else — `Conv2d`, `Softmax`, pooling,
  `BatchNorm`, `Custom`, etc. — is rejected with `CompileError::UnsupportedOperator`
  rather than silently skipped or miscompiled.

Nodes whose inputs are entirely compile-time constants (weight initializers, or chains
of other constant nodes) are evaluated during compilation and spliced into the
generated source as `f32` array literals instead of emitted runtime code — see the
[`fold`] module. Loop bounds of 8 elements or fewer are unrolled directly into
straight-line statements (see [`codegen`]); larger loop bounds are emitted as ordinary
compile-time-bounded `for` loops for LLVM to optimize.

`fpga_stub` is a **structural placeholder only** — an extension point reserved for a
future `tpt-crucible` (FPGA/photonic mesh) backend. Every method on it returns
`FpgaError::Unimplemented`; it performs no real instruction selection or scheduling.

## Errors

`CompileError` reports (non-exhaustively): `EmptyGraph`, `UnsupportedInputCount`,
`UnsupportedOutputCount`, `MissingInitializer`, `UnsupportedOperator`,
`UnsupportedRank` (e.g. `MatMul` on anything but rank 2), `ShapeMismatch`, and
propagated `GraphError`s.

## Relationship to the rest of the workspace

Depends on `tpt-infer-graph` for the `ComputationGraph` IR it compiles. It does **not**
depend on `tpt-infer-ops` or `tpt-infer-runtime` — generated code is fully self-contained
scalar Rust with no external calls. `tpt-infer-runtime` currently executes
`ComputationGraph` IR directly rather than depending on this crate's `CompiledModel`
(see the runtime crate's docs). See the [workspace README](../../README.md) for how the
two execution paths fit together.
