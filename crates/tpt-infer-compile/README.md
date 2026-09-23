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
- The operators `MatMul`, `Add`/`Sub`/`Mul`/`Div` (same-shape operands), `Relu`,
  `Sigmoid`, `Reshape`/`Flatten`, `Conv2d`, `Softmax`, `MaxPool2d`/`AveragePool2d`,
  `BatchNorm`, `Concat`, and `Transpose`:
  - `Conv2d` — direct/naive-loop NCHW convolution (no im2col), matching
    `NaiveBackend::conv2d`'s stride/padding/accumulation order exactly, plus an
    optional bias input broadcast per output channel. Only a `[1, oc, 1, 1]` bias
    shape is accepted — that's the only layout that actually broadcasts onto the
    channel axis under `tpt-infer-runtime`'s right-aligned broadcasting rules; a flat
    `[oc]` bias is rejected rather than silently mishandled.
  - `Softmax` — max-subtract/exp/normalize, generated only when the (negative-index
    normalized) axis is the last dimension, mirroring `tpt-infer-runtime`'s own
    `Backend::softmax` restriction. Other axes report `CompileError::UnsupportedRank`.
  - `MaxPool2d`/`AveragePool2d` — matching
    `tpt_infer_runtime::kernels::{max_pool2d, average_pool2d}`'s floor-division output
    geometry and padding-excluded-from-average convention.
  - `BatchNorm` — per-channel or per-element affine normalization, matching
    `tpt_infer_runtime::kernels::batch_norm`'s two accepted parameter layouts.
  - `Concat` — streams each operand into its channel-offset region of the output,
    matching `concat_copy_one`'s block layout.
  - `Transpose` — permutes via compile-time-constant strides, matching
    `tpt_infer_runtime::kernels::transpose`.

  Anything else (`Operator::Custom`, `Gelu`, ...) is rejected with
  `CompileError::UnsupportedOperator` rather than silently skipped or miscompiled.

Nodes whose inputs are entirely compile-time constants (weight initializers, or chains
of other constant nodes) are evaluated during compilation and spliced into the
generated source as `f32` array literals instead of emitted runtime code — see the
[`fold`] module, which covers the same operator set as `codegen` (including `Conv2d`,
pooling, `BatchNorm`, `Concat`, and `Transpose`; `Softmax` folding additionally
supports any axis, since it just evaluates in ordinary host Rust rather than being
constrained by what the generated code can express). Loop bounds of 8 elements or
fewer are unrolled directly into straight-line statements (see [`codegen`]); larger
loop bounds are emitted as ordinary compile-time-bounded `for` loops for LLVM to
optimize.

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
