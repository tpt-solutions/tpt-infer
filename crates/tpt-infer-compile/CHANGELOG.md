# Changelog

All notable changes to `tpt-infer-compile` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `aot_compile(graph: &ComputationGraph) -> Result<CompiledModel, CompileError>`:
  walks the graph in topological order and generates a self-contained
  `pub fn execute(input: &[f32]) -> Vec<f32>` Rust source string.
- Supported operators: `MatMul`, `Add`/`Sub`/`Mul`/`Div` (same-shape or
  numpy-broadcastable operands — an exact-shape fast path plus a general broadcasting
  path using compile-time-constant strides), `Relu`, `Sigmoid`, `Gelu`,
  `Reshape`/`Flatten`, `Conv2d` (direct/naive-loop, optional per-channel bias),
  `Softmax` (last axis only), `MaxPool2d`/`AveragePool2d`, `BatchNorm`, `Concat`, and
  `Transpose`. Unsupported operators (`Operator::Custom`, ...) are reported via
  `CompileError::UnsupportedOperator` rather than silently mis-generated.
- Constant folding (`fold::fold_constants`): nodes whose inputs are entirely
  compile-time constants are evaluated during compilation and spliced into the
  generated source as array literals. Covers the same operator set as `codegen`
  (`Conv2d`, `Softmax`, pooling, `BatchNorm`, `Concat`, `Transpose` included); folded
  `Softmax` additionally supports any axis, not just the last one, since folding just
  evaluates in ordinary host Rust rather than being constrained by what the generated
  code can express.
- Loop unrolling in `codegen`: dimensions of 8 elements or fewer are unrolled into
  straight-line statements; larger dimensions become compile-time-bounded `for` loops.
- `CompiledModel` metadata: `source()`, `input_shape()`, `output_shape()`,
  `node_count()`, `folded_node_count()`.
- `fpga_stub`: a structural placeholder extension point for a future `tpt-crucible`
  (FPGA/photonic mesh) backend — deliberately unimplemented
  (`FpgaError::Unimplemented`).
- Integration test (`aot_roundtrip`) that compiles a graph, writes the generated
  source to disk, invokes `rustc`, and checks the compiled binary's output — now
  covering `Conv2d` (with bias), `Softmax`, `MaxPool2d`, `AveragePool2d`, `BatchNorm`,
  `Concat`, and `Transpose` in addition to the original `MatMul`/`Relu` MLP graph.
- `criterion` benchmark (`codegen_vs_interpreted`) comparing AOT-generated code
  against interpreted execution.

### Known limitations

- Only single-input, single-output graphs are supported.
- `Reshape`/`Flatten` codegen only reads the data operand (`inputs[0]`) — the target
  shape must already be resolved into the node's own `Operator::Reshape` metadata by
  the loader, whether it came from an ONNX `shape` attribute or (the modern-export
  convention) a second constant *input* tensor.
- `Softmax` codegen only generates last-axis normalization, matching
  `tpt-infer-runtime`'s own `Backend::softmax` restriction; other axes report
  `CompileError::UnsupportedRank` rather than being silently miscompiled (constant
  folding is not subject to this restriction).
- `Conv2d` codegen only accepts a `[1, oc, 1, 1]`-shaped bias (the layout that
  actually broadcasts onto the channel axis under `tpt-infer-runtime`'s right-aligned
  broadcasting); a flat `[oc]` bias is rejected.
- `Operator::Custom` is not supported by codegen or folding, and never will be — it's
  an intentional escape hatch for unmapped ONNX ops, not a generatable operator.
