# Changelog

All notable changes to `tpt-infer-compile` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `aot_compile(graph: &ComputationGraph) -> Result<CompiledModel, CompileError>`:
  walks the graph in topological order and generates a self-contained
  `pub fn execute(input: &[f32]) -> Vec<f32>` Rust source string.
- Supported operators: `MatMul`, `Add`/`Sub`/`Mul`/`Div` (same-shape operands), `Relu`,
  `Sigmoid`, `Reshape`/`Flatten`. Unsupported operators are reported via
  `CompileError::UnsupportedOperator` rather than silently mis-generated.
- Constant folding (`fold::fold_constants`): nodes whose inputs are entirely
  compile-time constants are evaluated during compilation and spliced into the
  generated source as array literals.
- Loop unrolling in `codegen`: dimensions of 8 elements or fewer are unrolled into
  straight-line statements; larger dimensions become compile-time-bounded `for` loops.
- `CompiledModel` metadata: `source()`, `input_shape()`, `output_shape()`,
  `node_count()`, `folded_node_count()`.
- `fpga_stub`: a structural placeholder extension point for a future `tpt-crucible`
  (FPGA/photonic mesh) backend — deliberately unimplemented
  (`FpgaError::Unimplemented`).
- Integration test (`aot_roundtrip`) that compiles a graph, writes the generated
  source to disk, invokes `rustc`, and checks the compiled binary's output.
- `criterion` benchmark (`codegen_vs_interpreted`) comparing AOT-generated code
  against interpreted execution.

### Known limitations

- Only single-input, single-output graphs are supported.
- `Conv2d`, `Softmax`, pooling ops, `BatchNorm`, and `Operator::Custom` are not yet
  supported by codegen.
