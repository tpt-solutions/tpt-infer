# Changelog

All notable changes to `tpt-infer-graph` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `Operator` enum mirroring ONNX opset-17 operator semantics (`MatMul`, `Conv2d`,
  `Add`/`Sub`/`Mul`/`Div`, `Relu`, `Sigmoid`, `Gelu`, `Softmax`, `Reshape`, `Flatten`,
  `Transpose`, `Concat`, `MaxPool2d`, `AveragePool2d`, `BatchNorm`, `Custom`), marked
  `#[non_exhaustive]`.
- `Node`, `Edge`, `Initializer`, and `ComputationGraph` with `add_node`,
  `add_initializer`, `mark_output`, `infer_outputs`, and `topological_sort`.
- `GraphBuilder<S: ShapeMarker>`: a type-state builder tracking tensor shapes in the
  type system via `Sh<M, K>`, `ShapeMarker`, `MatMulShape`, `AddShape`, `FlattenShape`,
  and `TransposeShape`, so shape mismatches are compile errors.
- `GraphError` covering unknown node ids, node id mismatches, rank overflow, size
  mismatches, and cycle detection.
- `trybuild` compile-fail tests verifying shape mismatches are rejected at compile time.
- `#![no_std]` crate with a `std` feature (default) gating the `alloc`-backed
  builder/graph/operator modules.
