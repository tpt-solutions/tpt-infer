# Changelog

All notable changes to `tpt-infer-onnx` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `load` / `load_from_bytes`: decode an ONNX `ModelProto` via `prost` and lower it into
  a `tpt_infer_graph::ComputationGraph`.
- `graph_from_proto`: the shared lowering path used by both loaders.
- `registry::map_op` / `is_native`: opset-17 operator-name → `Operator` mapping, with
  unmapped operators preserved as `Operator::Custom(name)` instead of failing to load.
- `shapes::infer_shapes` / `infer_node_shape`: shape-inference pass over the parsed
  graph, representing dynamic dimensions (`-1` / `dim_param`) as `0`.
- `OnnxError` covering I/O failure, protobuf decode failure, malformed models, and
  graph-construction errors.
- `build.rs` using `prost-build` to generate Rust structs from the ONNX opset-17 schema
  at build time.

### Known limitations

- Only operators explicitly mapped in `registry::map_op` get native `Operator`
  treatment; everything else loads as `Operator::Custom` and is not executable by the
  runtime or AOT compiler.
