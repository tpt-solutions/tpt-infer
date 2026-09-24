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
- `load_from_bytes_with_limits`: explicit byte-size and node/attribute-count caps
  (`DEFAULT_MAX_MODEL_BYTES`, `DEFAULT_MAX_GRAPH_ITEMS`), opset-version enforcement
  (`MAX_SUPPORTED_OPSET`), and checked (overflow-safe) arithmetic throughout shape
  inference and initializer-size validation — hardening against malformed or
  adversarial `.onnx` files (a real trust boundary; see the workspace `SECURITY.md`).
- `INT64` initializers (shape/index tensors feeding `Reshape`, `Slice`, `Gather`, etc.
  — ubiquitous in real ONNX exports) are now read and bound as data, not silently left
  as unresolved free graph inputs; `Reshape`'s target shape is resolved from a second
  *input* tensor when present, matching the modern (opset ≥ 5) ONNX convention rather
  than only the legacy `shape` attribute.
- `auto_pad` (`SAME_UPPER`/`SAME_LOWER`) support for `Conv2d`/pooling, resolving actual
  symmetric padding from the input's spatial size once it's known — real exports
  commonly use this instead of an explicit `pads` attribute.
- Verified against a real (non-synthetic) ONNX export (MNIST, opset 12) in addition to
  the structurally-representative synthetic fixtures — see
  `tests/fixtures/real/README.md`.

### Known limitations

- Only operators explicitly mapped in `registry::map_op` get native `Operator`
  treatment; everything else loads as `Operator::Custom` and is not executable by the
  runtime or AOT compiler.
