# tpt-infer-onnx

**Pure-Rust ONNX model parser and operator registry (no C-FFI).**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-onnx` decodes an ONNX `ModelProto` with [`prost`] (a pure-Rust protobuf
implementation — no `onnxruntime`, no C/C++ toolchain), maps opset-17 operator names
onto `tpt_infer_graph::Operator` variants, attaches float initializers as graph weights,
runs a shape-inference pass, and hands back a ready-to-use
`tpt_infer_graph::ComputationGraph` for the AOT compiler (`tpt-infer-compile`) or the
interpreted runtime (`tpt-infer-runtime`).

## Usage

```rust,no_run
let graph = tpt_infer_onnx::load("model.onnx")?;
println!("{} nodes", graph.nodes().len());
# Ok::<(), tpt_infer_onnx::OnnxError>(())
```

`load_from_bytes(bytes: &[u8])` is available for models that are already in memory
(e.g. embedded via `include_bytes!` or fetched over the network) instead of a path.
`load_from_bytes_with_limits(bytes, max_bytes, max_graph_items)` additionally lets a
caller set explicit resource limits — useful on memory-constrained targets, or to
accept larger models than the defaults allow — since **an ONNX file is untrusted
input** (see the workspace `SECURITY.md`): both `load`/`load_from_bytes` apply generous
default limits (`DEFAULT_MAX_MODEL_BYTES`, `DEFAULT_MAX_GRAPH_ITEMS`) and reject models
whose default-domain opset exceeds `MAX_SUPPORTED_OPSET`.

## What it does

- `load` / `load_from_bytes` — decode the `ModelProto`, then `graph_from_proto` builds
  the `ComputationGraph`: one `Node` per ONNX node (mapped through `registry::map_op`),
  `Initializer`s for constant weight tensors (including `INT64` shape/index tensors,
  read and bound as data rather than left as unresolved free inputs), and graph
  inputs/outputs wired up from the model's `ValueInfoProto` lists. `Reshape`'s target
  shape is resolved from either a `shape` attribute (legacy) or a second constant
  *input* tensor (the modern, opset ≥ 5 convention most real exports use), and
  `Conv2d`/pooling's `auto_pad` (`SAME_UPPER`/`SAME_LOWER`) is resolved to concrete
  symmetric padding once the input's spatial size is known.
- `registry::map_op` — maps ONNX op-type strings to `tpt_infer_graph::Operator`
  variants (`MatMul`/`Gemm`, `Add`/`Sub`/`Mul`/`Div`, `Relu`, `Sigmoid`, `Gelu`/`Erf`,
  `Softmax`, `Reshape`, `Flatten`, `Transpose`, `Concat`, `Conv`, `MaxPool`/
  `AveragePool`/`GlobalAveragePool`, `BatchNormalization`). Anything unmapped
  (`LayerNormalization`, `Cast`, `Squeeze`, `Gather`, etc.) becomes
  `Operator::Custom(name)` rather than failing to load — `registry::is_native` reports
  whether a given op type got a first-class mapping.
- `shapes::infer_shapes` / `infer_node_shape` — a shape-inference pass over the parsed
  graph so downstream consumers have concrete dimensions to work with. Dynamic
  dimensions (batch size `-1` / ONNX `dim_param`) are represented as `0` in node shapes.

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `std` | off (crate's loader is inherently `std`-only regardless) | Additionally enables `tpt-infer-core/alloc` and `tpt-infer-graph/std` |

The parser itself always requires `std` (file I/O via `std::path::Path`/`prost`), so
`tpt-infer-graph`'s `std` feature is pulled in unconditionally as a dependency; this
crate's own `std` feature only controls whether `tpt-infer-core/alloc` is additionally
enabled for callers that need it. `prost-build` runs at build time (`build.rs`) to
generate the ONNX protobuf message types from the schema.

## Relationship to the rest of the workspace

Depends on `tpt-infer-core` and `tpt-infer-graph` (whose `Operator`/`ComputationGraph`
types it produces). `tpt-infer-quantize`'s PTQ pipeline and `tpt-infer-compile`'s AOT
codegen both consume the `ComputationGraph` this crate builds. See the
[workspace README](../../README.md) for the overall pipeline.

[`prost`]: https://docs.rs/prost
