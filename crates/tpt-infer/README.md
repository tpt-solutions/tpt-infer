# tpt-infer

**Edge AI & local inference runtime — facade crate unifying the tpt-infer workspace.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer` implements no inference logic itself. It re-exports the public API of the
workspace's individual crates (`tpt-infer-core`, `tpt-infer-graph`, `tpt-infer-ops`,
`tpt-infer-onnx`, `tpt-infer-compile`, `tpt-infer-runtime`, `tpt-infer-vision`,
`tpt-infer-quantize`) through a single `prelude` module, so downstream users depend on
one crate and one feature set instead of wiring up eight crates by hand.

## Pipeline

1. **Load or build** a `ComputationGraph` — either `load` an ONNX file (feature
   `onnx`) or assemble one programmatically with `GraphBuilder` (always available).
2. **Preprocess** image inputs with `preprocess_image` / `preprocess_raw_rgb`
   (feature `vision`, opt-in).
3. **Compile ahead-of-time** with `aot_compile` (feature `compile`) to get
   self-contained generated Rust source, or **execute** the graph directly with
   `execute` / `execute_graph` (feature `runtime`) against a `BumpArena` and a
   `Backend` (feature `ops-cpu`).
4. **Post-process** the output tensor with `argmax` for top-1 classification.

Optionally, `ptq` (feature `quantize`, opt-in) quantizes a graph's weights ahead of
either path.

## Usage

```rust
use tpt_infer::prelude::*;

// x [1,4] -> MatMul(w1 [4,3]) -> Relu -> MatMul(w2 [3,2])
let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
let w1 = b
    .add_initializer::<Sh<4, 3>>("w1", vec![0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1])
    .unwrap();
let mut b = b.matmul(w1).relu();
let w2 = b
    .add_initializer::<Sh<3, 2>>("w2", vec![0.2, -0.4, 0.1, 0.3, -0.25, 0.5])
    .unwrap();
let b = b.matmul(w2);
let graph = b.into_graph();

let input = [1.0f32, -2.0, 0.5, 3.0];
let mut mem = vec![0u8; required_arena_bytes(&graph) + 256];
let mut arena = BumpArena::new(&mut mem);
let backend = select_backend();

let output = execute(&graph, &input, &mut arena, &backend).unwrap();
let top1 = argmax(output.as_slice());
println!("top-1 class: {top1}");
```

See `examples/mobilenet.rs` for a larger end-to-end example that builds a structural
MobileNetV2-shaped graph (or loads a real `.onnx` file given on the command line) and
runs it through the facade.

## Feature flags

| Feature | Default | Unlocks |
|---------|---------|---------|
| `ops-cpu` | yes | `Backend`, `NaiveBackend`, `select_backend` |
| `ops-webgpu` | no | WebGPU backend stub (implies `ops-cpu`) |
| `onnx` | yes | `load`/`load_from_bytes` — parse `.onnx` files into a `ComputationGraph` |
| `compile` | yes | `aot_compile` — graph-to-Rust AOT codegen |
| `runtime` | yes | `execute`/`execute_graph`/`argmax` — interpreted graph execution |
| `vision` | no | `preprocess_image`/`preprocess_raw_rgb` — image preprocessing |
| `quantize` | no | `ptq` — post-training INT8/INT4 quantization |

`default = ["ops-cpu", "onnx", "compile", "runtime"]`: everything needed to load an
ONNX model and run it (interpreted or AOT-compiled) on the CPU. `vision` and `quantize`
are opt-in since not every deployment needs image preprocessing or quantization.

The core tensor/arena/dtype primitives (`Tensor`, `TensorVec`, `BumpArena`, `DType`,
`Shape`) and the graph IR (`GraphBuilder`, `ComputationGraph`, `Operator`, `Node`,
`Edge`) are always re-exported, regardless of feature flags.

## Relationship to the rest of the workspace

`tpt-infer` is the top of the dependency graph — it depends on every other crate in the
workspace (most of them optionally, gated by feature flag) and adds no logic of its
own beyond re-exports. See the [workspace README](../../README.md) for the architecture
diagram and per-crate roles.
