# tpt-infer

**Edge AI & Local Inference Runtime** · TPT Solutions · MIT / Apache-2.0

[![CI](https://github.com/tpt-solutions/tpt-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-infer/actions/workflows/ci.yml)
[![docs.rs](https://img.shields.io/docsrs/tpt-infer)](https://docs.rs/tpt-infer)
[![crates.io](https://img.shields.io/crates/v/tpt-infer.svg)](https://crates.io/crates/tpt-infer)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A zero-legacy, hardware-agnostic inference runtime that compiles and executes ML
models without CUDA, Python, C++ toolchains, or large C-FFI bindings.

> Not yet published to crates.io — the crates.io/docs.rs badges above will go live
> once the first release is tagged (see `todo.md`). Until then, depend on this repo
> directly via a `path`/`git` dependency.

```
┌─────────────────────────────────────────────────────────────┐
│                        tpt-infer                            │
│                     (facade + prelude)                      │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│   onnx   │ quantize │ compile  │ runtime  │     vision     │
│ (parser) │ (INT8/4) │ (AOT RS) │ (no_std) │ (preprocess)   │
├──────────┴────┬─────┴────┬─────┴──────────┴────────────────┤
│     ops       │   graph  │  core (Tensor, BumpArena, DType) │
│ (HAL + SIMD)  │ (const-  │        #![no_std]                │
│               │  generic)│                                  │
└───────────────┴──────────┴──────────────────────────────────┘
```

## Crates

| Crate | Role |
|---|---|
| `tpt-infer-core` | Const-generic `Tensor<T, N>`, `DType`, `BumpArena` (`no_std`) |
| `tpt-infer-ops` | `Backend` HAL: naive, AVX2, AVX-512, NEON, WASM SIMD, WebGPU |
| `tpt-infer-graph` | Type-state `GraphBuilder` — shape mismatches are compile errors |
| `tpt-infer-onnx` | Pure-Rust ONNX (opset-17) parser, no C-FFI |
| `tpt-infer-quantize` | INT8/INT4 PTQ, per-tensor & per-channel, nibble packing |
| `tpt-infer-compile` | AOT compiler: graph → optimized static Rust `fn` |
| `tpt-infer-runtime` | Zero-allocation execution over `BumpArena` (`no_std`) |
| `tpt-infer-vision` | Resize, normalize, letterbox, CHW↔HWC |
| `tpt-infer-cli` | `tpt-infer` binary: inspect/run/compile a model with no Rust required |
| `tpt-infer-wasm` | `wasm-bindgen` JS bindings: load and run ONNX models in the browser/Node.js |

## Try it without writing Rust

```sh
cargo install --path crates/tpt-infer-cli   # from a checkout, until published
tpt-infer inspect your-model.onnx           # node/operator summary + AOT-compile preflight
tpt-infer run your-model.onnx               # execute with a random input, print the output
```

See [`crates/tpt-infer-cli/README.md`](crates/tpt-infer-cli/README.md) for the full
command reference.

## Quick start (library)

```rust
use tpt_infer::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load an ONNX model into a computation graph.
    let graph = load("model.onnx")?;

    // 2. Execute it with the interpreted runtime, zero heap allocation
    //    during inference (everything comes from the arena).
    let input = vec![0.0f32; /* ... your model's flattened input ... */ 4];
    let mut mem = vec![0u8; required_arena_bytes(&graph) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let backend = select_backend();
    let output = execute(&graph, &input, &mut arena, &backend)?;
    println!("top-1 class: {}", argmax(output.as_slice()));

    // ...or ahead-of-time compile it to standalone Rust source instead of
    // interpreting it (when every operator in the graph has codegen support):
    let compiled = aot_compile(&graph)?;
    std::fs::write("generated.rs", compiled.source())?;

    Ok(())
}
```

This mirrors the tested example in [`crates/tpt-infer/src/lib.rs`](crates/tpt-infer/src/lib.rs)
(run via `cargo test --doc -p tpt-infer`), which builds its input graph programmatically
with `GraphBuilder` instead of loading a file, since no real `.onnx` model ships with
this repository.

## Feature matrix

| Feature | Default | Enables |
|---|---|---|
| `ops-cpu` | ✓ | CPU SIMD backends |
| `ops-webgpu` | | WebGPU via `wgpu` |
| `onnx` | ✓ | ONNX model loading |
| `compile` | ✓ | AOT codegen |
| `runtime` | ✓ | Execution engine |
| `vision` | | Image preprocessing |
| `quantize` | | INT8/INT4 PTQ |

## Building

```sh
cargo build --workspace
cargo test --workspace
cargo build -p tpt-infer-core --target thumbv7m-none-eabi   # no_std check
cargo deny check                                            # supply-chain policy (deny.toml)
```

## Security

`tpt-infer-onnx` and `tpt-infer-vision` parse untrusted input by design (arbitrary
`.onnx` model files and image bytes). See [`SECURITY.md`](SECURITY.md) for the trust
boundary and how to report a vulnerability.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
