# tpt-infer

**Edge AI & Local Inference Runtime** · TPT Solutions · MIT / Apache-2.0

A zero-legacy, hardware-agnostic inference runtime that compiles and executes ML
models without CUDA, Python, C++ toolchains, or large C-FFI bindings.

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

## Quick start

```rust
use tpt_infer::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load and compile an ONNX model
    let model = tpt_infer_onnx::load("model.onnx")?;
    let compiled = tpt_infer_compile::aot_compile(&model)?;

    // 2. Preprocess an image into a strictly typed tensor
    let input: Tensor<f32, 4> = tpt_infer_vision::preprocess_image(
        &image_bytes,
        &PreprocessConfig::imagenet(),
    )?;

    // 3. Execute with zero heap allocation during inference
    let mut buf = [0u8; 8 * 1024 * 1024];
    let mut arena = BumpArena::new(&mut buf);
    let output = compiled.execute(&input, &mut arena)?;

    println!("Top class: {}", argmax(&output));
    Ok(())
}
```

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
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
