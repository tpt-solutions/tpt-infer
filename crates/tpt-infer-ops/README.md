# tpt-infer-ops

**Hardware abstraction layer and SIMD-accelerated ML operators for tpt-infer.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-ops` defines the `Backend` trait — a small hardware abstraction layer (HAL)
covering `matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`, `sigmoid`, and `gelu`
— plus several concrete implementations, from a portable scalar reference up to
architecture-specific SIMD backends. It sits directly on top of `tpt-infer-core` and is
consumed by `tpt-infer-runtime` (interpreted execution) and, indirectly, by
`tpt-infer-quantize`'s benchmarks; `tpt-infer-compile`'s AOT codegen emits its own
inline Rust rather than calling into a `Backend` at runtime.

## Backends

| Backend | Module | Type | Availability |
|---|---|---|---|
| Naive scalar reference | `naive` | `NaiveBackend` | always |
| x86_64 AVX2 | `avx2` | `Avx2Backend` | `target_arch = "x86_64"`, runtime CPU-feature detected |
| x86_64 AVX-512 | `avx512` | `Avx512Backend` | feature `avx512` + CPU support |
| aarch64 NEON | `neon` | `NeonBackend` | `target_arch = "aarch64"` |
| wasm32 SIMD128 | `wasm` | `WasmSimdBackend` | `target_arch = "wasm32"` + `target_feature = "simd128"` |
| WebGPU | `webgpu` | `WebGpuBackend` | feature `webgpu`; real wgpu/WGSL compute dispatch for `matmul`, `conv2d`, `elementwise_add`, `relu`, `sigmoid`, `gelu`, `softmax` |

All operators write into a caller-provided output slice (no allocation) and validate
buffer lengths up front, returning `OpError` on mismatch instead of panicking.

`AnyBackend` (in `dispatch`) is an enum-dispatch wrapper — no trait objects — returned by
`select_backend()`, which probes the current architecture/CPU at runtime and picks the
fastest backend actually available, falling back to `NaiveBackend` everywhere else.
`WebGpuBackend` owns a real GPU `Device`/`Queue` and precompiled pipelines, so it doesn't
fit `AnyBackend`'s `Copy`-enum dispatch — construct it directly via
`WebGpuBackend::new() -> Option<Self>` (returns `None` if no adapter is available).

## Usage

```rust
use tpt_infer_ops::{select_backend, Backend};

let backend = select_backend();
let a = [1.0f32, 2.0, 3.0, 4.0];
let b = [5.0f32, 6.0, 7.0, 8.0];
let mut out = [0.0f32; 4];
backend.elementwise_add(&a, &b, &mut out).unwrap();
assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
println!("running on backend: {}", backend.name());
```

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `std` | yes | Runtime CPU-feature detection (`select_backend`) and native `f32::exp`/`tanh`; also enables `tpt-infer-core/alloc` |
| `avx512` | no | Compiles `Avx512Backend` (still runtime-detected before use) |
| `webgpu` | no | Compiles `WebGpuBackend` (real `wgpu`/WGSL dispatch) and its optional `wgpu`/`pollster` dependencies |

## `no_std`

The crate is `#![no_std]` unless the `std` feature is enabled (`std` is a default
feature, so opt out with `default-features = false` for a bare-metal build). Without
`std`, `select_backend`'s runtime CPU-feature probing is unavailable, but the concrete
backend types can still be constructed directly.

## Relationship to the rest of the workspace

Depends only on `tpt-infer-core`. `tpt-infer-runtime` dispatches graph nodes onto a
`Backend` chosen via `select_backend`; `tpt-infer-quantize` benchmarks quantized matmul
against this crate's f32 matmul. See the [workspace README](../../README.md) for the
overall architecture.
