# Changelog

All notable changes to `tpt-infer-ops` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `Backend` HAL trait: `matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`,
  `sigmoid`, `gelu`, all writing into caller-provided output slices.
- `NaiveBackend`: portable scalar reference implementation, used as the correctness
  baseline for every other backend.
- `Avx2Backend` (x86_64) and, behind the `avx512` feature, `Avx512Backend`.
- `NeonBackend` (aarch64) and `WasmSimdBackend` (wasm32 + `simd128`).
- `WebGpuBackend` behind the `webgpu` feature: owns a real `wgpu::Device`/`Queue`
  plus precompiled pipelines, dispatching actual WGSL compute shaders for `matmul`,
  `conv2d`, `elementwise_add`, `relu`, `sigmoid`, `gelu`, and `softmax`. `conv2d` is
  a direct (non-im2col) per-output-element dispatch matching `NaiveBackend`'s
  NCHW/OIHW semantics exactly. `WebGpuBackend::new()` returns `Option<Self>` (`None`
  if no GPU adapter is available) and is constructed directly rather than through
  `AnyBackend`, since it owns GPU resources instead of being a `Copy` unit type.
- `dispatch::select_backend` / `AnyBackend`: enum-dispatch runtime backend selection
  with no trait objects, preferring AVX-512 > AVX2 on x86_64, NEON on aarch64, SIMD128
  on wasm32, and falling back to `NaiveBackend`.
- `Conv2dOptions` (stride/padding) and `OpError` (size/shape mismatch, unsupported,
  invalid argument).
- `criterion` benchmark (`matmul`) comparing backend throughput.
- `no_std` support (default features off) with `std` as an opt-out default feature
  for runtime CPU-feature detection.
