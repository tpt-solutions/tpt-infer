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
- `WebGpuBackend` behind the `webgpu` feature: the type, dispatch plumbing, and
  optional `wgpu` dependency are wired up, but every operation currently returns
  `OpError::Unsupported` — GPU kernels are not implemented yet.
- `dispatch::select_backend` / `AnyBackend`: enum-dispatch runtime backend selection
  with no trait objects, preferring AVX-512 > AVX2 on x86_64, NEON on aarch64, SIMD128
  on wasm32, and falling back to `NaiveBackend`.
- `Conv2dOptions` (stride/padding) and `OpError` (size/shape mismatch, unsupported,
  invalid argument).
- `criterion` benchmark (`matmul`) comparing backend throughput.
- `no_std` support (default features off) with `std` as an opt-out default feature
  for runtime CPU-feature detection.

### Known limitations

- `WebGpuBackend` is a stub: it validates the trait shape but performs no GPU
  computation.
