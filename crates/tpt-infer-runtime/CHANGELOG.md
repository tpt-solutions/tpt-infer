# Changelog

All notable changes to `tpt-infer-runtime` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `execute` / `execute_graph` (feature `std`): topological planning, input binding,
  arena layout, and operator dispatch for a `ComputationGraph`, with zero heap
  allocation during the hot path.
- `required_arena_bytes`: sizes the `BumpArena` a graph execution needs up front.
- `kernels` module (`no_std`, always available): `binary`/`BinaryOp` (broadcasting
  add/sub/mul/div), plus pooling, batch normalization, transpose, concat, and data
  movement reference kernels for operators outside the `Backend` HAL.
- `argmax`: index of the largest element (top-1 classification helper).
- `RuntimeError` covering planning and execution failures (cycle detection, shape
  mismatch, missing initializers, etc.).
- `#![no_std]` core with `alloc`/`std` features layered on top; verified against
  `thumbv7m-none-eabi` with default (no-`alloc`) features.

### Fixed

- Corrected a transpose kernel coordinate-mapping bug in `kernels`.

### Known limitations

- Does not yet depend on `tpt-infer-compile`; graph execution walks
  `ComputationGraph` IR directly rather than a compiled model, pending that crate's
  `CompiledModel` gaining an execution-ready IR representation.
