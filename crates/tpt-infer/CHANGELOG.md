# Changelog

All notable changes to `tpt-infer` (the facade crate) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `prelude` module re-exporting the workspace's public API: core tensor/arena/dtype
  types (always), graph IR (always), CPU backends (feature `ops-cpu`, default), ONNX
  loading (feature `onnx`, default), AOT compilation (feature `compile`, default),
  interpreted execution (feature `runtime`, default), image preprocessing (feature
  `vision`, opt-in), and PTQ quantization (feature `quantize`, opt-in).
- Feature flags: `default = ["ops-cpu", "onnx", "compile", "runtime"]`, plus optional
  `ops-webgpu`, `vision`, `quantize`.
- Crate-level rustdoc example: build a 2-layer MLP graph with `GraphBuilder`, execute it
  via the interpreted runtime, and read the top-1 class with `argmax`.
- `examples/mobilenet.rs`: builds a structural MobileNetV2-shaped graph (or loads a
  real `.onnx` file given on the command line) and runs it end-to-end through the
  facade's re-exported API.
- Integration test (`facade_e2e`) covering the full load/build → compile/execute →
  argmax pipeline through the facade's public surface.
