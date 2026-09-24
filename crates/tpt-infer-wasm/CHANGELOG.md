# Changelog

All notable changes to `tpt-infer-wasm` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io / npm.

## [0.1.0] - Unreleased

### Added

- `TptInferModel`: a `wasm-bindgen` class wrapping `tpt_infer_onnx::load_from_bytes` +
  `tpt_infer_runtime::execute`, with `run(input)`, `nodeCount()`, and `outputCount()`.
- `init()` (`#[wasm_bindgen(start)]`): installs `console_error_panic_hook` so Rust panics
  surface as readable JS console errors instead of an opaque trap.
- Verified end to end against a real ONNX export (MNIST, opset 12) running under Node.js.
