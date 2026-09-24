# Changelog

All notable changes to `tpt-infer-cli` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `tpt-infer inspect <model.onnx>`: structural summary (node/operator counts,
  input/output shapes) and an AOT-compile preflight check.
- `tpt-infer run <model.onnx> [--input <raw_f32.bin>]`: execute a model via the
  interpreted runtime and print the output tensor and its argmax; falls back to a
  deterministic pseudo-random input of the model's declared shape when `--input` is
  omitted.
- `tpt-infer compile <model.onnx> -o <generated.rs>`: AOT-compile a model to a
  standalone Rust source file.
