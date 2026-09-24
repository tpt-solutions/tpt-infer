# tpt-infer-cli

**Command-line tool for smoke-testing and inspecting ONNX models with `tpt-infer`.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

Trying `tpt-infer` today means writing Rust: importing the prelude, loading a graph,
wiring up an arena and a backend. `tpt-infer-cli` gives you a single binary,
`tpt-infer`, so you can check whether *your own* exported `.onnx` file works with this
runtime — parses, executes, or AOT-compiles — in one command, with no Rust required.

## Install

```sh
cargo install --path crates/tpt-infer-cli   # from a checkout, until published
```

## Usage

```sh
# Structural summary: node/operator counts, input/output shapes, and whether
# the AOT compiler can handle every operator in the graph.
tpt-infer inspect model.onnx

# Execute the model via the interpreted runtime and print the output tensor
# and its argmax. Without --input, a deterministic pseudo-random input of the
# model's declared shape is used — enough to smoke-test that it runs end to
# end without needing real data.
tpt-infer run model.onnx
tpt-infer run model.onnx --input raw_f32.bin   # flattened, little-endian f32, row-major

# Ahead-of-time compile the model to a standalone Rust source file.
tpt-infer compile model.onnx -o generated.rs
```

`inspect` is the recommended first step for any model you haven't tried before — it
reports exactly which operators are present and whether `aot_compile` would succeed,
before you invest time in `run`/`compile`.

## What it doesn't do (yet)

- No image decoding — `run --input` expects a raw flattened `f32` buffer, not a
  `.png`/`.jpg`. Preprocess with `tpt-infer-vision` (or any tool that can dump raw
  floats) first.
- No batch/interactive mode — one model, one invocation.

## Relationship to the rest of the workspace

A thin wrapper over the `tpt-infer` facade crate's `prelude` (`onnx`, `compile`,
`runtime`, `ops-cpu` features) — essentially no logic of its own beyond argument
parsing and output formatting. See the [workspace README](../../README.md) for the
overall pipeline this CLI exercises.
