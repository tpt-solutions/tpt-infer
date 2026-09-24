# tpt-infer-py

**Python bindings for `tpt-infer`: load and run ONNX models via [PyO3](https://pyo3.rs/).**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

## Install (from source, until published)

```sh
pip install maturin
cd crates/tpt-infer-py
maturin develop --release   # builds and installs into the active virtualenv
```

## Usage

```python
import tpt_infer

model = tpt_infer.Model.load("model.onnx")
print(model.node_count(), model.output_count())

output = model.run([0.0] * input_length)
print(output)

# Or load from in-memory bytes:
with open("model.onnx", "rb") as f:
    model = tpt_infer.Model.load_bytes(f.read())

# Ahead-of-time compile to standalone Rust source:
rust_source = model.compile()
```

## What it doesn't do (yet)

- No NumPy integration — `run()` takes and returns plain Python lists of floats, not
  `numpy.ndarray`. Convert at the call site (`arr.flatten().tolist()` /
  `np.array(output)`).
- No image decoding — pass a flattened list of floats (preprocess with `tpt-infer-vision`
  natively, Pillow, or NumPy first).

## Relationship to the rest of the workspace

Depends on `tpt-infer-onnx`, `tpt-infer-graph`, `tpt-infer-ops`, `tpt-infer-runtime`, and
`tpt-infer-compile` directly (not through the `tpt-infer` facade, to keep the extension
module's dependency surface minimal and explicit) — the same pipeline `tpt-infer-cli` and
`tpt-infer-wasm` wrap for their respective platforms.
