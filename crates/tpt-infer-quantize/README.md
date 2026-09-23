# tpt-infer-quantize

**INT8/INT4 quantization and dequantization kernels for edge deployment.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-quantize` shrinks f32 model weights down to INT8 or INT4 for edge
deployment, and provides a post-training quantization (PTQ) pipeline that applies this
crate's kernels directly to a `tpt_infer_graph::ComputationGraph`'s initializers.

## What's in here

- **`quant`** — f32 → INT8/INT4, symmetric and asymmetric, per-tensor and per-channel:
  `quantize_i8_symmetric`, `quantize_i8_asymmetric`, `quantize_i4_symmetric`,
  `quantize_i4_symmetric_per_channel`, `quantize_i8_symmetric_per_channel`, plus
  `pack_i4`/`unpack_i4` for 2-values-per-byte nibble packing and the `QuantParams`
  (scale + zero-point) type.
- **`dequant`** — the inverse kernels: `dequantize_i8_symmetric`,
  `dequantize_i8_asymmetric`, `dequantize_i4_symmetric`,
  `dequantize_i8_symmetric_per_channel`, `dequantize_i4_symmetric_per_channel`.
- **`ptq`** — `ptq(graph, options) -> QuantizedGraph`: walks a `ComputationGraph` in
  topological order, finds `MatMul`/`Conv2d` nodes whose weights are float
  initializers, and quantizes those weights per `PtqOptions` (symmetric/asymmetric,
  per-tensor/per-channel, INT8/INT4). `quantized_matmul_i8` is a reference quantized
  matmul kernel used by the crate's benchmark and available for callers directly.

## Usage

```rust
use tpt_infer_quantize::{dequantize_i8_symmetric, quantize_i8_symmetric};

let data = [0.0f32, 0.5, -0.5, 1.0, -1.0];
let (q, scale) = quantize_i8_symmetric(&data);
let d = dequantize_i8_symmetric(&q, scale);
for (a, b) in data.iter().zip(&d) {
    assert!((a - b).abs() < 0.01, "{a} vs {b}");
}
```

Running PTQ over a whole graph:

```rust,ignore
use tpt_infer_quantize::{ptq, PtqOptions};

let quantized = ptq(&graph, PtqOptions::default())?;
```

`PtqOptions::default()` is symmetric, per-channel, INT8 quantization with calibration
disabled (weight-only PTQ; `calibrate` is reserved for a future activation-range pass).

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `std` | yes | `tpt-infer-core/alloc`, `tpt-infer-graph/std`, `tpt-infer-ops/std` |

## Relationship to the rest of the workspace

Depends on `tpt-infer-core`, `tpt-infer-graph` (the `ComputationGraph`/`Initializer`
types PTQ walks), and `tpt-infer-ops` (benchmarked against its f32 `matmul`). Produces a
`QuantizedGraph`/`QuantizedTensor` that a future INT8-aware execution path in
`tpt-infer-runtime` or `tpt-infer-compile` can consume. See the
[workspace README](../../README.md) for the overall pipeline.
