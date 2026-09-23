# Changelog

All notable changes to `tpt-infer-quantize` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `quantize_i8_symmetric` / `quantize_i8_asymmetric`: INT8 quantization, `scale =
  max(|x|) / 127` (symmetric) or a derived `QuantParams { scale, zero_point }`
  (asymmetric, `[min, max] -> [-128, 127]`).
- `quantize_i4_symmetric` / `quantize_i4_symmetric_per_channel` with `pack_i4`/
  `unpack_i4` nibble packing (two INT4 values per byte).
- `quantize_i8_symmetric_per_channel` for per-output-channel scales.
- Matching `dequant` kernels for every `quant` variant.
- `ptq`: post-training quantization pipeline walking a `ComputationGraph`'s `MatMul`/
  `Conv2d` weight initializers and replacing them with quantized tensors per
  `PtqOptions`.
- `quantized_matmul_i8`: reference INT8 matmul kernel.
- `criterion` benchmark (`quant_matmul`) comparing quantized vs. f32 matmul throughput.
- Doctest fix ensuring the PTQ example compiles against the current `ComputationGraph`
  API.
