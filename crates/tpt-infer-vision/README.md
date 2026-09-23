# tpt-infer-vision

**Pure-Rust image preprocessing (resize, normalize, letterbox) for vision models.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-vision` turns raw or encoded images into `tpt_infer_core::Tensor<f32, 4>`
batches ready for inference, without any C image-decoding dependency beyond the optional
pure-Rust `image` crate. The pipeline is:

1. (optional) **letterbox** — aspect-preserving resize with constant padding
2. **bilinear resize** to the configured target `width × height`
3. **rescale** (e.g. `1/255`) and per-channel **normalization** `(x - mean) / std`
4. optional RGB ↔ BGR channel swap
5. **layout** conversion (CHW ↔ HWC) with a leading batch dimension of 1

## Usage

Already-decoded pixels (works without `std`, given `alloc`):

```rust
use tpt_infer_vision::{preprocess_raw_rgb, PreprocessConfig};

// 2x2 RGB image: red, green, blue, white.
let rgb = [
    255u8, 0, 0, 0, 255, 0,
    0, 0, 255, 255, 255, 255,
];
let mut config = PreprocessConfig::imagenet();
config.width = 2;
config.height = 2;
let tensor = preprocess_raw_rgb(&rgb, 2, 2, &config).unwrap();
assert_eq!(tensor.shape(), [1, 3, 2, 2]);
assert_eq!(tensor.dtype(), tpt_infer_core::DType::F32);
```

Encoded bytes (PNG/JPEG/…, feature `std`, default):

```rust,ignore
use tpt_infer_vision::{preprocess_image, PreprocessConfig};

let bytes = std::fs::read("cat.jpg")?;
let tensor = preprocess_image(&bytes, &PreprocessConfig::imagenet())?;
```

## Main API

- `preprocess_raw_rgb(data, width, height, config)` — the core entry point; runs the
  full pipeline over already-decoded interleaved RGB8 pixels.
- `preprocess_image(bytes, config)` (feature `std`) — decodes with the `image` crate,
  then calls `preprocess_raw_rgb`.
- `preprocess_source(source, config)` — the `tpt-kinetix` integration boundary: accepts
  anything implementing `ImageSource` (`dimensions()` + `to_rgb8()`), including the
  built-in `RawRgbImage` (`no_std`-friendly) or `image::RgbImage` (feature `std`).
- `PreprocessConfig` — target `width`/`height`, `mean`/`std`, `Layout` (`Chw`/`Hwc`),
  `letterbox`, `rescale`, `ColorOrder` (`Rgb`/`Bgr`), and pad value; `imagenet()`
  provides a standard preset.
- Lower-level building blocks are also public: `bilinear_resize`, `letterbox`,
  `normalize_hwc`, `chw_to_hwc`/`hwc_to_chw`, `swap_rb`.

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `std` | yes | `preprocess_image` (decoding via the `image` crate) and an `ImageSource` impl for `image::RgbImage`; also enables `tpt-infer-core/alloc` |

Without `std` (but with `alloc`, which this crate always pulls in), `preprocess_raw_rgb`
and `preprocess_source` still work on already-decoded pixels — only encoded-format
decoding requires `std`.

## Relationship to the rest of the workspace

Depends on `tpt-infer-core` for `Tensor`. It has no dependency on `tpt-infer-graph` or
`tpt-infer-ops` (only pulled in as a dev-dependency for tests/benchmarks), so it can be
used standalone to prepare inputs before handing them to `tpt-infer-runtime` or a
`tpt-infer-compile`-generated function. See the [workspace README](../../README.md) for
the end-to-end pipeline.
