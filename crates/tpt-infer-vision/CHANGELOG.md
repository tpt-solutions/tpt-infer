# Changelog

All notable changes to `tpt-infer-vision` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `preprocess_raw_rgb`: full preprocessing pipeline (letterbox/resize → rescale +
  normalize → optional RGB/BGR swap → layout conversion) over already-decoded
  interleaved RGB8 pixels, producing a `Tensor<f32, 4>`.
- `preprocess_image` (feature `std`): decodes PNG/JPEG/etc. via the `image` crate, then
  runs the same pipeline.
- `preprocess_source` and the `ImageSource` trait: the `tpt-kinetix` integration
  boundary, with a built-in `RawRgbImage` (`no_std`-friendly) implementation and an
  `image::RgbImage` implementation under `std`.
- `bilinear_resize`, `letterbox`, `normalize_hwc`, `chw_to_hwc`/`hwc_to_chw`, `swap_rb`
  as independently usable building blocks.
- `PreprocessConfig` with `Layout` (`Chw`/`Hwc`), `ColorOrder` (`Rgb`/`Bgr`), and an
  `imagenet()` preset.
- `VisionError` covering invalid dimensions, size mismatches, zero standard deviation,
  invalid channel counts, decode failures, and tensor-construction failures.
- `no_std` support (with `alloc`) for the raw-pixel pipeline; `std` (default) adds
  encoded-image decoding.
