//! Image preprocessing for vision models.
//!
//! Pure-Rust pipeline that turns raw or encoded images into
//! [`Tensor<f32, 4>`](tpt_infer_core::Tensor) batches ready for inference:
//!
//! 1. (optional) **letterbox** — aspect-preserving resize + constant pad
//! 2. **bilinear resize** to the target `width × height`
//! 3. **rescale** (e.g. `1/255`) and per-channel **normalization**
//!    `(x - mean) / std`
//! 4. optional RGB ↔ BGR swap
//! 5. **layout** conversion (CHW ↔ HWC) and a leading batch dimension
//!
//! # Features
//!
//! - `std` (default): enables [`preprocess_image`], which decodes encoded
//!   bytes (PNG/JPEG/…) via the `image` crate, and an [`ImageSource`]
//!   implementation for [`image::RgbImage`].
//! - without `std` (but with `alloc`): [`preprocess_raw_rgb`] and
//!   [`preprocess_source`] still work on already-decoded pixels.
//!
//! # Example
//!
//! ```
//! use tpt_infer_vision::{preprocess_raw_rgb, PreprocessConfig};
//!
//! // 2×2 RGB image: red, green, blue, white.
//! let rgb = [
//!     255u8, 0, 0, 0, 255, 0, //
//!     0, 0, 255, 255, 255, 255,
//! ];
//! let mut config = PreprocessConfig::imagenet();
//! config.width = 2;
//! config.height = 2;
//! let tensor = preprocess_raw_rgb(&rgb, 2, 2, &config).unwrap();
//! assert_eq!(tensor.shape(), [1, 3, 2, 2]);
//! assert_eq!(tensor.dtype(), tpt_infer_core::DType::F32);
//! ```

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

extern crate alloc;

pub mod config;
pub mod error;
pub mod layout;
pub mod letterbox;
pub mod normalize;
pub mod resize;
pub mod source;

pub use config::{ColorOrder, Layout, PreprocessConfig};
pub use error::VisionError;
pub use layout::{chw_to_hwc, hwc_to_chw, swap_rb};
pub use letterbox::letterbox;
pub use normalize::normalize_hwc;
pub use resize::bilinear_resize;
pub use source::{ImageSource, RawRgbImage};

use alloc::vec::Vec;

use tpt_infer_core::Tensor;

/// Preprocess interleaved RGB8 pixels (`width * height * 3` bytes, row-major)
/// into a rank-4 `f32` tensor.
///
/// Output shape is `[1, 3, height, width]` for [`Layout::Chw`] or
/// `[1, height, width, 3]` for [`Layout::Hwc`], both using the config's
/// target dimensions. See the [crate documentation](crate) for the pipeline
/// order.
///
/// # Errors
/// - [`VisionError::InvalidDims`] if a source or target dimension is zero
/// - [`VisionError::SizeMismatch`] if `data.len() != width * height * 3`
/// - [`VisionError::ZeroStd`] if any config `std` entry is zero
/// - [`VisionError::InvalidChannels`] (internal layout failures)
/// - [`VisionError::Tensor`] if the output tensor cannot be constructed
pub fn preprocess_raw_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    config: &PreprocessConfig,
) -> Result<Tensor<f32, 4>, VisionError> {
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidDims { width, height });
    }
    if config.width == 0 || config.height == 0 {
        return Err(VisionError::InvalidDims {
            width: config.width,
            height: config.height,
        });
    }
    let expected = width as usize * height as usize * 3;
    if data.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: data.len(),
        });
    }

    let (src_w, src_h) = (width as usize, height as usize);
    let (dst_w, dst_h) = (config.width as usize, config.height as usize);

    let mut pixels: Vec<f32> = data.iter().map(|&v| f32::from(v)).collect();
    pixels = if config.letterbox {
        letterbox(&pixels, src_w, src_h, dst_w, dst_h, 3, config.pad)?
    } else {
        bilinear_resize(&pixels, src_w, src_h, dst_w, dst_h, 3)?
    };

    if config.color == ColorOrder::Bgr {
        swap_rb(&mut pixels, dst_h, dst_w)?;
    }
    normalize_hwc(
        &mut pixels,
        dst_h,
        dst_w,
        &config.mean,
        &config.std,
        config.rescale,
    )?;

    let (out, shape) = match config.layout {
        Layout::Chw => (hwc_to_chw(&pixels, dst_h, dst_w, 3)?, [1, 3, dst_h, dst_w]),
        Layout::Hwc => (pixels, [1, dst_h, dst_w, 3]),
    };
    Tensor::from_vec(out, shape).map_err(VisionError::Tensor)
}

/// Decode encoded image bytes (PNG, JPEG, …) and preprocess them.
///
/// Requires the `std` feature (default).
///
/// # Errors
/// - [`VisionError::Decode`] if the bytes are not a decodable image
/// - remaining errors as [`preprocess_raw_rgb`]
#[cfg(feature = "std")]
pub fn preprocess_image(
    bytes: &[u8],
    config: &PreprocessConfig,
) -> Result<Tensor<f32, 4>, VisionError> {
    let img = image::load_from_memory(bytes).map_err(|e| VisionError::Decode(e.to_string()))?;
    let rgb = img.to_rgb8();
    preprocess_raw_rgb(rgb.as_raw(), rgb.width(), rgb.height(), config)
}

/// Read pixels from any [`ImageSource`] and preprocess them.
///
/// This is the tpt-kinetix integration entry point: implement
/// [`ImageSource`] for your image type (or use [`RawRgbImage`] /
/// `image::RgbImage`) and call this.
///
/// # Errors
/// - [`ImageSource::to_rgb8`] errors from the source itself
/// - remaining errors as [`preprocess_raw_rgb`]
pub fn preprocess_source<S: ImageSource + ?Sized>(
    source: &S,
    config: &PreprocessConfig,
) -> Result<Tensor<f32, 4>, VisionError> {
    let (width, height) = source.dimensions();
    let rgb = source.to_rgb8()?;
    preprocess_raw_rgb(&rgb, width, height, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity(width as usize * height as usize * 3);
        for _ in 0..width * height {
            v.extend_from_slice(&rgb);
        }
        v
    }

    fn small_config() -> PreprocessConfig {
        let mut cfg = PreprocessConfig::imagenet();
        cfg.width = 8;
        cfg.height = 8;
        cfg.mean = [0.0; 3];
        cfg.std = [1.0; 3];
        cfg
    }

    #[test]
    fn preprocess_raw_chw_shape_and_values() {
        let data = solid(4, 4, [255, 128, 0]);
        let t = preprocess_raw_rgb(&data, 4, 4, &small_config()).unwrap();
        assert_eq!(t.shape(), [1, 3, 8, 8]);
        let plane = 8 * 8;
        let s = t.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[plane] - 128.0 / 255.0).abs() < 1e-6);
        assert!(s[2 * plane].abs() < 1e-6);
        // Interior values match (solid color, constant across the plane).
        assert!((s[plane - 1] - 1.0).abs() < 1e-6);
        assert!((s[2 * plane - 1] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn preprocess_raw_hwc_shape() {
        let data = solid(4, 4, [255, 128, 0]);
        let mut cfg = small_config();
        cfg.layout = Layout::Hwc;
        let t = preprocess_raw_rgb(&data, 4, 4, &cfg).unwrap();
        assert_eq!(t.shape(), [1, 8, 8, 3]);
        let s = t.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!(s[2].abs() < 1e-6);
    }

    #[test]
    fn preprocess_bgr_swaps_channels() {
        let data = solid(2, 2, [255, 128, 0]);
        let mut cfg = small_config();
        cfg.color = ColorOrder::Bgr;
        let t = preprocess_raw_rgb(&data, 2, 2, &cfg).unwrap();
        let plane = 8 * 8;
        let s = t.as_slice();
        assert!(s[0].abs() < 1e-6); // B first
        assert!((s[plane] - 128.0 / 255.0).abs() < 1e-6);
        assert!((s[2 * plane] - 1.0).abs() < 1e-6); // R last
    }

    #[test]
    fn preprocess_letterbox_keeps_dims() {
        // 8x4 image into 8x8 target: content 8x4 centered with pad rows.
        let data = solid(8, 4, [255, 255, 255]);
        let mut cfg = small_config();
        cfg.letterbox = true;
        cfg.pad = 0.0;
        let t = preprocess_raw_rgb(&data, 8, 4, &cfg).unwrap();
        assert_eq!(t.shape(), [1, 3, 8, 8]);
        let s = t.as_slice();
        // Rows 0..2 are pad (0 * rescale = 0 with mean 0 / std 1).
        assert!(s[..2 * 8].iter().all(|&v| v == 0.0));
        // Content rows: white → 1.0.
        assert!((s[3 * 8] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn preprocess_rejects_bad_input() {
        let data = solid(4, 4, [0, 0, 0]);
        // Zero source dims.
        assert!(matches!(
            preprocess_raw_rgb(&data, 0, 4, &small_config()),
            Err(VisionError::InvalidDims { .. })
        ));
        // Buffer length mismatch.
        assert!(matches!(
            preprocess_raw_rgb(&data[..3], 4, 4, &small_config()),
            Err(VisionError::SizeMismatch {
                expected: 48,
                actual: 3
            })
        ));
        // Zero target dims.
        let mut cfg = small_config();
        cfg.width = 0;
        assert!(matches!(
            preprocess_raw_rgb(&data, 4, 4, &cfg),
            Err(VisionError::InvalidDims { .. })
        ));
        // Zero std.
        let mut cfg = small_config();
        cfg.std = [1.0, 0.0, 1.0];
        assert_eq!(
            preprocess_raw_rgb(&data, 4, 4, &cfg).unwrap_err(),
            VisionError::ZeroStd
        );
    }

    #[test]
    fn preprocess_source_from_raw_rgb_image() {
        let data = solid(4, 4, [255, 128, 0]);
        let src = RawRgbImage::new(&data, 4, 4);
        let t = preprocess_source(&src, &small_config()).unwrap();
        assert_eq!(t.shape(), [1, 3, 8, 8]);
        let plane = 64;
        assert!((t.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((t.as_slice()[plane] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn preprocess_source_propagates_source_error() {
        let data = [0u8; 5];
        let src = RawRgbImage::new(&data, 2, 1);
        assert!(matches!(
            preprocess_source(&src, &small_config()),
            Err(VisionError::SizeMismatch { .. })
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn preprocess_image_decodes_png_bytes() {
        use std::io::Cursor;

        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([255u8, 0, 0]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        let t = preprocess_image(&png, &small_config()).unwrap();
        assert_eq!(t.shape(), [1, 3, 8, 8]);
        assert!((t.as_slice()[0] - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn preprocess_image_rejects_garbage() {
        assert!(matches!(
            preprocess_image(b"not an image", &small_config()),
            Err(VisionError::Decode(_))
        ));
    }
}
