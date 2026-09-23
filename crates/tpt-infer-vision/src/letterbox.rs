//! Aspect-preserving letterbox resize with constant padding.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::VisionError;
use crate::resize::bilinear_resize;

/// Resize `src` to fit inside `dst_w × dst_h` while preserving aspect ratio,
/// then center the result on a canvas filled with `pad`.
///
/// The content is scaled uniformly by `min(dst_w / src_w, dst_h / src_h)`
/// (rounded to whole pixels) so the aspect ratio is preserved exactly up to
/// rounding; the pad value is expressed in the same units as `src`
/// (input pixel units when used inside [`preprocess_raw_rgb`](crate::preprocess_raw_rgb)).
///
/// # Errors
/// - [`VisionError::InvalidDims`] if any dimension is zero
/// - [`VisionError::InvalidChannels`] if `channels == 0`
/// - [`VisionError::SizeMismatch`] if `src.len() != src_w * src_h * channels`
pub fn letterbox(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
    pad: f32,
) -> Result<Vec<f32>, VisionError> {
    if src_w == 0 || src_h == 0 {
        return Err(VisionError::InvalidDims {
            width: src_w as u32,
            height: src_h as u32,
        });
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(VisionError::InvalidDims {
            width: dst_w as u32,
            height: dst_h as u32,
        });
    }
    if channels == 0 {
        return Err(VisionError::InvalidChannels(channels));
    }
    let expected = src_w * src_h * channels;
    if src.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: src.len(),
        });
    }

    let scale = f64::min(dst_w as f64 / src_w as f64, dst_h as f64 / src_h as f64);
    // Round-half-up without `f64::round` (unavailable in `no_std`);
    // values are non-negative so truncation after +0.5 is equivalent.
    let new_w = ((src_w as f64 * scale + 0.5) as usize).clamp(1, dst_w);
    let new_h = ((src_h as f64 * scale + 0.5) as usize).clamp(1, dst_h);
    let resized = bilinear_resize(src, src_w, src_h, new_w, new_h, channels)?;

    let mut out = vec![pad; dst_w * dst_h * channels];
    let off_x = (dst_w - new_w) / 2;
    let off_y = (dst_h - new_h) / 2;
    let row_len = new_w * channels;
    for y in 0..new_h {
        let src_row = &resized[y * row_len..(y + 1) * row_len];
        let dst_start = ((off_y + y) * dst_w + off_x) * channels;
        out[dst_start..dst_start + row_len].copy_from_slice(src_row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_preserves_aspect_and_dims() {
        // 64x32 (2:1) into 100x100: content scales to 100x50 and is
        // centered vertically with (100 - 50) / 2 = 25 rows of pad top/bottom.
        let src = vec![7.0f32; 64 * 32 * 3];
        let out = letterbox(&src, 64, 32, 100, 100, 3, 0.0).unwrap();
        assert_eq!(out.len(), 100 * 100 * 3);
        assert!(out[..25 * 100 * 3].iter().all(|&v| v == 0.0));
        assert!(out[75 * 100 * 3..].iter().all(|&v| v == 0.0));
        // Content occupies rows 25..75 (aspect 100:50 = 2:1, like the source).
        assert!(out[25 * 100 * 3..75 * 100 * 3]
            .iter()
            .all(|&v| (v - 7.0).abs() < 1e-4));
    }

    #[test]
    fn letterbox_pads_left_and_right_with_pad_value() {
        // 32x64 (1:2) into 100x100: content scales to 50x100, centered
        // horizontally with 25 columns of pad on each side.
        let src = vec![7.0f32; 32 * 64 * 3];
        let out = letterbox(&src, 32, 64, 100, 100, 3, 114.0).unwrap();
        assert_eq!(out.len(), 100 * 100 * 3);
        for y in 0..100 {
            assert_eq!(out[y * 100 * 3], 114.0);
            assert_eq!(out[(y * 100 + 24) * 3], 114.0);
            assert_eq!(out[(y * 100 + 75) * 3], 114.0);
            assert_eq!(out[(y * 100 + 99) * 3], 114.0);
        }
        // Content columns 25..75 contain the resized source value.
        assert!(out[25 * 3..75 * 3].iter().all(|&v| (v - 7.0).abs() < 1e-4));
    }

    #[test]
    fn letterbox_exact_fit_has_no_pad() {
        let src = vec![3.0f32; 50 * 50 * 3];
        let out = letterbox(&src, 50, 50, 100, 100, 3, 114.0).unwrap();
        assert_eq!(out.len(), 100 * 100 * 3);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    }

    #[test]
    fn letterbox_downscales_into_target() {
        let src = vec![9.0f32; 640 * 640 * 3];
        let out = letterbox(&src, 640, 640, 320, 320, 3, 114.0).unwrap();
        assert_eq!(out.len(), 320 * 320 * 3);
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-4));
    }

    #[test]
    fn letterbox_rejects_bad_args() {
        let src = vec![0.0f32; 4];
        assert!(matches!(
            letterbox(&src, 0, 2, 4, 4, 1, 0.0),
            Err(VisionError::InvalidDims { .. })
        ));
        assert!(matches!(
            letterbox(&src, 2, 2, 4, 0, 1, 0.0),
            Err(VisionError::InvalidDims { .. })
        ));
        assert_eq!(
            letterbox(&src, 2, 2, 4, 4, 0, 0.0).unwrap_err(),
            VisionError::InvalidChannels(0)
        );
        assert!(matches!(
            letterbox(&src, 4, 4, 8, 8, 1, 0.0),
            Err(VisionError::SizeMismatch { .. })
        ));
    }
}
