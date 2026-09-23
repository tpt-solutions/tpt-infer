//! Classic bilinear resize over interleaved `channels`-plane f32 images.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::VisionError;

/// Bilinearly resize an interleaved image from `src_w × src_h` to
/// `dst_w × dst_h`.
///
/// Sampling uses half-pixel centers (OpenCV convention):
/// `src = (dst + 0.5) * (src_dim / dst_dim) - 0.5`, clamped to the source
/// bounds. When the dimensions already match, the data is copied unchanged.
///
/// # Errors
/// - [`VisionError::InvalidDims`] if any dimension is zero
/// - [`VisionError::InvalidChannels`] if `channels == 0`
/// - [`VisionError::SizeMismatch`] if `src.len() != src_w * src_h * channels`
pub fn bilinear_resize(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
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
    if src_w == dst_w && src_h == dst_h {
        return Ok(src.to_vec());
    }

    let mut out = vec![0.0f32; dst_w * dst_h * channels];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;
    let max_x = (src_w - 1) as f32;
    let max_y = (src_h - 1) as f32;

    for dy in 0..dst_h {
        let sy = ((dy as f32 + 0.5) * y_ratio - 0.5).clamp(0.0, max_y);
        // Truncation of a clamped non-negative value equals floor.
        let y0 = (sy as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        let row0 = y0 * src_w * channels;
        let row1 = y1 * src_w * channels;
        let out_row = dy * dst_w * channels;

        for dx in 0..dst_w {
            let sx = ((dx as f32 + 0.5) * x_ratio - 0.5).clamp(0.0, max_x);
            let x0 = (sx as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            let tl = row0 + x0 * channels;
            let tr = row0 + x1 * channels;
            let bl = row1 + x0 * channels;
            let br = row1 + x1 * channels;
            let o = out_row + dx * channels;
            for c in 0..channels {
                let p00 = src[tl + c];
                let p01 = src[tr + c];
                let p10 = src[bl + c];
                let p11 = src[br + c];
                let top = p00 + (p01 - p00) * wx;
                let bottom = p10 + (p11 - p10) * wx;
                out[o + c] = top + (bottom - top) * wy;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_64x48_to_224x224_output_dims() {
        let src = vec![0.5f32; 64 * 48 * 3];
        let out = bilinear_resize(&src, 64, 48, 224, 224, 3).unwrap();
        assert_eq!(out.len(), 224 * 224 * 3);
    }

    #[test]
    fn resize_solid_color_preserved() {
        let src = vec![7.0f32; 16 * 9 * 3];
        let out = bilinear_resize(&src, 16, 9, 33, 17, 3).unwrap();
        assert_eq!(out.len(), 33 * 17 * 3);
        assert!(out.iter().all(|&v| (v - 7.0).abs() < 1e-5));
    }

    #[test]
    fn resize_2x2_to_1x1_is_corner_average() {
        // Hand-computed: half-pixel sampling at (0.5, 0.5) bilinearly
        // averages all four pixels: (0 + 10 + 20 + 30) / 4 = 15.
        let src = [0.0f32, 10.0, 20.0, 30.0];
        let out = bilinear_resize(&src, 2, 2, 1, 1, 1).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn resize_identity_is_exact_copy() {
        let src: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let out = bilinear_resize(&src, 4, 4, 4, 4, 3).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn resize_rejects_bad_args() {
        let src = vec![0.0f32; 4];
        assert!(matches!(
            bilinear_resize(&src, 0, 2, 4, 4, 1),
            Err(VisionError::InvalidDims { .. })
        ));
        assert!(matches!(
            bilinear_resize(&src, 2, 2, 4, 0, 1),
            Err(VisionError::InvalidDims { .. })
        ));
        assert_eq!(
            bilinear_resize(&src, 2, 2, 4, 4, 0).unwrap_err(),
            VisionError::InvalidChannels(0)
        );
        assert!(matches!(
            bilinear_resize(&src, 4, 4, 8, 8, 1),
            Err(VisionError::SizeMismatch { .. })
        ));
    }
}
