//! CHW ↔ HWC layout conversion and RGB↔BGR channel swapping.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::VisionError;

/// Convert an interleaved HWC buffer to planar CHW.
///
/// Element `(y, x, c)` of the input lands at `c * (height * width) + y * width + x`.
///
/// # Errors
/// - [`VisionError::InvalidChannels`] if `channels == 0`
/// - [`VisionError::SizeMismatch`] if `data.len() != height * width * channels`
pub fn hwc_to_chw(
    data: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<Vec<f32>, VisionError> {
    if channels == 0 {
        return Err(VisionError::InvalidChannels(channels));
    }
    let plane = height * width;
    let expected = plane * channels;
    if data.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    let mut out = vec![0.0f32; expected];
    for y in 0..height {
        for x in 0..width {
            let hw = y * width + x;
            for (c, &v) in data[hw * channels..(hw + 1) * channels].iter().enumerate() {
                out[c * plane + hw] = v;
            }
        }
    }
    Ok(out)
}

/// Convert a planar CHW buffer to interleaved HWC.
///
/// Element `c * (height * width) + y * width + x` of the input lands at
/// `(y, x, c)` in the output.
///
/// # Errors
/// - [`VisionError::InvalidChannels`] if `channels == 0`
/// - [`VisionError::SizeMismatch`] if `data.len() != channels * height * width`
pub fn chw_to_hwc(
    data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, VisionError> {
    if channels == 0 {
        return Err(VisionError::InvalidChannels(channels));
    }
    let plane = height * width;
    let expected = channels * plane;
    if data.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    let mut out = vec![0.0f32; expected];
    for y in 0..height {
        for x in 0..width {
            let hw = y * width + x;
            for c in 0..channels {
                out[hw * channels + c] = data[c * plane + hw];
            }
        }
    }
    Ok(out)
}

/// Swap the R and B channels of an interleaved 3-channel HWC buffer
/// (RGB ↔ BGR; the operation is its own inverse).
///
/// # Errors
/// [`VisionError::SizeMismatch`] if `data.len() != height * width * 3`
pub fn swap_rb(data: &mut [f32], height: usize, width: usize) -> Result<(), VisionError> {
    let expected = height * width * 3;
    if data.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    for px in data.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hwc_to_chw_known_mapping() {
        // 2x2 RGB, values 0..12 in HWC order; plane = 4 spatial pixels.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let chw = hwc_to_chw(&data, 2, 2, 3).unwrap();
        let plane = 4;
        // Pixel (0, 1) is HWC indices 3..6 = [3, 4, 5] → plane * c + 1.
        assert_eq!(chw[1], 3.0); // R plane
        assert_eq!(chw[plane + 1], 4.0); // G plane
        assert_eq!(chw[2 * plane + 1], 5.0); // B plane
        // Pixel (1, 0) is HWC indices 6..9 = [6, 7, 8] → plane * c + 2.
        assert_eq!(chw[2], 6.0);
        assert_eq!(chw[plane + 2], 7.0);
        assert_eq!(chw[2 * plane + 2], 8.0);
    }

    #[test]
    fn hwc_chw_roundtrip() {
        let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.25).collect();
        let chw = hwc_to_chw(&data, 5, 2, 3).unwrap();
        assert_eq!(chw.len(), 30);
        let back = chw_to_hwc(&chw, 3, 5, 2).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn chw_to_hwc_known_mapping() {
        // CHW planes: R = [1, 2], G = [3, 4], B = [5, 6] for 1x2 image.
        let chw = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hwc = chw_to_hwc(&chw, 3, 1, 2).unwrap();
        assert_eq!(hwc, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn size_mismatch_errors() {
        let data = [0.0f32; 11];
        assert!(matches!(
            hwc_to_chw(&data, 2, 2, 3),
            Err(VisionError::SizeMismatch {
                expected: 12,
                actual: 11
            })
        ));
        assert!(matches!(
            chw_to_hwc(&data, 3, 2, 2),
            Err(VisionError::SizeMismatch {
                expected: 12,
                actual: 11
            })
        ));
        let mut one = [0.0f32; 5];
        assert!(matches!(
            swap_rb(&mut one, 2, 2),
            Err(VisionError::SizeMismatch {
                expected: 12,
                actual: 5
            })
        ));
    }

    #[test]
    fn zero_channels_rejected() {
        let data = [0.0f32; 4];
        assert_eq!(
            hwc_to_chw(&data, 2, 2, 0).unwrap_err(),
            VisionError::InvalidChannels(0)
        );
        assert_eq!(
            chw_to_hwc(&data, 0, 2, 2).unwrap_err(),
            VisionError::InvalidChannels(0)
        );
    }

    #[test]
    fn swap_rb_swaps_and_restores() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        swap_rb(&mut data, 1, 2).unwrap();
        assert_eq!(data, [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
        swap_rb(&mut data, 1, 2).unwrap();
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
