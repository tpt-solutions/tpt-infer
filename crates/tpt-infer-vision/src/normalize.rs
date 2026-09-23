//! Per-channel normalization of interleaved HWC f32 images.

use crate::error::VisionError;

/// Apply `out = (x * rescale - mean[c]) / std[c]` per channel of an
/// interleaved 3-channel HWC buffer, in place.
///
/// Mean/std must be non-zero-checked: any zero entry of `std` aborts with
/// [`VisionError::ZeroStd`] before the buffer is modified.
///
/// # Errors
/// - [`VisionError::SizeMismatch`] if `data.len() != height * width * 3`
/// - [`VisionError::ZeroStd`] if any `std` entry is zero
pub fn normalize_hwc(
    data: &mut [f32],
    height: usize,
    width: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    rescale: f32,
) -> Result<(), VisionError> {
    let expected = height * width * 3;
    if data.len() != expected {
        return Err(VisionError::SizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    if std.contains(&0.0) {
        return Err(VisionError::ZeroStd);
    }
    for px in data.chunks_exact_mut(3) {
        for (c, slot) in px.iter_mut().enumerate() {
            *slot = (*slot * rescale - mean[c]) / std[c];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    #[test]
    fn imagenet_pixel_255_matches_numpy_reference() {
        // Reference (hand-computed from numpy semantics):
        //   x = 255 / 255 = 1.0
        //   R: (1.0 - 0.485) / 0.229 = 0.515 / 0.229 = 2.248908...
        //   G: (1.0 - 0.456) / 0.224 = 0.544 / 0.224 = 2.428571...
        //   B: (1.0 - 0.406) / 0.225 = 0.594 / 0.225 = 2.640000
        let mut data = [255.0f32, 255.0, 255.0];
        normalize_hwc(&mut data, 1, 1, &IMAGENET_MEAN, &IMAGENET_STD, 1.0 / 255.0).unwrap();
        assert!((data[0] - 2.248_908).abs() < 1e-4, "R = {}", data[0]);
        assert!((data[1] - 2.428_571).abs() < 1e-4, "G = {}", data[1]);
        assert!((data[2] - 2.64).abs() < 1e-4, "B = {}", data[2]);
    }

    #[test]
    fn imagenet_pixel_0_matches_numpy_reference() {
        // Reference:
        //   R: (0 - 0.485) / 0.229 = -2.117903...
        //   G: (0 - 0.456) / 0.224 = -2.035714...
        //   B: (0 - 0.406) / 0.225 = -1.804444...
        let mut data = [0.0f32, 0.0, 0.0];
        normalize_hwc(&mut data, 1, 1, &IMAGENET_MEAN, &IMAGENET_STD, 1.0 / 255.0).unwrap();
        assert!((data[0] - -2.117_904).abs() < 1e-4, "R = {}", data[0]);
        assert!((data[1] - -2.035_714).abs() < 1e-4, "G = {}", data[1]);
        assert!((data[2] - -1.804_444).abs() < 1e-4, "B = {}", data[2]);
    }

    #[test]
    fn identity_norm_applies_only_rescale() {
        // YOLO-style: mean 0, std 1 → output is pixel * rescale.
        // 128 / 255 = 0.501960...
        let mut data = [128.0f32, 64.0, 0.0];
        normalize_hwc(&mut data, 1, 1, &[0.0; 3], &[1.0; 3], 1.0 / 255.0).unwrap();
        assert!((data[0] - 0.501_961).abs() < 1e-5, "R = {}", data[0]);
        assert!((data[1] - 0.250_980).abs() < 1e-5, "G = {}", data[1]);
        assert!((data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_one_mean_shift_only() {
        // rescale 1, mean 128, std 1 → x - 128.
        let mut data = [200.0f32, 128.0, 8.0];
        normalize_hwc(&mut data, 1, 1, &[128.0; 3], &[1.0; 3], 1.0).unwrap();
        assert_eq!(data, [72.0, 0.0, -120.0]);
    }

    #[test]
    fn per_pixel_application() {
        // Two pixels: [255,0,0] and [0,255,0] with identity norm.
        let mut data = [255.0f32, 0.0, 0.0, 0.0, 255.0, 0.0];
        normalize_hwc(&mut data, 1, 2, &[0.0; 3], &[1.0; 3], 1.0 / 255.0).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert_eq!(data[5], 0.0);
    }

    #[test]
    fn zero_std_rejected_before_mutation() {
        let mut data = [1.0f32, 2.0, 3.0];
        let err = normalize_hwc(&mut data, 1, 1, &[0.0; 3], &[1.0, 0.0, 1.0], 1.0).unwrap_err();
        assert_eq!(err, VisionError::ZeroStd);
        assert_eq!(data, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn size_mismatch_rejected() {
        let mut data = [0.0f32; 11];
        assert!(matches!(
            normalize_hwc(&mut data, 2, 2, &[0.0; 3], &[1.0; 3], 1.0),
            Err(VisionError::SizeMismatch {
                expected: 12,
                actual: 11
            })
        ));
    }
}
