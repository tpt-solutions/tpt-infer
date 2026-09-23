//! [`ImageSource`] trait — the integration boundary for tpt-kinetix.

use alloc::vec::Vec;

use crate::error::VisionError;

/// A source of 8-bit RGB image pixels.
///
/// This is the integration boundary for tpt-kinetix (and any other caller
/// that already holds decoded pixels or an encoded image type): implement
/// [`dimensions`](ImageSource::dimensions) and
/// [`to_rgb8`](ImageSource::to_rgb8), then feed the value to
/// [`preprocess_source`](crate::preprocess_source).
pub trait ImageSource {
    /// Pixel dimensions as `(width, height)`.
    fn dimensions(&self) -> (u32, u32);

    /// Interleaved RGB8 bytes, row-major, `width * height * 3` long.
    fn to_rgb8(&self) -> Result<Vec<u8>, VisionError>;
}

/// Borrowed raw RGB8 buffer implementing [`ImageSource`] without a decoder.
///
/// Available in `no_std` builds.
#[derive(Debug, Clone, Copy)]
pub struct RawRgbImage<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
}

impl<'a> RawRgbImage<'a> {
    /// Wrap an interleaved RGB8 buffer (`width * height * 3` bytes).
    pub const fn new(data: &'a [u8], width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }
}

impl ImageSource for RawRgbImage<'_> {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn to_rgb8(&self) -> Result<Vec<u8>, VisionError> {
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or(VisionError::DimensionOverflow)?;
        if self.data.len() != expected {
            return Err(VisionError::SizeMismatch {
                expected,
                actual: self.data.len(),
            });
        }
        Ok(self.data.to_vec())
    }
}

#[cfg(feature = "std")]
impl ImageSource for image::RgbImage {
    fn dimensions(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    fn to_rgb8(&self) -> Result<Vec<u8>, VisionError> {
        Ok(self.as_raw().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_rgb_image_roundtrip() {
        let data = [1u8, 2, 3, 4, 5, 6];
        let img = RawRgbImage::new(&data, 2, 1);
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.to_rgb8().unwrap(), data);
    }

    #[test]
    fn raw_rgb_image_length_mismatch() {
        let data = [0u8; 5];
        let img = RawRgbImage::new(&data, 2, 1);
        assert_eq!(
            img.to_rgb8().unwrap_err(),
            VisionError::SizeMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn raw_rgb_image_dimension_overflow_rejected() {
        // On a 32-bit `usize`, width * height * 3 can overflow for two large
        // (but individually valid `u32`) dimensions; this must be a hard
        // error rather than a silently wrapped `expected` that could defeat
        // the length check.
        let data = [0u8; 6];
        let img = RawRgbImage::new(&data, u32::MAX, u32::MAX);
        assert_eq!(img.to_rgb8().unwrap_err(), VisionError::DimensionOverflow);
    }

    #[cfg(feature = "std")]
    #[test]
    fn rgb_image_source_impl() {
        let img = image::RgbImage::from_pixel(3, 2, image::Rgb([9u8, 8, 7]));
        assert_eq!(ImageSource::dimensions(&img), (3, 2));
        let raw = img.to_rgb8().unwrap();
        assert_eq!(raw.len(), 18);
        assert_eq!(&raw[..3], &[9, 8, 7]);
    }
}
