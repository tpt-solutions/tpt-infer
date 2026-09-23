//! Error type for vision preprocessing.

use alloc::string::String;

use tpt_infer_core::TensorError;

/// Errors produced by image decoding and preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VisionError {
    /// The encoded image bytes could not be decoded.
    Decode(String),
    /// An image dimension was zero (source or target).
    InvalidDims {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// A buffer length does not match the expected size.
    SizeMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count provided.
        actual: usize,
    },
    /// The channel count was zero (or otherwise unsupported).
    InvalidChannels(usize),
    /// A [`PreprocessConfig`](crate::PreprocessConfig) `std` entry was zero.
    ZeroStd,
    /// Tensor construction failed.
    Tensor(TensorError),
}

impl core::fmt::Display for VisionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VisionError::Decode(msg) => write!(f, "failed to decode image: {msg}"),
            VisionError::InvalidDims { width, height } => {
                write!(f, "invalid image dimensions {width}x{height}")
            }
            VisionError::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} elements, got {actual}")
            }
            VisionError::InvalidChannels(ch) => write!(f, "invalid channel count: {ch}"),
            VisionError::ZeroStd => f.write_str("standard deviation must be non-zero"),
            VisionError::Tensor(e) => write!(f, "tensor error: {e}"),
        }
    }
}

impl core::error::Error for VisionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            VisionError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        assert_eq!(
            VisionError::ZeroStd.to_string(),
            "standard deviation must be non-zero"
        );
        assert_eq!(
            VisionError::InvalidDims {
                width: 0,
                height: 8
            }
            .to_string(),
            "invalid image dimensions 0x8"
        );
        assert_eq!(
            VisionError::SizeMismatch {
                expected: 6,
                actual: 5
            }
            .to_string(),
            "size mismatch: expected 6 elements, got 5"
        );
        assert_eq!(
            VisionError::InvalidChannels(0).to_string(),
            "invalid channel count: 0"
        );
        let e = VisionError::Decode(alloc::string::String::from("bad png"));
        assert_eq!(e.to_string(), "failed to decode image: bad png");
    }

    #[test]
    fn tensor_source_is_available() {
        let e = VisionError::Tensor(TensorError::Invalid);
        let src = core::error::Error::source(&e);
        assert!(src.is_some());
        let e2 = VisionError::ZeroStd;
        assert!(core::error::Error::source(&e2).is_none());
    }
}
