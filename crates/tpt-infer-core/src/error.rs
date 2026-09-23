//! Error types for tensor operations.

/// Errors produced by tensor construction and arena operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
    /// Provided data length does not match the product of the shape.
    ///
    /// Carries `(expected, actual)`.
    SizeMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count provided.
        actual: usize,
    },
    /// A dimension was zero where a positive dimension is required.
    ZeroDimension,
    /// Requested alignment could not be satisfied.
    Misaligned,
    /// Arena (or buffer) is out of space.
    OutOfMemory,
    /// Byte slice length is not a multiple of the element size.
    InvalidByteLength,
    /// Index out of bounds.
    OutOfBounds,
    /// Shape exceeds the maximum supported rank.
    RankTooLarge,
    /// Shape mismatch between two operands.
    ShapeMismatch,
    /// Generic invalid argument.
    Invalid,
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch: expected {expected} elements, got {actual}"
                )
            }
            TensorError::ZeroDimension => f.write_str("shape contains a zero dimension"),
            TensorError::Misaligned => f.write_str("buffer is misaligned for element type"),
            TensorError::OutOfMemory => f.write_str("arena out of memory"),
            TensorError::InvalidByteLength => f.write_str("invalid byte length for element type"),
            TensorError::OutOfBounds => f.write_str("index out of bounds"),
            TensorError::RankTooLarge => f.write_str("shape rank too large"),
            TensorError::ShapeMismatch => f.write_str("shape mismatch"),
            TensorError::Invalid => f.write_str("invalid argument"),
        }
    }
}

#[cfg(feature = "alloc")]
impl core::error::Error for TensorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        let e = TensorError::SizeMismatch {
            expected: 4,
            actual: 3,
        };
        assert!(e.to_string().contains("expected 4"));
        assert_eq!(
            TensorError::OutOfMemory.to_string(),
            "arena out of memory"
        );
    }
}
