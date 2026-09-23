//! Read-only and mutable tensor access traits.

use crate::dtype::DType;

/// Read-only view over tensor data: shape, dtype, and a flat element slice.
pub trait TensorView {
    /// Element data type.
    fn dtype(&self) -> DType;

    /// Number of dimensions (rank).
    fn rank(&self) -> usize;

    /// Shape dimensions.
    fn shape(&self) -> &[usize];

    /// Total number of elements (product of shape).
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Raw bytes backing the tensor (length = `numel * dtype.size_bytes()`, rounded for I4).
    fn as_bytes(&self) -> &[u8];

    /// Interpret the bytes as `f32` elements.
    ///
    /// Returns `None` if the dtype is not `F32`.
    fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype() != DType::F32 {
            return None;
        }
        let bytes = self.as_bytes();
        Some(bytes_as_slice::<f32>(bytes))
    }

    /// Interpret the bytes as `i8` elements.
    ///
    /// Returns `None` if the dtype is not `I8`.
    fn as_i8_slice(&self) -> Option<&[i8]> {
        if self.dtype() != DType::I8 {
            return None;
        }
        Some(bytes_as_slice::<i8>(self.as_bytes()))
    }
}

/// Mutable view over tensor data.
pub trait TensorMut: TensorView {
    /// Mutable raw bytes.
    fn as_bytes_mut(&mut self) -> &mut [u8];

    /// Mutable `f32` view, or `None` if dtype is not `F32`.
    fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype() != DType::F32 {
            return None;
        }
        let bytes = self.as_bytes_mut();
        Some(bytes_as_slice_mut::<f32>(bytes))
    }
}

/// Reinterpret a byte slice as `&[T]`.
///
/// # Panics
/// If `bytes.len()` is not a whole multiple of `size_of::<T>()`, or if
/// `bytes` is not aligned for `T`. These are real runtime checks (not
/// `debug_assert!`) because `bytes` can originate from parsed/untrusted
/// tensor data, and violating either invariant before the cast below would
/// be undefined behavior even in release builds.
pub(crate) fn bytes_as_slice<T>(bytes: &[u8]) -> &[T] {
    assert!(
        bytes.len() % core::mem::size_of::<T>() == 0,
        "byte slice length is not a whole multiple of the element size"
    );
    let ptr = bytes.as_ptr() as *const T;
    assert!(
        (ptr as usize) % core::mem::align_of::<T>() == 0,
        "byte slice is not aligned for the target element type"
    );
    let len = bytes.len() / core::mem::size_of::<T>();
    // SAFETY: length and alignment checked above; T has no invalid bit
    // patterns for F32/I8.
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

/// Reinterpret a mutable byte slice as `&mut [T]`.
///
/// # Panics
/// See [`bytes_as_slice`].
pub(crate) fn bytes_as_slice_mut<T>(bytes: &mut [u8]) -> &mut [T] {
    assert!(
        bytes.len() % core::mem::size_of::<T>() == 0,
        "byte slice length is not a whole multiple of the element size"
    );
    let ptr = bytes.as_mut_ptr() as *mut T;
    assert!(
        (ptr as usize) % core::mem::align_of::<T>() == 0,
        "byte slice is not aligned for the target element type"
    );
    let len = bytes.len() / core::mem::size_of::<T>();
    // SAFETY: as above.
    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_as_slice_accepts_well_formed_input() {
        let bytes = [0u8, 0, 0x80, 0x3f, 0, 0, 0, 0]; // [1.0f32, 0.0f32] little-endian
        let floats: &[f32] = bytes_as_slice(&bytes);
        assert_eq!(floats, &[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "whole multiple")]
    fn bytes_as_slice_rejects_non_multiple_length() {
        // A malformed/truncated buffer (e.g. from a corrupt model file) whose
        // length isn't a multiple of `size_of::<f32>()` must panic loudly
        // instead of silently truncating in a way that used to be masked by
        // `debug_assert!` (compiled out in release builds).
        let bytes = [0u8, 1, 2];
        let _: &[f32] = bytes_as_slice(&bytes);
    }
}
