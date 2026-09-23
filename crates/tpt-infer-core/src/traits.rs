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

/// Reinterpret a byte slice as `&[T]` (length truncated to a whole number of `T`s).
///
/// # Safety
/// Caller must ensure the bytes originated from (or are aligned/initialized for) `T`.
pub(crate) fn bytes_as_slice<T>(bytes: &[u8]) -> &[T] {
    debug_assert!(bytes.len() % core::mem::size_of::<T>() == 0);
    let len = bytes.len() / core::mem::size_of::<T>();
    let ptr = bytes.as_ptr() as *const T;
    // SAFETY: length computed from byte len; alignment guaranteed by tensor allocators
    // (debug_assert for tests), and T has no invalid bit patterns for F32/I8.
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

/// Reinterpret a mutable byte slice as `&mut [T]`.
///
/// # Safety
/// See [`bytes_as_slice`].
pub(crate) fn bytes_as_slice_mut<T>(bytes: &mut [u8]) -> &mut [T] {
    debug_assert!(bytes.len() % core::mem::size_of::<T>() == 0);
    let len = bytes.len() / core::mem::size_of::<T>();
    let ptr = bytes.as_mut_ptr() as *mut T;
    // SAFETY: as above.
    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}
