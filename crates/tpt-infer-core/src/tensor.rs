//! Const-generic `Tensor` and heap-backed `TensorVec`.

use crate::dtype::DType;
use crate::error::TensorError;
use crate::shape::{num_elements, strides};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Maps a Rust element type to its [`DType`].
///
/// Implemented for the primitive types tpt-infer supports.
pub trait DTypeOf {
    /// The corresponding [`DType`].
    const DTYPE: DType;
}

impl DTypeOf for f32 {
    const DTYPE: DType = DType::F32;
}
impl DTypeOf for i8 {
    const DTYPE: DType = DType::I8;
}
impl DTypeOf for u8 {
    const DTYPE: DType = DType::U8;
}
impl DTypeOf for i32 {
    const DTYPE: DType = DType::I32;
}
impl DTypeOf for i64 {
    const DTYPE: DType = DType::I64;
}
impl DTypeOf for bool {
    const DTYPE: DType = DType::Bool;
}
// f16 has no primitive; allow users to pass a newtype, but also provide
// a dedicated bit-pattern type below if needed. For now, `u16` maps to F16
// only when explicitly requested via `Tensor<u16, N>` — document carefully.
impl DTypeOf for u16 {
    const DTYPE: DType = DType::F16;
}

/// Shorthand for `<T as DTypeOf>::DTYPE`.
///
/// # Panics (const)
/// Does not panic; unknown types simply do not implement [`DTypeOf`].
pub const fn dtype_of<T: DTypeOf>() -> DType {
    T::DTYPE
}

/// A tensor with a compile-time fixed rank `N` and element type `T`.
///
/// Shape is `[usize; N]`. Element storage is a contiguous row-major buffer.
///
/// # Example
/// ```
/// use tpt_infer_core::Tensor;
/// let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
/// assert_eq!(t.shape(), [2, 2]);
/// assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T, const N: usize> {
    shape: [usize; N],
    data: TensorData<T>,
}

#[derive(Debug, Clone, PartialEq)]
enum TensorData<T> {
    #[cfg(not(feature = "alloc"))]
    Phantom(core::marker::PhantomData<T>),
    #[cfg(feature = "alloc")]
    Owned(Vec<T>),
}

impl<T: DTypeOf, const N: usize> Tensor<T, N> {
    /// Shape of the tensor.
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Number of dimensions (rank).
    pub const fn rank(&self) -> usize {
        N
    }

    /// Total number of elements (product of shape).
    pub fn numel(&self) -> usize {
        num_elements(self.shape)
    }

    /// Row-major strides for this shape.
    pub fn strides(&self) -> [usize; N] {
        strides(self.shape)
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }
}

#[cfg(feature = "alloc")]
impl<T: Copy + DTypeOf, const N: usize> Tensor<T, N> {
    /// Create a tensor by copying `data`, shaped `shape`.
    ///
    /// # Errors
    /// [`TensorError::SizeMismatch`] if `data.len()` != product(shape).
    pub fn new(data: &[T], shape: [usize; N]) -> Result<Self, TensorError> {
        let expected = num_elements(shape);
        if data.len() != expected {
            return Err(TensorError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            shape,
            data: TensorData::Owned(data.to_vec()),
        })
    }

    /// Create a tensor filled with `value`.
    pub fn full(shape: [usize; N], value: T) -> Self
    where
        T: Clone,
    {
        Self {
            shape,
            data: TensorData::Owned(alloc::vec![value; num_elements(shape)]),
        }
    }

    /// Create a zero-initialized tensor.
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Default + Clone,
    {
        Self::full(shape, T::default())
    }

    /// Immutable view of the elements (row-major).
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            TensorData::Owned(v) => v,
        }
    }

    /// Mutable view of the elements (row-major).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.data {
            TensorData::Owned(v) => v.as_mut_slice(),
        }
    }

    /// Element at a flat row-major index.
    ///
    /// # Panics
    /// Panics if `index >= numel()`.
    pub fn get(&self, index: usize) -> T {
        self.as_slice()[index]
    }

    /// Mutable element at a flat row-major index.
    ///
    /// # Panics
    /// Panics if `index >= numel()`.
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }

    /// Element at a multi-dimensional index.
    ///
    /// # Panics
    /// Panics if any index is out of bounds for its dimension.
    pub fn get_at(&self, index: [usize; N]) -> T {
        self.as_slice()[crate::shape::ravel_index(index, self.shape)]
    }

    /// Reshape, preserving row-major data order.
    ///
    /// # Errors
    /// [`TensorError::SizeMismatch`] if element counts differ.
    pub fn reshape<const M: usize>(
        &self,
        new_shape: [usize; M],
    ) -> Result<Tensor<T, M>, TensorError> {
        if num_elements(new_shape) != self.numel() {
            return Err(TensorError::SizeMismatch {
                expected: self.numel(),
                actual: num_elements(new_shape),
            });
        }
        Ok(Tensor {
            shape: new_shape,
            data: TensorData::Owned(self.as_slice().to_vec()),
        })
    }

    /// Consume the tensor and return the underlying `Vec`.
    pub fn into_vec(self) -> Vec<T> {
        match self.data {
            TensorData::Owned(v) => v,
        }
    }

    /// Raw pointer to the element buffer.
    pub fn data_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
}

#[cfg(feature = "alloc")]
impl<T: Copy + DTypeOf, const N: usize> Tensor<T, N> {
    /// Create a tensor by *moving* ownership of `data`.
    ///
    /// # Errors
    /// [`TensorError::SizeMismatch`] if lengths differ.
    pub fn from_vec(data: Vec<T>, shape: [usize; N]) -> Result<Self, TensorError> {
        let expected = num_elements(shape);
        if data.len() != expected {
            return Err(TensorError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            shape,
            data: TensorData::Owned(data),
        })
    }
}

/// Heap-allocated dynamically-shaped tensor (rank ≤ 8).
///
/// Useful for ONNX-loaded models whose shapes are only known at runtime.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorVec<T> {
    shape: crate::shape::Shape,
    data: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: Copy + DTypeOf> TensorVec<T> {
    /// Create from data and a dynamic shape (zero-padded to rank 8).
    ///
    /// # Errors
    /// - [`TensorError::RankTooLarge`] if the shape has more than 8 dims
    /// - [`TensorError::SizeMismatch`] if data length != product(shape)
    pub fn new(data: &[T], shape: &[usize]) -> Result<Self, TensorError> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Move ownership of `data`.
    ///
    /// # Errors
    /// Same validation as [`TensorVec::new`].
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorError> {
        if shape.len() > crate::shape::MAX_RANK {
            return Err(TensorError::RankTooLarge);
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let mut dyn_shape = [0; crate::shape::MAX_RANK];
        dyn_shape[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            shape: dyn_shape,
            data,
        })
    }

    /// Full dynamic shape (rank 8, trailing zeros pad unused dims).
    pub fn dyn_shape(&self) -> crate::shape::Shape {
        self.shape
    }

    /// Active dimensions (trailing zeros trimmed from the logical rank view).
    pub fn dims(&self) -> &[usize] {
        let mut end = self.shape.len();
        while end > 1 && self.shape[end - 1] == 0 {
            end -= 1;
        }
        &self.shape[..end]
    }

    /// Rank (number of active dimensions).
    pub fn rank(&self) -> usize {
        self.dims().len()
    }

    /// Total elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// Immutable element slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable element slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert to a fixed-rank tensor (copies data).
    ///
    /// # Errors
    /// - [`TensorError::ShapeMismatch`] if active rank != `N`
    pub fn to_tensor<const N: usize>(&self) -> Result<Tensor<T, N>, TensorError> {
        if self.rank() != N {
            return Err(TensorError::ShapeMismatch);
        }
        let mut shape = [0usize; N];
        shape.copy_from_slice(self.dims());
        Tensor::new(&self.data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_from_slice() {
        let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        assert_eq!(t.shape(), [2, 2]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.strides(), [2, 1]);
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn construct_size_mismatch() {
        let err = Tensor::new(&[1.0f32, 2.0, 3.0], [2, 2]).unwrap_err();
        assert_eq!(
            err,
            TensorError::SizeMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn full_and_zeros() {
        let t: Tensor<i32, 3> = Tensor::full([2, 3, 4], 7);
        assert_eq!(t.numel(), 24);
        assert!(t.as_slice().iter().all(|&x| x == 7));

        let z: Tensor<f32, 2> = Tensor::zeros([3, 3]);
        assert_eq!(z.numel(), 9);
        assert!(z.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn get_at_multi_index() {
        let t = Tensor::new(&[0u8, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        assert_eq!(t.get_at([0, 0]), 0);
        assert_eq!(t.get_at([0, 2]), 2);
        assert_eq!(t.get_at([1, 0]), 3);
        assert_eq!(t.get_at([1, 2]), 5);
    }

    #[test]
    fn reshape_preserves_order() {
        let t = Tensor::new(&[1i32, 2, 3, 4, 5, 6], [2, 3]).unwrap();
        let r = t.reshape([3, 2]).unwrap();
        assert_eq!(r.shape(), [3, 2]);
        assert_eq!(r.as_slice(), &[1, 2, 3, 4, 5, 6]);
        let bad = t.reshape([4, 2]);
        assert!(bad.is_err());
    }

    #[test]
    fn i8_dtype_detected() {
        let t = Tensor::new(&[1i8, 2], [2]).unwrap();
        assert_eq!(t.dtype(), DType::I8);
    }

    #[test]
    fn tensor_vec_roundtrip() {
        let v = TensorVec::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(v.rank(), 2);
        assert_eq!(v.dims(), &[2, 3]);
        assert_eq!(v.numel(), 6);
        let t = v.to_tensor::<2>().unwrap();
        assert_eq!(t.shape(), [2, 3]);
    }

    #[test]
    fn tensor_vec_rank_too_large() {
        let shape = [1usize; 9];
        let data = vec![0.0f32; 9];
        let err = TensorVec::new(&data, &shape).unwrap_err();
        assert_eq!(err, TensorError::RankTooLarge);
    }
}
