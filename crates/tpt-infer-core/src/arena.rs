//! `no_std` bump allocator over a pre-allocated byte slice.

use crate::error::TensorError;

/// A bump (arena) allocator that carves aligned chunks out of a caller-provided
/// byte slice. Ideal for zero-allocation inference on embedded targets.
///
/// # Example
/// ```
/// use tpt_infer_core::BumpArena;
/// let mut buf = [0u8; 1024];
/// let mut arena = BumpArena::new(&mut buf);
/// let a = arena.alloc_bytes(100, 1).unwrap();
/// let b = arena.alloc_bytes(100, 1).unwrap();
/// assert_eq!(a.len(), 100);
/// arena.reset();
/// assert_eq!(arena.used(), 0);
/// ```
#[derive(Debug)]
pub struct BumpArena<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> BumpArena<'a> {
    /// Create an arena over `buf`.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    /// Bytes used so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Total capacity of the underlying buffer.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Remaining free bytes.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.offset
    }

    /// Reset the bump pointer to zero (does not zero memory).
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate `len` bytes with the given `align` (must be a power of two).
    ///
    /// # Errors
    /// - [`TensorError::Invalid`] if `align` is zero or not a power of two
    /// - [`TensorError::OutOfMemory`] if the request does not fit
    pub fn alloc_bytes(&mut self, len: usize, align: usize) -> Result<&'a mut [u8], TensorError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(TensorError::Invalid);
        }
        let base = self.buf.as_ptr() as usize + self.offset;
        let aligned = base.checked_add(align - 1).ok_or(TensorError::OutOfMemory)? & !(align - 1);
        let new_off = aligned - self.buf.as_ptr() as usize;
        let end = new_off.checked_add(len).ok_or(TensorError::OutOfMemory)?;
        if end > self.buf.len() {
            return Err(TensorError::OutOfMemory);
        }
        self.offset = end;
        // SAFETY: [new_off, end) is in-bounds; we never hand out overlapping
        // regions because offset only advances. Lifetime 'a ties to the arena's
        // buffer borrow — we extend it deliberately (bump semantics).
        let ptr = unsafe { self.buf.as_mut_ptr().add(new_off) };
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
    }

    /// Allocate space for `count` elements of type `T`.
    ///
    /// # Errors
    /// [`TensorError::OutOfMemory`] if the request does not fit.
    pub fn alloc_slice<T>(
        &mut self,
        count: usize,
    ) -> Result<&'a mut [T], TensorError> {
        let size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let byte_len = size.checked_mul(count).ok_or(TensorError::OutOfMemory)?;
        let bytes = self.alloc_bytes(byte_len, align)?;
        let ptr = bytes.as_mut_ptr() as *mut T;
        // SAFETY: alignment and length established by alloc_bytes.
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Allocate an uninitialized-but-zeroed slot for a single `T`.
    ///
    /// # Errors
    /// [`TensorError::OutOfMemory`] if the request does not fit.
    pub fn alloc_one<T: Default>(&mut self) -> Result<&'a mut T, TensorError> {
        let s = self.alloc_slice::<T>(1)?;
        s[0] = T::default();
        Ok(&mut s[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_reset() {
        let mut buf = [0u8; 256];
        let mut arena = BumpArena::new(&mut buf);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.capacity(), 256);

        let a = arena.alloc_bytes(100, 1).unwrap();
        assert_eq!(a.len(), 100);
        assert_eq!(arena.used(), 100);

        let b = arena.alloc_bytes(50, 8).unwrap();
        assert_eq!(b.len(), 50);
        // Alignment: returned pointer must be 8-aligned.
        assert_eq!(b.as_ptr() as usize % 8, 0);

        assert_eq!(arena.remaining(), 256 - arena.used());
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 256);
    }

    #[test]
    fn oom_error() {
        let mut buf = [0u8; 16];
        let mut arena = BumpArena::new(&mut buf);
        assert!(arena.alloc_bytes(32, 1).is_err());
        assert_eq!(
            arena.alloc_bytes(32, 1).unwrap_err(),
            TensorError::OutOfMemory
        );
    }

    #[test]
    fn invalid_align() {
        let mut buf = [0u8; 64];
        let mut arena = BumpArena::new(&mut buf);
        assert_eq!(
            arena.alloc_bytes(4, 3).unwrap_err(),
            TensorError::Invalid
        );
    }

    #[test]
    fn typed_slice_alloc() {
        let mut buf = [0u8; 256];
        let mut arena = BumpArena::new(&mut buf);
        let s = arena.alloc_slice::<f32>(10).unwrap();
        assert_eq!(s.len(), 10);
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as f32;
        }
        assert_eq!(s[9], 9.0);
    }

    #[test]
    fn regions_do_not_overlap() {
        let mut buf = [0u8; 128];
        let mut arena = BumpArena::new(&mut buf);
        let a = arena.alloc_bytes(16, 1).unwrap();
        let a_start = a.as_ptr() as usize;
        a.fill(0xAB);
        let b = arena.alloc_bytes(16, 1).unwrap();
        let b_start = b.as_ptr() as usize;
        b.fill(0xCD);
        assert!(b_start >= a_start + 16 || a_start >= b_start + 16);
    }
}
