//! x86_64 AVX2 backend (`core::arch::x86_64`).
//!
//! Vectorized kernels: `matmul`, `elementwise_add`, `relu`. The remaining
//! operators delegate to the [`crate::naive`] reference kernels.
//! AVX2 is detected at runtime with the `std` feature, otherwise via
//! `target_feature = "avx2"` at compile time.

use crate::backend::{
    validate_binary, validate_matmul, validate_unary, Backend, Conv2dOptions, OpError,
};
use crate::naive;
use core::arch::x86_64::*;

#[cfg(feature = "std")]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(feature = "std"))]
fn has_avx2() -> bool {
    cfg!(target_feature = "avx2")
}

/// AVX2-accelerated CPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Avx2Backend;

impl Avx2Backend {
    /// Creates the backend.
    pub const fn new() -> Self {
        Self
    }

    /// True when AVX2 is usable on this machine.
    ///
    /// With `std`, detects the CPU feature at runtime; without `std`, uses
    /// compile-time `target_feature = "avx2"`. Operations fall back to the
    /// naive reference when this returns `false`.
    pub fn is_available() -> bool {
        has_avx2()
    }
}

/// # Safety
/// Caller must guarantee AVX2 is usable and the slices have validated lengths.
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2(a: &[f32], m: usize, k: usize, n: usize, b: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            let mut j = 0;
            while j + 8 <= n {
                let acc = _mm256_loadu_ps(o_row.as_ptr().add(j));
                let bv = _mm256_loadu_ps(b_row.as_ptr().add(j));
                let prod = _mm256_mul_ps(_mm256_set1_ps(av), bv);
                _mm256_storeu_ps(o_row.as_mut_ptr().add(j), _mm256_add_ps(acc, prod));
                j += 8;
            }
            for (slot, &bv) in o_row[j..].iter_mut().zip(b_row[j..].iter()) {
                *slot += av * bv;
            }
        }
    }
}

/// # Safety
/// Caller must guarantee AVX2 is usable and all slices have equal length.
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut i = 0;
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(av, bv));
        i += 8;
    }
    for ((&x, &y), slot) in a[i..].iter().zip(b[i..].iter()).zip(out[i..].iter_mut()) {
        *slot = x + y;
    }
}

/// # Safety
/// Caller must guarantee AVX2 is usable and `a.len() == out.len()`.
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(a: &[f32], out: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        let v = _mm256_loadu_ps(a.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    for (slot, &x) in out[i..].iter_mut().zip(a[i..].iter()) {
        *slot = if x > 0.0 { x } else { 0.0 };
    }
}

impl Backend for Avx2Backend {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
        out: &mut [f32],
    ) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::matmul(a, a_shape, b, b_shape, out);
        }
        let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;
        unsafe { matmul_avx2(a, m, k, n, b, out) };
        Ok(())
    }

    fn conv2d(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        weight: &[f32],
        w_shape: [usize; 4],
        out: &mut [f32],
        options: Conv2dOptions,
    ) -> Result<(), OpError> {
        naive::conv2d(input, in_shape, weight, w_shape, out, options)
    }

    fn elementwise_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::elementwise_add(a, b, out);
        }
        validate_binary(a, b, out)?;
        unsafe { add_avx2(a, b, out) };
        Ok(())
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::relu(a, out);
        }
        validate_unary(a, out)?;
        unsafe { relu_avx2(a, out) };
        Ok(())
    }

    fn softmax(&self, a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError> {
        naive::softmax(a, out, row_len)
    }

    fn sigmoid(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        naive::sigmoid(a, out)
    }

    fn gelu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        naive::gelu(a, out)
    }
}
