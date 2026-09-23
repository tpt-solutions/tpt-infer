//! x86_64 AVX-512 backend (feature `avx512`).
//!
//! Vectorized kernels: `matmul`, `elementwise_add`, `relu`. The remaining
//! operators delegate to the [`crate::naive`] reference kernels.
//! AVX-512F is detected at runtime with the `std` feature, otherwise via
//! `target_feature = "avx512f"` at compile time.
//!
//! This module is only compiled when both `feature = "avx512"` and
//! `target_arch = "x86_64"` hold; the target feature itself is still
//! runtime-checked before any kernel runs.
//!
//! AVX-512 intrinsics were stabilized in Rust 1.89, so this opt-in feature
//! effectively requires a newer compiler than the workspace MSRV of 1.75.
#![allow(clippy::incompatible_msrv)]

use crate::backend::{
    validate_binary, validate_matmul, validate_unary, Backend, Conv2dOptions, OpError,
};
use crate::naive;
use core::arch::x86_64::*;

#[cfg(feature = "std")]
fn has_avx512f() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(not(feature = "std"))]
fn has_avx512f() -> bool {
    cfg!(target_feature = "avx512f")
}

/// AVX-512-accelerated CPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Avx512Backend;

impl Avx512Backend {
    /// Creates the backend.
    pub const fn new() -> Self {
        Self
    }

    /// True when AVX-512F is usable on this machine.
    ///
    /// With `std`, detects the CPU feature at runtime; without `std`, uses
    /// compile-time `target_feature = "avx512f"`. Operations fall back to the
    /// naive reference when this returns `false`.
    pub fn is_available() -> bool {
        has_avx512f()
    }
}

/// # Safety
/// Caller must guarantee AVX-512F is usable and the slices have validated lengths.
#[target_feature(enable = "avx512f")]
unsafe fn matmul_avx512(a: &[f32], m: usize, k: usize, n: usize, b: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            let mut j = 0;
            while j + 16 <= n {
                let acc = _mm512_loadu_ps(o_row.as_ptr().add(j));
                let bv = _mm512_loadu_ps(b_row.as_ptr().add(j));
                let prod = _mm512_mul_ps(_mm512_set1_ps(av), bv);
                _mm512_storeu_ps(o_row.as_mut_ptr().add(j), _mm512_add_ps(acc, prod));
                j += 16;
            }
            for (slot, &bv) in o_row[j..].iter_mut().zip(b_row[j..].iter()) {
                *slot += av * bv;
            }
        }
    }
}

/// # Safety
/// Caller must guarantee AVX-512F is usable and all slices have equal length.
#[target_feature(enable = "avx512f")]
unsafe fn add_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut i = 0;
    while i + 16 <= a.len() {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_add_ps(av, bv));
        i += 16;
    }
    for ((&x, &y), slot) in a[i..].iter().zip(b[i..].iter()).zip(out[i..].iter_mut()) {
        *slot = x + y;
    }
}

/// # Safety
/// Caller must guarantee AVX-512F is usable and `a.len() == out.len()`.
#[target_feature(enable = "avx512f")]
unsafe fn relu_avx512(a: &[f32], out: &mut [f32]) {
    let zero = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        let v = _mm512_loadu_ps(a.as_ptr().add(i));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_max_ps(v, zero));
        i += 16;
    }
    for (slot, &x) in out[i..].iter_mut().zip(a[i..].iter()) {
        *slot = if x > 0.0 { x } else { 0.0 };
    }
}

impl Backend for Avx512Backend {
    fn name(&self) -> &'static str {
        "avx512"
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
        unsafe { matmul_avx512(a, m, k, n, b, out) };
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
        unsafe { add_avx512(a, b, out) };
        Ok(())
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::relu(a, out);
        }
        validate_unary(a, out)?;
        unsafe { relu_avx512(a, out) };
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
