//! aarch64 NEON backend (`core::arch::aarch64`).
//!
//! Vectorized kernels: `matmul`, `elementwise_add`, `relu`. The remaining
//! operators delegate to the [`crate::naive`] reference kernels.
//!
//! NEON is a baseline feature of every mainstream aarch64 target Rust
//! supports; availability requires `target_feature = "neon"` (which those
//! targets enable by default) and, with `std`, a runtime check. SVE is not
//! available on stable Rust and is therefore not implemented.
//!
//! The kernels intentionally omit `#[target_feature(enable = "neon")]`:
//! enabling NEON explicitly on soft-float aarch64 targets is unsound (ABI
//! mismatch), and those targets never pass the availability check anyway.

use crate::backend::{
    validate_binary, validate_matmul, validate_unary, Backend, Conv2dOptions, OpError,
};
use crate::naive;
use core::arch::aarch64::*;

#[cfg(feature = "std")]
fn has_neon() -> bool {
    cfg!(target_feature = "neon") && std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(feature = "std"))]
fn has_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// NEON-accelerated CPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NeonBackend;

impl NeonBackend {
    /// Creates the backend.
    pub const fn new() -> Self {
        Self
    }

    /// True when NEON is usable on this machine.
    ///
    /// Operations fall back to the naive reference when this returns `false`.
    pub fn is_available() -> bool {
        has_neon()
    }
}

/// # Safety
/// Caller must guarantee NEON is usable and the slices have validated lengths.
unsafe fn matmul_neon(a: &[f32], m: usize, k: usize, n: usize, b: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            let avv = vdupq_n_f32(av);
            let mut j = 0;
            while j + 4 <= n {
                let acc = vld1q_f32(o_row.as_ptr().add(j));
                let bv = vld1q_f32(b_row.as_ptr().add(j));
                let sum = vaddq_f32(acc, vmulq_f32(avv, bv));
                vst1q_f32(o_row.as_mut_ptr().add(j), sum);
                j += 4;
            }
            for (slot, &bv) in o_row[j..].iter_mut().zip(b_row[j..].iter()) {
                *slot += av * bv;
            }
        }
    }
}

/// # Safety
/// Caller must guarantee NEON is usable and all slices have equal length.
unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut i = 0;
    while i + 4 <= a.len() {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(av, bv));
        i += 4;
    }
    for ((&x, &y), slot) in a[i..].iter().zip(b[i..].iter()).zip(out[i..].iter_mut()) {
        *slot = x + y;
    }
}

/// # Safety
/// Caller must guarantee NEON is usable and `a.len() == out.len()`.
unsafe fn relu_neon(a: &[f32], out: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= a.len() {
        let v = vld1q_f32(a.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vmaxq_f32(v, zero));
        i += 4;
    }
    for (slot, &x) in out[i..].iter_mut().zip(a[i..].iter()) {
        *slot = if x > 0.0 { x } else { 0.0 };
    }
}

impl Backend for NeonBackend {
    fn name(&self) -> &'static str {
        "neon"
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
        unsafe { matmul_neon(a, m, k, n, b, out) };
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
        unsafe { add_neon(a, b, out) };
        Ok(())
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::relu(a, out);
        }
        validate_unary(a, out)?;
        unsafe { relu_neon(a, out) };
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
