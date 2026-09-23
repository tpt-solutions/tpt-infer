//! wasm32 SIMD128 backend (`core::arch::wasm32`).
//!
//! Vectorized kernels: `matmul`, `elementwise_add`, `relu`. The remaining
//! operators delegate to the [`crate::naive`] reference kernels.
//!
//! WebAssembly has no stable runtime feature detection, so SIMD128 support
//! is a compile-time gate: kernels are only used when the crate is built
//! with `target_feature = "simd128"` (e.g. `RUSTFLAGS="-C target-feature=+simd128"`).
//! Otherwise every operation falls back to the naive reference.

use crate::backend::{
    validate_binary, validate_matmul, validate_unary, Backend, Conv2dOptions, OpError,
};
use crate::naive;
use core::arch::wasm32::*;
use core::ptr::{read_unaligned, write_unaligned};

fn has_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

/// wasm32 SIMD128 backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WasmSimdBackend;

impl WasmSimdBackend {
    /// Creates the backend.
    pub const fn new() -> Self {
        Self
    }

    /// True when the crate was compiled with `target_feature = "simd128"`.
    ///
    /// Operations fall back to the naive reference when this returns `false`.
    pub fn is_available() -> bool {
        has_simd128()
    }
}

/// # Safety
/// Caller must guarantee SIMD128 is enabled and the slices have validated lengths.
#[target_feature(enable = "simd128")]
unsafe fn matmul_wasm(a: &[f32], m: usize, k: usize, n: usize, b: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            let mut j = 0;
            while j + 4 <= n {
                let acc: v128 = read_unaligned(o_row.as_ptr().add(j) as *const v128);
                let bv: v128 = read_unaligned(b_row.as_ptr().add(j) as *const v128);
                let prod = f32x4_mul(f32x4_splat(av), bv);
                write_unaligned(o_row.as_mut_ptr().add(j) as *mut v128, f32x4_add(acc, prod));
                j += 4;
            }
            for (slot, &bv) in o_row[j..].iter_mut().zip(b_row[j..].iter()) {
                *slot += av * bv;
            }
        }
    }
}

/// # Safety
/// Caller must guarantee SIMD128 is enabled and all slices have equal length.
#[target_feature(enable = "simd128")]
unsafe fn add_wasm(a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut i = 0;
    while i + 4 <= a.len() {
        let av: v128 = read_unaligned(a.as_ptr().add(i) as *const v128);
        let bv: v128 = read_unaligned(b.as_ptr().add(i) as *const v128);
        write_unaligned(out.as_mut_ptr().add(i) as *mut v128, f32x4_add(av, bv));
        i += 4;
    }
    for ((&x, &y), slot) in a[i..].iter().zip(b[i..].iter()).zip(out[i..].iter_mut()) {
        *slot = x + y;
    }
}

/// # Safety
/// Caller must guarantee SIMD128 is enabled and `a.len() == out.len()`.
#[target_feature(enable = "simd128")]
unsafe fn relu_wasm(a: &[f32], out: &mut [f32]) {
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= a.len() {
        let v: v128 = read_unaligned(a.as_ptr().add(i) as *const v128);
        write_unaligned(out.as_mut_ptr().add(i) as *mut v128, f32x4_max(v, zero));
        i += 4;
    }
    for (slot, &x) in out[i..].iter_mut().zip(a[i..].iter()) {
        *slot = if x > 0.0 { x } else { 0.0 };
    }
}

impl Backend for WasmSimdBackend {
    fn name(&self) -> &'static str {
        "wasm-simd128"
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
        unsafe { matmul_wasm(a, m, k, n, b, out) };
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
        unsafe { add_wasm(a, b, out) };
        Ok(())
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        if !Self::is_available() {
            return naive::relu(a, out);
        }
        validate_unary(a, out)?;
        unsafe { relu_wasm(a, out) };
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
