//! Runtime backend selection via enum dispatch (no trait objects).
//!
//! [`select_backend`] probes the architecture and CPU features and returns
//! the best [`AnyBackend`] available: AVX-512 (when the `avx512` feature is
//! enabled and supported) over AVX2 on x86_64, NEON on aarch64, SIMD128 on
//! wasm32, and the naive scalar reference otherwise.
//!
//! `crate::webgpu::WebGpuBackend` (feature `webgpu`) is deliberately not a
//! variant of [`AnyBackend`]: unlike every other backend here, it wraps a
//! real `wgpu::Device`/`Queue` pair, so it can't be a zero-sized, `Copy`,
//! unconditionally-constructible value, and acquiring one is async and
//! fallible (no GPU adapter on a headless machine). Construct it directly
//! via `WebGpuBackend::new()` (returns `Option`) when GPU dispatch is
//! wanted.

use crate::backend::{Backend, Conv2dOptions, OpError};
use crate::naive::NaiveBackend;

#[cfg(target_arch = "x86_64")]
use crate::avx2::Avx2Backend;
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
use crate::avx512::Avx512Backend;
#[cfg(target_arch = "aarch64")]
use crate::neon::NeonBackend;
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmSimdBackend;

/// A concrete backend chosen for this machine (enum dispatch, `no_std`-friendly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnyBackend {
    /// Portable scalar reference (always available).
    Naive,
    /// x86_64 AVX2 (selected when the CPU supports it).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// x86_64 AVX-512 (requires feature `avx512` and CPU support; never
    /// selected automatically without both).
    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    Avx512,
    /// aarch64 NEON.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// wasm32 SIMD128 (requires `target_feature = "simd128"`).
    #[cfg(target_arch = "wasm32")]
    WasmSimd,
}

impl Backend for AnyBackend {
    fn name(&self) -> &'static str {
        match self {
            AnyBackend::Naive => NaiveBackend.name(),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.name(),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.name(),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.name(),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.name(),
        }
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
        out: &mut [f32],
    ) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.matmul(a, a_shape, b, b_shape, out),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.matmul(a, a_shape, b, b_shape, out),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.matmul(a, a_shape, b, b_shape, out),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.matmul(a, a_shape, b, b_shape, out),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.matmul(a, a_shape, b, b_shape, out),
        }
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
        match self {
            AnyBackend::Naive => {
                NaiveBackend.conv2d(input, in_shape, weight, w_shape, out, options)
            }
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.conv2d(input, in_shape, weight, w_shape, out, options),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => {
                Avx512Backend.conv2d(input, in_shape, weight, w_shape, out, options)
            }
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.conv2d(input, in_shape, weight, w_shape, out, options),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => {
                WasmSimdBackend.conv2d(input, in_shape, weight, w_shape, out, options)
            }
        }
    }

    fn elementwise_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.elementwise_add(a, b, out),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.elementwise_add(a, b, out),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.elementwise_add(a, b, out),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.elementwise_add(a, b, out),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.elementwise_add(a, b, out),
        }
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.relu(a, out),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.relu(a, out),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.relu(a, out),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.relu(a, out),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.relu(a, out),
        }
    }

    fn softmax(&self, a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.softmax(a, out, row_len),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.softmax(a, out, row_len),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.softmax(a, out, row_len),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.softmax(a, out, row_len),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.softmax(a, out, row_len),
        }
    }

    fn sigmoid(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.sigmoid(a, out),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.sigmoid(a, out),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.sigmoid(a, out),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.sigmoid(a, out),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.sigmoid(a, out),
        }
    }

    fn gelu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        match self {
            AnyBackend::Naive => NaiveBackend.gelu(a, out),
            #[cfg(target_arch = "x86_64")]
            AnyBackend::Avx2 => Avx2Backend.gelu(a, out),
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            AnyBackend::Avx512 => Avx512Backend.gelu(a, out),
            #[cfg(target_arch = "aarch64")]
            AnyBackend::Neon => NeonBackend.gelu(a, out),
            #[cfg(target_arch = "wasm32")]
            AnyBackend::WasmSimd => WasmSimdBackend.gelu(a, out),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<AnyBackend> {
    #[cfg(feature = "avx512")]
    {
        if Avx512Backend::is_available() {
            return Some(AnyBackend::Avx512);
        }
    }
    if Avx2Backend::is_available() {
        return Some(AnyBackend::Avx2);
    }
    None
}

#[cfg(target_arch = "aarch64")]
fn detect() -> Option<AnyBackend> {
    if NeonBackend::is_available() {
        return Some(AnyBackend::Neon);
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn detect() -> Option<AnyBackend> {
    if WasmSimdBackend::is_available() {
        return Some(AnyBackend::WasmSimd);
    }
    None
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn detect() -> Option<AnyBackend> {
    None
}

/// Selects the best backend for this machine (AVX-512/AVX2, NEON, or
/// wasm SIMD128 when available, otherwise [`AnyBackend::Naive`]).
///
/// With `std`, x86_64 features are probed at runtime; without `std`,
/// compile-time `target_feature` settings decide.
pub fn select_backend() -> AnyBackend {
    detect().unwrap_or(AnyBackend::Naive)
}

/// Name of the backend [`select_backend`] would pick (e.g. `"avx2"`).
pub fn selected_backend() -> &'static str {
    select_backend().name()
}

/// Every backend that can compute on this machine, ordered worst-to-best
/// (always includes [`AnyBackend::Naive`]; excludes the WebGPU backend,
/// which is opt-in via the `webgpu` feature and requires an async device
/// init, so it isn't auto-selected here).
#[cfg(any(feature = "std", test))]
pub fn available_backends() -> Vec<AnyBackend> {
    let mut out = vec![AnyBackend::Naive];
    #[cfg(target_arch = "x86_64")]
    {
        if Avx2Backend::is_available() {
            out.push(AnyBackend::Avx2);
        }
        #[cfg(feature = "avx512")]
        {
            if Avx512Backend::is_available() {
                out.push(AnyBackend::Avx512);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if NeonBackend::is_available() {
            out.push(AnyBackend::Neon);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if WasmSimdBackend::is_available() {
            out.push(AnyBackend::WasmSimd);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(backend: &AnyBackend, op: &str, got: &[f32], want: &[f32]) {
        let name = backend.name();
        assert_eq!(got.len(), want.len(), "{name} {op}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-4 * (1.0 + w.abs());
            assert!(g.is_finite(), "{name} {op}[{i}]: {g} is not finite");
            assert!(
                (g - w).abs() <= tol,
                "{name} {op}[{i}]: got {g}, want {w} (tol {tol})"
            );
        }
    }

    fn reference_matmul(
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; a_shape[0] * b_shape[1]];
        crate::naive::matmul(a, a_shape, b, b_shape, &mut out).unwrap();
        out
    }

    #[test]
    fn selected_backend_has_name() {
        let name = selected_backend();
        assert!(!name.is_empty());
        assert_eq!(select_backend().name(), name);
    }

    #[test]
    fn matmul_small_matches_naive() {
        let a = lcg(3 * 4, 1);
        let b = lcg(4 * 5, 2);
        let want = reference_matmul(&a, [3, 4], &b, [4, 5]);
        for backend in available_backends() {
            let mut got = vec![0.0f32; 15];
            backend.matmul(&a, [3, 4], &b, [4, 5], &mut got).unwrap();
            assert_close(&backend, "matmul small", &got, &want);
        }
    }

    #[test]
    fn matmul_mlp_shape_matches_naive() {
        let a = lcg(784, 3);
        let b = lcg(784 * 10, 4);
        let want = reference_matmul(&a, [1, 784], &b, [784, 10]);
        for backend in available_backends() {
            let mut got = vec![0.0f32; 10];
            backend
                .matmul(&a, [1, 784], &b, [784, 10], &mut got)
                .unwrap();
            assert_close(&backend, "matmul mlp", &got, &want);
        }
    }

    #[test]
    fn matmul_odd_dims_match_naive() {
        let a = lcg(13 * 17, 5);
        let b = lcg(17 * 19, 6);
        let want = reference_matmul(&a, [13, 17], &b, [17, 19]);
        for backend in available_backends() {
            let mut got = vec![0.0f32; 13 * 19];
            backend
                .matmul(&a, [13, 17], &b, [17, 19], &mut got)
                .unwrap();
            assert_close(&backend, "matmul odd", &got, &want);
        }
    }

    #[test]
    fn conv2d_pad1_matches_naive() {
        let input = lcg(3 * 8 * 8, 7);
        let weight = lcg(4 * 3 * 3 * 3, 8);
        let options = Conv2dOptions {
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        };
        let mut want = vec![0.0f32; 4 * 8 * 8];
        crate::naive::conv2d(
            &input,
            [1, 3, 8, 8],
            &weight,
            [4, 3, 3, 3],
            &mut want,
            options,
        )
        .unwrap();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 4 * 8 * 8];
            backend
                .conv2d(
                    &input,
                    [1, 3, 8, 8],
                    &weight,
                    [4, 3, 3, 3],
                    &mut got,
                    options,
                )
                .unwrap();
            assert_close(&backend, "conv2d pad1", &got, &want);
        }
    }

    #[test]
    fn conv2d_stride2_matches_naive() {
        let input = lcg(2 * 3 * 10 * 10, 9);
        let weight = lcg(5 * 3 * 3 * 3, 10);
        let options = Conv2dOptions {
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        let mut want = vec![0.0f32; 2 * 5 * 4 * 4];
        crate::naive::conv2d(
            &input,
            [2, 3, 10, 10],
            &weight,
            [5, 3, 3, 3],
            &mut want,
            options,
        )
        .unwrap();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 2 * 5 * 4 * 4];
            backend
                .conv2d(
                    &input,
                    [2, 3, 10, 10],
                    &weight,
                    [5, 3, 3, 3],
                    &mut got,
                    options,
                )
                .unwrap();
            assert_close(&backend, "conv2d stride2", &got, &want);
        }
    }

    #[test]
    fn elementwise_add_matches_naive() {
        let a = lcg(64, 11);
        let b = lcg(64, 12);
        let want: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 64];
            backend.elementwise_add(&a, &b, &mut got).unwrap();
            assert_close(&backend, "add", &got, &want);
        }
    }

    #[test]
    fn relu_matches_naive() {
        let a = lcg(100, 13);
        let want: Vec<f32> = a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 100];
            backend.relu(&a, &mut got).unwrap();
            assert_close(&backend, "relu", &got, &want);
        }
    }

    #[test]
    fn softmax_matches_naive_and_rows_sum_to_one() {
        let a = lcg(2 * 7, 14);
        let mut want = vec![0.0f32; 14];
        crate::naive::softmax(&a, &mut want, 7).unwrap();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 14];
            backend.softmax(&a, &mut got, 7).unwrap();
            assert_close(&backend, "softmax", &got, &want);
            for row in got.chunks_exact(7) {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "{}: row sum {sum}",
                    backend.name()
                );
            }
        }
    }

    #[test]
    fn sigmoid_matches_naive() {
        let a = lcg(50, 15);
        let mut want = vec![0.0f32; 50];
        crate::naive::sigmoid(&a, &mut want).unwrap();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 50];
            backend.sigmoid(&a, &mut got).unwrap();
            assert_close(&backend, "sigmoid", &got, &want);
        }
    }

    #[test]
    fn gelu_matches_scalar_reference() {
        let a = lcg(50, 16);
        let want: Vec<f32> = a
            .iter()
            .map(|&x| {
                let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        for backend in available_backends() {
            let mut got = vec![0.0f32; 50];
            backend.gelu(&a, &mut got).unwrap();
            assert_close(&backend, "gelu", &got, &want);
        }
    }

    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    #[test]
    fn avx512_backend_matches_naive_when_available() {
        let backend = AnyBackend::Avx512;
        let a = lcg(32 * 64, 17);
        let b = lcg(64 * 48, 18);
        let want = reference_matmul(&a, [32, 64], &b, [64, 48]);
        let mut got = vec![0.0f32; 32 * 48];
        backend
            .matmul(&a, [32, 64], &b, [64, 48], &mut got)
            .unwrap();
        assert_close(&backend, "avx512 matmul", &got, &want);
    }
}
