//! Scalar (non-SIMD) reference implementation of every operator.
//!
//! This backend is always compiled and is the correctness baseline the
//! SIMD backends are tested against.

use crate::backend::{
    validate_binary, validate_conv2d, validate_matmul, validate_softmax, validate_unary, Backend,
    Conv2dOptions, OpError,
};
use crate::math::{exp_f32, tanh_f32};

/// Dense matrix multiply: `out[m,n] = a[m,k] @ b[k,n]` (row-major, no transpose).
pub fn matmul(
    a: &[f32],
    a_shape: [usize; 2],
    b: &[f32],
    b_shape: [usize; 2],
    out: &mut [f32],
) -> Result<(), OpError> {
    let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        for (j, slot) in o_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (kk, &av) in a_row.iter().enumerate() {
                acc += av * b[kk * n + j];
            }
            *slot = acc;
        }
    }
    Ok(())
}

/// 2D convolution, NCHW input, OIHW weights, groups = 1, no dilation or bias.
pub fn conv2d(
    input: &[f32],
    in_shape: [usize; 4],
    weight: &[f32],
    w_shape: [usize; 4],
    out: &mut [f32],
    options: Conv2dOptions,
) -> Result<(), OpError> {
    let [n, oc, oh, ow] = validate_conv2d(input, in_shape, weight, w_shape, out, options)?;
    let [_, c, h, w] = in_shape;
    let [_, _, kh, kw] = w_shape;
    let sh = options.stride_h;
    let sw = options.stride_w;
    let ph = options.pad_h;
    let pw = options.pad_w;

    for ni in 0..n {
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..c {
                        for ky in 0..kh {
                            let y = oy * sh + ky;
                            if y < ph || y - ph >= h {
                                continue;
                            }
                            let iy = y - ph;
                            for kx in 0..kw {
                                let x = ox * sw + kx;
                                if x < pw || x - pw >= w {
                                    continue;
                                }
                                let ix = x - pw;
                                let in_idx = ((ni * c + ci) * h + iy) * w + ix;
                                let w_idx = ((o * c + ci) * kh + ky) * kw + kx;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    let out_idx = ((ni * oc + o) * oh + oy) * ow + ox;
                    out[out_idx] = acc;
                }
            }
        }
    }
    Ok(())
}

/// Elementwise sum: `out = a + b`.
pub fn elementwise_add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError> {
    validate_binary(a, b, out)?;
    for ((&x, &y), slot) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *slot = x + y;
    }
    Ok(())
}

/// Rectified linear unit: `out = max(a, 0)`.
pub fn relu(a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
    validate_unary(a, out)?;
    for (&x, slot) in a.iter().zip(out.iter_mut()) {
        *slot = if x > 0.0 { x } else { 0.0 };
    }
    Ok(())
}

/// Softmax over rows of length `row_len`, with max subtraction for stability.
pub fn softmax(a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError> {
    validate_softmax(a, out, row_len)?;
    for (src, dst) in a.chunks_exact(row_len).zip(out.chunks_exact_mut(row_len)) {
        let mut max = f32::NEG_INFINITY;
        for &v in src {
            if v > max {
                max = v;
            }
        }
        for (slot, &v) in dst.iter_mut().zip(src.iter()) {
            *slot = exp_f32(v - max);
        }
        let mut sum = 0.0f32;
        for &e in dst.iter() {
            sum += e;
        }
        for slot in dst.iter_mut() {
            *slot /= sum;
        }
    }
    Ok(())
}

/// Elementwise logistic sigmoid: `out = 1 / (1 + exp(-a))`.
pub fn sigmoid(a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
    validate_unary(a, out)?;
    for (&x, slot) in a.iter().zip(out.iter_mut()) {
        *slot = 1.0 / (1.0 + exp_f32(-x));
    }
    Ok(())
}

/// Elementwise GELU (tanh approximation, see [`Backend::gelu`]).
pub fn gelu(a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
    validate_unary(a, out)?;
    const K: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;
    for (&x, slot) in a.iter().zip(out.iter_mut()) {
        let inner = K * (x + C * x * x * x);
        *slot = 0.5 * x * (1.0 + tanh_f32(inner));
    }
    Ok(())
}

/// Naive scalar [`Backend`] implementation (the correctness baseline).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NaiveBackend;

impl NaiveBackend {
    /// Creates the backend.
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for NaiveBackend {
    fn name(&self) -> &'static str {
        "naive"
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
        out: &mut [f32],
    ) -> Result<(), OpError> {
        matmul(a, a_shape, b, b_shape, out)
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
        conv2d(input, in_shape, weight, w_shape, out, options)
    }

    fn elementwise_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        elementwise_add(a, b, out)
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        relu(a, out)
    }

    fn softmax(&self, a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError> {
        softmax(a, out, row_len)
    }

    fn sigmoid(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        sigmoid(a, out)
    }

    fn gelu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        gelu(a, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_known_values() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        matmul(&a, [2, 2], &b, [2, 2], &mut out).unwrap();
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_shape_mismatch() {
        let a = [1.0f32; 6];
        let b = [1.0f32; 12];
        let mut out = [0.0f32; 6];
        let err = matmul(&a, [2, 3], &b, [4, 3], &mut out).unwrap_err();
        assert_eq!(err, OpError::ShapeMismatch);
    }

    #[test]
    fn matmul_size_mismatch() {
        let a = [1.0f32; 5];
        let b = [1.0f32; 9];
        let mut out = [0.0f32; 6];
        let err = matmul(&a, [2, 3], &b, [3, 3], &mut out).unwrap_err();
        assert_eq!(
            err,
            OpError::SizeMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn conv2d_known_values() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0f32, 0.0, 0.0, -1.0];
        let mut out = [0.0f32; 4];
        conv2d(
            &input,
            [1, 1, 3, 3],
            &weight,
            [1, 1, 2, 2],
            &mut out,
            Conv2dOptions::new(),
        )
        .unwrap();
        assert_eq!(out, [-4.0, -4.0, -4.0, -4.0]);
    }

    #[test]
    fn conv2d_stride2_pad0_output_shape() {
        let input = [1.0f32; 8 * 8];
        let weight = [1.0f32; 3 * 3];
        let mut out = vec![0.0f32; 3 * 3];
        conv2d(
            &input,
            [1, 1, 8, 8],
            &weight,
            [1, 1, 3, 3],
            &mut out,
            Conv2dOptions {
                stride_h: 2,
                stride_w: 2,
                pad_h: 0,
                pad_w: 0,
            },
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 9.0));
    }

    #[test]
    fn conv2d_channel_count_mismatch() {
        let input = [0.0f32; 8];
        let weight = [0.0f32; 6];
        let mut out = [0.0f32; 4];
        let err = conv2d(
            &input,
            [1, 2, 2, 2],
            &weight,
            [1, 3, 2, 2],
            &mut out,
            Conv2dOptions::new(),
        )
        .unwrap_err();
        assert_eq!(err, OpError::ShapeMismatch);
    }

    #[test]
    fn conv2d_zero_stride_rejected() {
        let input = [0.0f32; 16];
        let weight = [0.0f32; 9];
        let mut out = [0.0f32; 16];
        let err = conv2d(
            &input,
            [1, 1, 4, 4],
            &weight,
            [1, 1, 3, 3],
            &mut out,
            Conv2dOptions {
                stride_h: 0,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
        )
        .unwrap_err();
        assert_eq!(err, OpError::Invalid);
    }

    #[test]
    fn relu_clips_negatives() {
        let a = [-1.0f32, 0.0, 1.0, 0.5];
        let mut out = [0.0f32; 4];
        relu(&a, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 1.0, 0.5]);
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let a = [0.0f32];
        let mut out = [0.0f32; 1];
        sigmoid(&a, &mut out).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gelu_at_zero_is_zero() {
        let a = [0.0f32, -1.0, 1.0];
        let mut out = [1.0f32; 3];
        gelu(&a, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = [1.0f32, 2.0, 3.0, 1000.0, 1000.0, 1000.0];
        let mut out = [0.0f32; 6];
        softmax(&a, &mut out, 3).unwrap();
        let r0: f32 = out[..3].iter().sum();
        let r1: f32 = out[3..].iter().sum();
        assert!((r0 - 1.0).abs() < 1e-5);
        assert!((r1 - 1.0).abs() < 1e-5);
        assert!(out[3] > 0.3 && out[3] < 0.34);
    }

    #[test]
    fn softmax_invalid_row_len() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        assert_eq!(softmax(&a, &mut out, 0).unwrap_err(), OpError::Invalid);
        assert_eq!(softmax(&a, &mut out, 2).unwrap_err(), OpError::Invalid);
    }

    #[test]
    fn elementwise_add_length_check() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32];
        let mut out = [0.0f32; 2];
        assert_eq!(
            elementwise_add(&a, &b, &mut out).unwrap_err(),
            OpError::SizeMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn backend_delegates_and_names() {
        let backend = NaiveBackend::new();
        assert_eq!(backend.name(), "naive");
        let a = [1.0f32, -2.0];
        let mut out = [0.0f32; 2];
        backend.relu(&a, &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0]);
    }
}
