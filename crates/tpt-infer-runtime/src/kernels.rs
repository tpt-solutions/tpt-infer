//! Zero-allocation reference kernels for operators outside the `Backend` HAL.
//!
//! The [`Backend`] trait covers `matmul`, `conv2d`, `elementwise_add`,
//! `relu`, `softmax`, `sigmoid`, and `gelu`. The kernels here cover the
//! remaining operators the graph IR can express — broadcasting binary ops,
//! pooling, batch normalization, transpose, concat, and plain data
//! movement — with the same contract: **every function writes into a
//! caller-provided output slice and never allocates**, so they can run from
//! a bare-metal target with all buffers in a [`BumpArena`].
//!
//! All shapes are row-major and use numpy right-aligned broadcasting.
//!
//! [`Backend`]: tpt_infer_ops::Backend
//! [`BumpArena`]: tpt_infer_core::BumpArena

use tpt_infer_core::{ravel_index, unravel_index, MAX_RANK};

use crate::error::RuntimeError;
use tpt_infer_ops::OpError;

/// Binary element-wise operation performed by [`binary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `out = a + b`.
    Add,
    /// `out = a - b`.
    Sub,
    /// `out = a * b`.
    Mul,
    /// `out = a / b` (IEEE division by zero yields infinities).
    Div,
}

/// Product of `dims` (`1` for a rank-0 shape).
///
/// # Errors
/// [`RuntimeError::ShapeMismatch`] if the product overflows `usize`.
pub(crate) fn numel(dims: &[usize]) -> Result<usize, RuntimeError> {
    let mut acc = 1usize;
    for &d in dims {
        acc = acc.checked_mul(d).ok_or(RuntimeError::ShapeMismatch)?;
    }
    Ok(acc)
}

/// Zero-pads (with ones, preserving the element count) a shape to
/// [`MAX_RANK`] so the const-generic index helpers can be used uniformly.
fn pad(dims: &[usize]) -> Result<[usize; MAX_RANK], RuntimeError> {
    if dims.len() > MAX_RANK {
        return Err(RuntimeError::ShapeMismatch);
    }
    let mut out = [1usize; MAX_RANK];
    out[MAX_RANK - dims.len()..].copy_from_slice(dims);
    Ok(out)
}

/// Validates that `operand` broadcasts (numpy semantics) to `out`.
fn check_broadcast(operand: &[usize], out: &[usize]) -> Result<(), RuntimeError> {
    if operand.len() > out.len() {
        return Err(RuntimeError::ShapeMismatch);
    }
    let off = out.len() - operand.len();
    for (i, &d) in operand.iter().enumerate() {
        if d != 1 && d != out[off + i] {
            return Err(RuntimeError::ShapeMismatch);
        }
    }
    Ok(())
}

/// Maps an output coordinate onto an operand coordinate (right-aligned,
/// `1`-dims in the *operand's* padded shape broadcast to `0`).
fn map_coord(oc: &[usize; MAX_RANK], sd: &[usize; MAX_RANK]) -> [usize; MAX_RANK] {
    let mut out = [0usize; MAX_RANK];
    for d in 0..MAX_RANK {
        out[d] = if sd[d] == 1 { 0 } else { oc[d] };
    }
    out
}

/// Element-wise binary operation with numpy broadcasting:
/// `out = a op b`.
///
/// `out_dims` is the broadcast result shape; `out.len()` must equal its
/// element count.
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if an operand does not broadcast to
///   `out_dims`
/// - [`RuntimeError::SizeMismatch`] if a buffer length disagrees with its
///   shape
pub fn binary(
    op: BinaryOp,
    a: &[f32],
    a_dims: &[usize],
    b: &[f32],
    b_dims: &[usize],
    out: &mut [f32],
    out_dims: &[usize],
) -> Result<(), RuntimeError> {
    check_broadcast(a_dims, out_dims)?;
    check_broadcast(b_dims, out_dims)?;
    if a.len() != numel(a_dims)? {
        return Err(RuntimeError::SizeMismatch {
            expected: numel(a_dims)?,
            actual: a.len(),
        });
    }
    if b.len() != numel(b_dims)? {
        return Err(RuntimeError::SizeMismatch {
            expected: numel(b_dims)?,
            actual: b.len(),
        });
    }
    let n = numel(out_dims)?;
    if out.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: out.len(),
        });
    }
    let ad = pad(a_dims)?;
    let bd = pad(b_dims)?;
    let od = pad(out_dims)?;
    for (i, out_val) in out.iter_mut().enumerate().take(n) {
        let oc = unravel_index(i, od);
        let ai = ravel_index(map_coord(&oc, &ad), ad);
        let bi = ravel_index(map_coord(&oc, &bd), bd);
        *out_val = match op {
            BinaryOp::Add => a[ai] + b[bi],
            BinaryOp::Sub => a[ai] - b[bi],
            BinaryOp::Mul => a[ai] * b[bi],
            BinaryOp::Div => a[ai] / b[bi],
        };
    }
    Ok(())
}

/// In-place broadcasting add: `out += bias` where `bias` broadcasts to
/// `out_dims`.
///
/// Reading and writing `out[i]` happens at the same flat index, so the
/// element-wise update is safe in place; `bias` must not alias `out`.
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if `bias` does not broadcast to
///   `out_dims`
/// - [`RuntimeError::SizeMismatch`] if a buffer length disagrees with its
///   shape
pub fn add_inplace_broadcast(
    out: &mut [f32],
    out_dims: &[usize],
    bias: &[f32],
    bias_dims: &[usize],
) -> Result<(), RuntimeError> {
    check_broadcast(bias_dims, out_dims)?;
    let n = numel(out_dims)?;
    if out.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: out.len(),
        });
    }
    if bias.len() != numel(bias_dims)? {
        return Err(RuntimeError::SizeMismatch {
            expected: numel(bias_dims)?,
            actual: bias.len(),
        });
    }
    let bd = pad(bias_dims)?;
    let od = pad(out_dims)?;
    for (i, out_val) in out.iter_mut().enumerate().take(n) {
        let oc = unravel_index(i, od);
        let bi = ravel_index(map_coord(&oc, &bd), bd);
        *out_val += bias[bi];
    }
    Ok(())
}

/// Copies `src` into `dst` (used by `Reshape` / `Flatten`, which only move
/// data — geometry comes from the graph's node metadata).
///
/// # Errors
/// [`RuntimeError::SizeMismatch`] if the lengths differ.
pub fn copy_elements(src: &[f32], dst: &mut [f32]) -> Result<(), RuntimeError> {
    if src.len() != dst.len() {
        return Err(RuntimeError::SizeMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }
    dst.copy_from_slice(src);
    Ok(())
}

/// Validated geometry shared by the 2-D pooling kernels.
struct PoolGeom {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_len: usize,
    out_len: usize,
}

/// Validates ranks, window/stride/padding, and the declared output shape.
fn pool_geometry(
    in_dims: &[usize],
    out_dims: &[usize],
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Result<PoolGeom, RuntimeError> {
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(RuntimeError::ShapeMismatch);
    }
    let (n, c, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let (kh, kw) = (kernel[0], kernel[1]);
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(RuntimeError::Op(OpError::Invalid));
    }
    let two_pad = |p: usize| p.checked_mul(2).ok_or(RuntimeError::Op(OpError::Invalid));
    let padded_h = h
        .checked_add(two_pad(ph)?)
        .ok_or(RuntimeError::Op(OpError::Invalid))?;
    let padded_w = w
        .checked_add(two_pad(pw)?)
        .ok_or(RuntimeError::Op(OpError::Invalid))?;
    if padded_h < kh || padded_w < kw {
        return Err(RuntimeError::Op(OpError::Invalid));
    }
    let oh = (padded_h - kh) / sh + 1;
    let ow = (padded_w - kw) / sw + 1;
    if out_dims[0] != n || out_dims[1] != c || out_dims[2] != oh || out_dims[3] != ow {
        return Err(RuntimeError::ShapeMismatch);
    }
    let in_len = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .ok_or(RuntimeError::ShapeMismatch)?;
    let out_len = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(oh))
        .and_then(|v| v.checked_mul(ow))
        .ok_or(RuntimeError::ShapeMismatch)?;
    Ok(PoolGeom {
        n,
        c,
        h,
        w,
        oh,
        ow,
        in_len,
        out_len,
    })
}

/// 2-D max pooling over NCHW input (ONNX semantics: floor division,
/// padding treated as `-inf`).
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if ranks or output geometry disagree
/// - [`RuntimeError::Op`]`(`[`OpError::Invalid`]`)` for zero kernels or
///   strides and for an oversized kernel
/// - [`RuntimeError::SizeMismatch`] if buffer lengths disagree
pub fn max_pool2d(
    input: &[f32],
    in_dims: &[usize],
    out: &mut [f32],
    out_dims: &[usize],
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Result<(), RuntimeError> {
    let g = pool_geometry(in_dims, out_dims, kernel, strides, padding)?;
    if input.len() != g.in_len {
        return Err(RuntimeError::SizeMismatch {
            expected: g.in_len,
            actual: input.len(),
        });
    }
    if out.len() != g.out_len {
        return Err(RuntimeError::SizeMismatch {
            expected: g.out_len,
            actual: out.len(),
        });
    }
    let (kh, kw) = (kernel[0], kernel[1]);
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    for ni in 0..g.n {
        for ci in 0..g.c {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let mut best = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        let y = oy * sh + ky;
                        if y < ph || y - ph >= g.h {
                            continue;
                        }
                        let iy = y - ph;
                        for kx in 0..kw {
                            let x = ox * sw + kx;
                            if x < pw || x - pw >= g.w {
                                continue;
                            }
                            let ix = x - pw;
                            let idx = ((ni * g.c + ci) * g.h + iy) * g.w + ix;
                            if input[idx] > best {
                                best = input[idx];
                            }
                        }
                    }
                    out[((ni * g.c + ci) * g.oh + oy) * g.ow + ox] = best;
                }
            }
        }
    }
    Ok(())
}

/// 2-D average pooling over NCHW input (ONNX defaults: floor division,
/// padding excluded from the average — `count_include_pad = 0`).
///
/// Global average pooling is the special case `kernel == spatial` with
/// stride 1 and no padding.
///
/// # Errors
/// Same set as [`max_pool2d`].
pub fn average_pool2d(
    input: &[f32],
    in_dims: &[usize],
    out: &mut [f32],
    out_dims: &[usize],
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Result<(), RuntimeError> {
    let g = pool_geometry(in_dims, out_dims, kernel, strides, padding)?;
    if input.len() != g.in_len {
        return Err(RuntimeError::SizeMismatch {
            expected: g.in_len,
            actual: input.len(),
        });
    }
    if out.len() != g.out_len {
        return Err(RuntimeError::SizeMismatch {
            expected: g.out_len,
            actual: out.len(),
        });
    }
    let (kh, kw) = (kernel[0], kernel[1]);
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    for ni in 0..g.n {
        for ci in 0..g.c {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let mut acc = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        let y = oy * sh + ky;
                        if y < ph || y - ph >= g.h {
                            continue;
                        }
                        let iy = y - ph;
                        for kx in 0..kw {
                            let x = ox * sw + kx;
                            if x < pw || x - pw >= g.w {
                                continue;
                            }
                            let ix = x - pw;
                            let idx = ((ni * g.c + ci) * g.h + iy) * g.w + ix;
                            acc += input[idx];
                            count += 1;
                        }
                    }
                    out[((ni * g.c + ci) * g.oh + oy) * g.ow + ox] =
                        if count == 0 { 0.0 } else { acc / count as f32 };
                }
            }
        }
    }
    Ok(())
}

/// `no_std` square root: bit-trick initial guess plus Newton–Raphson
/// refinement (`f32::sqrt` lives in `std`, not `core`).
///
/// Returns `NaN` for negative inputs, `0.0` for `0.0`, and preserves
/// `+inf` / `NaN`.
pub(crate) fn sqrt_f32(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    // Halve the exponent bits for an approximate square root, then refine.
    let mut y = f32::from_bits((x.to_bits() >> 1) + (1 << 29));
    for _ in 0..4 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// Inference-mode batch normalization.
///
/// `params` is `[scale, bias, mean, var]`. Two layouts are accepted:
///
/// - **channel mode** — input is rank 4 (NCHW) and each parameter has one
///   element per channel (`x_dims[1]`);
/// - **element-wise mode** — each parameter has `numel(x)` elements.
///
/// Computes `y = (x - mean) / sqrt(var + eps) * scale + bias`.
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if ranks/parameter lengths disagree
/// - [`RuntimeError::SizeMismatch`] if buffer lengths disagree
pub fn batch_norm(
    x: &[f32],
    x_dims: &[usize],
    out: &mut [f32],
    params: [&[f32]; 4],
    eps: f32,
) -> Result<(), RuntimeError> {
    let [scale, bias, mean, var] = params;
    let n = numel(x_dims)?;
    if x.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if out.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: out.len(),
        });
    }
    let plen = scale.len();
    if bias.len() != plen || mean.len() != plen || var.len() != plen {
        return Err(RuntimeError::ShapeMismatch);
    }
    let channel_mode = x_dims.len() == 4 && plen == x_dims[1];
    let element_mode = plen == n;
    if !channel_mode && !element_mode {
        return Err(RuntimeError::ShapeMismatch);
    }
    if channel_mode {
        let (hw, c) = (x_dims[2] * x_dims[3], x_dims[1]);
        for ni in 0..x_dims[0] {
            for ci in 0..c {
                let inv = 1.0 / sqrt_f32(var[ci] + eps);
                let f = scale[ci] * inv;
                let g = bias[ci] - mean[ci] * f;
                let base = (ni * c + ci) * hw;
                for i in base..base + hw {
                    out[i] = x[i] * f + g;
                }
            }
        }
    } else {
        for i in 0..n {
            let inv = 1.0 / sqrt_f32(var[i] + eps);
            let f = scale[i] * inv;
            out[i] = (x[i] - mean[i]) * f + bias[i];
        }
    }
    Ok(())
}

/// Dimension permutation: `out[co] = x[perm^{-1}(co)]` with
/// `out_dims[i] = x_dims[perm[i]]`.
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if `perm` is not a permutation of
///   `0..rank`, if `out_dims` disagrees with `perm`, or on length mismatches
pub fn transpose(
    x: &[f32],
    x_dims: &[usize],
    out: &mut [f32],
    out_dims: &[usize],
    perm: &[usize],
) -> Result<(), RuntimeError> {
    let rank = x_dims.len();
    if out_dims.len() != rank || perm.len() != rank || rank > MAX_RANK {
        return Err(RuntimeError::ShapeMismatch);
    }
    let mut seen = [false; MAX_RANK];
    for &p in perm {
        if p >= rank || seen[p] {
            return Err(RuntimeError::ShapeMismatch);
        }
        seen[p] = true;
    }
    for i in 0..rank {
        if out_dims[i] != x_dims[perm[i]] {
            return Err(RuntimeError::ShapeMismatch);
        }
    }
    let n = numel(x_dims)?;
    if x.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if out.len() != n {
        return Err(RuntimeError::SizeMismatch {
            expected: n,
            actual: out.len(),
        });
    }
    let xd = pad(x_dims)?;
    let od = pad(out_dims)?;
    let offset = MAX_RANK - rank;
    for (i, out_val) in out.iter_mut().enumerate().take(n) {
        let oc = unravel_index(i, od);
        let mut xc = [0usize; MAX_RANK];
        for d in 0..rank {
            xc[offset + perm[d]] = oc[offset + d];
        }
        *out_val = x[ravel_index(xc, xd)];
    }
    Ok(())
}

/// Copies one concat operand into its strided region of `out`.
///
/// The operand is laid out as `outer` contiguous blocks of
/// `c_src * inner` elements; the output blocks are `c_total * inner`
/// elements wide and this operand occupies channels
/// `[c_offset, c_offset + c_src)` of each. Only one operand slice is needed
/// at a time, so concatenation can stream through the output without
/// buffering every input simultaneously.
///
/// # Errors
/// - [`RuntimeError::ShapeMismatch`] if the channel/window parameters are
///   inconsistent
/// - [`RuntimeError::SizeMismatch`] if `src` or `out` lengths disagree with
///   the implied geometry
pub fn concat_copy_one(
    out: &mut [f32],
    c_total: usize,
    c_offset: usize,
    c_src: usize,
    inner: usize,
    src: &[f32],
) -> Result<(), RuntimeError> {
    if c_src == 0 || inner == 0 || c_offset.saturating_add(c_src) > c_total {
        return Err(RuntimeError::ShapeMismatch);
    }
    let src_block = c_src
        .checked_mul(inner)
        .ok_or(RuntimeError::ShapeMismatch)?;
    if src_block == 0 || src.len() % src_block != 0 {
        return Err(RuntimeError::SizeMismatch {
            expected: src.len() / src_block.max(1) * src_block,
            actual: src.len(),
        });
    }
    let outer = src.len() / src_block;
    let out_block = c_total
        .checked_mul(inner)
        .ok_or(RuntimeError::ShapeMismatch)?;
    let expected_out = outer
        .checked_mul(out_block)
        .ok_or(RuntimeError::ShapeMismatch)?;
    if out.len() != expected_out {
        return Err(RuntimeError::SizeMismatch {
            expected: expected_out,
            actual: out.len(),
        });
    }
    for o in 0..outer {
        let dst_start = o * out_block + c_offset * inner;
        let src_start = o * src_block;
        out[dst_start..dst_start + src_block]
            .copy_from_slice(&src[src_start..src_start + src_block]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_broadcast_add() {
        let a = [1.0f32, 2.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 6];
        binary(BinaryOp::Add, &a, &[2, 1], &b, &[1, 3], &mut out, &[2, 3]).unwrap();
        assert_eq!(out, [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn binary_equal_shapes() {
        let a = [1.0f32, 4.0];
        let b = [3.0f32, 2.0];
        let mut out = [0.0f32; 2];
        binary(BinaryOp::Mul, &a, &[2], &b, &[2], &mut out, &[2]).unwrap();
        assert_eq!(out, [3.0, 8.0]);
        binary(BinaryOp::Sub, &a, &[2], &b, &[2], &mut out, &[2]).unwrap();
        assert_eq!(out, [-2.0, 2.0]);
        binary(BinaryOp::Div, &a, &[2], &b, &[2], &mut out, &[2]).unwrap();
        assert!((out[0] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn binary_incompatible_broadcast_errors() {
        let a = [1.0f32; 3];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        let err = binary(BinaryOp::Add, &a, &[3], &b, &[4], &mut out, &[4]);
        assert_eq!(err.unwrap_err(), RuntimeError::ShapeMismatch);
    }

    #[test]
    fn add_inplace_broadcast_bias() {
        let mut out = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = [10.0f32, 20.0];
        add_inplace_broadcast(&mut out, &[2, 3], &bias, &[2, 1]).unwrap();
        assert_eq!(out, [11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn copy_elements_checks_length() {
        let mut dst = [0.0f32; 2];
        assert_eq!(
            copy_elements(&[1.0, 2.0, 3.0], &mut dst).unwrap_err(),
            RuntimeError::SizeMismatch {
                expected: 2,
                actual: 3
            }
        );
        copy_elements(&[7.0, 8.0], &mut dst).unwrap();
        assert_eq!(dst, [7.0, 8.0]);
    }

    #[test]
    fn max_pool_2x2() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        max_pool2d(
            &input,
            &[1, 1, 4, 4],
            &mut out,
            &[1, 1, 2, 2],
            [2, 2],
            [2, 2],
            [0, 0],
        )
        .unwrap();
        assert_eq!(out, [5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn average_pool_global_mean() {
        let input = [1.0f32, 2.0, 3.0, 6.0];
        let mut out = [0.0f32; 1];
        average_pool2d(
            &input,
            &[1, 1, 2, 2],
            &mut out,
            &[1, 1, 1, 1],
            [2, 2],
            [1, 1],
            [0, 0],
        )
        .unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn pool_rejects_bad_geometry() {
        let input = [0.0f32; 16];
        let mut out = [0.0f32; 4];
        let err = max_pool2d(
            &input,
            &[1, 1, 4, 4],
            &mut out,
            &[1, 1, 3, 3],
            [2, 2],
            [2, 2],
            [0, 0],
        );
        assert_eq!(err.unwrap_err(), RuntimeError::ShapeMismatch);
    }

    #[test]
    fn batch_norm_channel_mode() {
        // [1, 2, 1, 2]: channels 0 and 1 with distinct statistics.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let scale = [2.0f32, 1.0];
        let bias = [0.0f32, 10.0];
        let mean = [1.0f32, 3.0];
        let var = [1.0f32, 1.0];
        let mut out = [0.0f32; 4];
        batch_norm(
            &x,
            &[1, 2, 1, 2],
            &mut out,
            [(&scale), (&bias), (&mean), (&var)],
            0.0,
        )
        .unwrap();
        // ch0: (x-1)*2 = 0, 2 ; ch1: (x-3) + 10 = 10, 11
        assert_eq!(out, [0.0, 2.0, 10.0, 11.0]);
    }

    #[test]
    fn sqrt_f32_matches_known_values() {
        assert_eq!(sqrt_f32(0.0), 0.0);
        assert!((sqrt_f32(1.0) - 1.0).abs() < 1e-6);
        assert!((sqrt_f32(4.0) - 2.0).abs() < 1e-5);
        assert!((sqrt_f32(2.0) - std::f32::consts::SQRT_2).abs() < 1e-5);
        assert!(sqrt_f32(-1.0).is_nan());
        assert_eq!(sqrt_f32(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn transpose_swaps_axes() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let mut out = [0.0f32; 6];
        transpose(&x, &[2, 3], &mut out, &[3, 2], &[1, 0]).unwrap();
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_rejects_non_permutation() {
        let x = [0.0f32; 6];
        let mut out = [0.0f32; 6];
        let err = transpose(&x, &[2, 3], &mut out, &[3, 2], &[0, 0]);
        assert_eq!(err.unwrap_err(), RuntimeError::ShapeMismatch);
    }

    #[test]
    fn concat_streams_operands() {
        // Two [1,1,2] operands concat on axis 1 → [1,2,2].
        let mut out = [0.0f32; 4];
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        // outer = 2, inner = 2, c_total = 2
        concat_copy_one(&mut out, 2, 0, 1, 2, &a).unwrap();
        concat_copy_one(&mut out, 2, 1, 1, 2, &b).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_rejects_overlapping_channels() {
        let mut out = [0.0f32; 4];
        let a = [1.0f32, 2.0];
        let err = concat_copy_one(&mut out, 2, 1, 2, 2, &a);
        assert_eq!(err.unwrap_err(), RuntimeError::ShapeMismatch);
    }
}
