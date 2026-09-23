//! The [`Backend`] operator HAL and shared operator types.

/// Error returned by operator implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpError {
    /// A buffer length does not match the product of its shape.
    SizeMismatch {
        /// Number of elements the operation expects.
        expected: usize,
        /// Number of elements actually provided.
        actual: usize,
    },
    /// Operand shapes are incompatible for the requested operation.
    ShapeMismatch,
    /// The backend does not implement the requested operation.
    Unsupported,
    /// An argument was invalid (e.g. a zero stride or empty softmax row).
    Invalid,
}

impl core::fmt::Display for OpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OpError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch: expected {expected} elements, got {actual}"
                )
            }
            OpError::ShapeMismatch => f.write_str("shape mismatch"),
            OpError::Unsupported => f.write_str("operation not supported by this backend"),
            OpError::Invalid => f.write_str("invalid argument"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OpError {}

/// Options for [`Backend::conv2d`].
///
/// Layout: input is NCHW, weights are OIHW (`[out_channels, in_channels, kh, kw]`),
/// output is NCHW. Groups = 1, dilation = 1, no bias.
///
/// Output spatial size is
/// `floor((dim + 2 * pad - kernel) / stride) + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dOptions {
    /// Stride over the height dimension.
    pub stride_h: usize,
    /// Stride over the width dimension.
    pub stride_w: usize,
    /// Zero-padding added to both sides of the height dimension.
    pub pad_h: usize,
    /// Zero-padding added to both sides of the width dimension.
    pub pad_w: usize,
}

impl Conv2dOptions {
    /// Stride 1, padding 0 (valid convolution).
    pub const fn new() -> Self {
        Self {
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        }
    }

    /// Explicit stride and zero-padding.
    pub const fn with_stride_padding(
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Self {
        Self {
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        }
    }
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Hardware abstraction layer for ML operators.
///
/// Shapes are row-major. Every method writes into a caller-provided output
/// slice and validates buffer lengths, returning [`OpError`] on mismatch.
/// Backends never allocate.
pub trait Backend {
    /// Stable backend name, e.g. `"naive"` or `"avx2"`.
    fn name(&self) -> &'static str;

    /// Dense matrix multiply: `out[m,n] = a[m,k] @ b[k,n]`.
    ///
    /// `a_shape = [m, k]`, `b_shape = [k, n]`, `out.len() == m * n`.
    /// No transposition is applied.
    fn matmul(
        &self,
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
        out: &mut [f32],
    ) -> Result<(), OpError>;

    /// 2D convolution over NCHW input with OIHW weights (see [`Conv2dOptions`]).
    ///
    /// Output spatial dims are derived from `in_shape`, `w_shape`, and
    /// `options`; `out.len()` must equal `N * OC * OH * OW`.
    fn conv2d(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        weight: &[f32],
        w_shape: [usize; 4],
        out: &mut [f32],
        options: Conv2dOptions,
    ) -> Result<(), OpError>;

    /// Elementwise sum: `out = a + b` (equal lengths).
    fn elementwise_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError>;

    /// Rectified linear unit: `out = max(a, 0)` (equal lengths).
    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError>;

    /// Softmax over rows of length `row_len` (the last dimension).
    ///
    /// `a.len() == out.len()` must be a multiple of `row_len`; each row of
    /// `out` sums to 1. The running max is subtracted for stability.
    fn softmax(&self, a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError>;

    /// Elementwise logistic sigmoid: `out = 1 / (1 + exp(-a))`.
    fn sigmoid(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError>;

    /// Elementwise GELU using the tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    fn gelu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError>;
}

pub(crate) fn validate_matmul(
    a: &[f32],
    a_shape: [usize; 2],
    b: &[f32],
    b_shape: [usize; 2],
    out: &[f32],
) -> Result<(usize, usize, usize), OpError> {
    let [m, k] = a_shape;
    let [kb, n] = b_shape;
    if k != kb {
        return Err(OpError::ShapeMismatch);
    }
    let expected_a = m.checked_mul(k).ok_or(OpError::Invalid)?;
    if a.len() != expected_a {
        return Err(OpError::SizeMismatch {
            expected: expected_a,
            actual: a.len(),
        });
    }
    let expected_b = k.checked_mul(n).ok_or(OpError::Invalid)?;
    if b.len() != expected_b {
        return Err(OpError::SizeMismatch {
            expected: expected_b,
            actual: b.len(),
        });
    }
    let expected_out = m.checked_mul(n).ok_or(OpError::Invalid)?;
    if out.len() != expected_out {
        return Err(OpError::SizeMismatch {
            expected: expected_out,
            actual: out.len(),
        });
    }
    Ok((m, k, n))
}

pub(crate) fn validate_binary(a: &[f32], b: &[f32], out: &[f32]) -> Result<(), OpError> {
    if a.len() != b.len() {
        return Err(OpError::SizeMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if out.len() != a.len() {
        return Err(OpError::SizeMismatch {
            expected: a.len(),
            actual: out.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_unary(a: &[f32], out: &[f32]) -> Result<(), OpError> {
    if out.len() != a.len() {
        return Err(OpError::SizeMismatch {
            expected: a.len(),
            actual: out.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_softmax(a: &[f32], out: &[f32], row_len: usize) -> Result<(), OpError> {
    validate_unary(a, out)?;
    if row_len == 0 || a.len() % row_len != 0 {
        return Err(OpError::Invalid);
    }
    Ok(())
}

pub(crate) fn validate_conv2d(
    input: &[f32],
    in_shape: [usize; 4],
    weight: &[f32],
    w_shape: [usize; 4],
    out: &[f32],
    options: Conv2dOptions,
) -> Result<[usize; 4], OpError> {
    let [n, c, h, w] = in_shape;
    let [oc, ic, kh, kw] = w_shape;
    if ic != c {
        return Err(OpError::ShapeMismatch);
    }
    if kh == 0 || kw == 0 {
        return Err(OpError::Invalid);
    }
    if options.stride_h == 0 || options.stride_w == 0 {
        return Err(OpError::Invalid);
    }
    let padded_h = h
        .checked_add(options.pad_h.checked_mul(2).ok_or(OpError::Invalid)?)
        .ok_or(OpError::Invalid)?;
    let padded_w = w
        .checked_add(options.pad_w.checked_mul(2).ok_or(OpError::Invalid)?)
        .ok_or(OpError::Invalid)?;
    if padded_h < kh || padded_w < kw {
        return Err(OpError::Invalid);
    }
    let oh = (padded_h - kh) / options.stride_h + 1;
    let ow = (padded_w - kw) / options.stride_w + 1;

    let expected_in = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w))
        .ok_or(OpError::Invalid)?;
    if input.len() != expected_in {
        return Err(OpError::SizeMismatch {
            expected: expected_in,
            actual: input.len(),
        });
    }
    let expected_w = oc
        .checked_mul(ic)
        .and_then(|v| v.checked_mul(kh))
        .and_then(|v| v.checked_mul(kw))
        .ok_or(OpError::Invalid)?;
    if weight.len() != expected_w {
        return Err(OpError::SizeMismatch {
            expected: expected_w,
            actual: weight.len(),
        });
    }
    let expected_out = n
        .checked_mul(oc)
        .and_then(|v| v.checked_mul(oh))
        .and_then(|v| v.checked_mul(ow))
        .ok_or(OpError::Invalid)?;
    if out.len() != expected_out {
        return Err(OpError::SizeMismatch {
            expected: expected_out,
            actual: out.len(),
        });
    }
    Ok([n, oc, oh, ow])
}
