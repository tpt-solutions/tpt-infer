//! Operators that describe the computation performed by a graph node.

use alloc::string::String;

use tpt_infer_core::{Shape, MAX_RANK};

/// The operation a [`Node`](crate::Node) performs.
///
/// Attribute payloads mirror the common ONNX opset-17 attributes for the
/// corresponding operators, so the ONNX loader can map op strings directly
/// onto variants. Unknown or unsupported ONNX ops fall back to
/// [`Operator::Custom`] with the original op name.
///
/// # Example
/// ```
/// use tpt_infer_graph::Operator;
/// let op = Operator::softmax(-1);
/// assert_eq!(op.name(), "Softmax");
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Operator {
    /// Graph input placeholder; produces no computation.
    Input,
    /// Matrix multiply over the last two dimensions: `[m, k] x [k, n] -> [m, n]`.
    MatMul,
    /// 2-D convolution (NCHW).
    Conv2d {
        /// Spatial stride `(stride_h, stride_w)`.
        strides: [usize; 2],
        /// Zero-padding `(pad_h, pad_w)` applied to height and width.
        padding: [usize; 2],
    },
    /// Element-wise addition (broadcasting is a runtime concern).
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Rectified linear unit, `max(0, x)`.
    Relu,
    /// Logistic sigmoid, `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Gaussian error linear unit.
    Gelu,
    /// Softmax along `axis` (ONNX default is `-1`).
    Softmax {
        /// Axis the softmax normalization is applied along (may be negative).
        axis: i32,
    },
    /// Reshape the tensor to `shape[..rank]` (element count must match).
    Reshape {
        /// Target shape, zero-padded to [`MAX_RANK`].
        shape: Shape,
        /// Number of active dimensions in `shape`.
        rank: usize,
    },
    /// Flatten to a 2-D tensor splitting after `axis` (ONNX semantics; may be negative).
    Flatten {
        /// Split axis (may be negative).
        axis: i32,
    },
    /// 2-D max pooling (NCHW).
    MaxPool2d {
        /// Pooling window `(kernel_h, kernel_w)`.
        kernel: [usize; 2],
        /// Spatial stride `(stride_h, stride_w)`.
        strides: [usize; 2],
        /// Zero-padding `(pad_h, pad_w)`.
        padding: [usize; 2],
    },
    /// 2-D average pooling (NCHW).
    AveragePool2d {
        /// Pooling window `(kernel_h, kernel_w)`.
        kernel: [usize; 2],
        /// Spatial stride `(stride_h, stride_w)`.
        strides: [usize; 2],
        /// Zero-padding `(pad_h, pad_w)`.
        padding: [usize; 2],
    },
    /// Batch normalization (inference mode uses running statistics).
    BatchNorm {
        /// Numerical-stability epsilon added to the variance.
        epsilon: f32,
    },
    /// Concatenation of all inputs along `axis` (may be negative).
    Concat {
        /// Concatenation axis (may be negative).
        axis: i32,
    },
    /// Dimension permutation.
    Transpose {
        /// Permutation, zero-padded to [`MAX_RANK`]: `perm[..rank]`.
        perm: Shape,
        /// Number of active dimensions in `perm`.
        rank: usize,
    },
    /// Forward-compatible placeholder for ops this crate does not model yet.
    ///
    /// Carries the original op name (e.g. from an ONNX model).
    Custom(String),
}

impl Operator {
    /// [`Operator::Softmax`] with the given axis.
    pub const fn softmax(axis: i32) -> Self {
        Self::Softmax { axis }
    }

    /// [`Operator::Conv2d`] with the given strides and padding.
    pub const fn conv2d(strides: [usize; 2], padding: [usize; 2]) -> Self {
        Self::Conv2d { strides, padding }
    }

    /// [`Operator::MaxPool2d`] with the given kernel, strides, and padding.
    pub const fn max_pool2d(kernel: [usize; 2], strides: [usize; 2], padding: [usize; 2]) -> Self {
        Self::MaxPool2d {
            kernel,
            strides,
            padding,
        }
    }

    /// [`Operator::AveragePool2d`] with the given kernel, strides, and padding.
    pub const fn average_pool2d(
        kernel: [usize; 2],
        strides: [usize; 2],
        padding: [usize; 2],
    ) -> Self {
        Self::AveragePool2d {
            kernel,
            strides,
            padding,
        }
    }

    /// [`Operator::Flatten`] with the given axis.
    pub const fn flatten(axis: i32) -> Self {
        Self::Flatten { axis }
    }

    /// [`Operator::Concat`] with the given axis.
    pub const fn concat(axis: i32) -> Self {
        Self::Concat { axis }
    }

    /// [`Operator::BatchNorm`] with the given epsilon.
    pub const fn batch_norm(epsilon: f32) -> Self {
        Self::BatchNorm { epsilon }
    }

    /// Wrap an arbitrary op name for forward compatibility.
    pub fn custom(name: impl Into<String>) -> Self {
        Self::Custom(name.into())
    }

    /// [`Operator::Reshape`] targeting `dims`.
    ///
    /// Returns `None` if `dims` exceeds [`MAX_RANK`].
    ///
    /// # Example
    /// ```
    /// use tpt_infer_graph::Operator;
    /// assert!(Operator::reshape(&[1, 784]).is_some());
    /// assert!(Operator::reshape(&[1; 9]).is_none());
    /// ```
    pub fn reshape(dims: &[usize]) -> Option<Self> {
        if dims.len() > MAX_RANK {
            return None;
        }
        let mut shape = [0; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Some(Self::Reshape {
            shape,
            rank: dims.len(),
        })
    }

    /// [`Operator::Transpose`] with permutation `perm`.
    ///
    /// Returns `None` if `perm` is empty, exceeds [`MAX_RANK`], or is not a
    /// permutation of `0..perm.len()`.
    ///
    /// # Example
    /// ```
    /// use tpt_infer_graph::Operator;
    /// assert!(Operator::transpose(&[2, 0, 1]).is_some());
    /// assert!(Operator::transpose(&[0, 0]).is_none());
    /// ```
    pub fn transpose(perm: &[usize]) -> Option<Self> {
        let rank = perm.len();
        if rank == 0 || rank > MAX_RANK {
            return None;
        }
        let mut seen = [false; MAX_RANK];
        for &p in perm {
            if p >= rank || seen[p] {
                return None;
            }
            seen[p] = true;
        }
        let mut packed = [0; MAX_RANK];
        packed[..rank].copy_from_slice(perm);
        Some(Self::Transpose { perm: packed, rank })
    }

    /// Canonical op name (e.g. `"MatMul"`), independent of attribute payloads.
    ///
    /// For [`Operator::Custom`] this is always `"Custom"`; use
    /// [`Display`](core::fmt::Display) to include the original op name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::MatMul => "MatMul",
            Self::Conv2d { .. } => "Conv2d",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Gelu => "Gelu",
            Self::Softmax { .. } => "Softmax",
            Self::Reshape { .. } => "Reshape",
            Self::Flatten { .. } => "Flatten",
            Self::MaxPool2d { .. } => "MaxPool2d",
            Self::AveragePool2d { .. } => "AveragePool2d",
            Self::BatchNorm { .. } => "BatchNorm",
            Self::Concat { .. } => "Concat",
            Self::Transpose { .. } => "Transpose",
            Self::Custom(_) => "Custom",
        }
    }

    /// Whether this operator is a graph input placeholder.
    pub const fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }
}

impl core::fmt::Display for Operator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Custom(name) => f.write_str(name),
            other => f.write_str(other.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_cover_all_variants() {
        assert_eq!(Operator::Input.name(), "Input");
        assert_eq!(Operator::MatMul.name(), "MatMul");
        assert_eq!(Operator::conv2d([1, 1], [0, 0]).name(), "Conv2d");
        assert_eq!(Operator::Add.name(), "Add");
        assert_eq!(Operator::Relu.name(), "Relu");
        assert_eq!(Operator::softmax(-1).name(), "Softmax");
        assert_eq!(Operator::reshape(&[1, 2]).unwrap().name(), "Reshape");
        assert_eq!(Operator::flatten(1).name(), "Flatten");
        assert_eq!(
            Operator::max_pool2d([2, 2], [2, 2], [0, 0]).name(),
            "MaxPool2d"
        );
        assert_eq!(
            Operator::average_pool2d([2, 2], [2, 2], [0, 0]).name(),
            "AveragePool2d"
        );
        assert_eq!(Operator::batch_norm(1e-5).name(), "BatchNorm");
        assert_eq!(Operator::concat(1).name(), "Concat");
        assert_eq!(Operator::transpose(&[1, 0]).unwrap().name(), "Transpose");
        assert_eq!(Operator::custom("Erf").name(), "Custom");
    }

    #[test]
    fn display_uses_custom_name() {
        assert_eq!(alloc::format!("{}", Operator::custom("Erf")), "Erf");
        assert_eq!(alloc::format!("{}", Operator::Relu), "Relu");
    }

    #[test]
    fn reshape_helper_validates_rank() {
        let op = Operator::reshape(&[1, 784]).unwrap();
        match op {
            Operator::Reshape { shape, rank } => {
                assert_eq!(rank, 2);
                assert_eq!(shape[0], 1);
                assert_eq!(shape[1], 784);
                assert_eq!(shape[2], 0);
            }
            other => panic!("expected Reshape, got {other:?}"),
        }
        assert!(Operator::reshape(&[1; 9]).is_none());
    }

    #[test]
    fn transpose_helper_validates_permutation() {
        assert!(Operator::transpose(&[2, 0, 1]).is_some());
        assert!(Operator::transpose(&[0]).is_some());
        assert!(Operator::transpose(&[]).is_none());
        assert!(Operator::transpose(&[0, 0]).is_none());
        assert!(Operator::transpose(&[2, 0]).is_none());
        assert!(Operator::transpose(&[1; 9]).is_none());
    }

    #[test]
    fn is_input_detection() {
        assert!(Operator::Input.is_input());
        assert!(!Operator::Relu.is_input());
        assert!(!Operator::custom("Erf").is_input());
    }
}
