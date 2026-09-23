//! Map ONNX opset-17 operator strings onto [`Operator`] variants.

use tpt_infer_graph::Operator;

/// Map an ONNX operator type name to a [`Graph`] operator.
///
/// Returns [`Operator::Custom`] (preserving the original name) for ops this
/// crate does not model natively, so unknown ops still load.
///
/// # Example
/// ```
/// use tpt_infer_graph::Operator;
/// use tpt_infer_onnx::registry::map_op;
/// assert!(matches!(map_op("Relu"), Operator::Relu));
/// assert!(matches!(map_op("TotallyUnknownOp"), Operator::Custom(_)));
/// ```
pub fn map_op(op_type: &str) -> Operator {
    match op_type {
        "Identity" | "Dropout" => Operator::Input,
        "MatMul" => Operator::MatMul,
        "Gemm" => Operator::MatMul,
        "Add" => Operator::Add,
        "Sub" => Operator::Sub,
        "Mul" => Operator::Mul,
        "Div" => Operator::Div,
        "Relu" => Operator::Relu,
        "Sigmoid" => Operator::Sigmoid,
        "Gelu" | "Erf" => Operator::Gelu,
        "Softmax" => Operator::Softmax { axis: -1 },
        "Reshape" => Operator::Reshape {
            shape: [0; tpt_infer_core::MAX_RANK],
            rank: 0,
        },
        "Flatten" => Operator::Flatten { axis: 1 },
        "Transpose" => Operator::Transpose {
            perm: [0; tpt_infer_core::MAX_RANK],
            rank: 0,
        },
        "Concat" => Operator::Concat { axis: 1 },
        "Conv" => Operator::Conv2d {
            strides: [1, 1],
            padding: [0, 0],
        },
        "MaxPool" => Operator::MaxPool2d {
            kernel: [2, 2],
            strides: [2, 2],
            padding: [0, 0],
        },
        "AveragePool" => Operator::AveragePool2d {
            kernel: [2, 2],
            strides: [2, 2],
            padding: [0, 0],
        },
        "GlobalAveragePool" => Operator::AveragePool2d {
            kernel: [0, 0],
            strides: [1, 1],
            padding: [0, 0],
        },
        "BatchNormalization" => Operator::BatchNorm {
            epsilon: 1e-5,
        },
        "LayerNormalization" => Operator::Custom("LayerNormalization".into()),
        "Cast" | "Squeeze" | "Unsqueeze" | "Slice" | "Gather" | "Shape" | "Constant"
        | "Range" | "Where" | "Equal" | "Greater" | "Less" | "Pow" | "Sqrt" | "Exp"
        | "Log" | "Tanh" | "Neg" | "Abs" | "Min" | "Max" | "Sum" | "ReduceMean" => {
            Operator::Custom(op_type.into())
        }
        other => Operator::Custom(other.into()),
    }
}

/// Whether `map_op` produced a first-class [`Operator`] (not [`Operator::Custom`]).
pub fn is_native(op_type: &str) -> bool {
    !matches!(map_op(op_type), Operator::Custom(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_core_ops() {
        assert!(matches!(map_op("MatMul"), Operator::MatMul));
        assert!(matches!(map_op("Conv"), Operator::Conv2d { .. }));
        assert!(matches!(map_op("Relu"), Operator::Relu));
        assert!(matches!(map_op("Softmax"), Operator::Softmax { axis: -1 }));
        assert!(matches!(map_op("Gemm"), Operator::MatMul));
    }

    #[test]
    fn unknown_becomes_custom() {
        match map_op("FancyAttn") {
            Operator::Custom(name) => assert_eq!(name, "FancyAttn"),
            other => panic!("expected Custom, got {other:?}"),
        }
        assert!(is_native("Relu"));
        assert!(!is_native("FancyAttn"));
    }
}
