//! Error types for the inference runtime.

use tpt_infer_core::TensorError;
use tpt_infer_ops::OpError;

/// Errors produced while planning or executing a computation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeError {
    /// The activation arena ran out of space during planning.
    OutOfMemory,
    /// An operator implementation rejected an operand (shape/size/argument).
    Op(OpError),
    /// A buffer was too small for the element count required by a node.
    SizeMismatch {
        /// Number of elements required.
        expected: usize,
        /// Number of elements provided.
        actual: usize,
    },
    /// Operand shapes were incompatible for the requested operation.
    ShapeMismatch,
    /// The graph declares more runtime inputs than slices were supplied.
    InputCountMismatch {
        /// Number of runtime inputs the graph expects.
        expected: usize,
        /// Number of input slices supplied.
        actual: usize,
    },
    /// An [`Input`](tpt_infer_graph::Operator::Input) node has neither an
    /// initializer nor a supplied input slice.
    MissingInput {
        /// Id of the input node that could not be bound.
        node: usize,
    },
    /// An input node references an initializer whose element count does not
    /// match the node's declared output shape.
    InitializerMismatch {
        /// Id of the input node.
        node: usize,
    },
    /// A graph input slice length does not match the declared input shape.
    InputDataMismatch {
        /// Id of the input node.
        node: usize,
    },
    /// A generic invalid argument (maps from lower-level tensor errors).
    Invalid,
    /// The operator is not implemented by this runtime.
    UnsupportedOp {
        /// Canonical operator name (e.g. `"Custom"`).
        name: &'static str,
    },
    /// A [`Custom`](tpt_infer_graph::Operator::Custom) operator is not
    /// implemented by this runtime. Only compiled with `std`/`test` because
    /// the original op name is a heap `String`.
    #[cfg(any(feature = "std", test))]
    UnsupportedCustom {
        /// Original operator name carried by the graph.
        name: String,
    },
    /// The graph contains a cycle (or an unknown node reference).
    CycleDetected,
    /// The graph declares no outputs.
    NoOutputs,
    /// A single-output entry point was used on a graph with a different
    /// number of outputs.
    OutputCountMismatch {
        /// Number of graph outputs found.
        actual: usize,
    },
    /// A graph-level error occurred during planning.
    #[cfg(any(feature = "std", test))]
    Graph(tpt_infer_graph::GraphError),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeError::OutOfMemory => f.write_str("arena out of memory"),
            RuntimeError::Op(e) => write!(f, "operator error: {e}"),
            RuntimeError::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} elements, got {actual}")
            }
            RuntimeError::ShapeMismatch => f.write_str("shape mismatch"),
            RuntimeError::Invalid => f.write_str("invalid argument"),
            RuntimeError::InputCountMismatch { expected, actual } => {
                write!(f, "input count mismatch: expected {expected}, got {actual}")
            }
            RuntimeError::MissingInput { node } => {
                write!(f, "input node {node} has no initializer and no supplied input")
            }
            RuntimeError::InitializerMismatch { node } => {
                write!(f, "initializer for input node {node} does not match its shape")
            }
            RuntimeError::InputDataMismatch { node } => {
                write!(f, "input data for node {node} does not match its shape")
            }
            RuntimeError::UnsupportedOp { name } => {
                write!(f, "operator {name} is not supported by this runtime")
            }
            #[cfg(any(feature = "std", test))]
            RuntimeError::UnsupportedCustom { name } => {
                write!(f, "custom operator {name} is not supported by this runtime")
            }
            RuntimeError::CycleDetected => f.write_str("graph contains a cycle"),
            RuntimeError::NoOutputs => f.write_str("graph has no marked outputs"),
            RuntimeError::OutputCountMismatch { actual } => {
                write!(f, "expected a single graph output, found {actual}")
            }
            #[cfg(any(feature = "std", test))]
            RuntimeError::Graph(e) => write!(f, "graph error: {e}"),
        }
    }
}

impl From<TensorError> for RuntimeError {
    fn from(e: TensorError) -> Self {
        match e {
            TensorError::OutOfMemory => RuntimeError::OutOfMemory,
            TensorError::SizeMismatch { expected, actual } => {
                RuntimeError::SizeMismatch { expected, actual }
            }
            TensorError::ShapeMismatch => RuntimeError::ShapeMismatch,
            _ => RuntimeError::Invalid,
        }
    }
}

impl From<OpError> for RuntimeError {
    fn from(e: OpError) -> Self {
        RuntimeError::Op(e)
    }
}

#[cfg(any(feature = "std", test))]
impl From<tpt_infer_graph::GraphError> for RuntimeError {
    fn from(e: tpt_infer_graph::GraphError) -> Self {
        match e {
            tpt_infer_graph::GraphError::CycleDetected => RuntimeError::CycleDetected,
            other => RuntimeError::Graph(other),
        }
    }
}

#[cfg(any(feature = "std", test))]
impl core::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        assert_eq!(
            RuntimeError::OutOfMemory.to_string(),
            "arena out of memory"
        );
        assert_eq!(
            RuntimeError::InputCountMismatch {
                expected: 1,
                actual: 0
            }
            .to_string(),
            "input count mismatch: expected 1, got 0"
        );
        assert_eq!(
            RuntimeError::UnsupportedOp { name: "Custom" }.to_string(),
            "operator Custom is not supported by this runtime"
        );
    }

    #[test]
    fn conversions_from_core_errors() {
        let e: RuntimeError = TensorError::OutOfMemory.into();
        assert_eq!(e, RuntimeError::OutOfMemory);
        let e: RuntimeError = OpError::ShapeMismatch.into();
        assert_eq!(e, RuntimeError::Op(OpError::ShapeMismatch));
        let e: RuntimeError = tpt_infer_graph::GraphError::CycleDetected.into();
        assert_eq!(e, RuntimeError::CycleDetected);
    }

    #[test]
    fn error_impl_is_object_safe_display() {
        let e: &dyn core::error::Error = &RuntimeError::NoOutputs;
        assert_eq!(e.to_string(), "graph has no marked outputs");
    }
}
