//! Pure-Rust ONNX model parser and operator registry (no C-FFI).
//!
//! Decodes `ModelProto` with [`prost`], maps opset-17 operators onto
//! [`tpt_infer_graph::Operator`], attaches initializers, and runs a shape
//! inference pass — producing a `ComputationGraph` ready for the AOT
//! compiler and runtime.
//!
//! # Example
//! ```no_run
//! let graph = tpt_infer_onnx::load("model.onnx")?;
//! println!("{} nodes", graph.nodes().len());
//! # Ok::<(), tpt_infer_onnx::OnnxError>(())
//! ```
//!
//! Dynamic dimensions (batch = `-1` / `dim_param`) are represented as `0` in
//! node shapes.

pub mod load;
pub mod registry;
pub mod shapes;

pub use load::{
    graph_from_proto, load, load_from_bytes, load_from_bytes_with_limits, proto, OnnxError,
    DEFAULT_MAX_GRAPH_ITEMS, DEFAULT_MAX_MODEL_BYTES, MAX_SUPPORTED_OPSET,
};
pub use registry::{is_native, map_op};
pub use shapes::{infer_node_shape, infer_shapes, ShapeInferError};
