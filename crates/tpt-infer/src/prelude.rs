//! Convenience re-exports of the tpt-infer workspace's public API.
//!
//! `use tpt_infer::prelude::*;` pulls in the tensor/graph core types plus
//! whichever optional pipeline stages this crate was built with (see the
//! [crate-level docs](crate) for the feature flags). Everything here is a
//! straight re-export — no new types are defined in this module.

// --- Core tensor / arena / dtype primitives (always available). ---
//
// `tpt-infer-graph` is a required dependency built with its `std` feature,
// which enables `tpt-infer-core/alloc` transitively (Cargo feature
// unification), so the heap-backed `TensorVec` is always available here too.
pub use tpt_infer_core::{BumpArena, DType, Shape, Tensor, TensorError, TensorVec, MAX_RANK};

// --- Computation graph IR and the type-state builder. ---
pub use tpt_infer_graph::GraphBuilder;
pub use tpt_infer_graph::{
    AddShape, ComputationGraph, Edge, FlattenShape, GraphError, Initializer, MatMulShape, Node,
    NodeRef, Operator, Sh, ShapeMarker, TransposeShape, TypedEdge,
};

// --- CPU operator backends (feature `ops-cpu`, default). ---
#[cfg(feature = "ops-cpu")]
pub use tpt_infer_ops::{dispatch::select_backend, Backend, Conv2dOptions, NaiveBackend, OpError};

// --- ONNX model loading (feature `onnx`, default). ---
#[cfg(feature = "onnx")]
pub use tpt_infer_onnx::{load, load_from_bytes, OnnxError};

// --- AOT graph-to-Rust compiler (feature `compile`, default). ---
#[cfg(feature = "compile")]
pub use tpt_infer_compile::{aot_compile, CompileError, CompiledModel};

// --- Interpreted graph runtime (feature `runtime`, default). ---
#[cfg(feature = "runtime")]
pub use tpt_infer_runtime::{argmax, execute, execute_graph, required_arena_bytes, RuntimeError};

// --- Vision preprocessing (feature `vision`, opt-in). ---
#[cfg(feature = "vision")]
pub use tpt_infer_vision::{
    preprocess_image, preprocess_raw_rgb, preprocess_source, ImageSource, PreprocessConfig,
    VisionError,
};

// --- Post-training quantization (feature `quantize`, opt-in). ---
#[cfg(feature = "quantize")]
pub use tpt_infer_quantize::{ptq, PtqError, PtqOptions, QuantizedGraph, QuantizedTensor};
