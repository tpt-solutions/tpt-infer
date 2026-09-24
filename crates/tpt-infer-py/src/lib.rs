//! Python bindings for `tpt-infer`, via [PyO3].
//!
//! ```python
//! import tpt_infer
//!
//! model = tpt_infer.Model.load("model.onnx")
//! print(model.node_count(), model.output_count())
//! output = model.run([0.0] * input_length)
//! ```
//!
//! Thin wrapper: loading is `tpt_infer_onnx::load`/`load_from_bytes`,
//! execution is `tpt_infer_runtime::execute` against
//! `tpt_infer_ops::dispatch::select_backend`'s auto-detected backend — the
//! same pipeline `tpt-infer-cli run` and `tpt-infer-wasm`'s `run` use.
//!
//! [PyO3]: https://pyo3.rs/

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tpt_infer_core::BumpArena;
use tpt_infer_graph::ComputationGraph;
use tpt_infer_ops::dispatch::select_backend;
use tpt_infer_runtime::{execute, required_arena_bytes};

/// A loaded ONNX model, ready to execute or AOT-compile.
#[pyclass]
struct Model {
    graph: ComputationGraph,
}

#[pymethods]
impl Model {
    /// Loads a model from a `.onnx` file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let graph = tpt_infer_onnx::load(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Model { graph })
    }

    /// Loads a model from in-memory `.onnx` bytes.
    #[staticmethod]
    fn load_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        let graph = tpt_infer_onnx::load_from_bytes(&bytes)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Model { graph })
    }

    /// Executes the model against a flattened `input` (row-major, matching
    /// the model's declared input shape) and returns the flattened output.
    fn run(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let mut mem = vec![0u8; required_arena_bytes(&self.graph) + 4096];
        let mut arena = BumpArena::new(&mut mem);
        let backend = select_backend();
        let output = execute(&self.graph, &input, &mut arena, &backend)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(output.as_slice().to_vec())
    }

    /// Ahead-of-time compiles the model to standalone Rust source (a
    /// `pub fn execute(input: &[f32]) -> Vec<f32>`), returned as a string.
    fn compile(&self) -> PyResult<String> {
        let compiled = tpt_infer_compile::aot_compile(&self.graph)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(compiled.source().to_string())
    }

    /// Total node count.
    fn node_count(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Number of graph outputs.
    fn output_count(&self) -> usize {
        self.graph.outputs().len()
    }
}

#[pymodule]
fn _tpt_infer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    Ok(())
}
