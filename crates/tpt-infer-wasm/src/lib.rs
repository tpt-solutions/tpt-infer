//! WASM/JS bindings for `tpt-infer`: load an ONNX model from bytes and run
//! it through the interpreted runtime, entirely in the browser or Node.js —
//! no server round-trip.
//!
//! ```js
//! import init, { TptInferModel } from "tpt-infer-wasm";
//!
//! await init();
//! const bytes = new Uint8Array(await (await fetch("model.onnx")).arrayBuffer());
//! const model = new TptInferModel(bytes);
//! const output = model.run(new Float32Array(inputLength).fill(0));
//! console.log(model.nodeCount(), output);
//! ```
//!
//! This crate is a thin wrapper: model loading is `tpt_infer_onnx::load_from_bytes`,
//! execution is `tpt_infer_runtime::execute` against `tpt_infer_ops`'s
//! auto-detected backend (falling back to the WASM SIMD backend or the
//! naive scalar backend, whichever `tpt-infer-ops::dispatch::select_backend`
//! picks for this target) — the same pipeline `tpt-infer-cli run` uses
//! natively.

use wasm_bindgen::prelude::*;

use tpt_infer_core::BumpArena;
use tpt_infer_graph::ComputationGraph;
use tpt_infer_ops::dispatch::select_backend;
use tpt_infer_runtime::{execute, required_arena_bytes};

/// Installs a panic hook that forwards Rust panics to the JS console
/// (instead of an opaque "unreachable executed") and reports the backend
/// this WASM binary will execute with. Call once, before using
/// [`TptInferModel`].
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// A loaded ONNX model, ready to execute.
#[wasm_bindgen]
pub struct TptInferModel {
    graph: ComputationGraph,
}

#[wasm_bindgen]
impl TptInferModel {
    /// Parses `onnx_bytes` (the raw contents of a `.onnx` file) into a
    /// computation graph.
    #[wasm_bindgen(constructor)]
    pub fn new(onnx_bytes: &[u8]) -> Result<TptInferModel, JsError> {
        let graph = tpt_infer_onnx::load_from_bytes(onnx_bytes)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(TptInferModel { graph })
    }

    /// Executes the model against a flattened `input` (row-major, matching
    /// the model's declared input shape) and returns the flattened output.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, JsError> {
        let mut mem = vec![0u8; required_arena_bytes(&self.graph) + 4096];
        let mut arena = BumpArena::new(&mut mem);
        let backend = select_backend();
        let output = execute(&self.graph, input, &mut arena, &backend)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(output.as_slice().to_vec())
    }

    /// Total node count — a quick sanity check that the model parsed as
    /// expected.
    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Number of graph outputs.
    #[wasm_bindgen(js_name = outputCount)]
    pub fn output_count(&self) -> usize {
        self.graph.outputs().len()
    }
}
