//! AOT graph-to-Rust compiler.
//!
//! [`aot_compile`] walks a [`tpt_infer_graph::ComputationGraph`] in
//! topological order and emits self-contained Rust source for a
//! `pub fn execute(input: &[f32]) -> Vec<f32>` function — see [`model`] for
//! the entry point and [`codegen`] for how the graph is translated to code.
//! [`fold`] evaluates initializer-only subgraphs ahead of time so the
//! generated code can skip them entirely. [`fpga_stub`] is a placeholder
//! extension point for a future `tpt-crucible` backend; it is not a real
//! implementation.
pub mod codegen;
pub mod fold;
pub mod fpga_stub;
pub mod model;
pub use model::{aot_compile, CompileError, CompiledModel};
