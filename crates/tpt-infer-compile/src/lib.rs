//! AOT graph-to-Rust compiler.
pub mod codegen;
pub mod fold;
pub mod model;
pub use model::{aot_compile, CompiledModel};
