//! Compile-time-checked computation graphs.
#![cfg_attr(not(test), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
pub mod builder;
pub mod graph;
pub mod operator;
pub use builder::GraphBuilder;
pub use graph::{ComputationGraph, Edge, Node};
pub use operator::Operator;
