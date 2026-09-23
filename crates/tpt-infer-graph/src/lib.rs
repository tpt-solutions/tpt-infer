//! Compile-time-checked computation graphs for neural network models.
//!
//! This crate offers two complementary ways to describe a model:
//!
//! - [`GraphBuilder`]: a type-state builder whose type parameters track
//!   tensor shapes, so incompatible operations (for example a
//!   `[1, 784] x [10, 10]` matmul) fail to compile instead of panicking at
//!   runtime.
//! - [`ComputationGraph`]: a dynamic graph (rank ≤ 8) with node/edge shape
//!   metadata and [`ComputationGraph::topological_sort`], consumed by the
//!   ONNX loader, AOT compiler, and runtime crates.
//!
//! # Compile-time shape checking
//!
//! ```
//! use tpt_infer_graph::{GraphBuilder, Sh};
//!
//! // Input tensor of shape [1, 784].
//! let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
//!
//! // [1, 784] x [784, 128] -> [1, 128], verified by the compiler.
//! let w1 = b.add_input::<Sh<784, 128>>("w1");
//! let mut b = b.matmul(w1).relu();
//!
//! // [1, 128] x [128, 10] -> [1, 10].
//! let w2 = b.add_input::<Sh<128, 10>>("w2");
//! let b = b.matmul(w2).softmax(-1);
//!
//! let g = b.into_graph();
//! assert_eq!(g.nodes().len(), 7);
//! assert_eq!(g.node(6).unwrap().dims(), &[1, 10]);
//! ```
//!
//! A mismatched shape is a compile error rather than a runtime panic — see
//! the `compile_fail` trybuild tests.
//!
//! # Dynamic graphs
//!
//! ```
//! use tpt_infer_graph::{ComputationGraph, Node, Operator};
//!
//! let mut g = ComputationGraph::new();
//! let x = g
//!     .add_node(Node::new(0, Operator::Input, vec![], &[1, 784]).unwrap())
//!     .unwrap();
//! let y = g
//!     .add_node(Node::new(1, Operator::Relu, vec![x], &[1, 784]).unwrap())
//!     .unwrap();
//! g.mark_output(y).unwrap();
//! assert_eq!(g.topological_sort().unwrap(), vec![0, 1]);
//! ```
//!
//! # `no_std`
//!
//! The crate is `#![no_std]` when built without the `std` feature; the
//! graph-building APIs are available whenever `std` (which enables `alloc`)
//! is enabled, which is the default.

#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

#[cfg(any(feature = "std", test))]
extern crate alloc;

#[cfg(any(feature = "std", test))]
pub mod builder;
#[cfg(any(feature = "std", test))]
pub mod graph;
#[cfg(any(feature = "std", test))]
pub mod operator;

#[cfg(any(feature = "std", test))]
pub use builder::{
    AddShape, FlattenShape, GraphBuilder, MatMulShape, NodeRef, Sh, ShapeMarker, TransposeShape,
    TypedEdge,
};
#[cfg(any(feature = "std", test))]
pub use graph::{ComputationGraph, Edge, GraphError, Initializer, Node};
#[cfg(any(feature = "std", test))]
pub use operator::Operator;
