//! Zero-allocation inference runtime.
//!
//! This crate walks a [`ComputationGraph`] (or a future compiled model —
//! see [Compile crate status](#compile-crate-status)) in topological order,
//! dispatches each node onto a [`Backend`] from `tpt-infer-ops`, and stores
//! every intermediate activation in a caller-provided
//! [`BumpArena`]. The inference hot path performs **no heap allocation**:
//! topological-sort scratch space, activation buffers, and all operator
//! outputs live in the arena; only the final result tensors are copied into
//! owned [`TensorVec`] values after execution.
//!
//! # Architecture
//!
//! - [`kernels`] — `no_std` reference kernels for operators the `Backend`
//!   HAL does not cover (broadcasting binary ops, pooling, batch norm,
//!   transpose, concat, data movement).
//! - [`argmax`] — index of the largest element (top-1 class helper).
//! - [`RuntimeError`] — planning/execution errors.
//! - `execute` / `execute_graph` (feature `std`) — planning, input binding,
//!   arena layout, and the operator dispatch table. Requires `std` because
//!   the graph IR itself (`tpt-infer-graph`) is only available with its
//!   `std` feature.
//!
//! # Example (`no_std` kernels)
//!
//! ```
//! use tpt_infer_runtime::{argmax, kernels::{binary, BinaryOp}};
//!
//! let a = [1.0f32, -2.0, 3.0];
//! let b = [10.0f32; 3];
//! let mut out = [0.0f32; 3];
//! binary(BinaryOp::Add, &a, &[3], &b, &[3], &mut out, &[3]).unwrap();
//! assert_eq!(out, [11.0, 8.0, 13.0]);
//! assert_eq!(argmax(&a), 2);
//! ```
//!
//! # Graph execution
//!
//! ```ignore
//! use tpt_infer_core::BumpArena;
//! use tpt_infer_ops::dispatch::select_backend;
//! use tpt_infer_runtime::{execute_graph, required_arena_bytes};
//!
//! let graph = /* a ComputationGraph with marked outputs */;
//! let input = [0.0f32; 1 * 3 * 224 * 224];
//! let mut mem = vec![0u8; required_arena_bytes(&graph)];
//! let mut arena = BumpArena::new(&mut mem);
//! let backend = select_backend();
//! let outputs = execute_graph(&graph, &[&input], &mut arena, &backend).unwrap();
//! // outputs[i] is a TensorVec<f32> with the shape declared by the graph.
//! ```
//! (Compile with `--features std` — graph execution is unavailable on the
//! default feature set.)
//!
//! # Features
//!
//! - `default = []` — `no_std`, kernels/`argmax`/errors only.
//! - `alloc` — enables `tpt-infer-core/alloc` (heap tensors in `core`).
//! - `std` — `alloc` plus `tpt-infer-graph/std` and `tpt-infer-ops/std`;
//!   unlocks `execute` / `execute_graph`.
//!
//! # `no_std`
//!
//! The crate is `#![cfg_attr(not(any(test, feature = "alloc")), no_std)]`.
//! `cargo build -p tpt-infer-runtime --target thumbv7m-none-eabi` (default
//! features) verifies the bare-metal build. On bare-metal targets the graph
//! IR is unavailable, so graph execution currently requires `std`; the
//! kernels are `no_std` and ready for a future plan-based entry point.
//!
//! # Compile crate status
//!
//! `tpt-infer-compile` is currently a placeholder: its `lib.rs` re-exports
//! `CompiledModel` / `aot_compile` from an empty `model` module and does not
//! compile. This crate therefore does **not** depend on it and executes
//! `ComputationGraph` IR directly via [`execute`] / [`execute_graph`].
//! When a real `CompiledModel` with embedded IR lands, an `execute` overload
//! accepting it will be added alongside the graph entry points.
//!
//! [`ComputationGraph`]: tpt_infer_graph::ComputationGraph
//! [`Backend`]: tpt_infer_ops::Backend
//! [`BumpArena`]: tpt_infer_core::BumpArena
//! [`TensorVec`]: tpt_infer_core::TensorVec

#![cfg_attr(not(any(test, feature = "alloc")), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod argmax;
pub mod error;
pub mod kernels;

#[cfg(any(feature = "std", test))]
mod exec;

#[cfg(test)]
mod tests;

pub use argmax::argmax;
pub use error::RuntimeError;
#[cfg(any(feature = "std", test))]
pub use exec::{execute, execute_graph, required_arena_bytes};
