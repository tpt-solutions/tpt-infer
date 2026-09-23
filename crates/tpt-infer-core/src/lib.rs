//! Const-generic tensor types, dtype definitions, and a `no_std` bump allocator.
//!
//! This crate is the foundation of the tpt-infer workspace. It is `#![no_std]`
//! by default and only requires the `alloc` feature for heap-backed tensors.
//!
//! # Example
//!
//! ```
//! use tpt_infer_core::{Tensor, DType};
//! let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
//! assert_eq!(t.shape(), [2, 2]);
//! assert_eq!(t.dtype(), DType::F32);
//! ```

#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod arena;
mod dtype;
mod error;
mod shape;
mod tensor;
mod traits;

pub use arena::BumpArena;
pub use dtype::DType;
pub use error::TensorError;
pub use shape::{broadcast, from_dyn, num_elements, ravel_index, strides, to_dyn, unravel_index, Shape, MAX_RANK};
pub use tensor::{dtype_of, DTypeOf, Tensor};
pub use traits::{TensorMut, TensorView};

#[cfg(feature = "alloc")]
pub use tensor::TensorVec;
