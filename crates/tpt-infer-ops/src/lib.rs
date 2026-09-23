//! Hardware abstraction layer and SIMD-accelerated ML operators.
//!
//! This crate provides the [`Backend`] trait — a small operator HAL covering
//! `matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`, `sigmoid`, and
//! `gelu` — plus several implementations:
//!
//! | Backend | Module | Availability |
//! |---------|--------|--------------|
//! | Naive scalar reference | `naive` | always |
//! | x86_64 AVX2 | `avx2` | `target_arch = "x86_64"`, runtime-detected with `std` |
//! | x86_64 AVX-512 | `avx512` | feature `avx512` + CPU support |
//! | aarch64 NEON | `neon` | `target_arch = "aarch64"` |
//! | wasm32 SIMD128 | `wasm` | `target_arch = "wasm32"` + `target_feature = "simd128"` |
//! | WebGPU | `webgpu` | feature `webgpu`; GPU compute via `wgpu`, `conv2d` still [`OpError::Unsupported`] |
//!
//! All operators write into caller-provided output slices, so hot paths can
//! run without allocating. Use [`select_backend`] to pick the best backend
//! available on the current machine (enum dispatch, `no_std`-friendly), or
//! construct a concrete backend directly.
//!
//! # Features
//!
//! - `std` (default): runtime CPU feature detection and native `f32::exp`/`tanh`.
//! - `avx512`: compile the AVX-512 backend (it is still runtime-detected).
//! - `webgpu`: compile the WebGPU backend (`WebGpuBackend::new` acquires a
//!   real `wgpu::Device`; it is not part of [`select_backend`]'s enum
//!   dispatch since construction is async/fallible).
//!
//! The crate is `no_std` unless the `std` feature is enabled.
//!
//! # Example
//!
//! ```
//! use tpt_infer_ops::{select_backend, Backend};
//!
//! let backend = select_backend();
//! let a = [1.0f32, 2.0, 3.0, 4.0];
//! let b = [5.0f32, 6.0, 7.0, 8.0];
//! let mut out = [0.0f32; 4];
//! backend.elementwise_add(&a, &b, &mut out).unwrap();
//! assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
//! assert!(!backend.name().is_empty());
//! ```

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![warn(missing_docs)]

pub mod backend;
pub mod dispatch;
mod math;
pub mod naive;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(feature = "webgpu")]
pub mod webgpu;

#[cfg(target_arch = "x86_64")]
pub use avx2::Avx2Backend;
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub use avx512::Avx512Backend;
pub use backend::{Backend, Conv2dOptions, OpError};
pub use dispatch::{select_backend, selected_backend, AnyBackend};
pub use naive::NaiveBackend;
#[cfg(target_arch = "aarch64")]
pub use neon::NeonBackend;
#[cfg(target_arch = "wasm32")]
pub use wasm::WasmSimdBackend;
#[cfg(feature = "webgpu")]
pub use webgpu::WebGpuBackend;
