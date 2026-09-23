# Changelog

All notable changes to `tpt-infer-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This crate is pre-release; versions have not yet been published to crates.io.

## [0.1.0] - Unreleased

### Added

- `Tensor<T, const N: usize>`: const-generic-rank tensor with row-major storage,
  `shape()`, `dtype()`, `as_slice()`, `new()`, and (with `alloc`) `from_vec()`.
- `TensorVec<T>` (feature `alloc`): heap-backed, dynamic-rank tensor for code paths
  that don't know their rank at compile time.
- `DType` enum covering `F32`, `F16` (stored as `u16` bit pattern), `I8`, `I4`
  (packed two-per-byte), `I32`, `I64`, `U8`, and `Bool`, with `size_bits()`/`size_bytes()`.
- `BumpArena`: a `no_std` bump allocator over a caller-supplied `&mut [u8]`, with
  `alloc_bytes`, `used`, `capacity`, `remaining`, and `reset`.
- `TensorView` / `TensorMut` traits for shape-and-slice access independent of storage.
- Shape utilities: `Shape`, `MAX_RANK`, `strides`, `num_elements`, `ravel_index`,
  `unravel_index`, `broadcast`, `to_dyn`/`from_dyn`.
- `TensorError` covering allocation and shape-construction failures.
- `#![no_std]` by default, with an `alloc` feature for heap-backed storage.
