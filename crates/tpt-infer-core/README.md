# tpt-infer-core

**Const-generic tensor types, `no_std` allocators, and dtype definitions for tpt-infer.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

`tpt-infer-core` is the foundation every other crate in the workspace builds on. It has
no dependencies of its own, is `#![no_std]` by default, and defines the shared vocabulary
(tensor storage, shapes, dtypes, and a bump allocator) that `tpt-infer-ops`,
`tpt-infer-graph`, `tpt-infer-runtime`, and the rest of the workspace all speak.

## What's in here

- [`Tensor<T, const N: usize>`] — a tensor with a compile-time fixed rank `N` and a
  runtime `[usize; N]` shape, backed by contiguous row-major storage.
- [`TensorVec<T>`] — a heap-backed, dynamic-rank tensor (requires the `alloc` feature).
- [`DType`] — the element data types tpt-infer understands: `F32`, `F16`, `I8`, `I4`,
  `I32`, `I64`, `U8`, `Bool`.
- [`BumpArena`] — a `no_std` bump (arena) allocator over a caller-provided byte slice,
  used by `tpt-infer-runtime` to execute inference with zero heap allocation.
- [`TensorView`] / [`TensorMut`] — read-only / mutable slice-and-shape access traits.
- Shape helpers ([`Shape`], [`strides`], [`num_elements`], [`ravel_index`],
  [`unravel_index`], [`broadcast`], [`to_dyn`]/[`from_dyn`]) for working with row-major
  layouts up to [`MAX_RANK`] dimensions.

## Usage

```rust
use tpt_infer_core::{BumpArena, DType, Tensor};

// A rank-2 tensor with owned storage (requires the `alloc` feature).
let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
assert_eq!(t.shape(), [2, 2]);
assert_eq!(t.dtype(), DType::F32);

// A bump allocator over a fixed byte buffer — no heap allocation at all.
let mut buf = [0u8; 1024];
let mut arena = BumpArena::new(&mut buf);
let scratch = arena.alloc_bytes(100, 1).unwrap();
assert_eq!(scratch.len(), 100);
arena.reset();
```

## Feature flags

| Feature | Default | Enables |
|---|---|---|
| `alloc` | off | Heap-backed tensor storage (`Tensor::new`/`from_vec`, `TensorVec`) via `extern crate alloc` |

Without `alloc`, the crate still exposes `DType`, `Shape` arithmetic, `BumpArena`, and
the `TensorView`/`TensorMut` traits — everything needed to build `no_std`, arena-backed
tensor storage on embedded targets.

## `no_std`

`tpt-infer-core` is `#![no_std]` (only `#![cfg_attr(not(test), no_std)]` for local unit
tests). It has zero required dependencies. Sibling crates that need heap-backed tensors
enable this crate's `alloc` feature rather than pulling in `std` directly.

## Relationship to the rest of the workspace

Every other crate in the workspace depends on `tpt-infer-core` directly or transitively:
`tpt-infer-ops` operates on its `f32` slices, `tpt-infer-graph`/`tpt-infer-onnx` use
`Shape`/`MAX_RANK` for node metadata, `tpt-infer-runtime` executes into `TensorVec`
buffers allocated from a `BumpArena`, and `tpt-infer-vision`/`tpt-infer-quantize` produce
or consume `Tensor` values. See the [workspace README](../../README.md) for the full
architecture diagram.

[`Tensor<T, const N: usize>`]: https://docs.rs/tpt-infer-core
[`TensorVec<T>`]: https://docs.rs/tpt-infer-core
[`DType`]: https://docs.rs/tpt-infer-core
[`BumpArena`]: https://docs.rs/tpt-infer-core
[`TensorView`]: https://docs.rs/tpt-infer-core
[`TensorMut`]: https://docs.rs/tpt-infer-core
[`Shape`]: https://docs.rs/tpt-infer-core
[`strides`]: https://docs.rs/tpt-infer-core
[`num_elements`]: https://docs.rs/tpt-infer-core
[`ravel_index`]: https://docs.rs/tpt-infer-core
[`unravel_index`]: https://docs.rs/tpt-infer-core
[`broadcast`]: https://docs.rs/tpt-infer-core
[`to_dyn`]: https://docs.rs/tpt-infer-core
[`from_dyn`]: https://docs.rs/tpt-infer-core
[`MAX_RANK`]: https://docs.rs/tpt-infer-core
