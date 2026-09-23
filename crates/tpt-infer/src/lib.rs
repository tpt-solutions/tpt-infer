//! `tpt-infer` — facade crate unifying the tpt-infer edge-inference workspace.
//!
//! This crate does not implement any inference logic itself. It re-exports
//! the public API of the workspace's individual crates through
//! [`prelude`] so downstream users depend on a single crate (and feature
//! set) instead of wiring up `tpt-infer-core`, `tpt-infer-graph`,
//! `tpt-infer-onnx`, `tpt-infer-compile`, `tpt-infer-runtime`,
//! `tpt-infer-vision`, and `tpt-infer-quantize` by hand.
//!
//! # Pipeline
//!
//! A typical flow through the re-exported API looks like:
//!
//! 1. **Load or build** a [`ComputationGraph`](prelude::ComputationGraph) —
//!    either `prelude::load` an ONNX file (feature `onnx`) or assemble one
//!    programmatically with [`prelude::GraphBuilder`] (always available).
//! 2. **Preprocess** image inputs with `prelude::preprocess_image` /
//!    `prelude::preprocess_raw_rgb` (feature `vision`, opt-in).
//! 3. **Compile ahead-of-time** with [`prelude::aot_compile`] (feature
//!    `compile`) to get self-contained generated Rust source, or **execute**
//!    the graph directly with [`prelude::execute`] / [`prelude::execute_graph`]
//!    (feature `runtime`) against a [`prelude::BumpArena`] and a
//!    [`prelude::Backend`] (feature `ops-cpu`).
//! 4. **Post-process** the output tensor with [`prelude::argmax`] for
//!    top-1 classification.
//!
//! Optionally, `prelude::ptq` (feature `quantize`, opt-in) quantizes a
//! graph's weights ahead of either path.
//!
//! # Feature flags
//!
//! | Feature | Default | Unlocks |
//! |---------|---------|---------|
//! | `ops-cpu` | yes | [`prelude::Backend`], [`prelude::NaiveBackend`], [`prelude::select_backend`] |
//! | `ops-webgpu` | no | WebGPU backend (real WGSL compute kernels: matmul/conv2d/elementwise/relu/sigmoid/gelu/softmax; implies `ops-cpu`) |
//! | `onnx` | yes | `prelude::load` — parse `.onnx` files into a [`prelude::ComputationGraph`] |
//! | `compile` | yes | [`prelude::aot_compile`] — graph-to-Rust AOT codegen |
//! | `runtime` | yes | [`prelude::execute`] / [`prelude::execute_graph`] / [`prelude::argmax`] — interpreted graph execution |
//! | `vision` | no | `prelude::preprocess_image` / `prelude::preprocess_raw_rgb` — image preprocessing |
//! | `quantize` | no | `prelude::ptq` — post-training INT8/INT4 quantization |
//!
//! `default = ["ops-cpu", "onnx", "compile", "runtime"]`: everything needed
//! to load an ONNX model and run it (interpreted or AOT-compiled) on the
//! CPU. `vision` and `quantize` are opt-in since not every deployment needs
//! image preprocessing or quantization.
//!
//! # Example
//!
//! No real `.onnx` file ships with this repository, so this example builds a
//! tiny 2-layer MLP graph programmatically with [`prelude::GraphBuilder`]
//! (mirroring the synthetic fixtures used by `tpt-infer-runtime`'s own
//! tests), executes it via the interpreted runtime, and reports the top-1
//! class:
//!
//! ```
//! use tpt_infer::prelude::*;
//!
//! // x [1,4] -> MatMul(w1 [4,3]) -> Relu -> MatMul(w2 [3,2])
//! let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
//! let w1 = b
//!     .add_initializer::<Sh<4, 3>>("w1", vec![0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1])
//!     .unwrap();
//! let mut b = b.matmul(w1).relu();
//! let w2 = b
//!     .add_initializer::<Sh<3, 2>>("w2", vec![0.2, -0.4, 0.1, 0.3, -0.25, 0.5])
//!     .unwrap();
//! let b = b.matmul(w2);
//! let graph = b.into_graph();
//!
//! let input = [1.0f32, -2.0, 0.5, 3.0];
//! let mut mem = vec![0u8; required_arena_bytes(&graph) + 256];
//! let mut arena = BumpArena::new(&mut mem);
//! let backend = select_backend();
//!
//! let output = execute(&graph, &input, &mut arena, &backend).unwrap();
//! assert_eq!(output.dims(), &[1, 2][..]);
//!
//! let top1 = argmax(output.as_slice());
//! assert!(top1 < 2, "top-1 class {top1} must be one of the 2 outputs");
//! ```

pub mod prelude;
