# tpt-infer — Task Checklist

> Edge AI & Local Inference Runtime · TPT Solutions · MIT / Apache 2.0

---

## Setup & Infrastructure

- [x] Initialize Cargo workspace (`Cargo.toml`)
- [x] Create crate skeletons for all 8 crates + facade
- [x] `LICENSE-MIT`
- [x] `LICENSE-APACHE`
- [x] `README.md`
- [x] `.gitignore`
- [x] `.github/workflows/ci.yml` (fmt + clippy + test, 3 platforms)
- [x] Verify `cargo build --workspace` passes (fixed: implemented `tpt-infer-compile`; `cargo test --workspace` also passes clean)

---

## Phase 1 · Months 1–3: Core + Ops + Graph

### tpt-infer-core (no_std)

- [x] `Tensor<T, const N: usize>` struct with const-generic shape array
- [x] `Shape` type alias and shape arithmetic helpers
- [x] `DType` enum (f32, f16, i8, i4)
- [x] `TensorView` trait (read-only slice access + shape)
- [x] `TensorMut` trait (mutable slice access)
- [x] `BumpArena` no_std bump allocator (pre-allocated byte slice)
- [x] `#![no_std]` with `alloc` feature for heap-backed tensors
- [x] Unit tests: tensor shape construction
- [x] Unit tests: BumpArena alloc + reset
- [ ] Build with `--target thumbv7m-none-eabi` to confirm no_std

### tpt-infer-ops

- [x] `Backend` HAL trait (`matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`, `sigmoid`, `gelu`)
- [x] Naive CPU reference backend (correctness baseline)
- [x] x86_64 AVX2 SIMD backend (`std::arch::x86_64`)
- [x] x86_64 AVX-512 SIMD backend (feature-gated)
- [x] ARM NEON/SVE backend (`std::arch::aarch64`)
- [x] WASM SIMD backend (`std::arch::wasm32`)
- [ ] WebGPU backend via `wgpu` (feature = "webgpu") — stub only, every op returns `Unsupported`
- [x] Runtime dispatch: `#[cfg(target_feature)]` + `#[cfg(target_arch)]`
- [x] `criterion` benchmark: matmul at [1,784]×[784,10] and [1,3,224,224] conv shapes
- [x] Unit tests: all backends produce identical results to naive reference

### tpt-infer-graph

- [x] `Operator` enum (MatMul, Conv2d, Add, Relu, Softmax, Reshape, Flatten, …)
- [x] `Node` struct (operator + input/output shape metadata)
- [x] `Edge` type carrying tensor shape at compile time
- [x] `GraphBuilder` with const-generic type state threading shapes
- [x] Compile-time shape mismatch = type error (not runtime panic)
- [x] `ComputationGraph` with topological sort
- [x] Unit tests: valid graph builds successfully
- [x] Unit tests: shape mismatch caught at compile time (compile-fail test, via `trybuild`)
- [x] Integration test: build 2-layer MLP graph, verify node order and shapes

---

## Phase 2 · Months 4–5: ONNX + Quantize

### tpt-infer-onnx

- [x] Add `proto/onnx.proto3` (official ONNX schema)
- [x] `build.rs` using `prost-build` to generate Rust structs
- [x] `load(path: &Path) -> Result<ComputationGraph>`: ModelProto → tpt-infer-graph
- [x] Operator registry: map opset-17 ONNX op strings → `Operator` variants
- [x] Handle ONNX initializers (pre-trained weight tensors)
- [x] Shape inference pass over parsed graph
- [x] Support dynamic dimensions (batch size = -1)
- [ ] Integration test: load MobileNetV2.onnx, verify graph node count & shapes — only synthetic/hand-built `ModelProto` fixtures exist; no real `.onnx` file in repo
- [ ] Integration test: load BERT-tiny.onnx, verify graph structure — same gap as above

### tpt-infer-quantize

- [x] INT8 symmetric quantization (`scale = max(|x|) / 127`)
- [x] INT8 asymmetric quantization (`scale` + `zero_point`)
- [x] INT4 quantization with nibble packing (2 × i4 per byte)
- [x] Per-tensor and per-channel variants for both INT8/INT4
- [x] Dequantization kernels (INT8 → f32, INT4 → f32)
- [x] Post-training quantization (PTQ) pipeline: walk graph, replace f32 nodes
- [x] Unit tests: round-trip accuracy (max abs error < 0.01 for INT8)
- [x] Benchmark: quantized matmul vs f32 matmul throughput

---

## Phase 3 · Months 6–8: Compile + Runtime + Vision

### tpt-infer-compile

- [x] `aot_compile(graph: &ComputationGraph) -> Result<CompiledModel>`
- [x] Walk graph and emit static Rust `fn execute(...)` using `quote` + `proc-macro2` (covers MatMul, Add/Sub/Mul/Div, Relu, Sigmoid, Reshape/Flatten; other ops cleanly error via `CompileError::UnsupportedOperator`)
- [x] Loop unrolling for shapes fully known at compile time (dims ≤ 8 fully unrolled; larger dims emit a const-bound `for` loop for LLVM to unroll)
- [x] Constant folding for initializer-only subgraphs
- [x] FPGA/photonic mesh instruction emitter stub (tpt-crucible seed) — `fpga_stub.rs`, intentionally unimplemented
- [x] Integration test: compile graph → write `.rs` → `rustc` it → run → check output
- [x] Benchmark: generated code throughput vs interpreted runtime

### tpt-infer-runtime (no_std)

- [x] `execute(model: &CompiledModel, input: &Tensor, arena: &mut BumpArena) -> Result<Tensor>`
- [x] Operator dispatch table backed by `tpt-infer-ops` HAL
- [x] Zero heap allocation during inference (all intermediates from arena)
- [x] Intermediate tensor lifetime management within arena
- [x] `#![no_std]` verification build (`thumbv7m-none-eabi`)
- [ ] Integration test: end-to-end MobileNetV2 inference, verify top-1 class — current test uses a synthetic structural fixture, not a real `.onnx` model (real model gitignored/unavailable in CI)
- [x] Benchmark: inference latency (median + p99) for MobileNetV2 on CPU

### tpt-infer-vision

- [x] `preprocess_image(bytes: &[u8], config: &PreprocessConfig) -> Result<Tensor<f32, [1, 3, 224, 224]>>`
- [x] Bilinear resize
- [x] Per-channel normalization (configurable mean/std)
- [x] Letterbox with zero-padding
- [x] CHW ↔ HWC layout conversion
- [x] `ImageSource` trait for tpt-kinetix integration boundary
- [x] Unit tests: resize output dimensions correct
- [x] Unit tests: normalization values match reference numpy output
- [x] Integration test: preprocess → infer end-to-end on sample image

---

## tpt-infer (Facade Crate)

- [ ] `pub mod prelude` re-exporting all public APIs — `prelude.rs` is currently `// placeholder`
- [ ] Feature flags: `default = ["ops-cpu"]`, optional `"ops-webgpu"`, `"vision"`, `"quantize"` — flags are declared in `Cargo.toml`, but default features include `compile`, which pulls in the broken `tpt-infer-compile` stub; nothing re-exports the gated crates yet
- [ ] Integration test: full Target API from spec (load → compile → preprocess → execute → argmax)
- [ ] `examples/mobilenet.rs` demonstrating end-to-end usage
- [ ] Rustdoc examples in `lib.rs`

---

## Documentation & Publishing

- [ ] Crate-level `#![doc]` for all 8 crates
- [ ] `README.md` with architecture diagram, quick-start, and feature matrix
- [ ] `docs.rs` metadata in each `Cargo.toml`
- [ ] Verify `cargo doc --workspace --no-deps` builds clean
- [ ] Set up `CHANGELOG.md`
- [ ] Tag `v0.1.0` pre-release on GitHub
- [ ] Publish to crates.io (when ready)
