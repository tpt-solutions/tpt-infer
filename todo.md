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
- [ ] Verify `cargo build --workspace` passes (currently fails: `tpt-infer-graph` skeleton is missing `GraphBuilder`/`ComputationGraph`/`Edge`/`Node`/`Operator` definitions, and its `alloc` cfg feature isn't declared)

---

## Phase 1 · Months 1–3: Core + Ops + Graph

### tpt-infer-core (no_std)

- [ ] `Tensor<T, const N: usize>` struct with const-generic shape array
- [ ] `Shape` type alias and shape arithmetic helpers
- [ ] `DType` enum (f32, f16, i8, i4)
- [ ] `TensorView` trait (read-only slice access + shape)
- [ ] `TensorMut` trait (mutable slice access)
- [ ] `BumpArena` no_std bump allocator (pre-allocated byte slice)
- [ ] `#![no_std]` with `alloc` feature for heap-backed tensors
- [ ] Unit tests: tensor shape construction
- [ ] Unit tests: BumpArena alloc + reset
- [ ] Build with `--target thumbv7m-none-eabi` to confirm no_std

### tpt-infer-ops

- [ ] `Backend` HAL trait (`matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`, `sigmoid`, `gelu`)
- [ ] Naive CPU reference backend (correctness baseline)
- [ ] x86_64 AVX2 SIMD backend (`std::arch::x86_64`)
- [ ] x86_64 AVX-512 SIMD backend (feature-gated)
- [ ] ARM NEON/SVE backend (`std::arch::aarch64`)
- [ ] WASM SIMD backend (`std::arch::wasm32`)
- [ ] WebGPU backend via `wgpu` (feature = "webgpu")
- [ ] Runtime dispatch: `#[cfg(target_feature)]` + `#[cfg(target_arch)]`
- [ ] `criterion` benchmark: matmul at [1,784]×[784,10] and [1,3,224,224] conv shapes
- [ ] Unit tests: all backends produce identical results to naive reference

### tpt-infer-graph

- [ ] `Operator` enum (MatMul, Conv2d, Add, Relu, Softmax, Reshape, Flatten, …)
- [ ] `Node` struct (operator + input/output shape metadata)
- [ ] `Edge` type carrying tensor shape at compile time
- [ ] `GraphBuilder` with const-generic type state threading shapes
- [ ] Compile-time shape mismatch = type error (not runtime panic)
- [ ] `ComputationGraph` with topological sort
- [ ] Unit tests: valid graph builds successfully
- [ ] Unit tests: shape mismatch caught at compile time (compile-fail test)
- [ ] Integration test: build 2-layer MLP graph, verify node order and shapes

---

## Phase 2 · Months 4–5: ONNX + Quantize

### tpt-infer-onnx

- [ ] Add `proto/onnx.proto3` (official ONNX schema)
- [ ] `build.rs` using `prost-build` to generate Rust structs
- [ ] `load(path: &Path) -> Result<ComputationGraph>`: ModelProto → tpt-infer-graph
- [ ] Operator registry: map opset-17 ONNX op strings → `Operator` variants
- [ ] Handle ONNX initializers (pre-trained weight tensors)
- [ ] Shape inference pass over parsed graph
- [ ] Support dynamic dimensions (batch size = -1)
- [ ] Integration test: load MobileNetV2.onnx, verify graph node count & shapes
- [ ] Integration test: load BERT-tiny.onnx, verify graph structure

### tpt-infer-quantize

- [ ] INT8 symmetric quantization (`scale = max(|x|) / 127`)
- [ ] INT8 asymmetric quantization (`scale` + `zero_point`)
- [ ] INT4 quantization with nibble packing (2 × i4 per byte)
- [ ] Per-tensor and per-channel variants for both INT8/INT4
- [ ] Dequantization kernels (INT8 → f32, INT4 → f32)
- [ ] Post-training quantization (PTQ) pipeline: walk graph, replace f32 nodes
- [ ] Unit tests: round-trip accuracy (max abs error < 0.01 for INT8)
- [ ] Benchmark: quantized matmul vs f32 matmul throughput

---

## Phase 3 · Months 6–8: Compile + Runtime + Vision

### tpt-infer-compile

- [ ] `aot_compile(graph: &ComputationGraph) -> Result<CompiledModel>`
- [ ] Walk graph and emit static Rust `fn execute(...)` using `quote` + `proc-macro2`
- [ ] Loop unrolling for shapes fully known at compile time
- [ ] Constant folding for initializer-only subgraphs
- [ ] FPGA/photonic mesh instruction emitter stub (tpt-crucible seed)
- [ ] Integration test: compile graph → write `.rs` → `rustc` it → run → check output
- [ ] Benchmark: generated code throughput vs interpreted runtime

### tpt-infer-runtime (no_std)

- [ ] `execute(model: &CompiledModel, input: &Tensor, arena: &mut BumpArena) -> Result<Tensor>`
- [ ] Operator dispatch table backed by `tpt-infer-ops` HAL
- [ ] Zero heap allocation during inference (all intermediates from arena)
- [ ] Intermediate tensor lifetime management within arena
- [ ] `#![no_std]` verification build (`thumbv7m-none-eabi`)
- [ ] Integration test: end-to-end MobileNetV2 inference, verify top-1 class
- [ ] Benchmark: inference latency (median + p99) for MobileNetV2 on CPU

### tpt-infer-vision

- [ ] `preprocess_image(bytes: &[u8], config: &PreprocessConfig) -> Result<Tensor<f32, [1, 3, 224, 224]>>`
- [ ] Bilinear resize
- [ ] Per-channel normalization (configurable mean/std)
- [ ] Letterbox with zero-padding
- [ ] CHW ↔ HWC layout conversion
- [ ] `ImageSource` trait for tpt-kinetix integration boundary
- [ ] Unit tests: resize output dimensions correct
- [ ] Unit tests: normalization values match reference numpy output
- [ ] Integration test: preprocess → infer end-to-end on sample image

---

## tpt-infer (Facade Crate)

- [ ] `pub mod prelude` re-exporting all public APIs
- [ ] Feature flags: `default = ["ops-cpu"]`, optional `"ops-webgpu"`, `"vision"`, `"quantize"`
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
