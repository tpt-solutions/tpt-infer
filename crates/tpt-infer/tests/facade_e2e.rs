//! End-to-end integration test for the `tpt-infer` facade.
//!
//! Builds a small synthetic 2-layer MLP graph via the facade's re-exported
//! [`GraphBuilder`], then exercises both target execution paths reachable
//! through `tpt_infer::prelude`:
//!
//! 1. **AOT compile** ([`aot_compile`]) — confirm the facade's `compile`
//!    re-export produces self-contained Rust source for the graph.
//! 2. **Interpreted execution** ([`execute`]) — actually run the graph
//!    through the facade's `runtime` re-export and check the output shape,
//!    finiteness, and a sane [`argmax`] top-1 class.
//!
//! There is no real `.onnx` file in this repository (see
//! `tpt-infer-runtime`'s own `mobilenet_v2_end_to_end` fixture for the same
//! rationale), so the graph is built programmatically rather than loaded
//! from disk — `tpt_infer::prelude::load` is covered by `tpt-infer-onnx`'s
//! own test suite.

use tpt_infer::prelude::*;

/// Builds `x [1,4] -> MatMul(w1 [4,8]) -> Relu -> MatMul(w2 [8,3])`.
fn mlp_graph() -> ComputationGraph {
    let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
    let w1 = b
        .add_initializer::<Sh<4, 8>>(
            "w1",
            vec![
                0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1, -0.15, 0.2,
                0.3, -0.1, 0.05, -0.25, 0.1, 0.2, -0.3, 0.15, 0.4, -0.05, 0.1, -0.2, 0.05, 0.3,
                -0.1, 0.2, -0.15, 0.25,
            ],
        )
        .unwrap();
    let mut b = b.matmul(w1).relu();
    let w2 = b
        .add_initializer::<Sh<8, 3>>(
            "w2",
            vec![
                0.2, -0.4, 0.1, 0.3, -0.25, 0.5, -0.1, 0.2, 0.05, 0.15, -0.3, 0.4, 0.2, -0.1, 0.3,
                -0.05, 0.25, -0.2, 0.1, -0.15, 0.35, -0.4, 0.2, 0.1,
            ],
        )
        .unwrap();
    let b = b.matmul(w2);
    b.into_graph()
}

#[test]
fn aot_compile_produces_runnable_source_description() {
    let graph = mlp_graph();
    let model = aot_compile(&graph).expect("mlp graph is within aot_compile's supported subset");

    assert_eq!(model.input_shape(), &[1, 4]);
    assert_eq!(model.output_shape(), &[1, 3]);
    assert!(model.source().contains("pub fn execute"));
    // Both weight `Input` nodes are constant-folded (they carry initializers).
    assert_eq!(model.folded_node_count(), 2);
}

#[test]
fn facade_builds_compiles_and_executes_end_to_end() {
    let graph = mlp_graph();

    // Compile path (feature `compile`, re-exported by the facade).
    let model = aot_compile(&graph).unwrap();
    assert!(model.source().contains("pub fn execute"));

    // Execute path (feature `runtime` + `ops-cpu`, re-exported by the facade).
    let input = [1.0f32, -2.0, 0.5, 3.0];
    let mut mem = vec![0u8; required_arena_bytes(&graph) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let backend = select_backend();

    let output = execute(&graph, &input, &mut arena, &backend).expect("mlp graph executes");
    assert_eq!(output.dims(), &[1, 3][..]);
    assert!(output.as_slice().iter().all(|v| v.is_finite()));

    let top1 = argmax(output.as_slice());
    assert!(top1 < 3, "top-1 class {top1} must be in 0..3");

    // Determinism: re-running on a reset arena gives the identical result.
    arena.reset();
    let output2 = execute(&graph, &input, &mut arena, &backend).unwrap();
    assert_eq!(output.as_slice(), output2.as_slice());
    assert_eq!(argmax(output2.as_slice()), top1);
}

/// Exercises the `vision` re-exports end-to-end: preprocess a tiny synthetic
/// RGB image into a tensor, then feed its flattened data through a graph
/// shaped to match. Only compiled when the `vision` feature is enabled
/// (`cargo test -p tpt-infer --features vision` or `--all-features`); it is
/// not part of the facade's default feature set.
#[cfg(feature = "vision")]
#[test]
fn facade_vision_preprocess_feeds_graph_execution() {
    // 2x2 RGB image (red, green, blue, white) preprocessed to a 1x3x2x2 CHW
    // tensor, i.e. 12 f32 elements — matches the graph's [1, 12] input.
    let rgb: [u8; 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
    let mut config = PreprocessConfig::imagenet();
    config.width = 2;
    config.height = 2;
    config.mean = [0.0; 3];
    config.std = [1.0; 3];
    let tensor = preprocess_raw_rgb(&rgb, 2, 2, &config).expect("preprocess succeeds");
    assert_eq!(tensor.shape(), [1, 3, 2, 2]);

    // A minimal 1-layer graph over the flattened 12-element pixel tensor.
    let mut b = GraphBuilder::<Sh<1, 12>>::input("x");
    let w = b.add_initializer::<Sh<12, 4>>("w", vec![0.05; 48]).unwrap();
    let graph = b.matmul(w).into_graph();

    let input = tensor.as_slice();
    let mut mem = vec![0u8; required_arena_bytes(&graph) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let backend = select_backend();
    let output = execute(&graph, input, &mut arena, &backend).expect("graph executes");

    assert_eq!(output.dims(), &[1, 4][..]);
    assert!(output.as_slice().iter().all(|v| v.is_finite()));
    let top1 = argmax(output.as_slice());
    assert!(top1 < 4);
}

/// Exercises the `quantize` re-export end-to-end over the same MLP graph.
/// Only compiled when the `quantize` feature is enabled; not part of the
/// facade's default feature set.
#[cfg(feature = "quantize")]
#[test]
fn facade_ptq_quantizes_mlp_weights() {
    let graph = mlp_graph();
    let quantized = ptq(&graph, PtqOptions::default()).expect("ptq succeeds on mlp graph");
    // Both `w1` and `w2` are MatMul weight initializers, so PTQ quantizes both.
    assert!(!quantized.is_empty());
    assert_eq!(quantized.len(), 2);
    assert!(quantized.weights.contains_key("w1"));
    assert!(quantized.weights.contains_key("w2"));
}
