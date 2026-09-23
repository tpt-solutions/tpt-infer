//! Closes the "Integration test: end-to-end MobileNetV2 inference" gap for
//! real: unlike `mobilenet_v2_end_to_end` in `src/tests.rs` (which builds a
//! MobileNetV2-*shaped* graph directly via `tpt_infer_graph`, bypassing
//! ONNX loading entirely), this test genuinely goes
//! ONNX file on disk -> `tpt_infer_onnx::load(path)` -> `ComputationGraph`
//! -> `tpt_infer_runtime::execute` -> output tensor.
//!
//! The `.onnx` file itself is a small, structurally-representative
//! MobileNetV2-style fixture (conv/batchnorm/relu stem, an
//! inverted-residual-ish block with a skip connection, global average
//! pool, MatMul+Add classification head) — not the real pretrained model.
//! See `crates/tpt-infer-onnx/tests/common/mod.rs` for exactly what it
//! stands in for.

use std::path::PathBuf;

use tpt_infer_core::BumpArena;
use tpt_infer_ops::NaiveBackend;
use tpt_infer_runtime::{argmax, execute, required_arena_bytes};

/// Path to the fixture `.onnx` file, shared (not duplicated) with the
/// `tpt-infer-onnx` crate's own integration tests / fixture generator.
fn mobilenet_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tpt-infer-onnx/tests/fixtures/mobilenet_v2_style.onnx")
}

#[test]
fn onnx_file_to_graph_to_execution_end_to_end() {
    let path = mobilenet_fixture_path();
    assert!(
        path.exists(),
        "fixture missing: {} (regenerate via `cargo test -p tpt-infer-onnx \
         --test generate_fixtures -- --ignored`)",
        path.display()
    );

    // ONNX file on disk -> ComputationGraph, via the real file-loading API.
    let g = tpt_infer_onnx::load(&path).expect("real ONNX file load must succeed");
    assert!(g.nodes().len() >= 20);
    assert_eq!(g.outputs().len(), 1);

    // Graph -> execution, via the real runtime execution API.
    let input_len: usize = 3 * 16 * 16;
    let input: Vec<f32> = (0..input_len)
        .map(|i| ((i % 13) as f32 / 13.0) - 0.5)
        .collect();

    let mut mem = vec![0u8; required_arena_bytes(&g) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &NaiveBackend::new())
        .expect("ONNX-loaded MobileNetV2-style graph must execute");

    // 10-class logits head (scaled down from the real model's 1000).
    assert_eq!(out.dims(), &[1, 10][..]);
    assert_eq!(out.numel(), 10);
    assert!(
        out.as_slice().iter().all(|v| v.is_finite()),
        "all logits must be finite"
    );
    let top1 = argmax(out.as_slice());
    assert!(top1 < 10, "top-1 class {top1} must be in 0..10");

    // Determinism: a second run on the reset arena is bit-identical.
    arena.reset();
    let out2 = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap();
    assert_eq!(out.as_slice(), out2.as_slice());
    assert_eq!(argmax(out2.as_slice()), top1);
}
