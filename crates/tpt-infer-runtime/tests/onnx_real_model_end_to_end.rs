//! Real (non-synthetic) counterpart to `onnx_mobilenet_end_to_end.rs`:
//! loads the genuine MNIST `.onnx` export cached by
//! `tpt-infer-onnx/tests/real_model_fixtures.rs` (see that file and
//! `tpt-infer-onnx/tests/fixtures/real/README.md`) and executes it through
//! the real interpreted runtime.
//!
//! Skipped (not failed) if the fixture isn't cached yet, keeping a plain
//! `cargo test` offline/hermetic — run `TPT_INFER_FETCH_FIXTURES=1 cargo
//! test -p tpt-infer-onnx --test real_model_fixtures` first, or rely on the
//! `real-model-validation` CI job, which runs both in order.

use std::path::PathBuf;

use tpt_infer_core::BumpArena;
use tpt_infer_ops::NaiveBackend;
use tpt_infer_runtime::{argmax, execute, required_arena_bytes};

/// Path to the cached fixture, shared (not duplicated) with `tpt-infer-onnx`.
fn real_mnist_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tpt-infer-onnx/tests/fixtures/real/mnist-12.onnx")
}

#[test]
fn real_mnist_onnx_executes_end_to_end() {
    let path = real_mnist_fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: {} not cached; run `TPT_INFER_FETCH_FIXTURES=1 cargo test \
             -p tpt-infer-onnx --test real_model_fixtures` first",
            path.display()
        );
        return;
    }

    // ONNX file on disk -> ComputationGraph, via the real file-loading API.
    let g = tpt_infer_onnx::load(&path).expect("real ONNX file load must succeed");
    assert_eq!(g.nodes().len(), 21);
    assert_eq!(g.outputs().len(), 1);

    // Deterministic pseudo-random 28x28 grayscale "digit".
    let input: Vec<f32> = (0..28 * 28)
        .map(|i| ((i % 17) as f32 / 17.0) - 0.5)
        .collect();

    let mut mem = vec![0u8; required_arena_bytes(&g) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &NaiveBackend::new())
        .expect("real MNIST ONNX export must execute end to end");

    assert_eq!(out.dims(), &[1, 10][..]);
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
