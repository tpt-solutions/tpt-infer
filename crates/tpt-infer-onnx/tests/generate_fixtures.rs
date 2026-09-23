//! Regenerates the committed `.onnx` fixture files under `tests/fixtures/`.
//!
//! The fixtures themselves are committed binaries (a few KB each) so that
//! `real_model_fixtures.rs` doesn't need to regenerate them on every test
//! run, but this generator is kept around — and re-runnable — so they stay
//! reproducible and auditable rather than opaque blobs. To regenerate:
//!
//! ```text
//! cargo test -p tpt-infer-onnx --test generate_fixtures -- --ignored
//! ```

mod common;

#[test]
#[ignore = "regenerates committed fixtures; run explicitly with `-- --ignored`"]
fn regenerate_fixtures() {
    let dir = common::fixtures_dir();
    std::fs::create_dir_all(&dir).unwrap();

    let mobilenet = common::mobilenet_v2_style_bytes();
    std::fs::write(dir.join("mobilenet_v2_style.onnx"), &mobilenet).unwrap();

    let bert = common::bert_tiny_style_bytes();
    std::fs::write(dir.join("bert_tiny_style.onnx"), &bert).unwrap();

    // Sanity: both fixtures round-trip through the real loader immediately
    // after being written, so a bad regeneration fails loudly right here.
    tpt_infer_onnx::load(dir.join("mobilenet_v2_style.onnx")).expect("mobilenet fixture loads");
    tpt_infer_onnx::load(dir.join("bert_tiny_style.onnx")).expect("bert fixture loads");
}
