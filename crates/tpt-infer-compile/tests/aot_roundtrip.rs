//! Integration test: compile a small MLP graph to Rust source, `rustc` it
//! to a standalone binary, run it, and check the output against the
//! interpreted `tpt-infer-runtime` execution of the same graph.
//!
//! Graph topology (mirrors `crates/tpt-infer-graph/tests/mlp.rs`, shrunk so
//! the emitted `rustc` invocation stays fast):
//!
//! ```text
//! x  [1,4] ─┐
//!           ├─ MatMul ─ [1,3] ─ Relu ─┐
//! w1 [4,3] ─┘                         ├─ MatMul ─ [1,2]
//!                             w2 [3,2]┘
//! ```
//!
//! If `rustc` is not on `PATH` (e.g. a minimal CI image with only `cargo`
//! wired up), the test prints a message and returns early instead of
//! failing — the rest of the suite still exercises `aot_compile` itself.

use std::process::Command;

use tpt_infer_compile::aot_compile;
use tpt_infer_core::BumpArena;
use tpt_infer_graph::{ComputationGraph, Initializer, Node, Operator};
use tpt_infer_ops::NaiveBackend;
use tpt_infer_runtime::{execute, required_arena_bytes};

fn build_mlp() -> (ComputationGraph, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut g = ComputationGraph::new();

    let x = g
        .add_node(
            Node::new(0, Operator::Input, vec![], &[1, 4])
                .unwrap()
                .with_name("x"),
        )
        .unwrap();
    let w1 = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[4, 3])
                .unwrap()
                .with_name("w1"),
        )
        .unwrap();
    let w1_data: Vec<f32> = vec![
        0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1,
    ];
    g.add_initializer(Initializer::new("w1", &[4, 3], w1_data.clone()).unwrap());

    let m1 = g
        .add_node(Node::new(2, Operator::MatMul, vec![x, w1], &[1, 3]).unwrap())
        .unwrap();
    let r = g
        .add_node(Node::new(3, Operator::Relu, vec![m1], &[1, 3]).unwrap())
        .unwrap();

    let w2 = g
        .add_node(
            Node::new(4, Operator::Input, vec![], &[3, 2])
                .unwrap()
                .with_name("w2"),
        )
        .unwrap();
    let w2_data: Vec<f32> = vec![0.2, -0.4, 0.1, 0.3, -0.25, 0.5];
    g.add_initializer(Initializer::new("w2", &[3, 2], w2_data.clone()).unwrap());

    let m2 = g
        .add_node(Node::new(5, Operator::MatMul, vec![r, w2], &[1, 2]).unwrap())
        .unwrap();
    g.mark_output(m2).unwrap();

    let input = vec![1.0f32, -2.0, 0.5, 3.0];
    (g, input, w1_data, w2_data)
}

fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Shared round-trip helper for the per-operator tests below: `aot_compile`s
/// `graph`, writes the generated source (plus a `main` that calls
/// `execute(&input)` and prints comma-separated results) to a temp file,
/// invokes `rustc -O`, runs the binary, and parses its stdout back into
/// `f32`s.
///
/// Returns `None` (with a diagnostic on stderr) if `rustc` is not on
/// `PATH`, matching the crate's existing skip-if-unavailable convention.
fn run_compiled(model: &tpt_infer_compile::CompiledModel, input: &[f32]) -> Option<Vec<f32>> {
    if !rustc_available() {
        eprintln!("skipping rustc round-trip: `rustc` not found on PATH");
        return None;
    }

    let input_literal = input
        .iter()
        .map(|v| format!("{v:?}f32"))
        .collect::<Vec<_>>()
        .join(", ");
    let program = format!(
        "{source}\n\nfn main() {{\n    let input: Vec<f32> = vec![{input_literal}];\n    \
         let out = execute(&input);\n    let strs: Vec<String> = out.iter().map(|v| v.to_string()).collect();\n    \
         println!(\"{{}}\", strs.join(\",\"));\n}}\n",
        source = model.source(),
    );

    let dir = tempfile::tempdir().expect("create temp dir");
    let src_path = dir.path().join("generated.rs");
    std::fs::write(&src_path, &program).expect("write generated source");

    let bin_name = if cfg!(windows) {
        "generated.exe"
    } else {
        "generated"
    };
    let bin_path = dir.path().join(bin_name);

    let compile_output = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("-O")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("spawn rustc");
    assert!(
        compile_output.status.success(),
        "rustc failed to compile generated source:\nstdout: {}\nstderr: {}\n--- source ---\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr),
        program,
    );

    let run_output = Command::new(&bin_path)
        .output()
        .expect("run generated binary");
    assert!(
        run_output.status.success(),
        "generated binary exited with failure: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    if stdout.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        stdout
            .trim()
            .split(',')
            .map(|s| {
                s.parse::<f32>()
                    .expect("parse f32 from generated binary output")
            })
            .collect(),
    )
}

/// Compiles `graph`, runs it both ways (interpreted via `tpt-infer-runtime`
/// and AOT-compiled via `rustc`), and asserts the two agree within `tol`.
/// Skips the `rustc` half (returning early) if `rustc` is unavailable.
fn assert_roundtrip_matches(graph: &ComputationGraph, input: &[f32], tol: f32) {
    let bytes = required_arena_bytes(graph);
    let mut mem = vec![0u8; bytes];
    let mut arena = BumpArena::new(&mut mem);
    let expected = execute(graph, input, &mut arena, &NaiveBackend::new()).unwrap();
    let expected: Vec<f32> = expected.as_slice().to_vec();

    let model = aot_compile(graph).unwrap();
    let Some(actual) = run_compiled(&model, input) else {
        return;
    };

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < tol, "compiled={a} interpreted={e}");
    }
}

#[test]
fn compiled_source_matches_interpreted_runtime() {
    let (graph, input, _w1, _w2) = build_mlp();

    // Expected output via the interpreted runtime (tpt-infer-runtime + the
    // naive CPU backend), independent of aot_compile.
    let bytes = required_arena_bytes(&graph);
    let mut mem = vec![0u8; bytes];
    let mut arena = BumpArena::new(&mut mem);
    let expected = execute(&graph, &input, &mut arena, &NaiveBackend::new()).unwrap();
    let expected: Vec<f32> = expected.as_slice().to_vec();
    assert_eq!(expected.len(), 2);

    // AOT-compile the same graph to Rust source.
    let model = aot_compile(&graph).unwrap();
    assert_eq!(model.input_shape(), &[1, 4]);
    assert_eq!(model.output_shape(), &[1, 2]);
    // Both weight `Input` nodes are constant-folded (they have initializers).
    assert_eq!(model.folded_node_count(), 2);

    if !rustc_available() {
        eprintln!("skipping rustc round-trip: `rustc` not found on PATH");
        return;
    }

    let input_literal = input
        .iter()
        .map(|v| format!("{v:?}f32"))
        .collect::<Vec<_>>()
        .join(", ");
    let program = format!(
        "{source}\n\nfn main() {{\n    let input: Vec<f32> = vec![{input_literal}];\n    \
         let out = execute(&input);\n    let strs: Vec<String> = out.iter().map(|v| v.to_string()).collect();\n    \
         println!(\"{{}}\", strs.join(\",\"));\n}}\n",
        source = model.source(),
    );

    let dir = tempfile::tempdir().expect("create temp dir");
    let src_path = dir.path().join("generated.rs");
    std::fs::write(&src_path, &program).expect("write generated source");

    let bin_name = if cfg!(windows) {
        "generated.exe"
    } else {
        "generated"
    };
    let bin_path = dir.path().join(bin_name);

    let compile_output = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("-O")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("spawn rustc");
    assert!(
        compile_output.status.success(),
        "rustc failed to compile generated source:\nstdout: {}\nstderr: {}\n--- source ---\n{}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr),
        program,
    );

    let run_output = Command::new(&bin_path)
        .output()
        .expect("run generated binary");
    assert!(
        run_output.status.success(),
        "generated binary exited with failure: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    let actual: Vec<f32> = stdout
        .trim()
        .split(',')
        .map(|s| {
            s.parse::<f32>()
                .expect("parse f32 from generated binary output")
        })
        .collect();

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4, "compiled={a} interpreted={e}");
    }
}

#[test]
fn unsupported_operator_reports_error_instead_of_panicking() {
    // `Operator::Custom` (an ONNX op with no native `Operator` mapping) is
    // the remaining genuinely-unsupported case now that `Gelu` and
    // `Softmax` (last axis only) both have codegen.
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(
            Node::new(
                1,
                Operator::Custom("LayerNormalization".into()),
                vec![x],
                &[1, 4],
            )
            .unwrap(),
        )
        .unwrap();
    g.mark_output(y).unwrap();

    let err = aot_compile(&g).unwrap_err();
    match err {
        tpt_infer_compile::CompileError::UnsupportedOperator { name, node } => {
            assert_eq!(name, "Custom");
            assert_eq!(node, 1);
        }
        other => panic!("expected UnsupportedOperator, got {other:?}"),
    }
}

#[test]
fn conv2d_with_bias_matches_interpreted_runtime() {
    // x [1,1,4,4] (runtime) --Conv2d(stride=1,pad=1,bias)--> [1,2,4,4]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(
            Node::new(0, Operator::Input, vec![], &[1, 1, 4, 4])
                .unwrap()
                .with_name("x"),
        )
        .unwrap();
    let weight = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[2, 1, 3, 3])
                .unwrap()
                .with_name("weight"),
        )
        .unwrap();
    let w_data: Vec<f32> = (0..18).map(|i| (i as f32) * 0.05 - 0.4).collect();
    g.add_initializer(Initializer::new("weight", &[2, 1, 3, 3], w_data).unwrap());
    let bias = g
        .add_node(
            Node::new(2, Operator::Input, vec![], &[1, 2, 1, 1])
                .unwrap()
                .with_name("bias"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("bias", &[1, 2, 1, 1], vec![0.5, -0.25]).unwrap());
    let conv = g
        .add_node(
            Node::new(
                3,
                Operator::conv2d([1, 1], [1, 1]),
                vec![x, weight, bias],
                &[1, 2, 4, 4],
            )
            .unwrap(),
        )
        .unwrap();
    g.mark_output(conv).unwrap();

    let input: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
    assert_roundtrip_matches(&g, &input, 1e-3);
}

#[test]
fn broadcast_bias_add_matches_interpreted_runtime() {
    // x [1,2,2,2] (runtime, standing in for a conv activation)
    // --Add(bias [1,2,1,1])--> [1,2,2,2]: a per-channel bias broadcasting
    // over the spatial axes, as real ONNX exports commonly emit as a
    // standalone `Add` (rather than a `Conv2d`-with-bias input) — this is
    // exactly the pattern the real MNIST fixture (see
    // crates/tpt-infer-onnx/tests/fixtures/real/) hits.
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 2, 2, 2]).unwrap())
        .unwrap();
    let bias = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[1, 2, 1, 1])
                .unwrap()
                .with_name("bias"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("bias", &[1, 2, 1, 1], vec![10.0, -10.0]).unwrap());
    let y = g
        .add_node(Node::new(2, Operator::Add, vec![x, bias], &[1, 2, 2, 2]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    assert_roundtrip_matches(&g, &input, 1e-5);
}

#[test]
fn broadcast_scalar_mul_matches_interpreted_runtime() {
    // x [2,3] (runtime) --Mul(scalar [1,1])--> [2,3]: broadcasting on every
    // axis at once, the most extreme case of the same mechanism.
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
        .unwrap();
    let scale = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[1, 1])
                .unwrap()
                .with_name("scale"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("scale", &[1, 1], vec![2.5]).unwrap());
    let y = g
        .add_node(Node::new(2, Operator::Mul, vec![x, scale], &[2, 3]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let input: Vec<f32> = (0..6).map(|i| i as f32 - 3.0).collect();
    assert_roundtrip_matches(&g, &input, 1e-5);
}

#[test]
fn softmax_last_axis_matches_interpreted_runtime() {
    // x [2,4] (runtime) --Softmax(axis=-1)--> [2,4]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[2, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(1, Operator::softmax(-1), vec![x], &[2, 4]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let input = vec![1.0f32, 2.0, -1.0, 0.5, -3.0, 4.0, 0.0, 1.5];
    assert_roundtrip_matches(&g, &input, 1e-4);
}

#[test]
fn gelu_matches_interpreted_runtime() {
    // x [1,4] (runtime) --Gelu--> [1,4]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(1, Operator::Gelu, vec![x], &[1, 4]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let input = vec![-2.0f32, -0.5, 0.5, 2.0];
    assert_roundtrip_matches(&g, &input, 1e-6);
}

#[test]
fn max_pool2d_matches_interpreted_runtime() {
    // x [1,1,4,4] (runtime) --MaxPool2d(k=2,s=2)--> [1,1,2,2]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 1, 4, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(
            Node::new(
                1,
                Operator::max_pool2d([2, 2], [2, 2], [0, 0]),
                vec![x],
                &[1, 1, 2, 2],
            )
            .unwrap(),
        )
        .unwrap();
    g.mark_output(y).unwrap();

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert_roundtrip_matches(&g, &input, 1e-4);
}

#[test]
fn average_pool2d_matches_interpreted_runtime() {
    // x [1,1,4,4] (runtime) --AveragePool2d(k=2,s=2)--> [1,1,2,2]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 1, 4, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(
            Node::new(
                1,
                Operator::average_pool2d([2, 2], [2, 2], [0, 0]),
                vec![x],
                &[1, 1, 2, 2],
            )
            .unwrap(),
        )
        .unwrap();
    g.mark_output(y).unwrap();

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert_roundtrip_matches(&g, &input, 1e-4);
}

#[test]
fn batch_norm_channel_mode_matches_interpreted_runtime() {
    // x [1,2,2,2] (runtime) --BatchNorm--> [1,2,2,2]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(
            Node::new(0, Operator::Input, vec![], &[1, 2, 2, 2])
                .unwrap()
                .with_name("x"),
        )
        .unwrap();
    let scale = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[2])
                .unwrap()
                .with_name("scale"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("scale", &[2], vec![2.0, 0.5]).unwrap());
    let bias = g
        .add_node(
            Node::new(2, Operator::Input, vec![], &[2])
                .unwrap()
                .with_name("bias"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("bias", &[2], vec![1.0, -1.0]).unwrap());
    let mean = g
        .add_node(
            Node::new(3, Operator::Input, vec![], &[2])
                .unwrap()
                .with_name("mean"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("mean", &[2], vec![0.5, 1.5]).unwrap());
    let var = g
        .add_node(
            Node::new(4, Operator::Input, vec![], &[2])
                .unwrap()
                .with_name("var"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("var", &[2], vec![1.0, 4.0]).unwrap());
    let bn = g
        .add_node(
            Node::new(
                5,
                Operator::batch_norm(1e-5),
                vec![x, scale, bias, mean, var],
                &[1, 2, 2, 2],
            )
            .unwrap(),
        )
        .unwrap();
    g.mark_output(bn).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
    assert_roundtrip_matches(&g, &input, 1e-3);
}

#[test]
fn concat_matches_interpreted_runtime() {
    // x [1,2,2] (runtime) concat w [1,1,2] (constant) on axis=1 -> [1,3,2]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(
            Node::new(0, Operator::Input, vec![], &[1, 2, 2])
                .unwrap()
                .with_name("x"),
        )
        .unwrap();
    let w = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[1, 1, 2])
                .unwrap()
                .with_name("w"),
        )
        .unwrap();
    g.add_initializer(Initializer::new("w", &[1, 1, 2], vec![9.0, 10.0]).unwrap());
    let cat = g
        .add_node(Node::new(2, Operator::concat(1), vec![x, w], &[1, 3, 2]).unwrap())
        .unwrap();
    g.mark_output(cat).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    assert_roundtrip_matches(&g, &input, 1e-4);
}

#[test]
fn transpose_matches_interpreted_runtime() {
    // x [2,3] (runtime) --Transpose(perm=[1,0])--> [3,2]
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(1, Operator::transpose(&[1, 0]).unwrap(), vec![x], &[3, 2]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_roundtrip_matches(&g, &input, 1e-4);
}

#[test]
fn unsupported_softmax_axis_reports_error_instead_of_panicking() {
    // Codegen only generates last-axis softmax (matching
    // tpt-infer-runtime's `Backend::softmax` restriction); a non-last axis
    // must be reported, not miscompiled.
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(1, Operator::softmax(0), vec![x], &[2, 3]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let err = aot_compile(&g).unwrap_err();
    assert!(matches!(
        err,
        tpt_infer_compile::CompileError::UnsupportedRank {
            node: 1,
            op: "Softmax",
            ..
        }
    ));
}

/// Regression test for the real (non-synthetic) MNIST fixture (see
/// `crates/tpt-infer-onnx/tests/fixtures/real/README.md`): it previously
/// failed to `aot_compile` at all — first on a `Reshape` node whose second
/// input is the (now int64-initializer-bound) target-shape tensor, then on
/// a per-channel bias `Add` broadcasting `[1,C,1,1]` against `[1,C,H,W]`,
/// which the exact-same-shape-only fast path rejected. Both are fixed; this
/// locks in that the real model round-trips through AOT compilation too,
/// not just the interpreted runtime (see `onnx_real_model_end_to_end.rs` in
/// `tpt-infer-runtime`).
///
/// Skipped (not failed) if the fixture isn't cached — run
/// `TPT_INFER_FETCH_FIXTURES=1 cargo test -p tpt-infer-onnx --test
/// real_model_fixtures` first.
#[test]
fn real_mnist_onnx_aot_compiles_and_matches_interpreted_runtime() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tpt-infer-onnx/tests/fixtures/real/mnist-12.onnx");
    if !path.exists() {
        eprintln!(
            "skipping: {} not cached; run `TPT_INFER_FETCH_FIXTURES=1 cargo test \
             -p tpt-infer-onnx --test real_model_fixtures` first",
            path.display()
        );
        return;
    }

    let g = tpt_infer_onnx::load(&path).expect("real ONNX file load must succeed");
    let input: Vec<f32> = (0..28 * 28)
        .map(|i| ((i % 17) as f32 / 17.0) - 0.5)
        .collect();
    assert_roundtrip_matches(&g, &input, 1e-4);
}
