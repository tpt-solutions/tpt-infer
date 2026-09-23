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
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(1, Operator::softmax(-1), vec![x], &[1, 4]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let err = aot_compile(&g).unwrap_err();
    match err {
        tpt_infer_compile::CompileError::UnsupportedOperator { name, node } => {
            assert_eq!(name, "Softmax");
            assert_eq!(node, 1);
        }
        other => panic!("expected UnsupportedOperator, got {other:?}"),
    }
}
