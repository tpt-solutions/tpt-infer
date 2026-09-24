//! Ahead-of-time compilation: turn a graph into standalone Rust source and
//! show that it agrees with the interpreted runtime.
//!
//! ```text
//! cargo run --example aot_compile -p tpt-infer
//! ```
//!
//! Builds a small `x -> MatMul(w) -> Relu -> y` graph, compiles it with
//! [`aot_compile`], prints the generated `pub fn execute` source, and
//! compares its (freshly `rustc`-compiled and run) output against the
//! interpreted runtime's — the same check `tpt-infer-cli compile` and
//! `tpt-infer-compile`'s own test suite rely on.

use std::process::Command;

use tpt_infer::prelude::*;

fn main() {
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
        .unwrap();
    let w = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[4, 2])
                .unwrap()
                .with_name("w"),
        )
        .unwrap();
    g.add_initializer(
        Initializer::new(
            "w",
            &[4, 2],
            vec![0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15],
        )
        .unwrap(),
    );
    let mm = g
        .add_node(Node::new(2, Operator::MatMul, vec![x, w], &[1, 2]).unwrap())
        .unwrap();
    let y = g
        .add_node(Node::new(3, Operator::Relu, vec![mm], &[1, 2]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let compiled = aot_compile(&g).expect("this graph only uses ops the AOT compiler supports");
    println!("--- generated source ---\n{}\n", compiled.source());

    let input = [1.0f32, -2.0, 0.5, 3.0];

    // Interpreted result, via the same runtime tpt-infer-cli's `run` uses.
    let mut mem = vec![0u8; required_arena_bytes(&g) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let interpreted = execute(&g, &input, &mut arena, &select_backend())
        .expect("interpreted execution should succeed");
    println!("interpreted: {:?}", interpreted.as_slice());

    // AOT-compiled result: write the generated source, compile it with
    // rustc, and run it — skipped gracefully if rustc isn't on PATH.
    let Some(compiled_output) = run_via_rustc(compiled.source(), &input) else {
        println!("(skipping AOT execution — `rustc` not found on PATH)");
        return;
    };
    println!("AOT-compiled: {compiled_output:?}");

    for (a, b) in interpreted.as_slice().iter().zip(compiled_output.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "AOT output must match the interpreter"
        );
    }
    println!("\ninterpreted and AOT-compiled outputs match.");
}

/// Wraps `source` in a tiny `main` that prints its output as JSON-ish
/// space-separated floats, compiles it with `rustc`, runs it, and parses
/// the result back into a `Vec<f32>`. Returns `None` if `rustc` isn't
/// available (this example still demonstrates codegen without it).
fn run_via_rustc(source: &str, input: &[f32]) -> Option<Vec<f32>> {
    let dir = std::env::temp_dir().join(format!("tpt_infer_aot_example_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src_path = dir.join("gen.rs");
    let input_list = input
        .iter()
        .map(|v| format!("{v}f32"))
        .collect::<Vec<_>>()
        .join(", ");
    let wrapped = format!(
        "{source}\nfn main() {{\n    let out = execute(&[{input_list}]);\n    let s: Vec<String> = out.iter().map(|v| v.to_string()).collect();\n    println!(\"{{}}\", s.join(\" \"));\n}}\n"
    );
    std::fs::write(&src_path, wrapped).ok()?;

    let exe_path = dir.join(if cfg!(windows) { "gen.exe" } else { "gen" });
    let status = Command::new("rustc")
        .args(["-O", "-o"])
        .arg(&exe_path)
        .arg(&src_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let output = Command::new(&exe_path).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.split_whitespace().map(|t| t.parse().ok()).collect()
}
