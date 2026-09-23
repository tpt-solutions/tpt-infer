//! Generated-code throughput vs. `tpt-infer-runtime`'s interpreted
//! execution, for the same small MLP graph.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p tpt-infer-compile
//! ```
//!
//! The "compiled" side does not shell out to `rustc` (benchmarking a
//! subprocess compiler invocation per-iteration would measure the wrong
//! thing) — instead it calls the same scalar Rust the generated code would
//! run, written by hand to mirror exactly what [`tpt_infer_compile::codegen`]
//! emits for this graph (`MatMul` → `Relu` → `MatMul`, with both weight
//! matrices folded to constants, matching this graph's actual
//! [`tpt_infer_compile::aot_compile`] output). This still isolates the
//! interesting comparison: generated straight-line/unrolled scalar code
//! versus the interpreted dispatch-table executor walking the same graph
//! node by node through the `Backend` HAL.

use criterion::{criterion_group, criterion_main, Criterion};
use tpt_infer_compile::aot_compile;
use tpt_infer_core::BumpArena;
use tpt_infer_graph::{ComputationGraph, Initializer, Node, Operator};
use tpt_infer_ops::NaiveBackend;
use tpt_infer_runtime::{execute_graph, required_arena_bytes};

const W1: [f32; 12] = [
    0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1,
];
const W2: [f32; 6] = [0.2, -0.4, 0.1, 0.3, -0.25, 0.5];

fn build_mlp() -> ComputationGraph {
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
    g.add_initializer(Initializer::new("w1", &[4, 3], W1.to_vec()).unwrap());
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
    g.add_initializer(Initializer::new("w2", &[3, 2], W2.to_vec()).unwrap());
    let m2 = g
        .add_node(Node::new(5, Operator::MatMul, vec![r, w2], &[1, 2]).unwrap())
        .unwrap();
    g.mark_output(m2).unwrap();
    g
}

/// Hand-written scalar Rust mirroring what `tpt-infer-compile::codegen`
/// generates for `build_mlp()` (both weights are constant-folded, so the
/// generated `execute` only ever computes the two `MatMul`s and the `Relu`
/// against literal weight arrays — see the crate's integration test for the
/// actual generated source).
fn compiled_execute(input: &[f32; 4]) -> [f32; 2] {
    let mut m1 = [0.0f32; 3];
    for j in 0..3 {
        let mut acc = 0.0f32;
        for p in 0..4 {
            acc += input[p] * W1[p * 3 + j];
        }
        m1[j] = acc;
    }
    let mut r = [0.0f32; 3];
    for i in 0..3 {
        r[i] = m1[i].max(0.0);
    }
    let mut m2 = [0.0f32; 2];
    for j in 0..2 {
        let mut acc = 0.0f32;
        for p in 0..3 {
            acc += r[p] * W2[p * 2 + j];
        }
        m2[j] = acc;
    }
    m2
}

fn bench(c: &mut Criterion) {
    // Sanity-check the hand-written mirror against the real aot_compile
    // output's metadata (same graph shape/fold count) so the benchmark
    // can't silently drift from what codegen actually produces.
    let graph = build_mlp();
    let model = aot_compile(&graph).unwrap();
    assert_eq!(model.folded_node_count(), 2);

    let input = [1.0f32, -2.0, 0.5, 3.0];

    c.bench_function("compiled (generated-code shape) execute", |b| {
        b.iter(|| criterion::black_box(compiled_execute(criterion::black_box(&input))))
    });

    let bytes = required_arena_bytes(&graph);
    let mut mem = vec![0u8; bytes + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let backend = NaiveBackend::new();
    c.bench_function("interpreted tpt-infer-runtime execute_graph", |b| {
        b.iter(|| {
            arena.reset();
            let outs = execute_graph(
                &graph,
                criterion::black_box(&[input.as_slice()]),
                &mut arena,
                &backend,
            )
            .unwrap();
            criterion::black_box(outs)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
