//! Inference latency benchmark for a MobileNetV2-shaped graph.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p tpt-infer-runtime --features std
//! ```
//!
//! (The `std` feature is required: graph execution lives behind it.)
//!
//! The graph mirrors the structural fixture in the crate's tests —
//! conv stem → inverted-residual-ish blocks → global average pool →
//! MatMul head to 1000 classes — with spatial dims shrunk to 16×16 so a
//! full 224×224 conv stack does not dominate iteration time. Criterion
//! reports the sample distribution; an explicit **median (p50) and p99**
//! over `SAMPLES` timed runs is printed alongside it. The arena backing
//! buffer is reused (`reset`) across runs, so the measured path is the
//! zero-heap hot path plus the final output copy.

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use tpt_infer_core::BumpArena;
use tpt_infer_graph::{ComputationGraph, Initializer, Node, Operator};
use tpt_infer_ops::dispatch::select_backend;
use tpt_infer_runtime::{execute_graph, required_arena_bytes};

/// Timed samples collected for the median / p99 report.
const SAMPLES: usize = 200;

fn add_input(g: &mut ComputationGraph, dims: &[usize], name: &str) -> usize {
    let id = g.nodes().len();
    g.add_node(
        Node::new(id, Operator::Input, vec![], dims)
            .unwrap()
            .with_name(name),
    )
    .unwrap()
}

fn add_weight(g: &mut ComputationGraph, name: &str, dims: &[usize], scale: f32, seed: u32) -> usize {
    let len: usize = dims.iter().product();
    let mut s = seed;
    let data: Vec<f32> = (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (s >> 8) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect();
    let id = add_input(g, dims, name);
    g.add_initializer(Initializer::new(name, dims, data).unwrap());
    id
}

#[allow(clippy::too_many_arguments)]
fn add_conv(
    g: &mut ComputationGraph,
    x: usize,
    wname: &str,
    wdims: &[usize],
    scale: f32,
    seed: u32,
    strides: [usize; 2],
    padding: [usize; 2],
    odims: &[usize],
) -> usize {
    let w = add_weight(g, wname, wdims, scale, seed);
    let id = g.nodes().len();
    g.add_node(Node::new(id, Operator::conv2d(strides, padding), vec![x, w], odims).unwrap())
        .unwrap()
}

fn add_unary(g: &mut ComputationGraph, op: Operator, x: usize, odims: &[usize]) -> usize {
    let id = g.nodes().len();
    g.add_node(Node::new(id, op, vec![x], odims).unwrap()).unwrap()
}

fn add_binary(g: &mut ComputationGraph, op: Operator, a: usize, b: usize, odims: &[usize]) -> usize {
    let id = g.nodes().len();
    g.add_node(Node::new(id, op, vec![a, b], odims).unwrap())
        .unwrap()
}

/// Shrunk MobileNetV2-shaped graph: `[1,8,16,16]` input, two residual
/// blocks, global average pool, `[16,1000]` head.
fn mobilenet_v2_small() -> ComputationGraph {
    let mut g = ComputationGraph::new();
    let mut seed = 0u32;
    let mut next = || {
        seed += 1;
        seed
    };

    let x = add_input(&mut g, &[1, 8, 16, 16], "data");
    let mut cur = add_conv(
        &mut g,
        x,
        "stem_w",
        &[16, 8, 3, 3],
        0.05,
        next(),
        [2, 2],
        [1, 1],
        &[1, 16, 8, 8],
    );
    cur = add_unary(&mut g, Operator::Relu, cur, &[1, 16, 8, 8]);

    // Block 1: stride 2, no residual.
    let mut t = add_conv(&mut g, cur, "b1_expand_w", &[32, 16, 1, 1], 0.05, next(), [1, 1], [0, 0], &[1, 32, 8, 8]);
    t = add_unary(&mut g, Operator::Relu, t, &[1, 32, 8, 8]);
    t = add_conv(&mut g, t, "b1_spatial_w", &[32, 32, 3, 3], 0.05, next(), [2, 2], [1, 1], &[1, 32, 4, 4]);
    t = add_unary(&mut g, Operator::Relu, t, &[1, 32, 4, 4]);
    let b1 = add_conv(&mut g, t, "b1_proj_w", &[16, 32, 1, 1], 0.05, next(), [1, 1], [0, 0], &[1, 16, 4, 4]);

    // Block 2: stride 1 with residual add.
    let mut t = add_conv(&mut g, b1, "b2_expand_w", &[32, 16, 1, 1], 0.05, next(), [1, 1], [0, 0], &[1, 32, 4, 4]);
    t = add_unary(&mut g, Operator::Relu, t, &[1, 32, 4, 4]);
    t = add_conv(&mut g, t, "b2_spatial_w", &[32, 32, 3, 3], 0.05, next(), [1, 1], [1, 1], &[1, 32, 4, 4]);
    t = add_unary(&mut g, Operator::Relu, t, &[1, 32, 4, 4]);
    let b2p = add_conv(&mut g, t, "b2_proj_w", &[16, 32, 1, 1], 0.05, next(), [1, 1], [0, 0], &[1, 16, 4, 4]);
    let cur = add_binary(&mut g, Operator::Add, b1, b2p, &[1, 16, 4, 4]);

    let gap = add_unary(
        &mut g,
        Operator::average_pool2d([4, 4], [1, 1], [0, 0]),
        cur,
        &[1, 16, 1, 1],
    );
    let flat = add_unary(&mut g, Operator::flatten(1), gap, &[1, 16]);
    let fc = add_weight(&mut g, "fc_w", &[16, 1000], 0.1, next());
    let head = add_binary(&mut g, Operator::MatMul, flat, fc, &[1, 1000]);
    g.mark_output(head).unwrap();
    g
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_latency(c: &mut Criterion) {
    let graph = mobilenet_v2_small();
    let input: Vec<f32> = (0..1 * 8 * 16 * 16)
        .map(|i| (i % 255) as f32 / 255.0)
        .collect();
    let mut mem = vec![0u8; required_arena_bytes(&graph) + 4096];
    let backend = select_backend();
    eprintln!(
        "tpt-infer-runtime latency bench: backend={}, nodes={}, arena={} bytes",
        backend.name(),
        graph.nodes().len(),
        mem.len()
    );

    // Explicit median / p99 over SAMPLES timed runs (hot arena reuse).
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut arena = BumpArena::new(&mut mem);
    for _ in 0..SAMPLES {
        arena.reset();
        let start = Instant::now();
        let outs = execute_graph(&graph, &[&input], &mut arena, &backend).unwrap();
        samples.push(start.elapsed());
        criterion::black_box(outs);
    }
    samples.sort_unstable();
    eprintln!(
        "inference latency over {SAMPLES} runs: median={:?}, p99={:?}, min={:?}, max={:?}",
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        samples.first().copied().unwrap_or_default(),
        samples.last().copied().unwrap_or_default(),
    );

    c.bench_function("mobilenet_v2_small execute_graph", |ben| {
        ben.iter(|| {
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

criterion_group!(benches, bench_latency);
criterion_main!(benches);
