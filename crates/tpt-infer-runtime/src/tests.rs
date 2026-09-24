//! Structural fixtures and end-to-end execution tests.
//!
//! The flagship test, `mobilenet_v2_end_to_end`, builds a **synthetic
//! MobileNetV2-shaped graph programmatically** (the real `.onnx` file is
//! gitignored and unavailable in CI): a 3×3 stride-2 convolution stem,
//! four inverted-residual-ish blocks simplified to
//! Conv → Relu → Conv → Relu → Conv [→ residual Add], a global average
//! pool, and a MatMul classification head to 1000 classes. Spatial
//! dimensions (32×32, 8 base channels) are shrunk from the real
//! 224×224 / 3-channel model so the fixture runs quickly in debug builds —
//! this is a *structural* fixture: it validates shapes, binding, arena
//! layout, dispatch, and the top-1 contract, not learned weights.

use tpt_infer_core::BumpArena;
use tpt_infer_graph::{ComputationGraph, Initializer, Node, Operator};
use tpt_infer_ops::{dispatch::select_backend, Backend, NaiveBackend};

use crate::{argmax, execute, execute_graph, required_arena_bytes, RuntimeError};

/// Deterministic LCG-distributed weights in `[-scale, scale]`.
fn lcg(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (s >> 8) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn next_seed(s: &mut u32) -> u32 {
    *s += 1;
    *s
}

/// Appends an `Operator::Input` node with `name`.
fn add_input(g: &mut ComputationGraph, dims: &[usize], name: &str) -> usize {
    let id = g.nodes().len();
    g.add_node(
        Node::new(id, Operator::Input, vec![], dims)
            .unwrap()
            .with_name(name),
    )
    .unwrap()
}

/// Appends an input node bound to a freshly generated named initializer.
fn add_weight(
    g: &mut ComputationGraph,
    name: &str,
    dims: &[usize],
    seed: u32,
    scale: f32,
) -> usize {
    let len: usize = dims.iter().product();
    let id = add_input(g, dims, name);
    g.add_initializer(Initializer::new(name, dims, lcg(len, seed, scale)).unwrap());
    id
}

/// Appends `Conv2d(x, weight)` with a fresh weight initializer.
#[allow(clippy::too_many_arguments)]
fn add_conv(
    g: &mut ComputationGraph,
    x: usize,
    wname: &str,
    wdims: &[usize],
    seed: u32,
    strides: [usize; 2],
    padding: [usize; 2],
    odims: &[usize],
) -> usize {
    let w = add_weight(g, wname, wdims, seed, 0.05);
    let id = g.nodes().len();
    g.add_node(Node::new(id, Operator::conv2d(strides, padding), vec![x, w], odims).unwrap())
        .unwrap()
}

/// Appends a unary operator node.
fn add_unary(g: &mut ComputationGraph, op: Operator, x: usize, odims: &[usize]) -> usize {
    let id = g.nodes().len();
    g.add_node(Node::new(id, op, vec![x], odims).unwrap())
        .unwrap()
}

/// Appends a binary operator node.
fn add_binary(
    g: &mut ComputationGraph,
    op: Operator,
    a: usize,
    b: usize,
    odims: &[usize],
) -> usize {
    let id = g.nodes().len();
    g.add_node(Node::new(id, op, vec![a, b], odims).unwrap())
        .unwrap()
}

/// Inverted-residual-ish block: expand 1×1 → Relu → spatial 3×3 → Relu →
/// project 1×1 → (residual Add when `stride == 1` and channel counts
/// match). Returns the output node id and its `[n, c, h, w]` dims.
#[allow(clippy::too_many_arguments)]
fn add_ir_block(
    g: &mut ComputationGraph,
    x: usize,
    x_dims: [usize; 4],
    c_expand: usize,
    c_out: usize,
    stride: usize,
    tag: &str,
    seed: &mut u32,
) -> (usize, [usize; 4]) {
    let [n, c_in, h, w] = x_dims;
    let oh = (h + 2 - 3) / stride + 1;
    let ow = (w + 2 - 3) / stride + 1;
    let dims_e = [n, c_expand, h, w];
    let dims_s = [n, c_expand, oh, ow];
    let dims_p = [n, c_out, oh, ow];

    let mut t = add_conv(
        g,
        x,
        &format!("{tag}_expand_w"),
        &[c_expand, c_in, 1, 1],
        next_seed(seed),
        [1, 1],
        [0, 0],
        &dims_e,
    );
    t = add_unary(g, Operator::Relu, t, &dims_e);
    t = add_conv(
        g,
        t,
        &format!("{tag}_spatial_w"),
        &[c_expand, c_expand, 3, 3],
        next_seed(seed),
        [stride, stride],
        [1, 1],
        &dims_s,
    );
    t = add_unary(g, Operator::Relu, t, &dims_s);
    t = add_conv(
        g,
        t,
        &format!("{tag}_proj_w"),
        &[c_out, c_expand, 1, 1],
        next_seed(seed),
        [1, 1],
        [0, 0],
        &dims_p,
    );
    if stride == 1 && c_in == c_out {
        t = add_binary(g, Operator::Add, x, t, &dims_p);
    }
    (t, dims_p)
}

/// Builds the structural MobileNetV2-like graph (see module docs).
fn mobilenet_v2_fixture() -> ComputationGraph {
    let mut g = ComputationGraph::new();
    let mut seed = 0u32;

    let x = add_input(&mut g, &[1, 8, 32, 32], "data");
    let mut cur = add_conv(
        &mut g,
        x,
        "stem_w",
        &[16, 8, 3, 3],
        next_seed(&mut seed),
        [2, 2],
        [1, 1],
        &[1, 16, 16, 16],
    );
    cur = add_unary(&mut g, Operator::Relu, cur, &[1, 16, 16, 16]);

    let (b1, d1) = add_ir_block(&mut g, cur, [1, 16, 16, 16], 32, 16, 2, "b1", &mut seed);
    let (b2, d2) = add_ir_block(&mut g, b1, d1, 32, 16, 1, "b2", &mut seed);
    let (b3, d3) = add_ir_block(&mut g, b2, d2, 48, 16, 2, "b3", &mut seed);
    let (b4, d4) = add_ir_block(&mut g, b3, d3, 48, 16, 1, "b4", &mut seed);
    debug_assert_eq!(d2, d1);
    debug_assert_eq!(d4, [1, 16, 4, 4]);

    // Global average pool: kernel covers the whole feature map.
    let [n, c, h, w] = d4;
    let gap_id = g.nodes().len();
    let gap = g
        .add_node(
            Node::new(
                gap_id,
                Operator::average_pool2d([h, w], [1, 1], [0, 0]),
                vec![b4],
                &[n, c, 1, 1],
            )
            .unwrap(),
        )
        .unwrap();
    let flat = add_unary(&mut g, Operator::flatten(1), gap, &[n, c]);

    let fc = add_weight(&mut g, "fc_w", &[c, 1000], next_seed(&mut seed), 0.1);
    let head = add_binary(&mut g, Operator::MatMul, flat, fc, &[n, 1000]);
    g.mark_output(head).unwrap();
    g
}

/// End-to-end structural MobileNetV2 inference: build the synthetic
/// fixture, execute it from the arena, and verify the `[1, 1000]` output
/// and top-1 contract (see module docs for the fixture rationale).
#[test]
fn mobilenet_v2_end_to_end() {
    let g = mobilenet_v2_fixture();
    assert!(g.nodes().len() > 30, "fixture should be multi-block");
    assert_eq!(g.outputs().len(), 1);

    let input = lcg(8 * 32 * 32, 7, 0.5);
    let mut mem = vec![0u8; required_arena_bytes(&g) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &NaiveBackend::new()).expect("fixture executes");

    assert_eq!(out.dims(), &[1, 1000][..]);
    assert_eq!(out.numel(), 1000);
    assert!(
        out.as_slice().iter().all(|v| v.is_finite()),
        "all logits must be finite"
    );
    let top1 = argmax(out.as_slice());
    assert!(top1 < 1000, "top-1 class {top1} must be in 0..1000");
    assert!(arena.used() <= arena.capacity());

    // Determinism: a second run on the reset arena is bit-identical.
    arena.reset();
    let out2 = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap();
    assert_eq!(out.as_slice(), out2.as_slice());
    assert_eq!(argmax(out2.as_slice()), top1);
}

/// Exercises the remaining dispatch arms (MaxPool, Concat, Flatten,
/// Transpose, Sub/Mul/Div, Sigmoid, GELU, Softmax) plus multi-output
/// collection.
#[test]
fn dispatch_operator_coverage() {
    let mut g = ComputationGraph::new();
    let x = add_input(&mut g, &[1, 1, 4, 4], "x");
    let mp_dims = &[1, 1, 2, 2];
    let mp = add_unary(
        &mut g,
        Operator::max_pool2d([2, 2], [2, 2], [0, 0]),
        x,
        mp_dims,
    );
    let cat = add_binary(&mut g, Operator::concat(1), mp, mp, &[1, 2, 2, 2]);
    let flat = add_unary(&mut g, Operator::flatten(1), cat, &[1, 8]);
    let tr = add_unary(
        &mut g,
        Operator::transpose(&[0, 1, 3, 2]).unwrap(),
        mp,
        mp_dims,
    );
    let sub = add_binary(&mut g, Operator::Sub, mp, mp, mp_dims);
    let mul = add_binary(&mut g, Operator::Mul, mp, mp, mp_dims);
    let div = add_binary(&mut g, Operator::Div, mp, mp, mp_dims);
    let sig = add_unary(&mut g, Operator::Sigmoid, mp, mp_dims);
    let gelu = add_unary(&mut g, Operator::Gelu, mp, mp_dims);
    for &id in &[flat, tr, sub, mul, div, sig, gelu] {
        g.mark_output(id).unwrap();
    }

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut mem = vec![0u8; required_arena_bytes(&g) + 1024];
    let mut arena = BumpArena::new(&mut mem);
    let outs = execute_graph(&g, &[&input], &mut arena, &NaiveBackend::new()).unwrap();
    assert_eq!(outs.len(), 7);

    // flatten output keeps concat data: [5,7,13,15] duplicated.
    assert_eq!(
        outs[0].as_slice(),
        &[5.0, 7.0, 13.0, 15.0, 5.0, 7.0, 13.0, 15.0]
    );
    // transpose swaps the spatial axes of the pool output.
    assert_eq!(outs[1].as_slice(), &[5.0, 13.0, 7.0, 15.0]);
    // sub = 0, mul = x^2, div = 1
    assert_eq!(outs[2].as_slice(), &[0.0; 4]);
    let mpv = [5.0f32, 7.0, 13.0, 15.0];
    for (i, &v) in mpv.iter().enumerate() {
        assert!((outs[3].as_slice()[i] - v * v).abs() < 1e-4);
        assert!((outs[4].as_slice()[i] - 1.0).abs() < 1e-5);
    }
    // sigmoid / gelu are finite and in range.
    assert!(outs[5].as_slice().iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(outs[6].as_slice().iter().all(|v| v.is_finite()));
}

/// Conv with a third (bias) input: conv output + broadcast bias.
#[test]
fn conv_bias_is_added() {
    let mut g = ComputationGraph::new();
    let x = add_input(&mut g, &[1, 1, 3, 3], "x");
    let w = add_input(&mut g, &[1, 1, 2, 2], "w");
    g.add_initializer(Initializer::new("w", &[1, 1, 2, 2], vec![1.0, 0.0, 0.0, -1.0]).unwrap());
    let b = add_input(&mut g, &[1], "b");
    g.add_initializer(Initializer::new("b", &[1], vec![5.0]).unwrap());
    let id = g.nodes().len();
    g.add_node(
        Node::new(
            id,
            Operator::conv2d([1, 1], [0, 0]),
            vec![x, w, b],
            &[1, 1, 2, 2],
        )
        .unwrap(),
    )
    .unwrap();
    g.mark_output(id).unwrap();

    let input: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let mut mem = vec![0u8; required_arena_bytes(&g) + 512];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap();
    // naive conv result (see tpt-infer-ops) is [-4,-4,-4,-4], + bias 5.
    assert_eq!(out.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
}

/// BatchNorm through the dispatch table with initializer parameters.
#[test]
fn batch_norm_dispatches() {
    let mut g = ComputationGraph::new();
    let x = add_input(&mut g, &[1, 2, 1, 2], "x");
    let param = |g: &mut ComputationGraph, name: &str, vals: &[f32]| {
        let id = add_input(g, &[2], name);
        g.add_initializer(Initializer::new(name, &[2], vals.to_vec()).unwrap());
        id
    };
    let scale = param(&mut g, "scale", &[2.0, 1.0]);
    let bias = param(&mut g, "bias", &[0.0, 10.0]);
    let mean = param(&mut g, "mean", &[1.0, 3.0]);
    let var = param(&mut g, "var", &[1.0, 1.0]);

    let id = g.nodes().len();
    g.add_node(
        Node::new(
            id,
            Operator::batch_norm(0.0),
            vec![x, scale, bias, mean, var],
            &[1, 2, 1, 2],
        )
        .unwrap(),
    )
    .unwrap();
    g.mark_output(id).unwrap();

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut mem = vec![0u8; required_arena_bytes(&g) + 512];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap();
    assert_eq!(out.as_slice(), &[0.0, 2.0, 10.0, 11.0]);
}

/// Softmax on a non-final axis is rejected by the HAL row contract.
#[test]
fn softmax_non_last_axis_is_unsupported() {
    let mut g = ComputationGraph::new();
    let x = add_input(&mut g, &[2, 3], "x");
    let y = add_unary(&mut g, Operator::softmax(0), x, &[2, 3]);
    g.mark_output(y).unwrap();
    let mut mem = vec![0u8; required_arena_bytes(&g) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let input = [1.0f32; 6];
    let err = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap_err();
    assert_eq!(err, RuntimeError::UnsupportedOp { name: "Softmax" });
}

/// Runs the relu/softmax chain on whatever backend `select_backend`
/// picks (AVX2/NEON when available, naive otherwise).
#[test]
fn selected_backend_executes() {
    let mut g = ComputationGraph::new();
    let x = add_input(&mut g, &[1, 4], "x");
    let h = add_unary(&mut g, Operator::Relu, x, &[1, 4]);
    let y = add_unary(&mut g, Operator::softmax(-1), h, &[1, 4]);
    g.mark_output(y).unwrap();

    let backend = select_backend();
    let input = [-1.0f32, 2.0, -3.0, 4.0];
    let mut mem = vec![0u8; required_arena_bytes(&g) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let out = execute(&g, &input, &mut arena, &backend).unwrap();
    let sum: f32 = out.as_slice().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "backend {} softmax rows",
        backend.name()
    );
    assert_eq!(argmax(out.as_slice()), 3);
}
