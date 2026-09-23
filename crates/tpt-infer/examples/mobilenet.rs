//! End-to-end example: load or build a graph, run it through the facade,
//! and print the top-1 class.
//!
//! Usage:
//!
//! ```text
//! cargo run --example mobilenet -p tpt-infer
//! cargo run --example mobilenet -p tpt-infer -- path/to/model.onnx
//! ```
//!
//! No real MobileNetV2 `.onnx` file ships with this repository (there are no
//! pretrained weights checked into the workspace), so by default this
//! example builds a small **structural stand-in** for MobileNetV2
//! programmatically: a 3x3 stride-2 convolution stem, two
//! inverted-residual-style blocks (expand 1x1 -> ReLU -> spatial 3x3 -> ReLU
//! -> project 1x1, with a residual add when shapes allow it), a global
//! average pool, and a MatMul classification head — the same shape of graph
//! `tpt-infer-runtime`'s `mobilenet_v2_end_to_end` test fixture uses, just
//! reproduced here through the facade's public re-exports instead of
//! `tpt-infer-runtime`'s internal helpers.
//!
//! If a path is given on the command line, that file is loaded with
//! [`tpt_infer::prelude::load`] (feature `onnx`) instead.

use tpt_infer::prelude::*;

/// Deterministic LCG-distributed pseudo-random weights in `[-scale, scale]`,
/// so the example is reproducible without pulling in a `rand` dependency.
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

fn add_input(g: &mut ComputationGraph, dims: &[usize], name: &str) -> usize {
    let id = g.nodes().len();
    g.add_node(
        Node::new(id, Operator::Input, vec![], dims)
            .unwrap()
            .with_name(name),
    )
    .unwrap()
}

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

fn add_unary(g: &mut ComputationGraph, op: Operator, x: usize, odims: &[usize]) -> usize {
    let id = g.nodes().len();
    g.add_node(Node::new(id, op, vec![x], odims).unwrap())
        .unwrap()
}

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

/// One inverted-residual-style block: expand -> ReLU -> spatial -> ReLU ->
/// project (-> residual add when shapes match). Returns the output node id
/// and its `[n, c, h, w]` dims.
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

    *seed += 1;
    let mut t = add_conv(
        g,
        x,
        &format!("{tag}_expand_w"),
        &[c_expand, c_in, 1, 1],
        *seed,
        [1, 1],
        [0, 0],
        &dims_e,
    );
    t = add_unary(g, Operator::Relu, t, &dims_e);
    *seed += 1;
    t = add_conv(
        g,
        t,
        &format!("{tag}_spatial_w"),
        &[c_expand, c_expand, 3, 3],
        *seed,
        [stride, stride],
        [1, 1],
        &dims_s,
    );
    t = add_unary(g, Operator::Relu, t, &dims_s);
    *seed += 1;
    t = add_conv(
        g,
        t,
        &format!("{tag}_proj_w"),
        &[c_out, c_expand, 1, 1],
        *seed,
        [1, 1],
        [0, 0],
        &dims_p,
    );
    if stride == 1 && c_in == c_out {
        t = add_binary(g, Operator::Add, x, t, &dims_p);
    }
    (t, dims_p)
}

/// Builds the structural MobileNetV2-style stand-in graph (see module docs).
fn synthetic_mobilenet_v2() -> ComputationGraph {
    let mut g = ComputationGraph::new();
    let mut seed = 0u32;

    let x = add_input(&mut g, &[1, 8, 32, 32], "data");
    seed += 1;
    let mut cur = add_conv(
        &mut g,
        x,
        "stem_w",
        &[16, 8, 3, 3],
        seed,
        [2, 2],
        [1, 1],
        &[1, 16, 16, 16],
    );
    cur = add_unary(&mut g, Operator::Relu, cur, &[1, 16, 16, 16]);

    let (b1, d1) = add_ir_block(&mut g, cur, [1, 16, 16, 16], 32, 16, 2, "b1", &mut seed);
    let (b2, d2) = add_ir_block(&mut g, b1, d1, 32, 16, 1, "b2", &mut seed);

    let [n, c, h, w] = d2;
    let gap_id = g.nodes().len();
    let gap = g
        .add_node(
            Node::new(
                gap_id,
                Operator::average_pool2d([h, w], [1, 1], [0, 0]),
                vec![b2],
                &[n, c, 1, 1],
            )
            .unwrap(),
        )
        .unwrap();
    let flat = add_unary(&mut g, Operator::flatten(1), gap, &[n, c]);

    seed += 1;
    let fc = add_weight(&mut g, "fc_w", &[c, 1000], seed, 0.1);
    let head = add_binary(&mut g, Operator::MatMul, flat, fc, &[n, 1000]);
    g.mark_output(head).unwrap();
    g
}

fn main() {
    let path = std::env::args().nth(1);

    let (graph, input): (ComputationGraph, Vec<f32>) = match path {
        Some(p) => {
            #[cfg(feature = "onnx")]
            {
                println!("loading ONNX model from {p}");
                let graph = load(&p).expect("failed to load ONNX model");
                let dims = graph
                    .node(graph.inputs()[0])
                    .expect("graph has an input node")
                    .dims()
                    .to_vec();
                let numel: usize = dims.iter().product();
                let input = lcg(numel, 42, 0.5);
                (graph, input)
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = p;
                panic!("loading an ONNX file requires the `onnx` feature");
            }
        }
        None => {
            println!("no model path given; using a synthetic MobileNetV2-style graph");
            let graph = synthetic_mobilenet_v2();
            let input = lcg(8 * 32 * 32, 7, 0.5);
            (graph, input)
        }
    };

    println!(
        "graph: {} nodes, {} inputs, {} outputs",
        graph.nodes().len(),
        graph.inputs().len(),
        graph.outputs().len()
    );

    let mut mem = vec![0u8; required_arena_bytes(&graph) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let backend = select_backend();
    println!("executing on backend: {}", backend.name());

    let output = execute(&graph, &input, &mut arena, &backend).expect("graph execution failed");
    println!("output shape: {:?}", output.dims());

    let top1 = argmax(output.as_slice());
    println!("top-1 class: {top1} (score {:.4})", output.as_slice()[top1]);
}
