//! Integration test: build a 2-layer MLP, verify topological order and shapes.
//!
//! Graph topology (dynamic [`ComputationGraph`] construction):
//!
//! ```text
//! x  [1,784] ─┐
//!             ├─ MatMul ─ [1,128] ─ Relu ─┐
//! w1 [784,128]─┘                          ├─ MatMul ─ [1,10] ─ Softmax
//!                              w2 [128,10]─┘
//! ```

use tpt_infer_graph::{ComputationGraph, Edge, GraphBuilder, Node, Operator, Sh};

/// Build the MLP dynamically: Input → MatMul → Relu → MatMul → Softmax.
fn build_mlp() -> ComputationGraph {
    let mut g = ComputationGraph::new();

    let x = g
        .add_node(
            Node::new(0, Operator::Input, vec![], &[1, 784])
                .unwrap()
                .with_name("x"),
        )
        .unwrap();
    let w1 = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[784, 128])
                .unwrap()
                .with_name("w1"),
        )
        .unwrap();
    let m1 = g
        .add_node(Node::new(2, Operator::MatMul, vec![x, w1], &[1, 128]).unwrap())
        .unwrap();
    let r = g
        .add_node(Node::new(3, Operator::Relu, vec![m1], &[1, 128]).unwrap())
        .unwrap();
    let w2 = g
        .add_node(
            Node::new(4, Operator::Input, vec![], &[128, 10])
                .unwrap()
                .with_name("w2"),
        )
        .unwrap();
    let m2 = g
        .add_node(Node::new(5, Operator::MatMul, vec![r, w2], &[1, 10]).unwrap())
        .unwrap();
    let out = g
        .add_node(Node::new(6, Operator::softmax(-1), vec![m2], &[1, 10]).unwrap())
        .unwrap();

    g.add_edge(Edge::new(x, 0, m1, 0, &[1, 784]).unwrap())
        .unwrap();
    g.add_edge(Edge::new(w1, 0, m1, 1, &[784, 128]).unwrap())
        .unwrap();
    g.add_edge(Edge::new(m1, 0, r, 0, &[1, 128]).unwrap())
        .unwrap();
    g.add_edge(Edge::new(r, 0, m2, 0, &[1, 128]).unwrap())
        .unwrap();
    g.add_edge(Edge::new(w2, 0, m2, 1, &[128, 10]).unwrap())
        .unwrap();
    g.add_edge(Edge::new(m2, 0, out, 0, &[1, 10]).unwrap())
        .unwrap();

    g.mark_output(out).unwrap();
    g
}

#[test]
fn mlp_topological_order_respects_dependencies() {
    let g = build_mlp();
    let order = g.topological_sort().unwrap();
    assert_eq!(order.len(), g.nodes().len());

    let pos = |id: usize| order.iter().position(|&n| n == id).unwrap();
    // Every consumer must come after both of its producers.
    assert!(pos(0) < pos(2), "x before first MatMul");
    assert!(pos(1) < pos(2), "w1 before first MatMul");
    assert!(pos(2) < pos(3), "MatMul before Relu");
    assert!(pos(3) < pos(5), "Relu before second MatMul");
    assert!(pos(4) < pos(5), "w2 before second MatMul");
    assert!(pos(5) < pos(6), "MatMul before Softmax");
}

#[test]
fn mlp_shapes_are_as_expected() {
    let g = build_mlp();
    assert_eq!(g.node(0).unwrap().dims(), &[1, 784]);
    assert_eq!(g.node(1).unwrap().dims(), &[784, 128]);
    assert_eq!(g.node(2).unwrap().dims(), &[1, 128]);
    assert_eq!(g.node(3).unwrap().dims(), &[1, 128]);
    assert_eq!(g.node(4).unwrap().dims(), &[128, 10]);
    assert_eq!(g.node(5).unwrap().dims(), &[1, 10]);
    assert_eq!(g.node(6).unwrap().dims(), &[1, 10]);
}

#[test]
fn mlp_inputs_outputs_and_names() {
    let g = build_mlp();
    // Operator::Input nodes are auto-registered, in insertion order.
    assert_eq!(g.inputs(), &[0, 1, 4]);
    assert_eq!(g.outputs(), &[6]);
    assert_eq!(g.node(0).unwrap().name.as_deref(), Some("x"));
    assert_eq!(g.node(1).unwrap().name.as_deref(), Some("w1"));
    assert_eq!(g.node(4).unwrap().name.as_deref(), Some("w2"));
    assert_eq!(g.node(2).unwrap().name, None);
}

#[test]
fn mlp_edges_carry_shapes() {
    let g = build_mlp();
    let edges = g.edges();
    assert_eq!(edges.len(), 6);

    let shapes: Vec<Vec<usize>> = edges.iter().map(|e| e.dims().to_vec()).collect();
    assert_eq!(
        shapes,
        vec![
            vec![1, 784],
            vec![784, 128],
            vec![1, 128],
            vec![1, 128],
            vec![128, 10],
            vec![1, 10],
        ]
    );

    // Port wiring: weights enter MatMul on input port 1.
    let w1_edge = edges.iter().find(|e| e.from == 1).unwrap();
    assert_eq!((w1_edge.to, w1_edge.to_port), (2, 1));
}

#[test]
fn mlp_type_state_builder_matches_dynamic_graph() {
    let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
    let w1 = b.add_input::<Sh<784, 128>>("w1");
    let mut b = b.matmul(w1).relu();
    let w2 = b.add_input::<Sh<128, 10>>("w2");
    let g = b.matmul(w2).softmax(-1).into_graph();

    let dynamic = build_mlp();

    assert_eq!(g.nodes().len(), dynamic.nodes().len());
    assert_eq!(g.inputs(), dynamic.inputs());
    assert_eq!(g.outputs(), dynamic.outputs());
    for (a, b) in g.nodes().iter().zip(dynamic.nodes()) {
        assert_eq!(a.operator, b.operator, "node {}", a.id);
        assert_eq!(a.dims(), b.dims(), "node {}", a.id);
        assert_eq!(a.inputs, b.inputs, "node {}", a.id);
    }

    let order = g.topological_sort().unwrap();
    let pos = |id: usize| order.iter().position(|&n| n == id).unwrap();
    assert!(pos(0) < pos(2) && pos(2) < pos(3) && pos(3) < pos(5) && pos(5) < pos(6));
}
