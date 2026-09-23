//! Integration tests that exercise `tpt_infer_onnx::load(path)` — the real
//! file-reading entry point (as opposed to `load_from_bytes`) — against
//! actual `.onnx` files on disk under `tests/fixtures/`.
//!
//! The fixtures are small-but-real ONNX protobuf files that are
//! *structurally representative* of MobileNetV2 and BERT-tiny (see
//! `tests/common/mod.rs` for exactly what they stand in for and what they
//! deliberately are not). Closes the "load MobileNetV2.onnx" / "load
//! BERT-tiny.onnx" integration test gaps in todo.md.

mod common;

use tpt_infer_graph::Operator;

#[test]
fn load_mobilenet_v2_style_fixture_from_disk() {
    let path = common::fixtures_dir().join("mobilenet_v2_style.onnx");
    assert!(path.exists(), "fixture missing: {}", path.display());

    let g = tpt_infer_onnx::load(&path).expect("real file-path load must succeed");

    // conv1_w, bn1 x4 params, relu1's conv, bn, relu nodes, conv2_w, bn2 x4
    // params, block conv/bn/relu, residual add, gap, flatten, fc_w, matmul,
    // fc_b, add-bias, plus the graph input "x" = plenty of nodes.
    assert!(
        g.nodes().len() >= 20,
        "expected a multi-node graph, got {}",
        g.nodes().len()
    );
    assert_eq!(g.initializers().len(), 12, "12 weight/param tensors");

    // Graph input is the [1, 3, 16, 16] image-like tensor.
    assert_eq!(
        g.inputs().len(),
        1 + 12,
        "1 activation input + 12 weight inputs"
    );
    let x = g
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("x"))
        .expect("input node named x");
    assert_eq!(x.dims(), &[1, 3, 16, 16]);

    // Output is the 10-class logits vector.
    assert_eq!(g.outputs().len(), 1);
    let out_node = &g.nodes()[g.outputs()[0]];
    assert_eq!(out_node.dims(), &[1, 10]);

    // The op sequence includes the conv/batchnorm/relu stem, an
    // inverted-residual-ish block, a global average pool, and a
    // MatMul-based classification head.
    let has = |op: fn(&Operator) -> bool| g.nodes().iter().any(|n| op(&n.operator));
    assert!(has(|o| matches!(o, Operator::Conv2d { .. })), "has Conv2d");
    assert!(
        has(|o| matches!(o, Operator::BatchNorm { .. })),
        "has BatchNorm"
    );
    assert!(has(|o| matches!(o, Operator::Relu)), "has Relu");
    assert!(has(|o| matches!(o, Operator::Add)), "has residual/bias Add");
    // GlobalAveragePool is mapped to `AveragePool2d` with its kernel resolved
    // to the input's actual spatial extent (not the `[0, 0]` sentinel) so
    // the runtime can execute it; see the comment in `load.rs`.
    assert!(
        has(|o| matches!(
            o,
            Operator::AveragePool2d {
                kernel: [k0, k1],
                ..
            } if *k0 > 0 && *k1 > 0
        )),
        "has GlobalAveragePool resolved to a concrete kernel"
    );
    assert!(
        has(|o| matches!(o, Operator::Flatten { .. })),
        "has Flatten"
    );
    assert!(has(|o| matches!(o, Operator::MatMul)), "has MatMul head");

    // Shape inference + topological sort both succeed end to end.
    let order = g.topological_sort().expect("acyclic graph sorts");
    assert_eq!(order.len(), g.nodes().len());
}

#[test]
fn load_bert_tiny_style_fixture_from_disk() {
    let path = common::fixtures_dir().join("bert_tiny_style.onnx");
    assert!(path.exists(), "fixture missing: {}", path.display());

    let g = tpt_infer_onnx::load(&path).expect("real file-path load must succeed");

    assert!(
        g.nodes().len() >= 15,
        "expected a multi-node transformer-block graph, got {}",
        g.nodes().len()
    );
    assert_eq!(
        g.initializers().len(),
        10,
        "q/k/v weights+biases + 4 norm params"
    );

    let x = g
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("x"))
        .expect("input node named x");
    assert_eq!(x.dims(), &[1, 8]);

    assert_eq!(g.outputs().len(), 1);
    let out_node = &g.nodes()[g.outputs()[0]];
    assert_eq!(out_node.dims(), &[1, 8]);

    // QKV projections (MatMul+Add), attention-ish matmul/softmax/matmul,
    // residual add, and a normalization stand-in are all present.
    let count = |op: fn(&Operator) -> bool| g.nodes().iter().filter(|n| op(&n.operator)).count();
    assert!(
        count(|o| matches!(o, Operator::MatMul)) >= 5,
        "q/k/v projections + attention scores + context matmuls"
    );
    assert!(
        count(|o| matches!(o, Operator::Add)) >= 4,
        "q/k/v biases + residual add"
    );
    assert_eq!(count(|o| matches!(o, Operator::Softmax { .. })), 1);
    assert_eq!(
        count(|o| matches!(o, Operator::BatchNorm { .. })),
        1,
        "layernorm stand-in"
    );
    assert_eq!(count(|o| matches!(o, Operator::Relu)), 1);

    let order = g.topological_sort().expect("acyclic graph sorts");
    assert_eq!(order.len(), g.nodes().len());
}
