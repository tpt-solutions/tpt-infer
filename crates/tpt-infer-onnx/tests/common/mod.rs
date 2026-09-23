//! Shared helpers for building tiny, *structurally representative* ONNX
//! `ModelProto` fixtures via `prost` struct literals, and encoding them to
//! real `.onnx` bytes on disk.
//!
//! This module is compiled separately into each `tests/*.rs` integration
//! test binary (the standard `tests/common/mod.rs` pattern), so any given
//! binary only exercises a subset of these helpers — hence `dead_code` is
//! allowed here rather than per-binary warnings being treated as bugs.
#![allow(dead_code)]
//!
//! These are **not** the real published MobileNetV2 / BERT-tiny models —
//! see the doc comments on [`mobilenet_v2_style_bytes`] and
//! [`bert_tiny_style_bytes`] for exactly what each fixture stands in for.
//! They exist so integration tests can exercise `tpt_infer_onnx::load`
//! (the real file-path loading entry point) against an actual file on
//! disk, instead of only ever calling `load_from_bytes` on an in-memory
//! `ModelProto`.

use prost::Message;
use tpt_infer_onnx::proto::{
    tensor_proto::DataType, tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto,
    NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
};

/// Path to the checked-in fixture directory.
pub fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn f32_vi(name: &str, dims: &[i64]) -> ValueInfoProto {
    let shape = TensorShapeProto {
        dim: dims
            .iter()
            .map(|&d| tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                denotation: String::new(),
            })
            .collect(),
    };
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(shape),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
        metadata_props: vec![],
    }
}

pub fn node(input: &[&str], output: &[&str], name: &str, op_type: &str) -> NodeProto {
    NodeProto {
        input: input.iter().map(|s| s.to_string()).collect(),
        output: output.iter().map(|s| s.to_string()).collect(),
        name: name.into(),
        op_type: op_type.into(),
        domain: String::new(),
        overload: String::new(),
        attribute: vec![],
        doc_string: String::new(),
        metadata_props: vec![],
        device_configurations: vec![],
    }
}

fn empty_attr() -> AttributeProto {
    AttributeProto {
        name: String::new(),
        ref_attr_name: String::new(),
        doc_string: String::new(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
        r#type: 0,
    }
}

pub fn attr_ints(name: &str, ints: Vec<i64>) -> AttributeProto {
    let mut a = empty_attr();
    a.name = name.into();
    a.ints = ints;
    a
}

pub fn attr_f32(name: &str, f: f32) -> AttributeProto {
    let mut a = empty_attr();
    a.name = name.into();
    a.f = f;
    a
}

/// Single-`int` ONNX attribute (e.g. `axis`), as opposed to the `ints`-list
/// attributes built by [`attr_ints`] (e.g. `strides`, `pads`).
pub fn attr_int(name: &str, i: i64) -> AttributeProto {
    let mut a = empty_attr();
    a.name = name.into();
    a.i = i;
    a
}

/// Builds a float32 initializer tensor with deterministic (non-random,
/// reproducible) values so fixtures are byte-stable across regenerations.
pub fn f32_tensor(name: &str, dims: &[i64], values: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: DataType::Float as i32,
        float_data: values,
        name: name.into(),
        ..Default::default()
    }
}

/// Deterministic small values in `[-scale, scale]`, no external RNG crate
/// needed: a fixed-point sawtooth is enough for a structural fixture.
pub fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = (i % 17) as f32 / 17.0; // in [0, 1)
            (t * 2.0 - 1.0) * scale
        })
        .collect()
}

pub fn graph_of(
    name: &str,
    nodes: Vec<NodeProto>,
    input: Vec<ValueInfoProto>,
    output: Vec<ValueInfoProto>,
    initializer: Vec<TensorProto>,
) -> GraphProto {
    GraphProto {
        node: nodes,
        name: name.into(),
        initializer,
        sparse_initializer: vec![],
        doc_string: String::new(),
        input,
        output,
        value_info: vec![],
        quantization_annotation: vec![],
        metadata_props: vec![],
    }
}

pub fn model_bytes(graph: GraphProto) -> Vec<u8> {
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "tpt-infer-onnx-fixture-generator".into(),
        graph: Some(graph),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).unwrap();
    bytes
}

/// Builds a tiny, *structurally representative* MobileNetV2-style ONNX
/// model: a conv stem (Conv → BatchNorm → Relu), one inverted-residual-ish
/// block (Conv → BatchNorm → Relu, then a residual `Add` back to the stem
/// output), a `GlobalAveragePool`, a `Flatten`, and a `MatMul` + `Add`
/// classification head.
///
/// This is **not** the real pretrained MobileNetV2: spatial dims (16×16),
/// channel counts (8), and class count (10, vs. the real model's 1000) are
/// all shrunk to keep the fixture a few KB, and weights are deterministic
/// placeholder values, not trained parameters. It exists purely to give
/// `tpt_infer_onnx::load(path)` a real `.onnx` file on disk with a
/// MobileNetV2-shaped op sequence (conv/batchnorm/relu stem +
/// inverted-residual block + global-avg-pool + FC head) to parse.
pub fn mobilenet_v2_style_bytes() -> Vec<u8> {
    // Stem: Conv(3->8, k3 s2 p1) -> BatchNorm -> Relu.
    let conv1_w = f32_tensor(
        "conv1_w",
        &[8, 3, 3, 3],
        deterministic_values(8 * 3 * 3 * 3, 0.05),
    );
    let bn1_scale = f32_tensor("bn1_scale", &[8], vec![1.0; 8]);
    let bn1_bias = f32_tensor("bn1_bias", &[8], vec![0.0; 8]);
    let bn1_mean = f32_tensor("bn1_mean", &[8], vec![0.0; 8]);
    let bn1_var = f32_tensor("bn1_var", &[8], vec![1.0; 8]);

    let mut conv1 = node(&["x", "conv1_w"], &["c1"], "stem_conv", "Conv");
    conv1.attribute = vec![
        attr_ints("strides", vec![2, 2]),
        attr_ints("pads", vec![1, 1, 1, 1]),
        attr_ints("kernel_shape", vec![3, 3]),
    ];
    let bn1 = node(
        &["c1", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
        &["bn1"],
        "stem_bn",
        "BatchNormalization",
    );
    let relu1 = node(&["bn1"], &["r1"], "stem_relu", "Relu");

    // Inverted-residual-ish block: Conv(8->8, k3 s1 p1) -> BatchNorm -> Relu
    // -> residual Add back to the stem's Relu output (channels/spatial dims
    // match at stride 1, mirroring MobileNetV2's skip connection).
    let conv2_w = f32_tensor(
        "conv2_w",
        &[8, 8, 3, 3],
        deterministic_values(8 * 8 * 3 * 3, 0.05),
    );
    let bn2_scale = f32_tensor("bn2_scale", &[8], vec![1.0; 8]);
    let bn2_bias = f32_tensor("bn2_bias", &[8], vec![0.0; 8]);
    let bn2_mean = f32_tensor("bn2_mean", &[8], vec![0.0; 8]);
    let bn2_var = f32_tensor("bn2_var", &[8], vec![1.0; 8]);

    let mut conv2 = node(&["r1", "conv2_w"], &["c2"], "block_conv", "Conv");
    conv2.attribute = vec![
        attr_ints("strides", vec![1, 1]),
        attr_ints("pads", vec![1, 1, 1, 1]),
        attr_ints("kernel_shape", vec![3, 3]),
    ];
    let bn2 = node(
        &["c2", "bn2_scale", "bn2_bias", "bn2_mean", "bn2_var"],
        &["bn2"],
        "block_bn",
        "BatchNormalization",
    );
    let relu2 = node(&["bn2"], &["r2"], "block_relu", "Relu");
    let residual = node(&["r1", "r2"], &["res"], "block_residual_add", "Add");

    // Head: GlobalAveragePool -> Flatten -> MatMul -> Add (bias) -> logits.
    let gap = node(&["res"], &["gap"], "gap", "GlobalAveragePool");
    let flatten_node = {
        let mut n = node(&["gap"], &["flat"], "flatten", "Flatten");
        n.attribute = vec![attr_int("axis", 1)];
        n
    };
    let fc_w = f32_tensor("fc_w", &[8, 10], deterministic_values(8 * 10, 0.05));
    let fc_b = f32_tensor("fc_b", &[10], vec![0.0; 10]);
    let matmul = node(&["flat", "fc_w"], &["logits_pre"], "fc_matmul", "MatMul");
    let add_bias = node(&["logits_pre", "fc_b"], &["logits"], "fc_bias", "Add");

    let gproto = graph_of(
        "mobilenet_v2_style",
        vec![
            conv1,
            bn1,
            relu1,
            conv2,
            bn2,
            relu2,
            residual,
            gap,
            flatten_node,
            matmul,
            add_bias,
        ],
        vec![f32_vi("x", &[1, 3, 16, 16])],
        vec![f32_vi("logits", &[1, 10])],
        vec![
            conv1_w, bn1_scale, bn1_bias, bn1_mean, bn1_var, conv2_w, bn2_scale, bn2_bias,
            bn2_mean, bn2_var, fc_w, fc_b,
        ],
    );
    model_bytes(gproto)
}

/// Builds a tiny, *structurally representative* single-transformer-block
/// BERT-tiny-style ONNX model: Q/K/V linear projections (`MatMul` + `Add`
/// each), a simplified single-token self-attention
/// (`Transpose` + `MatMul` + `Softmax` + `MatMul`), a residual `Add` back
/// to the input embedding, a `BatchNormalization` node standing in for
/// `LayerNormalization` (not natively modeled by this crate's registry —
/// see `registry.rs`), and a final `Relu` activation.
///
/// This is **not** the real pretrained BERT-tiny: hidden size is 8 (vs.
/// BERT-tiny's 128), sequence length is collapsed to a single token
/// (vs. real multi-token attention over a sequence), and weights are
/// deterministic placeholder values, not trained parameters. It exists
/// purely to give `tpt_infer_onnx::load(path)` a real `.onnx` file on disk
/// with a transformer-block-shaped tensor flow (QKV projections,
/// attention-style matmul/softmax/matmul, residual add, normalization
/// stand-in) to parse.
pub fn bert_tiny_style_bytes() -> Vec<u8> {
    let hidden = 8i64;

    let mk_proj = |prefix: &str| {
        (
            f32_tensor(
                &format!("{prefix}_w"),
                &[hidden, hidden],
                deterministic_values((hidden * hidden) as usize, 0.05),
            ),
            f32_tensor(
                &format!("{prefix}_b"),
                &[hidden],
                vec![0.0; hidden as usize],
            ),
        )
    };

    let (q_w, q_b) = mk_proj("q");
    let (k_w, k_b) = mk_proj("k");
    let (v_w, v_b) = mk_proj("v");

    let q_mm = node(&["x", "q_w"], &["q_pre"], "q_matmul", "MatMul");
    let q_add = node(&["q_pre", "q_b"], &["q"], "q_bias", "Add");
    let k_mm = node(&["x", "k_w"], &["k_pre"], "k_matmul", "MatMul");
    let k_add = node(&["k_pre", "k_b"], &["k"], "k_bias", "Add");
    let v_mm = node(&["x", "v_w"], &["v_pre"], "v_matmul", "MatMul");
    let v_add = node(&["v_pre", "v_b"], &["v"], "v_bias", "Add");

    // Single-token "attention": scores = q @ k^T, softmax, context = scores @ v.
    let k_t = {
        let mut n = node(&["k"], &["k_t"], "k_transpose", "Transpose");
        n.attribute = vec![attr_ints("perm", vec![1, 0])];
        n
    };
    let scores = node(&["q", "k_t"], &["scores"], "attn_scores", "MatMul");
    let attn = {
        let mut n = node(&["scores"], &["attn"], "attn_softmax", "Softmax");
        n.attribute = vec![attr_int("axis", -1)];
        n
    };
    let context = node(&["attn", "v"], &["context"], "attn_context", "MatMul");

    // Residual add back to the input embedding.
    let residual = node(&["context", "x"], &["res"], "residual_add", "Add");

    // LayerNorm stand-in: BatchNormalization is the closest natively-mapped
    // normalization op in registry.rs (LayerNormalization itself maps to
    // `Operator::Custom`, which the runtime cannot execute yet).
    let ln_scale = f32_tensor("ln_scale", &[hidden], vec![1.0; hidden as usize]);
    let ln_bias = f32_tensor("ln_bias", &[hidden], vec![0.0; hidden as usize]);
    let ln_mean = f32_tensor("ln_mean", &[hidden], vec![0.0; hidden as usize]);
    let ln_var = f32_tensor("ln_var", &[hidden], vec![1.0; hidden as usize]);
    let norm = {
        let mut n = node(
            &["res", "ln_scale", "ln_bias", "ln_mean", "ln_var"],
            &["normed"],
            "layernorm_stand_in",
            "BatchNormalization",
        );
        n.attribute = vec![attr_f32("epsilon", 1e-5)];
        n
    };

    let act = node(&["normed"], &["y"], "act", "Relu");

    let gproto = graph_of(
        "bert_tiny_style",
        vec![
            q_mm, q_add, k_mm, k_add, v_mm, v_add, k_t, scores, attn, context, residual, norm, act,
        ],
        vec![f32_vi("x", &[1, hidden])],
        vec![f32_vi("y", &[1, hidden])],
        vec![
            q_w, q_b, k_w, k_b, v_w, v_b, ln_scale, ln_bias, ln_mean, ln_var,
        ],
    );
    model_bytes(gproto)
}
