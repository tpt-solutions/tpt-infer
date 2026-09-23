//! Load ONNX `ModelProto` files into a [`ComputationGraph`].

use std::collections::HashMap;
use std::path::Path;

use prost::Message;
use tpt_infer_graph::{ComputationGraph, Edge, Initializer, Node, Operator};

use crate::registry::map_op;
use crate::shapes::infer_node_shape;

/// Include the prost-generated ONNX message types.
#[allow(clippy::all, clippy::pedantic)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use proto::{
    tensor_proto::DataType, AttributeProto, GraphProto, ModelProto, OperatorSetIdProto,
    TensorProto, ValueInfoProto,
};

/// Errors from loading an ONNX model.
#[derive(Debug)]
#[non_exhaustive]
pub enum OnnxError {
    /// Filesystem or read failure.
    Io(std::io::Error),
    /// Protobuf decode failure.
    Decode(prost::DecodeError),
    /// Model graph missing or malformed.
    InvalidModel(String),
    /// Graph construction failed.
    Graph(tpt_infer_graph::GraphError),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "io error: {e}"),
            OnnxError::Decode(e) => write!(f, "protobuf decode error: {e}"),
            OnnxError::InvalidModel(m) => write!(f, "invalid model: {m}"),
            OnnxError::Graph(e) => write!(f, "graph: {e}"),
        }
    }
}

impl std::error::Error for OnnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxError::Io(e) => Some(e),
            OnnxError::Decode(e) => Some(e),
            OnnxError::Graph(e) => Some(e),
            OnnxError::InvalidModel(_) => None,
        }
    }
}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}

impl From<prost::DecodeError> for OnnxError {
    fn from(e: prost::DecodeError) -> Self {
        OnnxError::Decode(e)
    }
}

impl From<tpt_infer_graph::GraphError> for OnnxError {
    fn from(e: tpt_infer_graph::GraphError) -> Self {
        OnnxError::Graph(e)
    }
}

/// Load an ONNX model from `path`.
///
/// # Errors
/// [`OnnxError`] on IO, decode, or graph construction failure.
///
/// # Example
/// ```no_run
/// let g = tpt_infer_onnx::load("model.onnx")?;
/// assert!(!g.nodes().is_empty());
/// # Ok::<(), tpt_infer_onnx::OnnxError>(())
/// ```
pub fn load(path: impl AsRef<Path>) -> Result<ComputationGraph, OnnxError> {
    let bytes = std::fs::read(path)?;
    load_from_bytes(&bytes)
}

/// Load an ONNX model from in-memory bytes.
///
/// # Errors
/// See [`load`].
pub fn load_from_bytes(bytes: &[u8]) -> Result<ComputationGraph, OnnxError> {
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| OnnxError::InvalidModel("model has no graph".into()))?;
    graph_from_proto(&graph, model.opset_import)
}

/// Convert a decoded [`GraphProto`] into a [`ComputationGraph`].
pub fn graph_from_proto(
    graph: &GraphProto,
    _opsets: Vec<OperatorSetIdProto>,
) -> Result<ComputationGraph, OnnxError> {
    if graph.node.is_empty() {
        return Err(OnnxError::InvalidModel("graph has no nodes".into()));
    }

    let mut value_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for vi in &graph.input {
        if let Some(dims) = vi_dims(vi) {
            value_shapes.insert(vi.name.clone(), dims);
        }
    }
    for vi in &graph.output {
        if let Some(dims) = vi_dims(vi) {
            value_shapes.entry(vi.name.clone()).or_insert(dims);
        }
    }
    for vi in &graph.value_info {
        if let Some(dims) = vi_dims(vi) {
            value_shapes.entry(vi.name.clone()).or_insert(dims);
        }
    }

    let mut out = ComputationGraph::new();
    let mut name_to_node: HashMap<String, usize> = HashMap::new();

    // True graph inputs (names that are not initializers).
    let init_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();
    for vi in &graph.input {
        if init_names.contains(vi.name.as_str()) {
            continue;
        }
        let dims = value_shapes
            .get(&vi.name)
            .cloned()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| vec![1]);
        let node = Node::new(
            out.nodes().len(),
            Operator::Input,
            vec![],
            &dims,
        )?
        .with_name(&vi.name);
        let id = out.add_node(node)?;
        name_to_node.insert(vi.name.clone(), id);
    }

    // Initializer weight inputs.
    for init in &graph.initializer {
        let mut dims: Vec<usize> = init.dims.iter().map(|&d| d.max(0) as usize).collect();
        if dims.is_empty() {
            dims.push(1);
        }
        let node = Node::new(
            out.nodes().len(),
            Operator::Input,
            vec![],
            &dims,
        )?
        .with_name(&init.name);
        let id = out.add_node(node)?;
        name_to_node.insert(init.name.clone(), id);
        if let Some(data) = tensor_f32(init) {
            // Size must match; skip mismatched (e.g. raw_data empty).
            let expected: usize = dims.iter().product();
            if data.len() == expected {
                out.add_initializer(Initializer::new(&init.name, &dims, data)?);
            }
        }
    }

    // ONNX graphs are topologically sorted.
    for onnx_node in &graph.node {
        let op = build_operator(onnx_node, graph);
        let mut inputs = Vec::with_capacity(onnx_node.input.len());
        for in_name in &onnx_node.input {
            if in_name.is_empty() {
                continue;
            }
            let id = if let Some(&id) = name_to_node.get(in_name.as_str()) {
                id
            } else {
                let node = Node::new(out.nodes().len(), Operator::Input, vec![], &[1])?
                    .with_name(in_name.as_str());
                out.add_node(node)?
            };
            inputs.push(id);
        }

        let in_dims: Vec<Vec<usize>> = inputs
            .iter()
            .map(|&i| out.nodes()[i].dims().to_vec())
            .collect();
        let declared = onnx_node
            .output
            .first()
            .and_then(|n| value_shapes.get(n))
            .cloned();
        let dims = match declared {
            Some(d) if !d.is_empty() => d,
            _ => infer_node_shape(&op, &in_dims, &[1]),
        };

        let mut node = Node::new(out.nodes().len(), op, inputs.clone(), &dims)?;
        node.name = Some(onnx_node.name.clone());
        let id = out.add_node(node)?;

        if let Some(out_name) = onnx_node.output.first() {
            if !out_name.is_empty() {
                name_to_node.insert(out_name.clone(), id);
            }
        }

        for (port, &src) in inputs.iter().enumerate() {
            let shape = out.nodes()[src].output_shape;
            let rank = out.nodes()[src].output_rank;
            out.add_edge(Edge {
                from: src,
                from_port: 0,
                to: id,
                to_port: port,
                shape,
                rank,
            })?;
        }
    }

    let mut any_marked = false;
    for oi in &graph.output {
        if let Some(&id) = name_to_node.get(oi.name.as_str()) {
            out.mark_output(id)?;
            any_marked = true;
        }
    }
    if !any_marked {
        out.infer_outputs();
    }

    Ok(out)
}

fn build_operator(node: &proto::NodeProto, _graph: &GraphProto) -> Operator {
    let mut strides = [1usize, 1usize];
    let mut padding = [0usize, 2usize]; // placeholder, overwritten below
    let mut padding_set = false;
    let mut kernel = [0usize, 2usize];
    let mut axis_i: Option<i32> = None;
    let mut epsilon = 1e-5f32;
    let mut perm: Vec<i64> = Vec::new();
    let mut reshape_shape: Option<Vec<i64>> = None;

    for a in &node.attribute {
        match a.name.as_str() {
            "strides" => {
                let v = ints_of(a);
                if v.len() >= 2 {
                    strides = [v[0].max(1) as usize, v[1].max(1) as usize];
                }
            }
            "pads" => {
                let v = ints_of(a);
                if v.len() >= 2 {
                    padding = [v[0].max(0) as usize, v[1].max(0) as usize];
                    padding_set = true;
                }
            }
            "kernel_shape" => {
                let v = ints_of(a);
                if v.len() >= 2 {
                    kernel = [v[0] as usize, v[1] as usize];
                }
            }
            "axis" => {
                axis_i = Some(a.i as i32);
            }
            "epsilon" => {
                epsilon = a.f;
            }
            "perm" => {
                perm = ints_of(a);
            }
            "shape" => {
                reshape_shape = Some(ints_of(a));
            }
            _ => {}
        }
    }
    if !padding_set {
        padding = [0, 0];
    }

    // Reshape target can come from the second input initializer (int64 shape
    // tensor). Float weight blobs are not reshape shapes; skip them.
    if reshape_shape.is_none() {
        // Left as map_op default; shape attr is the common path above.
    }

    let op_type = node.op_type.as_str();
    let op = map_op(op_type);

    match op_type {
        "Conv" => Operator::Conv2d {
            strides,
            padding,
        },
        "MaxPool" => Operator::MaxPool2d {
            kernel: if kernel == [0, 0] {
                [2, 2]
            } else {
                kernel
            },
            strides,
            padding,
        },
        "AveragePool" => Operator::AveragePool2d {
            kernel: if kernel == [0, 0] {
                [2, 2]
            } else {
                kernel
            },
            strides,
            padding,
        },
        "GlobalAveragePool" => Operator::AveragePool2d {
            kernel: [0, 0],
            strides: [1, 1],
            padding: [0, 0],
        },
        "Softmax" => Operator::Softmax {
            axis: axis_i.unwrap_or(-1),
        },
        "BatchNormalization" => Operator::BatchNorm { epsilon },
        "Flatten" => Operator::Flatten {
            axis: axis_i.unwrap_or(1),
        },
        "Concat" => Operator::Concat {
            axis: axis_i.unwrap_or(1),
        },
        "Transpose" => {
            if perm.is_empty() {
                Operator::Transpose {
                    perm: [0; tpt_infer_core::MAX_RANK],
                    rank: 0,
                }
            } else {
                let mut p = [0; tpt_infer_core::MAX_RANK];
                let rank = perm.len().min(tpt_infer_core::MAX_RANK);
                for (i, &v) in perm.iter().take(rank).enumerate() {
                    p[i] = v.max(0) as usize;
                }
                Operator::Transpose {
                    perm: p,
                    rank: perm.len(),
                }
            }
        }
        "Reshape" => {
            if let Some(s) = reshape_shape {
                let mut shape = [0; tpt_infer_core::MAX_RANK];
                let rank = s.len().min(tpt_infer_core::MAX_RANK);
                for (i, &d) in s.iter().take(rank).enumerate() {
                    shape[i] = if d < 0 { 0 } else { d as usize };
                }
                Operator::Reshape {
                    shape,
                    rank: s.len(),
                }
            } else {
                op
            }
        }
        _ => op,
    }
}

fn ints_of(a: &AttributeProto) -> Vec<i64> {
    if !a.ints.is_empty() {
        a.ints.clone()
    } else if a.i != 0 {
        vec![a.i]
    } else {
        Vec::new()
    }
}

fn vi_dims(vi: &ValueInfoProto) -> Option<Vec<usize>> {
    let tt = vi.r#type.as_ref()?;
    let tensor = tt.value.as_ref()?;
    use proto::type_proto::Value;
    let Value::TensorType(t) = tensor else {
        return None;
    };
    let shape = t.shape.as_ref()?;
    let mut dims = Vec::new();
    for d in &shape.dim {
        use proto::tensor_shape_proto::dimension::Value as DimValue;
        match d.value.as_ref() {
            Some(DimValue::DimValue(v)) if *v > 0 => dims.push(*v as usize),
            _ => dims.push(0),
        }
    }
    Some(dims)
}

fn tensor_f32(t: &TensorProto) -> Option<Vec<f32>> {
    if t.data_type != DataType::Float as i32 {
        return None;
    }
    if !t.raw_data.is_empty() {
        let bytes = &t.raw_data;
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        return Some(out);
    }
    if !t.float_data.is_empty() {
        return Some(t.float_data.clone());
    }
    Some(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{
        type_proto, tensor_shape_proto, NodeProto, TensorShapeProto, TypeProto,
    };

    fn f32_vi(name: &str, dims: &[i64]) -> ValueInfoProto {
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

    fn node(input: &[&str], output: &[&str], name: &str, op_type: &str) -> NodeProto {
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

    fn empty_attrs() -> AttributeProto {
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

    fn attr_ints(name: &str, ints: Vec<i64>) -> AttributeProto {
        let mut a = empty_attrs();
        a.name = name.into();
        a.ints = ints;
        a
    }

    fn model(graph: GraphProto) -> Vec<u8> {
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "test".into(),
            graph: Some(graph),
            ..Default::default()
        };
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        bytes
    }

    fn graph_of(name: &str, nodes: Vec<NodeProto>, input: Vec<ValueInfoProto>, output: Vec<ValueInfoProto>) -> GraphProto {
        GraphProto {
            node: nodes,
            name: name.into(),
            initializer: vec![],
            sparse_initializer: vec![],
            doc_string: String::new(),
            input,
            output,
            value_info: vec![],
            quantization_annotation: vec![],
            metadata_props: vec![],
        }
    }

    #[test]
    fn decode_simple_relu() {
        let bytes = model(graph_of(
            "relu",
            vec![node(&["x"], &["y"], "relu0", "Relu")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 4])],
        ));
        let g = load_from_bytes(&bytes).unwrap();
        assert_eq!(g.nodes().len(), 2);
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.outputs(), &[1]);
        assert!(matches!(g.nodes()[1].operator, Operator::Relu));
        assert_eq!(g.nodes()[1].dims(), &[1, 4]);
        assert_eq!(g.topological_sort().unwrap(), vec![0, 1]);
    }

    #[test]
    fn load_with_initializer_matmul() {
        let w_data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let w_init = TensorProto {
            dims: vec![4, 5],
            data_type: DataType::Float as i32,
            float_data: w_data,
            name: "w".into(),
            ..Default::default()
        };
        let mut gproto = graph_of(
            "mm",
            vec![node(&["x", "w"], &["y"], "mm0", "MatMul")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 5])],
        );
        gproto.initializer = vec![w_init];
        let bytes = model(gproto);
        let g = load_from_bytes(&bytes).unwrap();
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.initializers().len(), 1);
        assert_eq!(g.initializers()[0].name, "w");
        assert_eq!(g.initializers()[0].data.len(), 20);
        assert_eq!(g.nodes()[2].dims(), &[1, 5]);
    }

    #[test]
    fn load_conv_mobilenet_style_stem() {
        let oc = 32usize;
        let n_w = oc * 3 * 3 * 3;
        let w = TensorProto {
            dims: vec![oc as i64, 3, 3, 3],
            data_type: DataType::Float as i32,
            float_data: vec![0.1; n_w],
            name: "conv_w".into(),
            ..Default::default()
        };
        let mut conv = node(&["x", "conv_w"], &["c"], "conv", "Conv");
        conv.attribute = vec![
            attr_ints("strides", vec![2, 2]),
            attr_ints("pads", vec![1, 1, 1, 1]),
            attr_ints("kernel_shape", vec![3, 3]),
        ];
        let mut gproto = graph_of(
            "mobilenet_stem",
            vec![
                conv,
                node(&["c"], &["r"], "relu", "Relu"),
            ],
            vec![f32_vi("x", &[1, 3, 224, 224])],
            vec![f32_vi("r", &[1, 32, 112, 112])],
        );
        gproto.initializer = vec![w];
        let bytes = model(gproto);
        let g = load_from_bytes(&bytes).unwrap();
        assert_eq!(g.nodes().len(), 4);
        assert_eq!(g.initializers().len(), 1);
        let conv_node = g
            .nodes()
            .iter()
            .find(|n| matches!(n.operator, Operator::Conv2d { .. }))
            .unwrap();
        assert_eq!(conv_node.dims(), &[1, 32, 112, 112]);
        assert_eq!(
            conv_node.operator,
            Operator::Conv2d {
                strides: [2, 2],
                padding: [1, 1]
            }
        );
        assert_eq!(g.topological_sort().unwrap().len(), 4);
    }

    #[test]
    fn load_bert_tiny_structure() {
        let w1 = TensorProto {
            dims: vec![4, 8],
            data_type: DataType::Float as i32,
            float_data: vec![0.01; 32],
            name: "w1".into(),
            ..Default::default()
        };
        let b1 = TensorProto {
            dims: vec![8],
            data_type: DataType::Float as i32,
            float_data: vec![0.0; 8],
            name: "b1".into(),
            ..Default::default()
        };
        let mut gproto = graph_of(
            "bert_tiny_mlp",
            vec![
                node(&["x", "w1"], &["h"], "mm", "MatMul"),
                node(&["h", "b1"], &["hb"], "bias", "Add"),
                node(&["hb"], &["y"], "gelu", "Gelu"),
            ],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 8])],
        );
        gproto.initializer = vec![w1, b1];
        let bytes = model(gproto);
        let g = load_from_bytes(&bytes).unwrap();
        assert_eq!(g.nodes().len(), 6);
        assert_eq!(g.initializers().len(), 2);
        assert_eq!(g.topological_sort().unwrap().len(), 6);
        let last = g.nodes().last().unwrap();
        assert_eq!(last.dims(), &[1, 8]);
        assert!(matches!(last.operator, Operator::Gelu));
        assert_eq!(g.outputs(), &[5]);
    }

    #[test]
    fn dynamic_batch_zero() {
        let bytes = model(graph_of(
            "dyn",
            vec![node(&["x"], &["y"], "r", "Relu")],
            vec![f32_vi("x", &[-1, 4])],
            vec![f32_vi("y", &[-1, 4])],
        ));
        let g = load_from_bytes(&bytes).unwrap();
        assert_eq!(g.nodes()[0].dims(), &[0, 4]);
        assert_eq!(g.nodes()[1].dims(), &[0, 4]);
    }
}
