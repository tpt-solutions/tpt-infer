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

impl From<crate::shapes::ShapeInferError> for OnnxError {
    fn from(e: crate::shapes::ShapeInferError) -> Self {
        OnnxError::InvalidModel(format!("shape inference failed: {e}"))
    }
}

/// Default maximum accepted size, in bytes, for a model passed to [`load`]/
/// [`load_from_bytes`]. Guards against decoding an excessively large or
/// corrupt file; use [`load_from_bytes_with_limits`] to override.
pub const DEFAULT_MAX_MODEL_BYTES: usize = 512 * 1024 * 1024;

/// Default maximum accepted total node + attribute count across the graph,
/// checked after decoding. Bounds the work a crafted file can force.
pub const DEFAULT_MAX_GRAPH_ITEMS: usize = 1_000_000;

/// Highest ONNX opset version (default `""` domain) this parser targets.
pub const MAX_SUPPORTED_OPSET: i64 = 17;

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

/// Load an ONNX model from in-memory bytes, using [`DEFAULT_MAX_MODEL_BYTES`]
/// and [`DEFAULT_MAX_GRAPH_ITEMS`] as resource limits.
///
/// # Errors
/// See [`load`].
pub fn load_from_bytes(bytes: &[u8]) -> Result<ComputationGraph, OnnxError> {
    load_from_bytes_with_limits(bytes, DEFAULT_MAX_MODEL_BYTES, DEFAULT_MAX_GRAPH_ITEMS)
}

/// Load an ONNX model from in-memory bytes, with explicit resource limits.
///
/// `max_bytes` bounds the size of `bytes` itself (checked before decoding).
/// `max_graph_items` bounds the total number of nodes plus attributes in the
/// decoded graph (checked after decoding, before further processing). Use
/// this directly — with tighter limits — on memory-constrained targets, or
/// to accept models larger than the defaults allow.
///
/// # Errors
/// [`OnnxError::InvalidModel`] if either limit is exceeded; see [`load`] for
/// other error cases.
pub fn load_from_bytes_with_limits(
    bytes: &[u8],
    max_bytes: usize,
    max_graph_items: usize,
) -> Result<ComputationGraph, OnnxError> {
    if bytes.len() > max_bytes {
        return Err(OnnxError::InvalidModel(format!(
            "model is {} bytes, exceeding the {max_bytes}-byte limit",
            bytes.len()
        )));
    }
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| OnnxError::InvalidModel("model has no graph".into()))?;
    let attr_count: usize = graph.node.iter().map(|n| n.attribute.len()).sum();
    let item_count = graph.node.len().saturating_add(attr_count);
    if item_count > max_graph_items {
        return Err(OnnxError::InvalidModel(format!(
            "graph has {item_count} nodes+attributes, exceeding the {max_graph_items}-item limit"
        )));
    }
    graph_from_proto(&graph, model.opset_import)
}

/// Convert a decoded [`GraphProto`] into a [`ComputationGraph`].
///
/// # Errors
/// [`OnnxError::InvalidModel`] if the graph has no nodes, or if `opsets`
/// declares a default-domain opset version newer than [`MAX_SUPPORTED_OPSET`].
pub fn graph_from_proto(
    graph: &GraphProto,
    opsets: Vec<OperatorSetIdProto>,
) -> Result<ComputationGraph, OnnxError> {
    if graph.node.is_empty() {
        return Err(OnnxError::InvalidModel("graph has no nodes".into()));
    }
    for opset in &opsets {
        if opset.domain.is_empty() && opset.version > MAX_SUPPORTED_OPSET {
            return Err(OnnxError::InvalidModel(format!(
                "unsupported opset version {} for the default domain (this parser targets opset {MAX_SUPPORTED_OPSET})",
                opset.version
            )));
        }
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
        let node =
            Node::new(out.nodes().len(), Operator::Input, vec![], &dims)?.with_name(&vi.name);
        let id = out.add_node(node)?;
        name_to_node.insert(vi.name.clone(), id);
    }

    // Initializer weight inputs.
    for init in &graph.initializer {
        if init.dims.iter().any(|&d| d < 0) {
            return Err(OnnxError::InvalidModel(format!(
                "initializer '{}' has a negative declared dimension",
                init.name
            )));
        }
        let mut dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        if dims.is_empty() {
            dims.push(1);
        }
        let node =
            Node::new(out.nodes().len(), Operator::Input, vec![], &dims)?.with_name(&init.name);
        let id = out.add_node(node)?;
        name_to_node.insert(init.name.clone(), id);

        // `INT64` initializers (shape/index tensors feeding e.g. `Reshape`,
        // `Slice`, `Gather` — ubiquitous in real ONNX exports) have no
        // dedicated storage in `Initializer` (which is `f32`-only), but
        // every value in a shape/index tensor is a small integer that
        // round-trips losslessly through `f32`. Binding them this way
        // (rather than leaving them unbound) is what lets `execute`/
        // `aot_compile` treat this node as a resolved constant instead of a
        // dangling free input the caller could never supply data for.
        let data = tensor_f32(init).or_else(|| {
            tensor_i64(init).map(|ints| ints.into_iter().map(|v| v as f32).collect())
        });
        if let Some(data) = data {
            let expected: usize = dims
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "initializer '{}' declared dimensions overflow",
                        init.name
                    ))
                })?;
            if data.len() != expected {
                return Err(OnnxError::InvalidModel(format!(
                    "initializer '{}' declares {expected} elements but has {}",
                    init.name,
                    data.len()
                )));
            }
            out.add_initializer(Initializer::new(&init.name, &dims, data)?);
        }
    }

    // ONNX graphs are topologically sorted.
    for onnx_node in &graph.node {
        let mut op = build_operator(onnx_node, graph);
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

        // `GlobalAveragePool` is mapped to `AveragePool2d` with a `[0, 0]`
        // sentinel kernel (see `build_operator`/`registry::map_op`) since the
        // real spatial extent isn't known until the input's shape is. Now
        // that `in_dims` is available, resolve it to the actual `[H, W]` of
        // the input feature map so the runtime (which pools over exactly
        // `kernel` elements, not a sentinel) executes it correctly.
        if let Operator::AveragePool2d {
            kernel: [0, 0],
            strides,
            padding,
        } = op
        {
            if let Some(x) = in_dims.first().filter(|d| d.len() == 4) {
                op = Operator::AveragePool2d {
                    kernel: [x[2], x[3]],
                    strides,
                    padding,
                };
            }
        }

        // `auto_pad = SAME_UPPER`/`SAME_LOWER` (common in real exports —
        // e.g. models converted from frameworks whose native op computes
        // "same" padding rather than emitting explicit `pads`) means the
        // padding actually applied depends on the input's spatial size,
        // which (like `GlobalAveragePool` above) is only known now that
        // `in_dims` is available. `VALID`/`NOTSET` need no fixup (`NOTSET`
        // already used the explicit `pads` attribute, if any, in
        // `build_operator`; `VALID` means zero padding).
        if let Some(auto_pad) = auto_pad_attr(onnx_node) {
            if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
                match &mut op {
                    Operator::Conv2d { strides, padding } => {
                        if let (Some(x), Some(w)) = (
                            in_dims.first().filter(|d| d.len() == 4),
                            in_dims.get(1).filter(|d| d.len() == 4),
                        ) {
                            *padding = [
                                same_padding(x[2], w[2], strides[0]),
                                same_padding(x[3], w[3], strides[1]),
                            ];
                        }
                    }
                    Operator::MaxPool2d {
                        kernel,
                        strides,
                        padding,
                    }
                    | Operator::AveragePool2d {
                        kernel,
                        strides,
                        padding,
                    } => {
                        if let Some(x) = in_dims.first().filter(|d| d.len() == 4) {
                            *padding = [
                                same_padding(x[2], kernel[0], strides[0]),
                                same_padding(x[3], kernel[1], strides[1]),
                            ];
                        }
                    }
                    _ => {}
                }
            }
        }

        let declared = onnx_node
            .output
            .first()
            .and_then(|n| value_shapes.get(n))
            .cloned();
        let dims = match declared {
            Some(d) if !d.is_empty() => d,
            _ => infer_node_shape(&op, &in_dims, &[1])?,
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

fn build_operator(node: &proto::NodeProto, graph: &GraphProto) -> Operator {
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

    // Modern ONNX exports (opset >= 5) pass Reshape's target shape as its
    // second *input* (an int64 initializer) rather than a `shape`
    // attribute. Resolve it here so `Operator::Reshape` always carries a
    // concrete compile-time shape when one is statically known.
    if reshape_shape.is_none() && node.op_type == "Reshape" {
        if let Some(shape_name) = node.input.get(1) {
            reshape_shape = int64_initializer(graph, shape_name);
        }
    }

    let op_type = node.op_type.as_str();
    let op = map_op(op_type);

    match op_type {
        "Conv" => Operator::Conv2d { strides, padding },
        "MaxPool" => Operator::MaxPool2d {
            kernel: if kernel == [0, 0] { [2, 2] } else { kernel },
            strides,
            padding,
        },
        "AveragePool" => Operator::AveragePool2d {
            kernel: if kernel == [0, 0] { [2, 2] } else { kernel },
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

/// The `auto_pad` string attribute (`"NOTSET"`/`"VALID"`/`"SAME_UPPER"`/
/// `"SAME_LOWER"`), if present.
fn auto_pad_attr(node: &proto::NodeProto) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == "auto_pad")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
}

/// Symmetric padding that approximates ONNX's `SAME_UPPER`/`SAME_LOWER`
/// `auto_pad` modes for one spatial dimension.
///
/// ONNX's real semantics can be *asymmetric* (a different amount of padding
/// at the start vs. end of the axis), which `Operator::Conv2d`'s
/// single-`usize`-per-axis `padding` can't represent — but the total
/// padding is only odd when `kernel` is even (rare for real conv/pool
/// kernels, which are overwhelmingly odd-sized), so `total_pad / 2` is
/// exact in the common case and a one-off-per-side approximation otherwise.
fn same_padding(input: usize, kernel: usize, stride: usize) -> usize {
    let stride = stride.max(1);
    let out = input.div_ceil(stride);
    let needed = out.saturating_sub(1) * stride + kernel;
    needed.saturating_sub(input) / 2
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

/// Reads an `INT64` tensor's data (e.g. a `Reshape`/`Shape`/`Slice` target,
/// which ONNX always stores as int64, never as an attribute in modern
/// exports). Returns `None` for any other dtype.
fn tensor_i64(t: &TensorProto) -> Option<Vec<i64>> {
    if t.data_type != DataType::Int64 as i32 {
        return None;
    }
    if !t.raw_data.is_empty() {
        let bytes = &t.raw_data;
        let mut out = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            out.push(i64::from_le_bytes(chunk.try_into().expect("8-byte chunk")));
        }
        return Some(out);
    }
    if !t.int64_data.is_empty() {
        return Some(t.int64_data.clone());
    }
    Some(Vec::new())
}

/// Finds `name` among `graph`'s initializers and, if it's an `INT64`
/// tensor, reads its data — used to resolve a `Reshape` node's target shape
/// when it comes from a graph input rather than a `shape` attribute (the
/// common case for models exported by modern ONNX converters).
fn int64_initializer(graph: &GraphProto, name: &str) -> Option<Vec<i64>> {
    graph
        .initializer
        .iter()
        .find(|init| init.name == name)
        .and_then(tensor_i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{tensor_shape_proto, type_proto, NodeProto, TensorShapeProto, TypeProto};

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

    fn attr_str(name: &str, value: &str) -> AttributeProto {
        let mut a = empty_attrs();
        a.name = name.into();
        a.s = value.as_bytes().to_vec();
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

    fn graph_of(
        name: &str,
        nodes: Vec<NodeProto>,
        input: Vec<ValueInfoProto>,
        output: Vec<ValueInfoProto>,
    ) -> GraphProto {
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
            vec![conv, node(&["c"], &["r"], "relu", "Relu")],
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
    fn conv_auto_pad_same_upper_resolves_symmetric_padding() {
        // Real exports (e.g. from frameworks whose native op computes "same"
        // padding) often use `auto_pad` instead of an explicit `pads`
        // attribute. For an odd kernel with stride 1, SAME_UPPER/SAME_LOWER
        // padding is exactly symmetric: total_pad = kernel - 1 (always
        // even), split evenly on both sides.
        let oc = 8usize;
        let w = TensorProto {
            dims: vec![oc as i64, 1, 5, 5],
            data_type: DataType::Float as i32,
            float_data: vec![0.1; oc * 25],
            name: "conv_w".into(),
            ..Default::default()
        };
        let mut conv = node(&["x", "conv_w"], &["c"], "conv", "Conv");
        conv.attribute = vec![
            attr_ints("strides", vec![1, 1]),
            attr_ints("kernel_shape", vec![5, 5]),
            attr_str("auto_pad", "SAME_UPPER"),
        ];
        let mut gproto = graph_of(
            "auto_pad_conv",
            vec![conv],
            vec![f32_vi("x", &[1, 1, 28, 28])],
            vec![f32_vi("c", &[1, 8, 28, 28])],
        );
        gproto.initializer = vec![w];
        let bytes = model(gproto);
        let g = load_from_bytes(&bytes).unwrap();
        let conv_node = g
            .nodes()
            .iter()
            .find(|n| matches!(n.operator, Operator::Conv2d { .. }))
            .unwrap();
        assert_eq!(
            conv_node.operator,
            Operator::Conv2d {
                strides: [1, 1],
                padding: [2, 2],
            }
        );
        // Output spatial size matches input (that's the point of "same" padding).
        assert_eq!(conv_node.dims(), &[1, 8, 28, 28]);
    }

    #[test]
    fn reshape_target_from_int64_initializer_input() {
        // Modern ONNX exports pass Reshape's target shape as a second
        // *input* (an int64 initializer), not a `shape` attribute. This
        // must resolve to a concrete `Operator::Reshape` shape, and the
        // int64 tensor must be bound as an initializer (not left as a
        // dangling free runtime input the caller could never supply data for).
        let shape_init = TensorProto {
            dims: vec![2],
            data_type: DataType::Int64 as i32,
            int64_data: vec![1, 12],
            name: "target_shape".into(),
            ..Default::default()
        };
        let reshape = node(&["x", "target_shape"], &["y"], "reshape0", "Reshape");
        let mut gproto = graph_of(
            "reshape_from_input",
            vec![reshape],
            vec![f32_vi("x", &[1, 3, 4])],
            vec![f32_vi("y", &[1, 12])],
        );
        gproto.initializer = vec![shape_init];
        let bytes = model(gproto);
        let g = load_from_bytes(&bytes).unwrap();

        let reshape_node = g
            .nodes()
            .iter()
            .find(|n| matches!(n.operator, Operator::Reshape { .. }))
            .expect("reshape node present");
        match &reshape_node.operator {
            Operator::Reshape { shape, rank } => {
                assert_eq!(&shape[..*rank], &[1, 12]);
            }
            _ => unreachable!(),
        }

        assert!(
            g.initializers().iter().any(|i| i.name == "target_shape"),
            "int64 shape tensor must be bound as an initializer"
        );
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

    #[test]
    fn oversized_model_bytes_rejected() {
        let bytes = model(graph_of(
            "relu",
            vec![node(&["x"], &["y"], "relu0", "Relu")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 4])],
        ));
        let err = load_from_bytes_with_limits(&bytes, bytes.len() - 1, DEFAULT_MAX_GRAPH_ITEMS)
            .unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }

    #[test]
    fn oversized_graph_item_count_rejected() {
        let bytes = model(graph_of(
            "relu",
            vec![node(&["x"], &["y"], "relu0", "Relu")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 4])],
        ));
        let err = load_from_bytes_with_limits(&bytes, DEFAULT_MAX_MODEL_BYTES, 0).unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }

    #[test]
    fn unsupported_opset_version_rejected() {
        let graph = graph_of(
            "relu",
            vec![node(&["x"], &["y"], "relu0", "Relu")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 4])],
        );
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: MAX_SUPPORTED_OPSET + 1,
            }],
            producer_name: "test".into(),
            graph: Some(graph),
            ..Default::default()
        };
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        let err = load_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }

    #[test]
    fn negative_initializer_dim_rejected() {
        let w = TensorProto {
            dims: vec![-4, 5],
            data_type: DataType::Float as i32,
            float_data: vec![0.0; 20],
            name: "w".into(),
            ..Default::default()
        };
        let mut gproto = graph_of(
            "mm",
            vec![node(&["x", "w"], &["y"], "mm0", "MatMul")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 5])],
        );
        gproto.initializer = vec![w];
        let bytes = model(gproto);
        let err = load_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }

    #[test]
    fn mismatched_initializer_length_rejected() {
        // Previously this silently dropped the initializer (leaving a
        // dangling weight-input node with no bound data) instead of
        // rejecting the malformed model.
        let w = TensorProto {
            dims: vec![4, 5], // expects 20 elements
            data_type: DataType::Float as i32,
            float_data: vec![0.0; 3],
            name: "w".into(),
            ..Default::default()
        };
        let mut gproto = graph_of(
            "mm",
            vec![node(&["x", "w"], &["y"], "mm0", "MatMul")],
            vec![f32_vi("x", &[1, 4])],
            vec![f32_vi("y", &[1, 5])],
        );
        gproto.initializer = vec![w];
        let bytes = model(gproto);
        let err = load_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }

    #[test]
    fn conv_kernel_larger_than_input_rejected() {
        let w = TensorProto {
            dims: vec![1, 1, 999, 999],
            data_type: DataType::Float as i32,
            float_data: vec![0.0; 999 * 999],
            name: "w".into(),
            ..Default::default()
        };
        let mut conv = node(&["x", "w"], &["y"], "conv", "Conv");
        conv.attribute = vec![
            attr_ints("strides", vec![1, 1]),
            attr_ints("pads", vec![0, 0, 0, 0]),
            attr_ints("kernel_shape", vec![999, 999]),
        ];
        // No declared shape for "y": this forces `infer_node_shape` to run
        // conv_shape's checked geometry arithmetic instead of short-circuiting
        // on a caller-declared output shape.
        let unshaped_output = ValueInfoProto {
            name: "y".into(),
            r#type: None,
            doc_string: String::new(),
            metadata_props: vec![],
        };
        let mut gproto = graph_of(
            "conv_bad",
            vec![conv],
            vec![f32_vi("x", &[1, 1, 4, 4])],
            vec![unshaped_output],
        );
        gproto.initializer = vec![w];
        let bytes = model(gproto);
        let err = load_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, OnnxError::InvalidModel(_)));
    }
}
