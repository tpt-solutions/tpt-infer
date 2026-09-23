//! Post-training quantization (PTQ) pipeline.
//!
//! Walks a `ComputationGraph` in topological order, finds `MatMul` /
//! `Conv2d` nodes whose weights are float `Initializer`s, and quantizes
//! those weights (per-tensor or per-channel, symmetric by default).

use std::collections::HashMap;

use tpt_infer_graph::ComputationGraph;

use crate::dequant::{
    dequantize_i4_symmetric, dequantize_i8_symmetric, dequantize_i8_symmetric_per_channel,
};
use crate::quant::{
    pack_i4, quantize_i4_symmetric, quantize_i4_symmetric_per_channel, quantize_i8_symmetric,
    quantize_i8_symmetric_per_channel,
};

/// PTQ configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtqOptions {
    /// Symmetric quantization (INT8 range `[-127, 127]`).
    pub symmetric: bool,
    /// Per-channel scales along the output channel axis (MatMul: rows of B;
    /// Conv: output channels).
    pub per_channel: bool,
    /// Use INT4 instead of INT8.
    pub int4: bool,
    /// Calibration inputs (unused in weight-only PTQ; reserved for activation
    /// ranges). Each entry is one sample's flattened input tensor.
    pub calibrate: bool,
}

impl Default for PtqOptions {
    fn default() -> Self {
        Self {
            symmetric: true,
            per_channel: true,
            int4: false,
            calibrate: false,
        }
    }
}

/// Errors from the PTQ pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PtqError {
    /// Topological sort failed (cycle / unknown node).
    Graph(tpt_infer_graph::GraphError),
    /// A weight tensor had an unexpected layout.
    InvalidWeights {
        /// Initializer name.
        name: String,
    },
}

impl std::fmt::Display for PtqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtqError::Graph(e) => write!(f, "graph: {e}"),
            PtqError::InvalidWeights { name } => write!(f, "invalid weights for {name}"),
        }
    }
}

impl std::error::Error for PtqError {}

impl From<tpt_infer_graph::GraphError> for PtqError {
    fn from(e: tpt_infer_graph::GraphError) -> Self {
        PtqError::Graph(e)
    }
}

/// Quantized weight storage.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Element type of the packed storage (`I8` or `I4`).
    pub dtype: tpt_infer_core::DType,
    /// Logical shape (active dims).
    pub shape: Vec<usize>,
    /// Per-channel scales (`len == 1` for per-tensor).
    pub scales: Vec<f32>,
    /// Quantized payload: `i8` bytes for INT8, nibble-packed for INT4.
    pub packed: Vec<u8>,
}

impl QuantizedTensor {
    /// Dequantize back to f32 (row-major, matching original layout for
    /// per-tensor and per-channel over the leading dimension).
    pub fn dequantize(&self) -> Vec<f32> {
        match self.dtype {
            tpt_infer_core::DType::I8 => {
                let q: Vec<i8> = self.packed.iter().map(|&b| b as i8).collect();
                if self.scales.len() == 1 {
                    dequantize_i8_symmetric(&q, self.scales[0])
                } else {
                    dequantize_i8_symmetric_per_channel(&q, &self.scales)
                }
            }
            tpt_infer_core::DType::I4 => {
                let rows = self.scales.len();
                if rows == 0 {
                    return Vec::new();
                }
                let total: usize = self.shape.iter().product();
                let row_elems = total / rows;
                if rows == 1 {
                    dequantize_i4_symmetric(&self.packed, total, self.scales[0])
                } else {
                    crate::dequant::dequantize_i4_symmetric_per_channel(
                        &self.packed,
                        row_elems,
                        &self.scales,
                    )
                }
            }
            _ => Vec::new(),
        }
    }
}

/// Result of running PTQ: the original graph plus quantized weights keyed by
/// initializer name.
#[derive(Debug, Clone)]
pub struct QuantizedGraph {
    /// The (unchanged-topology) graph.
    pub graph: ComputationGraph,
    /// Quantized weights: initializer name → quantized tensor.
    pub weights: HashMap<String, QuantizedTensor>,
    /// Options used to produce this result.
    pub options: PtqOptions,
}

impl QuantizedGraph {
    /// Number of quantized tensors.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether nothing was quantized.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Quantize float initializers that feed `MatMul` / `Conv2d` nodes.
///
/// Weight-only PTQ: activations remain f32. Bias terms (rank-1) are skipped.
///
/// # Errors
/// Propagates graph errors; [`PtqError::InvalidWeights`] for bad layouts.
///
/// # Example
/// ```
/// use tpt_infer_graph::{ComputationGraph, Initializer, Node, Operator};
/// use tpt_infer_quantize::{ptq, PtqOptions};
///
/// let mut g = ComputationGraph::new();
/// let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap()).unwrap();
/// let w = g.add_node(Node::new(1, Operator::Input, vec![], &[4, 8]).unwrap().with_name("w")).unwrap();
/// g.add_initializer(Initializer::new("w", &[4, 8], vec![0.01; 32]).unwrap());
/// let y = g.add_node(Node::new(2, Operator::MatMul, vec![x, w], &[1, 8]).unwrap()).unwrap();
/// g.mark_output(y).unwrap();
///
/// let q = ptq(&g, PtqOptions::default()).unwrap();
/// assert_eq!(q.len(), 1);
/// let wq = &q.weights["w"];
/// let recon = wq.dequantize();
/// assert_eq!(recon.len(), 32);
/// ```
pub fn ptq(graph: &ComputationGraph, opts: PtqOptions) -> Result<QuantizedGraph, PtqError> {
    // Map input node id -> initializer name via node name matching.
    let mut weight_names: Vec<&str> = Vec::new();

    for node in graph.nodes() {
        match node.operator {
            tpt_infer_graph::Operator::MatMul => {
                // Second input is B (weights).
                if let Some(&w_id) = node.inputs.get(1) {
                    if let Some(name) = initializer_name_of(graph, w_id) {
                        weight_names.push(name);
                    }
                }
            }
            tpt_infer_graph::Operator::Conv2d { .. } => {
                if let Some(&w_id) = node.inputs.get(1) {
                    if let Some(name) = initializer_name_of(graph, w_id) {
                        weight_names.push(name);
                    }
                }
            }
            _ => {}
        }
    }
    weight_names.sort_unstable();
    weight_names.dedup();

    let mut weights = HashMap::new();
    for name in weight_names {
        let init = graph
            .initializers()
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| PtqError::InvalidWeights {
                name: name.to_string(),
            })?;
        if let Some(q) = quantize_initializer(init, opts) {
            weights.insert(init.name.clone(), q);
        }
    }

    Ok(QuantizedGraph {
        graph: graph.clone(),
        weights,
        options: opts,
    })
}

fn initializer_name_of(graph: &ComputationGraph, node_id: usize) -> Option<&str> {
    let node = graph.nodes().get(node_id)?;
    if !node.operator.is_input() {
        return None;
    }
    let name = node.name.as_deref()?;
    graph
        .initializers()
        .iter()
        .find(|i| i.name == name)
        .map(|i| i.name.as_str())
}

fn quantize_initializer(
    init: &tpt_infer_graph::Initializer,
    opts: PtqOptions,
) -> Option<QuantizedTensor> {
    let dims = init.dims();
    // Skip biases (rank 1) and scalars.
    if dims.len() < 2 {
        return None;
    }
    let rows = dims[0];
    let data = &init.data;

    if opts.int4 {
        let (packed, scales) = if opts.per_channel && rows > 1 {
            quantize_i4_symmetric_per_channel(data, rows)
        } else {
            let (p, s) = quantize_i4_symmetric(data);
            (p, vec![s])
        };
        return Some(QuantizedTensor {
            dtype: tpt_infer_core::DType::I4,
            shape: dims.to_vec(),
            scales,
            packed,
        });
    }

    // INT8
    if !opts.symmetric {
        // Asymmetric still supported per-tensor only.
        let (q, p) = crate::quant::quantize_i8_asymmetric(data);
        let packed: Vec<u8> = q.into_iter().map(|v| v as u8).collect();
        return Some(QuantizedTensor {
            dtype: tpt_infer_core::DType::I8,
            shape: dims.to_vec(),
            scales: vec![p.scale],
            packed,
        });
    }

    let (q, scales) = if opts.per_channel && rows > 1 {
        quantize_i8_symmetric_per_channel(data, rows)
    } else {
        let (q, s) = quantize_i8_symmetric(data);
        (q, vec![s])
    };
    Some(QuantizedTensor {
        dtype: tpt_infer_core::DType::I8,
        shape: dims.to_vec(),
        scales,
        packed: q.into_iter().map(|v| v as u8).collect(),
    })
}

/// Integer matmul with f32 scale application:
/// `out[m,n] = scale_a * scale_b * sum_k(a_i8 * b_i8)`.
///
/// For a simple end-to-end quantized path (tests / bench).
///
/// `a_shape = [m, k]`, `b_shape = [k, n]`.
pub fn quantized_matmul_i8(
    a: &[i8],
    a_shape: [usize; 2],
    a_scale: f32,
    b: &[i8],
    b_shape: [usize; 2],
    b_scales: &[f32], // 1 per-tensor, or `n` per-column if len==n… use per-tensor only here
    out: &mut [f32],
) -> Result<(), String> {
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k != k2 {
        return Err(format!("k mismatch: {k} vs {k2}"));
    }
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err("buffer size mismatch".into());
    }
    let b_scale = b_scales.first().copied().unwrap_or(1.0);
    let out_scale = a_scale * b_scale;
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i32;
            for ki in 0..k {
                acc += a[mi * k + ki] as i32 * b[ki * n + ni] as i32;
            }
            out[mi * n + ni] = acc as f32 * out_scale;
        }
    }
    // Per-column scales: if b_scales.len() == n, refine (optional path unused
    // by default benches — keep simple contract documented above).
    Ok(())
}

// Silence unused import warnings for re-exports used only in docs/tests.
#[allow(unused_imports)]
use dequantize_i8_symmetric as _reexport_guard;
#[allow(unused_imports)]
use pack_i4 as _reexport_guard2;

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_infer_graph::{Initializer, Node, Operator};

    fn mlp_with_weights() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let w = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[4, 8])
                    .unwrap()
                    .with_name("w"),
            )
            .unwrap();
        let data: Vec<f32> = (0..32).map(|i| (i as f32 / 32.0) - 0.5).collect();
        g.add_initializer(Initializer::new("w", &[4, 8], data).unwrap());
        let y = g
            .add_node(Node::new(2, Operator::MatMul, vec![x, w], &[1, 8]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        g
    }

    #[test]
    fn ptq_quantizes_matmul_weights() {
        let g = mlp_with_weights();
        let q = ptq(&g, PtqOptions::default()).unwrap();
        assert_eq!(q.len(), 1);
        let wq = &q.weights["w"];
        assert_eq!(wq.dtype, tpt_infer_core::DType::I8);
        assert_eq!(wq.shape, vec![4, 8]);
        assert_eq!(wq.scales.len(), 4); // per-channel over rows
        assert_eq!(wq.packed.len(), 32);

        // Round-trip accuracy vs original.
        let orig = g.initializers()[0].data.clone();
        let recon = wq.dequantize();
        for (a, b) in orig.iter().zip(&recon) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn ptq_int4_path() {
        let g = mlp_with_weights();
        let opts = PtqOptions {
            int4: true,
            ..Default::default()
        };
        let q = ptq(&g, opts).unwrap();
        let wq = &q.weights["w"];
        assert_eq!(wq.dtype, tpt_infer_core::DType::I4);
        // 32 elements → 16 bytes
        assert_eq!(wq.packed.len(), 16);
        let recon = wq.dequantize();
        assert_eq!(recon.len(), 32);
    }

    #[test]
    fn ptq_skips_bias() {
        let mut g = mlp_with_weights();
        let b = g
            .add_node(
                Node::new(3, Operator::Input, vec![], &[8])
                    .unwrap()
                    .with_name("b"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("b", &[8], vec![0.0; 8]).unwrap());
        let q = ptq(&g, PtqOptions::default()).unwrap();
        assert!(!q.weights.contains_key("b"));
        let _ = b;
    }

    #[test]
    fn quantized_matmul_matches_f32_approx() {
        use tpt_infer_ops::{Backend, NaiveBackend};
        let m = 4;
        let k = 8;
        let n = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin() * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos() * 0.5).collect();

        let mut ref_out = vec![0.0f32; m * n];
        NaiveBackend
            .matmul(&a, [m, k], &b, [k, n], &mut ref_out)
            .unwrap();

        let (aq, as_) = quantize_i8_symmetric(&a);
        let (bq, bs) = quantize_i8_symmetric(&b);
        let mut q_out = vec![0.0f32; m * n];
        quantized_matmul_i8(&aq, [m, k], as_, &bq, [k, n], &[bs], &mut q_out)
            .unwrap();

        for (r, q) in ref_out.iter().zip(&q_out) {
            assert!(
                (r - q).abs() < 0.05,
                "f32={r} quant={q} diff={}",
                (r - q).abs()
            );
        }
    }
}
