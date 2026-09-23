//! Shape inference over an ONNX graph once nodes are mapped to [`Operator`]s.
//!
//! Dynamic dimensions (ONNX dim_param / `dim_value = 0`) are represented as
//! `0` in [`Node::output_shape`](tpt_infer_graph::Node::output_shape).

use std::collections::HashMap;

use tpt_infer_graph::{ComputationGraph, Operator};

/// Infer each node's output shape from its inputs.
///
/// Returns a map from node id to the inferred active dims.
///
/// Input and initializer-seeded nodes use declared/known shapes first.
pub fn infer_shapes(
    graph: &ComputationGraph,
    value_shapes: &HashMap<String, Vec<usize>>,
) -> Result<HashMap<usize, Vec<usize>>, ShapeInferError> {
    let mut out = HashMap::new();
    // Seed inputs from declared value shapes / initializers.
    let mut init_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for init in graph.initializers() {
        init_shapes.insert(init.name.clone(), init.dims().to_vec());
    }
    for &id in graph.inputs() {
        let node = &graph.nodes()[id];
        let dims: Vec<usize> = if let Some(name) = &node.name {
            value_shapes
                .get(name)
                .or_else(|| init_shapes.get(name))
                .cloned()
                .unwrap_or_else(|| node.dims().to_vec())
        } else {
            node.dims().to_vec()
        };
        out.insert(id, dims);
    }

    for id in 0..graph.nodes().len() {
        if out.contains_key(&id) {
            continue;
        }
        let node = &graph.nodes()[id];
        let in_dims: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .map(|&i| out.get(&i).cloned().unwrap_or_default())
            .collect();
        let dims = infer_node_shape(&node.operator, &in_dims, node.dims())?;
        out.insert(id, dims);
    }
    Ok(out)
}

/// Shape inference for a single operator given input dims.
///
/// Falls back to `fallback` (the node's current dims) when inference is not
/// possible (unknown inputs or unsupported op).
///
/// # Errors
/// [`ShapeInferError::InvalidGeometry`] if a convolution/pooling kernel does
/// not fit within its (padded) input — this can only happen with
/// attacker-controlled or otherwise malformed `dims`/`kernel_shape`/`pads`.
pub fn infer_node_shape(
    op: &Operator,
    inputs: &[Vec<usize>],
    fallback: &[usize],
) -> Result<Vec<usize>, ShapeInferError> {
    match op {
        Operator::Input => Ok(fallback.to_vec()),
        Operator::MatMul => Ok(matmul_shape(inputs, fallback)),
        Operator::Conv2d {
            strides, padding, ..
        } => conv_shape(inputs, *strides, *padding, fallback),
        Operator::Add
        | Operator::Sub
        | Operator::Mul
        | Operator::Div
        | Operator::Relu
        | Operator::Sigmoid
        | Operator::Gelu
        | Operator::BatchNorm { .. }
        | Operator::Custom(_) => Ok(inputs
            .first()
            .filter(|d| !d.is_empty())
            .cloned()
            .unwrap_or_else(|| fallback.to_vec())),
        Operator::Softmax { .. } => Ok(inputs
            .first()
            .filter(|d| !d.is_empty())
            .cloned()
            .unwrap_or_else(|| fallback.to_vec())),
        Operator::Reshape { shape, rank, .. } => Ok(if *rank > 0 {
            shape[..*rank].to_vec()
        } else {
            fallback.to_vec()
        }),
        Operator::Flatten { axis } => {
            let Some(inp) = inputs.first().filter(|d| !d.is_empty()) else {
                return Ok(fallback.to_vec());
            };
            let rank = inp.len() as i32;
            let ax = if *axis < 0 { rank + *axis } else { *axis };
            let ax = ax.max(0) as usize;
            let outer: usize = inp[..ax.min(inp.len())].iter().product();
            let inner: usize = inp[ax.min(inp.len())..].iter().product();
            Ok(vec![outer, inner])
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
        } => pool_shape(inputs, *kernel, *strides, *padding, fallback),
        Operator::Concat { axis } => {
            let Some(first) = inputs.first().filter(|d| !d.is_empty()) else {
                return Ok(fallback.to_vec());
            };
            if inputs.len() == 1 {
                return Ok(first.clone());
            }
            let rank = first.len() as i32;
            let ax = if *axis < 0 { rank + *axis } else { *axis };
            let ax = ax.max(0) as usize;
            let mut out = first.clone();
            if ax >= out.len() {
                return Ok(out);
            }
            let mut sum = 0usize;
            for d in inputs {
                if d.len() > ax {
                    sum += d[ax];
                }
            }
            out[ax] = sum;
            Ok(out)
        }
        Operator::Transpose { perm, rank, .. } => {
            let Some(inp) = inputs.first().filter(|d| !d.is_empty()) else {
                return Ok(fallback.to_vec());
            };
            if *rank == inp.len() {
                let mut out = vec![0; inp.len()];
                for i in 0..*rank {
                    out[i] = inp[perm[i]];
                }
                Ok(out)
            } else {
                // default reverse
                let mut out = inp.clone();
                out.reverse();
                Ok(out)
            }
        }
        _ => Ok(fallback.to_vec()),
    }
}

fn matmul_shape(inputs: &[Vec<usize>], fallback: &[usize]) -> Vec<usize> {
    if inputs.len() < 2 {
        return fallback.to_vec();
    }
    let (a, b) = (&inputs[0], &inputs[1]);
    if a.len() < 2 || b.len() < 2 {
        return fallback.to_vec();
    }
    // Broadcast batch dims, then [m,k]x[k,n].
    let a_batch = &a[..a.len() - 2];
    let a_m = a[a.len() - 2];
    let a_k = a[a.len() - 1];
    let b_batch = &b[..b.len() - 2];
    let b_k = b[b.len() - 2];
    let b_n = b[b.len() - 1];
    if a_k != b_k && a_k != 0 && b_k != 0 {
        // mismatch — leave fallback for caller to validate
        return fallback.to_vec();
    }
    let mut batch = broadcast_dims(a_batch, b_batch);
    batch.push(a_m);
    batch.push(b_n);
    batch
}

fn broadcast_dims(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < a.len() { a[i] } else { 1 };
        let db = if i < b.len() { b[i] } else { 1 };
        out.push(if da == 1 { db } else { da });
    }
    out
}

/// Compute a single spatial output dimension for conv/pool:
/// `(input + 2*pad - kernel) / stride + 1`, using checked arithmetic
/// throughout since every operand can originate from an attacker-controlled
/// ONNX file.
fn conv_out_dim(
    input: usize,
    pad: usize,
    kernel: usize,
    stride: usize,
) -> Result<usize, ShapeInferError> {
    let padded = pad
        .checked_mul(2)
        .and_then(|p| input.checked_add(p))
        .ok_or(ShapeInferError::InvalidGeometry)?;
    let numerator = padded
        .checked_sub(kernel)
        .ok_or(ShapeInferError::InvalidGeometry)?;
    Ok(numerator / stride + 1)
}

fn conv_shape(
    inputs: &[Vec<usize>],
    strides: [usize; 2],
    padding: [usize; 2],
    fallback: &[usize],
) -> Result<Vec<usize>, ShapeInferError> {
    if inputs.len() < 2 {
        return Ok(fallback.to_vec());
    }
    let x = &inputs[0];
    let w = &inputs[1];
    if x.len() != 4 || w.len() != 4 {
        return Ok(fallback.to_vec());
    }
    let n = x[0];
    let oc = w[0];
    let (sh, sw) = (strides[0].max(1), strides[1].max(1));
    let (ph, pw) = (padding[0], padding[1]);
    let oh = if x[2] == 0 {
        0
    } else {
        conv_out_dim(x[2], ph, w[2], sh)?
    };
    let ow = if x[3] == 0 {
        0
    } else {
        conv_out_dim(x[3], pw, w[3], sw)?
    };
    Ok(vec![n, oc, oh, ow])
}

fn pool_shape(
    inputs: &[Vec<usize>],
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
    fallback: &[usize],
) -> Result<Vec<usize>, ShapeInferError> {
    let Some(x) = inputs.first().filter(|d| d.len() == 4) else {
        return Ok(fallback.to_vec());
    };
    if kernel[0] == 0 {
        // GlobalAveragePool → [N, C, 1, 1]
        return Ok(vec![x[0], x[1], 1, 1]);
    }
    let (sh, sw) = (strides[0].max(1), strides[1].max(1));
    let (ph, pw) = (padding[0], padding[1]);
    let oh = conv_out_dim(x[2], ph, kernel[0], sh)?;
    let ow = conv_out_dim(x[3], pw, kernel[1], sw)?;
    Ok(vec![x[0], x[1], oh, ow])
}

/// Errors from the shape inference pass.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShapeInferError {
    /// An input shape was required but not known.
    UnknownInput {
        /// Node id whose input shape was missing.
        node: usize,
    },
    /// A convolution/pooling kernel does not fit its (padded) input, or the
    /// geometry arithmetic would otherwise overflow. This can only happen
    /// with malformed or adversarial `dims`/`kernel_shape`/`pads`.
    InvalidGeometry,
}

impl std::fmt::Display for ShapeInferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeInferError::UnknownInput { node } => {
                write!(f, "unknown input shape for node {node}")
            }
            ShapeInferError::InvalidGeometry => {
                write!(
                    f,
                    "convolution/pooling kernel does not fit the (padded) input dimensions"
                )
            }
        }
    }
}

impl std::error::Error for ShapeInferError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_shapes() {
        let d = infer_node_shape(&Operator::MatMul, &[vec![1, 784], vec![784, 10]], &[]).unwrap();
        assert_eq!(d, vec![1, 10]);
    }

    #[test]
    fn conv_shapes() {
        let d = infer_node_shape(
            &Operator::Conv2d {
                strides: [1, 1],
                padding: [1, 1],
            },
            &[vec![1, 3, 224, 224], vec![8, 3, 3, 3]],
            &[],
        )
        .unwrap();
        assert_eq!(d, vec![1, 8, 224, 224]);
    }

    #[test]
    fn flatten_shapes() {
        let d =
            infer_node_shape(&Operator::Flatten { axis: 1 }, &[vec![1, 3, 32, 32]], &[]).unwrap();
        assert_eq!(d, vec![1, 3072]);
    }

    #[test]
    fn dynamic_batch_stays_zero() {
        // dim 0 = dynamic
        let d = infer_node_shape(&Operator::Relu, &[vec![0, 128]], &[]).unwrap();
        assert_eq!(d, vec![0, 128]);
    }

    #[test]
    fn conv_kernel_larger_than_input_is_rejected() {
        // A malicious/malformed ONNX file could declare a kernel far larger
        // than the (padded) input; this must be a hard error, not a
        // debug-mode panic or a release-mode `usize` wraparound.
        let err = infer_node_shape(
            &Operator::Conv2d {
                strides: [1, 1],
                padding: [0, 0],
            },
            &[vec![1, 3, 4, 4], vec![8, 3, 999, 999]],
            &[],
        )
        .unwrap_err();
        assert_eq!(err, ShapeInferError::InvalidGeometry);
    }

    #[test]
    fn pool_kernel_larger_than_input_is_rejected() {
        let err = infer_node_shape(
            &Operator::MaxPool2d {
                kernel: [999, 999],
                strides: [1, 1],
                padding: [0, 0],
            },
            &[vec![1, 3, 4, 4]],
            &[],
        )
        .unwrap_err();
        assert_eq!(err, ShapeInferError::InvalidGeometry);
    }
}
