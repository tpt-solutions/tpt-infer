//! Constant folding: evaluate initializer-only subgraphs at compile time.
//!
//! A node is "constant" if every input it consumes is itself constant. The
//! base case is an [`Operator::Input`] node bound to a graph
//! [`Initializer`](tpt_infer_graph::Initializer) (a pre-trained weight
//! tensor with no runtime dependency). Constant-ness then propagates forward
//! through any operator this module knows how to evaluate host-side.
//!
//! The result is a `node id -> Vec<f32>` map. [`crate::codegen`] splices
//! these values into the generated source as literal arrays instead of
//! emitting runtime computation for them, which both skips work at inference
//! time and lets the folded operator's code path be omitted entirely.
//!
//! This is a best-effort pass, not a full constant-propagation engine: only
//! the operators [`crate::codegen`] itself generates code for are folded
//! (see `try_eval`). Anything else simply is not folded — its inputs (if
//! constant) are still folded, so a later operator can still short-circuit
//! once it reaches a foldable node.

use std::collections::BTreeMap;

use tpt_infer_graph::{ComputationGraph, Operator};

/// Constant values computed at compile time, keyed by node id.
pub type FoldedConstants = BTreeMap<usize, Vec<f32>>;

/// Walks `graph` in `order` and evaluates every node whose inputs are all
/// already known constants (initializers, or the output of an earlier fold).
///
/// Returns a map from node id to its folded `f32` values. Nodes not present
/// in the map must be computed at runtime by the generated code.
pub fn fold_constants(graph: &ComputationGraph, order: &[usize]) -> FoldedConstants {
    let mut values: FoldedConstants = BTreeMap::new();

    for &id in order {
        let Some(node) = graph.node(id) else {
            continue;
        };

        if node.operator.is_input() {
            if let Some(name) = &node.name {
                if let Some(init) = graph.initializers().iter().find(|i| &i.name == name) {
                    values.insert(id, init.data.clone());
                }
            }
            continue;
        }

        // Gather constant inputs, bailing out (leaving this node unfolded)
        // as soon as one input is not itself constant.
        let mut inputs: Vec<&[f32]> = Vec::with_capacity(node.inputs.len());
        let mut all_const = true;
        for &inp in &node.inputs {
            match values.get(&inp) {
                Some(v) => inputs.push(v.as_slice()),
                None => {
                    all_const = false;
                    break;
                }
            }
        }
        if !all_const {
            continue;
        }

        let dims: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .filter_map(|&inp| graph.node(inp).map(|n| n.dims().to_vec()))
            .collect();

        if let Some(result) = try_eval(&node.operator, &inputs, &dims, node.dims()) {
            values.insert(id, result);
        }
    }

    values
}

/// Evaluates a single operator against constant inputs, if this module knows
/// how to. Mirrors the runtime semantics of the same operators in
/// [`crate::codegen`].
fn try_eval(
    op: &Operator,
    inputs: &[&[f32]],
    input_dims: &[Vec<usize>],
    out_dims: &[usize],
) -> Option<Vec<f32>> {
    match op {
        Operator::MatMul => {
            let a = inputs.first()?;
            let b = inputs.get(1)?;
            let ad = input_dims.first()?;
            let bd = input_dims.get(1)?;
            if ad.len() != 2 || bd.len() != 2 {
                return None;
            }
            let (m, k) = (ad[0], ad[1]);
            let (k2, n) = (bd[0], bd[1]);
            if k != k2 {
                return None;
            }
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a[i * k + p] * b[p * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            Some(out)
        }
        Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
            let a = inputs.first()?;
            let b = inputs.get(1)?;
            if a.len() != b.len() {
                return None;
            }
            Some(
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| match op {
                        Operator::Add => x + y,
                        Operator::Sub => x - y,
                        Operator::Mul => x * y,
                        _ => x / y,
                    })
                    .collect(),
            )
        }
        Operator::Relu => {
            let a = inputs.first()?;
            Some(a.iter().map(|&x| x.max(0.0)).collect())
        }
        Operator::Sigmoid => {
            let a = inputs.first()?;
            Some(a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
        }
        Operator::Reshape { .. } | Operator::Flatten { .. } => {
            let a = inputs.first()?;
            let expected: usize = out_dims.iter().product();
            if a.len() != expected {
                return None;
            }
            Some(a.to_vec())
        }
        Operator::Conv2d { strides, padding } => {
            fold_conv2d(inputs, input_dims, out_dims, *strides, *padding)
        }
        Operator::Softmax { axis } => fold_softmax(inputs, input_dims, *axis),
        Operator::MaxPool2d {
            kernel,
            strides,
            padding,
        } => fold_pool(
            inputs, input_dims, out_dims, true, *kernel, *strides, *padding,
        ),
        Operator::AveragePool2d {
            kernel,
            strides,
            padding,
        } => fold_pool(
            inputs, input_dims, out_dims, false, *kernel, *strides, *padding,
        ),
        Operator::BatchNorm { epsilon } => fold_batch_norm(inputs, input_dims, *epsilon),
        Operator::Concat { axis } => fold_concat(inputs, input_dims, out_dims, *axis),
        Operator::Transpose { perm, rank } => {
            fold_transpose(inputs, input_dims, out_dims, &perm[..*rank], *rank)
        }
        // `Custom` is an intentional escape hatch this crate never generates
        // code for (see `codegen::emit_op`), so there is nothing to fold.
        _ => None,
    }
}

/// Row-major strides for `dims` (`strides[i] = product(dims[i+1..])`).
fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Host-side evaluation of `Conv2d`, mirroring `NaiveBackend::conv2d` plus
/// the optional per-channel bias `tpt_infer_runtime::exec` adds afterward.
fn fold_conv2d(
    inputs: &[&[f32]],
    input_dims: &[Vec<usize>],
    out_dims: &[usize],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Option<Vec<f32>> {
    let xa = *inputs.first()?;
    let wt = *inputs.get(1)?;
    let bias = inputs.get(2).copied();
    let xd = input_dims.first()?;
    let wd = input_dims.get(1)?;
    if xd.len() != 4 || wd.len() != 4 {
        return None;
    }
    let (n, c, ih, iw) = (xd[0], xd[1], xd[2], xd[3]);
    let (oc, c2, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
    if c != c2 {
        return None;
    }
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    if sh == 0 || sw == 0 || kh == 0 || kw == 0 {
        return None;
    }
    let padded_h = ih + 2 * ph;
    let padded_w = iw + 2 * pw;
    if padded_h < kh || padded_w < kw {
        return None;
    }
    let oh = (padded_h - kh) / sh + 1;
    let ow = (padded_w - kw) / sw + 1;
    if out_dims != [n, oc, oh, ow].as_slice() {
        return None;
    }
    // Only a `[1, oc, 1, 1]` bias actually broadcasts onto the channel axis
    // under `tpt_infer_runtime`'s right-aligned broadcasting (see
    // `codegen::emit_conv2d`'s doc comment) — a flat `[oc]` bias would not,
    // so it is rejected here too rather than folding to a different result
    // than the interpreted runtime would produce.
    let bias_dims = input_dims.get(2);
    let bias_ok = match (bias, bias_dims) {
        (Some(b), Some(bd)) => {
            bd.as_slice() == [1, oc, 1, 1] && b.len() == bd.iter().product::<usize>()
        }
        (None, None) => true,
        _ => false,
    };
    if !bias_ok {
        return None;
    }

    let mut out = vec![0.0f32; n * oc * oh * ow];
    for ni in 0..n {
        for o in 0..oc {
            let base_bias = bias.map(|b| b[o]).unwrap_or(0.0);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = base_bias;
                    for ci in 0..c {
                        for ky in 0..kh {
                            let y = oy * sh + ky;
                            if y < ph || y - ph >= ih {
                                continue;
                            }
                            let iy = y - ph;
                            for kx in 0..kw {
                                let x = ox * sw + kx;
                                if x < pw || x - pw >= iw {
                                    continue;
                                }
                                let ix = x - pw;
                                let in_idx = ((ni * c + ci) * ih + iy) * iw + ix;
                                let w_idx = ((o * c + ci) * kh + ky) * kw + kx;
                                acc += xa[in_idx] * wt[w_idx];
                            }
                        }
                    }
                    out[((ni * oc + o) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    Some(out)
}

/// Host-side evaluation of `Softmax` along an arbitrary axis (max-subtract,
/// exp, normalize), matching `NaiveBackend::softmax`'s per-row formula
/// generalized to any axis (folding is not limited to the last-axis
/// restriction `codegen::emit_softmax` inherits from the interpreted
/// runtime, since this just evaluates in ordinary host Rust).
fn fold_softmax(inputs: &[&[f32]], input_dims: &[Vec<usize>], axis: i32) -> Option<Vec<f32>> {
    let a = *inputs.first()?;
    let ad = input_dims.first()?;
    if ad.is_empty() {
        return None;
    }
    let rank = ad.len() as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return None;
    }
    let axis_us = normalized as usize;
    let axis_len = ad[axis_us];
    if axis_len == 0 {
        return None;
    }
    let inner: usize = ad[axis_us + 1..].iter().product();
    let outer: usize = ad[..axis_us].iter().product();

    let mut out = vec![0.0f32; a.len()];
    for o in 0..outer {
        for p in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = (o * axis_len + k) * inner + p;
                if a[idx] > max_v {
                    max_v = a[idx];
                }
            }
            let mut sum_v = 0.0f32;
            for k in 0..axis_len {
                let idx = (o * axis_len + k) * inner + p;
                let e = (a[idx] - max_v).exp();
                out[idx] = e;
                sum_v += e;
            }
            for k in 0..axis_len {
                let idx = (o * axis_len + k) * inner + p;
                out[idx] /= sum_v;
            }
        }
    }
    Some(out)
}

/// Host-side evaluation of `MaxPool2d`/`AveragePool2d`, mirroring
/// `tpt_infer_runtime::kernels::{max_pool2d, average_pool2d}`.
fn fold_pool(
    inputs: &[&[f32]],
    input_dims: &[Vec<usize>],
    out_dims: &[usize],
    is_max: bool,
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Option<Vec<f32>> {
    let a = *inputs.first()?;
    let ad = input_dims.first()?;
    if ad.len() != 4 {
        return None;
    }
    let (n, c, h, w) = (ad[0], ad[1], ad[2], ad[3]);
    let (kh, kw) = (kernel[0], kernel[1]);
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return None;
    }
    let padded_h = h + 2 * ph;
    let padded_w = w + 2 * pw;
    if padded_h < kh || padded_w < kw {
        return None;
    }
    let oh = (padded_h - kh) / sh + 1;
    let ow = (padded_w - kw) / sw + 1;
    if out_dims != [n, c, oh, ow].as_slice() {
        return None;
    }

    let mut out = vec![0.0f32; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut acc = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        let y = oy * sh + ky;
                        if y < ph || y - ph >= h {
                            continue;
                        }
                        let iy = y - ph;
                        for kx in 0..kw {
                            let x = ox * sw + kx;
                            if x < pw || x - pw >= w {
                                continue;
                            }
                            let ix = x - pw;
                            let idx = ((ni * c + ci) * h + iy) * w + ix;
                            if is_max {
                                if a[idx] > best {
                                    best = a[idx];
                                }
                            } else {
                                acc += a[idx];
                                count += 1;
                            }
                        }
                    }
                    let v = if is_max {
                        best
                    } else if count == 0 {
                        0.0
                    } else {
                        acc / count as f32
                    };
                    out[((ni * c + ci) * oh + oy) * ow + ox] = v;
                }
            }
        }
    }
    Some(out)
}

/// Host-side evaluation of `BatchNorm`, matching
/// `tpt_infer_runtime::kernels::batch_norm`'s two parameter layouts.
fn fold_batch_norm(inputs: &[&[f32]], input_dims: &[Vec<usize>], epsilon: f32) -> Option<Vec<f32>> {
    let x = *inputs.first()?;
    let scale = *inputs.get(1)?;
    let bias = *inputs.get(2)?;
    let mean = *inputs.get(3)?;
    let var = *inputs.get(4)?;
    let xd = input_dims.first()?;
    let numel: usize = xd.iter().product();
    let plen = scale.len();
    if bias.len() != plen || mean.len() != plen || var.len() != plen {
        return None;
    }
    let channel_mode = xd.len() == 4 && plen == xd[1];
    let element_mode = plen == numel;
    if !channel_mode && !element_mode {
        return None;
    }

    let mut out = vec![0.0f32; numel];
    if channel_mode {
        let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
        let hw = h * w;
        for ni in 0..n {
            for ci in 0..c {
                let inv = 1.0 / (var[ci] + epsilon).sqrt();
                let f = scale[ci] * inv;
                let g = bias[ci] - mean[ci] * f;
                let base = (ni * c + ci) * hw;
                for i in base..base + hw {
                    out[i] = x[i] * f + g;
                }
            }
        }
    } else {
        for i in 0..numel {
            let inv = 1.0 / (var[i] + epsilon).sqrt();
            let f = scale[i] * inv;
            out[i] = (x[i] - mean[i]) * f + bias[i];
        }
    }
    Some(out)
}

/// Host-side evaluation of `Concat`, matching
/// `tpt_infer_runtime::kernels::concat_copy_one`'s block layout.
fn fold_concat(
    inputs: &[&[f32]],
    input_dims: &[Vec<usize>],
    out_dims: &[usize],
    axis: i32,
) -> Option<Vec<f32>> {
    if inputs.is_empty() {
        return None;
    }
    let rank = out_dims.len();
    if rank == 0 {
        return None;
    }
    let ax = if axis < 0 { axis + rank as i32 } else { axis };
    if ax < 0 || ax as usize >= rank {
        return None;
    }
    let axis_us = ax as usize;
    let c_total = out_dims[axis_us];
    let inner: usize = out_dims[axis_us + 1..].iter().product();
    let numel: usize = out_dims.iter().product();

    let mut out = vec![0.0f32; numel];
    let mut c_offset = 0usize;
    for (idx, &src) in inputs.iter().enumerate() {
        let dims = input_dims.get(idx)?;
        if dims.len() != rank {
            return None;
        }
        for (d, &v) in dims.iter().enumerate() {
            if d != axis_us && v != out_dims[d] {
                return None;
            }
        }
        let c_src = dims[axis_us];
        let outer: usize = dims[..axis_us].iter().product();
        for o in 0..outer {
            for cc in 0..c_src {
                for p in 0..inner {
                    let dst = o * c_total * inner + (c_offset + cc) * inner + p;
                    let s = o * c_src * inner + cc * inner + p;
                    out[dst] = src[s];
                }
            }
        }
        c_offset += c_src;
    }
    if c_offset != c_total {
        return None;
    }
    Some(out)
}

/// Host-side evaluation of `Transpose`, matching
/// `tpt_infer_runtime::kernels::transpose`.
fn fold_transpose(
    inputs: &[&[f32]],
    input_dims: &[Vec<usize>],
    out_dims: &[usize],
    perm: &[usize],
    rank: usize,
) -> Option<Vec<f32>> {
    let a = *inputs.first()?;
    let xd = input_dims.first()?;
    if xd.len() != rank || out_dims.len() != rank || perm.len() != rank {
        return None;
    }
    for d in 0..rank {
        if out_dims[d] != xd[perm[d]] {
            return None;
        }
    }

    let x_strides = strides_of(xd);
    let out_strides = strides_of(out_dims);
    let numel: usize = out_dims.iter().product();
    let mut out = vec![0.0f32; numel];
    for (t, slot) in out.iter_mut().enumerate() {
        let mut in_idx = 0usize;
        for d in 0..rank {
            let coord = (t / out_strides[d]) % out_dims[d];
            in_idx += coord * x_strides[perm[d]];
        }
        *slot = a[in_idx];
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_infer_graph::{ComputationGraph, Initializer, Node};

    #[test]
    fn folds_initializer_only_matmul() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[1, 2])
                    .unwrap()
                    .with_name("a"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("a", &[1, 2], vec![1.0, 2.0]).unwrap());
        let b = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[2, 1])
                    .unwrap()
                    .with_name("b"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("b", &[2, 1], vec![3.0, 4.0]).unwrap());
        let m = g
            .add_node(Node::new(2, Operator::MatMul, vec![a, b], &[1, 1]).unwrap())
            .unwrap();
        g.mark_output(m).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        assert_eq!(folded.get(&2), Some(&vec![11.0f32]));
    }

    #[test]
    fn does_not_fold_runtime_dependent_nodes() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 2]).unwrap())
            .unwrap();
        let r = g
            .add_node(Node::new(1, Operator::Relu, vec![x], &[1, 2]).unwrap())
            .unwrap();
        g.mark_output(r).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        assert!(folded.is_empty());
    }

    #[test]
    fn folds_initializer_only_softmax() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[1, 3])
                    .unwrap()
                    .with_name("a"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("a", &[1, 3], vec![1.0, 2.0, 3.0]).unwrap());
        let s = g
            .add_node(Node::new(1, Operator::softmax(-1), vec![a], &[1, 3]).unwrap())
            .unwrap();
        g.mark_output(s).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        let out = folded.get(&s).expect("softmax should fold");
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Monotonic: larger logits get larger probabilities.
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn folds_initializer_only_transpose() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[2, 3])
                    .unwrap()
                    .with_name("a"),
            )
            .unwrap();
        g.add_initializer(
            Initializer::new("a", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let t = g
            .add_node(
                Node::new(1, Operator::transpose(&[1, 0]).unwrap(), vec![a], &[3, 2]).unwrap(),
            )
            .unwrap();
        g.mark_output(t).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        assert_eq!(folded.get(&t), Some(&vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn folds_initializer_only_concat() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[1, 2])
                    .unwrap()
                    .with_name("a"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("a", &[1, 2], vec![1.0, 2.0]).unwrap());
        let b = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[1, 2])
                    .unwrap()
                    .with_name("b"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("b", &[1, 2], vec![3.0, 4.0]).unwrap());
        let c = g
            .add_node(Node::new(2, Operator::concat(1), vec![a, b], &[1, 4]).unwrap())
            .unwrap();
        g.mark_output(c).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        assert_eq!(folded.get(&c), Some(&vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn folds_initializer_only_max_pool2d() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[1, 1, 4, 4])
                    .unwrap()
                    .with_name("a"),
            )
            .unwrap();
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        g.add_initializer(Initializer::new("a", &[1, 1, 4, 4], data).unwrap());
        let p = g
            .add_node(
                Node::new(
                    1,
                    Operator::max_pool2d([2, 2], [2, 2], [0, 0]),
                    vec![a],
                    &[1, 1, 2, 2],
                )
                .unwrap(),
            )
            .unwrap();
        g.mark_output(p).unwrap();

        let order = g.topological_sort().unwrap();
        let folded = fold_constants(&g, &order);
        assert_eq!(folded.get(&p), Some(&vec![5.0, 7.0, 13.0, 15.0]));
    }
}
