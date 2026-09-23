//! Graph-to-Rust code generator.
//!
//! [`generate`] walks a [`ComputationGraph`] in topological order and emits
//! a `TokenStream` of Rust statements that compute the graph's marked output
//! from its single runtime input, plus a trailing return expression. The
//! caller ([`crate::model::aot_compile`]) wraps this in a
//! `pub fn execute(input: &[f32]) -> Vec<f32>` function signature.
//!
//! Every node produces one `let v{id}: Vec<f32> = ...;` binding (or, for the
//! runtime input node, is referenced directly as the `input` parameter), so
//! downstream nodes can freely read earlier bindings by index without
//! fighting the borrow checker — nothing is ever moved out from under a
//! consumer.
//!
//! # Loop unrolling
//!
//! Every generated loop bound is known at compile time (that is the whole
//! point of `tpt-infer-graph`'s shape-carrying nodes). Dimensions of size
//! `UNROLL_THRESHOLD` or smaller are unrolled directly into repeated
//! straight-line statements instead of a `for` loop; larger dimensions are
//! emitted as ordinary `for` loops with a compile-time-constant bound, which
//! LLVM is free to unroll itself at the chosen optimization level. This is a
//! deliberately simple policy — not a cost-model-driven unroller — that
//! still demonstrates the mechanism the AOT compiler is meant to exploit.

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};

use tpt_infer_graph::{ComputationGraph, Node, Operator};

use crate::fold::FoldedConstants;
use crate::model::CompileError;

/// Dimensions at or below this size are unrolled into straight-line code
/// rather than emitted as a runtime `for` loop.
const UNROLL_THRESHOLD: usize = 8;

/// Generates the statement sequence (plus trailing return expression) that
/// computes `output_id` from `input_id`, walking `graph` in `order`.
///
/// `folded` supplies compile-time-known values (see [`crate::fold`]) that
/// are spliced in as literals instead of generated computation.
pub fn generate(
    graph: &ComputationGraph,
    order: &[usize],
    input_id: usize,
    output_id: usize,
    folded: &FoldedConstants,
) -> Result<TokenStream, CompileError> {
    let mut stmts = Vec::new();

    for &id in order {
        if id == input_id {
            // Referenced directly via the `input` parameter; no binding needed.
            continue;
        }
        let node = graph.node(id).ok_or(CompileError::Graph(
            tpt_infer_graph::GraphError::UnknownNode { id },
        ))?;

        if let Some(data) = folded.get(&id) {
            stmts.push(emit_literal(id, data));
            continue;
        }

        if node.operator.is_input() {
            // A weight input with no initializer and no runtime binding:
            // nothing left to generate code from.
            return Err(CompileError::MissingInitializer { node: id });
        }

        stmts.push(emit_op(graph, node, input_id)?);
    }

    let ret = var_ref(output_id, input_id);
    let ret_expr = if output_id == input_id {
        quote! { #ret.to_vec() }
    } else {
        quote! { #ret }
    };

    Ok(quote! {
        #(#stmts)*
        #ret_expr
    })
}

/// The identifier a node's output is bound to: `v{id}`.
fn var_ident(id: usize) -> proc_macro2::Ident {
    format_ident!("v{}", id)
}

/// A token-stream reference to node `id`'s value: the `input` parameter
/// itself if `id` is the runtime input, otherwise its `v{id}` binding.
fn var_ref(id: usize, input_id: usize) -> TokenStream {
    if id == input_id {
        quote! { input }
    } else {
        let ident = var_ident(id);
        quote! { #ident }
    }
}

/// `let v{id}: Vec<f32> = vec![lit, lit, ...];` for a compile-time-constant node.
fn emit_literal(id: usize, data: &[f32]) -> TokenStream {
    let out = var_ident(id);
    let lits = data.iter().map(|&x| float_literal(x));
    quote! {
        let #out: Vec<f32> = std::vec![#(#lits),*];
    }
}

/// Renders `x` as an unambiguous `f32` literal, including `NaN`/`inf` via
/// `f32::` constructors since Rust has no literal syntax for them.
fn float_literal(x: f32) -> TokenStream {
    if x.is_nan() {
        quote! { f32::NAN }
    } else if x.is_infinite() {
        if x > 0.0 {
            quote! { f32::INFINITY }
        } else {
            quote! { f32::NEG_INFINITY }
        }
    } else {
        // `{:?}` always prints a form that round-trips and parses as f32
        // (e.g. "1.0", "-0.5", "3.0").
        let text = format!("{x:?}f32");
        let lit: proc_macro2::TokenStream = text.parse().expect("valid f32 literal");
        lit
    }
}

/// Emits code for one non-constant, non-input node.
fn emit_op(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
) -> Result<TokenStream, CompileError> {
    match &node.operator {
        Operator::MatMul => emit_matmul(graph, node, input_id),
        Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
            emit_elementwise_binary(graph, node, input_id)
        }
        Operator::Relu => emit_unary(node, input_id, |i| quote! { #i.max(0.0) }, graph),
        Operator::Sigmoid => emit_unary(
            node,
            input_id,
            |i| quote! { 1.0f32 / (1.0f32 + (-(#i)).exp()) },
            graph,
        ),
        // Tanh approximation, matching `tpt_infer_ops::naive::gelu` exactly
        // (with `std`'s `f32::tanh`) so AOT-compiled output is numerically
        // identical to the interpreted runtime.
        Operator::Gelu => emit_unary(
            node,
            input_id,
            |i| {
                quote! {
                    {
                        let x: f32 = #i;
                        const K: f32 = 0.797_884_6;
                        const C: f32 = 0.044_715;
                        0.5f32 * x * (1.0f32 + (K * (x + C * x * x * x)).tanh())
                    }
                }
            },
            graph,
        ),
        Operator::Reshape { .. } | Operator::Flatten { .. } => emit_reshape(graph, node, input_id),
        Operator::Conv2d { strides, padding } => {
            emit_conv2d(graph, node, input_id, *strides, *padding)
        }
        Operator::Softmax { axis } => emit_softmax(graph, node, input_id, *axis),
        Operator::MaxPool2d {
            kernel,
            strides,
            padding,
        } => emit_pool(
            graph,
            node,
            input_id,
            PoolKind::Max,
            *kernel,
            *strides,
            *padding,
        ),
        Operator::AveragePool2d {
            kernel,
            strides,
            padding,
        } => emit_pool(
            graph,
            node,
            input_id,
            PoolKind::Average,
            *kernel,
            *strides,
            *padding,
        ),
        Operator::BatchNorm { epsilon } => emit_batch_norm(graph, node, input_id, *epsilon),
        Operator::Concat { axis } => emit_concat(graph, node, input_id, *axis),
        Operator::Transpose { perm, rank } => {
            emit_transpose(graph, node, input_id, &perm[..*rank], *rank)
        }
        other => Err(CompileError::UnsupportedOperator {
            name: other.name().to_string(),
            node: node.id,
        }),
    }
}

/// Which reduction [`emit_pool`] generates: running max, or a running sum
/// plus count for the mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    /// [`Operator::MaxPool2d`].
    Max,
    /// [`Operator::AveragePool2d`].
    Average,
}

/// Row-major strides for `dims` (`strides[i] = product(dims[i+1..])`).
fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = std::vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn producer_dims(graph: &ComputationGraph, id: usize) -> Result<&[usize], CompileError> {
    graph.node(id).map(Node::dims).ok_or(CompileError::Graph(
        tpt_infer_graph::GraphError::UnknownNode { id },
    ))
}

fn emit_matmul(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 2 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let (a_id, b_id) = (node.inputs[0], node.inputs[1]);
    let a_dims = producer_dims(graph, a_id)?;
    let b_dims = producer_dims(graph, b_id)?;
    if a_dims.len() != 2 {
        return Err(CompileError::UnsupportedRank {
            node: a_id,
            op: "MatMul",
            rank: a_dims.len(),
        });
    }
    if b_dims.len() != 2 {
        return Err(CompileError::UnsupportedRank {
            node: b_id,
            op: "MatMul",
            rank: b_dims.len(),
        });
    }
    let (m, k) = (a_dims[0], a_dims[1]);
    let (k2, n) = (b_dims[0], b_dims[1]);
    if k != k2 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }

    let out = var_ident(node.id);
    let a = var_ref(a_id, input_id);
    let b = var_ref(b_id, input_id);
    let len = usize_lit(m * n);
    let k_lit = usize_lit(k);
    let n_lit = usize_lit(n);

    let body = for_or_unroll("i", m, |i| {
        for_or_unroll("j", n, |j| {
            let acc_loop = for_or_unroll("p", k, |p| {
                quote! { acc += #a[(#i) * #k_lit + (#p)] * #b[(#p) * #n_lit + (#j)]; }
            });
            quote! {
                {
                    let mut acc: f32 = 0.0;
                    #acc_loop
                    #out[(#i) * #n_lit + (#j)] = acc;
                }
            }
        })
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

fn emit_elementwise_binary(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 2 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let (a_id, b_id) = (node.inputs[0], node.inputs[1]);
    let a_dims = producer_dims(graph, a_id)?;
    let b_dims = producer_dims(graph, b_id)?;
    if a_dims != b_dims || a_dims != node.dims() {
        // Broadcasting binary ops are not generated (yet); only the exact
        // same-shape case tpt-infer-runtime treats as the fast path here.
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let numel: usize = node.dims().iter().product();

    let out = var_ident(node.id);
    let a = var_ref(a_id, input_id);
    let b = var_ref(b_id, input_id);
    let len = usize_lit(numel);
    let op_tok = match node.operator {
        Operator::Add => quote! { + },
        Operator::Sub => quote! { - },
        Operator::Mul => quote! { * },
        Operator::Div => quote! { / },
        _ => unreachable!("caller only routes Add/Sub/Mul/Div here"),
    };

    let body = for_or_unroll("i", numel, |i| {
        quote! { #out[#i] = #a[#i] #op_tok #b[#i]; }
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

fn emit_unary(
    node: &Node,
    input_id: usize,
    map_expr: impl Fn(&TokenStream) -> TokenStream,
    graph: &ComputationGraph,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 1 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let a_id = node.inputs[0];
    let _ = producer_dims(graph, a_id)?; // validate the producer exists
    let numel: usize = node.dims().iter().product();

    let out = var_ident(node.id);
    let a = var_ref(a_id, input_id);
    let len = usize_lit(numel);

    let body = for_or_unroll("i", numel, |i| {
        let idx = quote! { #a[#i] };
        let mapped = map_expr(&idx);
        quote! { #out[#i] = #mapped; }
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

fn emit_reshape(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 1 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let a_id = node.inputs[0];
    let a_numel: usize = producer_dims(graph, a_id)?.iter().product();
    let out_numel: usize = node.dims().iter().product();
    if a_numel != out_numel {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let out = var_ident(node.id);
    let a = var_ref(a_id, input_id);
    Ok(quote! {
        let #out: Vec<f32> = #a.to_vec();
    })
}

/// `Conv2d`: direct (non-im2col) NCHW convolution, matching
/// `NaiveBackend::conv2d`'s accumulation order and padding/stride
/// conventions, plus an optional bias input (`node.inputs[2]`) broadcast
/// per output channel. `tpt_infer_runtime::exec` adds the bias via
/// `kernels::add_inplace_broadcast`, which uses numpy right-aligned
/// broadcasting against the rank-4 `[n, oc, oh, ow]` output — so only a
/// `[1, oc, 1, 1]` bias actually broadcasts onto the channel axis (a flat
/// `[oc]` bias would right-align against the *width* axis instead and be
/// rejected by the runtime, so this compiler rejects it too rather than
/// generating code that would disagree with the interpreted reference).
fn emit_conv2d(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    strides: [usize; 2],
    padding: [usize; 2],
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 2 && node.inputs.len() != 3 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let x_id = node.inputs[0];
    let w_id = node.inputs[1];
    let x_dims = producer_dims(graph, x_id)?;
    let w_dims = producer_dims(graph, w_id)?;
    if x_dims.len() != 4 {
        return Err(CompileError::UnsupportedRank {
            node: x_id,
            op: "Conv2d",
            rank: x_dims.len(),
        });
    }
    if w_dims.len() != 4 {
        return Err(CompileError::UnsupportedRank {
            node: w_id,
            op: "Conv2d",
            rank: w_dims.len(),
        });
    }
    let (n, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (oc, c2, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    if c != c2 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    if sh == 0 || sw == 0 || kh == 0 || kw == 0 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let padded_h = h + 2 * ph;
    let padded_w = w + 2 * pw;
    if padded_h < kh || padded_w < kw {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let oh = (padded_h - kh) / sh + 1;
    let ow = (padded_w - kw) / sw + 1;
    if node.dims() != [n, oc, oh, ow].as_slice() {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }

    let bias_id = node.inputs.get(2).copied();
    if let Some(bid) = bias_id {
        let bd = producer_dims(graph, bid)?;
        let ok = bd == [1, oc, 1, 1].as_slice();
        if !ok {
            return Err(CompileError::UnsupportedRank {
                node: bid,
                op: "Conv2d bias",
                rank: bd.len(),
            });
        }
    }

    let out = var_ident(node.id);
    let xr = var_ref(x_id, input_id);
    let wr = var_ref(w_id, input_id);
    let len = usize_lit(n * oc * oh * ow);
    let c_lit = usize_lit(c);
    let h_lit = usize_lit(h);
    let w_lit = usize_lit(w);
    let kh_lit = usize_lit(kh);
    let kw_lit = usize_lit(kw);
    let oh_lit = usize_lit(oh);
    let ow_lit = usize_lit(ow);
    let oc_lit = usize_lit(oc);
    let sh_lit = usize_lit(sh);
    let sw_lit = usize_lit(sw);
    let ph_lit = usize_lit(ph);
    let pw_lit = usize_lit(pw);

    let body = for_or_unroll("ni", n, |ni| {
        for_or_unroll("oi", oc, |oi| {
            let acc_init = match bias_id {
                Some(bid) => {
                    let br = var_ref(bid, input_id);
                    quote! { #br[(#oi)] }
                }
                None => quote! { 0.0f32 },
            };
            for_or_unroll("oy", oh, |oy| {
                for_or_unroll("ox", ow, |ox| {
                    let inner = for_or_unroll("ci", c, |ci| {
                        for_or_unroll("ky", kh, |ky| {
                            for_or_unroll("kx", kw, |kx| {
                                quote! {
                                    {
                                        let y = (#oy) * #sh_lit + (#ky);
                                        let x = (#ox) * #sw_lit + (#kx);
                                        if y >= #ph_lit && y - #ph_lit < #h_lit && x >= #pw_lit && x - #pw_lit < #w_lit {
                                            let iy = y - #ph_lit;
                                            let ix = x - #pw_lit;
                                            let in_idx = (((#ni) * #c_lit + (#ci)) * #h_lit + iy) * #w_lit + ix;
                                            let w_idx = (((#oi) * #c_lit + (#ci)) * #kh_lit + (#ky)) * #kw_lit + (#kx);
                                            acc += #xr[in_idx] * #wr[w_idx];
                                        }
                                    }
                                }
                            })
                        })
                    });
                    quote! {
                        {
                            let mut acc: f32 = #acc_init;
                            #inner
                            let out_idx = (((#ni) * #oc_lit + (#oi)) * #oh_lit + (#oy)) * #ow_lit + (#ox);
                            #out[out_idx] = acc;
                        }
                    }
                })
            })
        })
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

/// `Softmax`: max-subtract, exp, normalize. Only generates code when `axis`
/// (after negative-index normalization) is the last dimension, mirroring
/// `tpt_infer_runtime`'s `Backend::softmax` restriction (see `exec.rs`) —
/// normalizing any other axis would require a transpose-like gather this
/// compiler does not emit for `Softmax` itself.
fn emit_softmax(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    axis: i32,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 1 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let a_id = node.inputs[0];
    let a_dims = producer_dims(graph, a_id)?;
    if a_dims.is_empty() {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let rank = a_dims.len() as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized != rank - 1 {
        return Err(CompileError::UnsupportedRank {
            node: node.id,
            op: "Softmax",
            rank: normalized.max(0) as usize,
        });
    }
    let row_len = *a_dims.last().unwrap();
    if row_len == 0 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let numel: usize = a_dims.iter().product();
    let num_rows = numel / row_len;

    let out = var_ident(node.id);
    let a = var_ref(a_id, input_id);
    let len = usize_lit(numel);
    let row_len_lit = usize_lit(row_len);

    let body = for_or_unroll("r", num_rows, |r| {
        let max_loop = for_or_unroll("j", row_len, |j| {
            quote! {
                {
                    let v = #a[base + (#j)];
                    if v > max_v { max_v = v; }
                }
            }
        });
        let exp_loop = for_or_unroll("j", row_len, |j| {
            quote! { #out[base + (#j)] = (#a[base + (#j)] - max_v).exp(); }
        });
        let sum_loop = for_or_unroll("j", row_len, |j| {
            quote! { sum_v += #out[base + (#j)]; }
        });
        let div_loop = for_or_unroll("j", row_len, |j| {
            quote! { #out[base + (#j)] /= sum_v; }
        });
        quote! {
            {
                let base = (#r) * #row_len_lit;
                let mut max_v: f32 = f32::NEG_INFINITY;
                #max_loop
                #exp_loop
                let mut sum_v: f32 = 0.0;
                #sum_loop
                #div_loop
            }
        }
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

/// `MaxPool2d` / `AveragePool2d`: NCHW pooling matching
/// `tpt_infer_runtime::kernels::{max_pool2d, average_pool2d}` (floor
/// division for output geometry, padding excluded from the average).
fn emit_pool(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    kind: PoolKind,
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: [usize; 2],
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 1 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let a_id = node.inputs[0];
    let x_dims = producer_dims(graph, a_id)?;
    let op_name = match kind {
        PoolKind::Max => "MaxPool2d",
        PoolKind::Average => "AveragePool2d",
    };
    if x_dims.len() != 4 {
        return Err(CompileError::UnsupportedRank {
            node: a_id,
            op: op_name,
            rank: x_dims.len(),
        });
    }
    let (n, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (kh, kw) = (kernel[0], kernel[1]);
    let (sh, sw) = (strides[0], strides[1]);
    let (ph, pw) = (padding[0], padding[1]);
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let padded_h = h + 2 * ph;
    let padded_w = w + 2 * pw;
    if padded_h < kh || padded_w < kw {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let oh = (padded_h - kh) / sh + 1;
    let ow = (padded_w - kw) / sw + 1;
    if node.dims() != [n, c, oh, ow].as_slice() {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }

    let out = var_ident(node.id);
    let xr = var_ref(a_id, input_id);
    let len = usize_lit(n * c * oh * ow);
    let c_lit = usize_lit(c);
    let h_lit = usize_lit(h);
    let w_lit = usize_lit(w);
    let oh_lit = usize_lit(oh);
    let ow_lit = usize_lit(ow);
    let sh_lit = usize_lit(sh);
    let sw_lit = usize_lit(sw);
    let ph_lit = usize_lit(ph);
    let pw_lit = usize_lit(pw);

    let body = for_or_unroll("ni", n, |ni| {
        for_or_unroll("ci", c, |ci| {
            for_or_unroll("oy", oh, |oy| {
                for_or_unroll("ox", ow, |ox| {
                    let window = for_or_unroll("ky", kh, |ky| {
                        for_or_unroll("kx", kw, |kx| match kind {
                            PoolKind::Max => quote! {
                                {
                                    let y = (#oy) * #sh_lit + (#ky);
                                    let x = (#ox) * #sw_lit + (#kx);
                                    if y >= #ph_lit && y - #ph_lit < #h_lit && x >= #pw_lit && x - #pw_lit < #w_lit {
                                        let iy = y - #ph_lit;
                                        let ix = x - #pw_lit;
                                        let in_idx = ((#ni) * #c_lit + (#ci)) * #h_lit * #w_lit + iy * #w_lit + ix;
                                        let v = #xr[in_idx];
                                        if v > best { best = v; }
                                    }
                                }
                            },
                            PoolKind::Average => quote! {
                                {
                                    let y = (#oy) * #sh_lit + (#ky);
                                    let x = (#ox) * #sw_lit + (#kx);
                                    if y >= #ph_lit && y - #ph_lit < #h_lit && x >= #pw_lit && x - #pw_lit < #w_lit {
                                        let iy = y - #ph_lit;
                                        let ix = x - #pw_lit;
                                        let in_idx = ((#ni) * #c_lit + (#ci)) * #h_lit * #w_lit + iy * #w_lit + ix;
                                        let v = #xr[in_idx];
                                        acc += v;
                                        count += 1usize;
                                    }
                                }
                            },
                        })
                    });
                    let out_idx =
                        quote! { (((#ni) * #c_lit + (#ci)) * #oh_lit + (#oy)) * #ow_lit + (#ox) };
                    match kind {
                        PoolKind::Max => quote! {
                            {
                                let mut best: f32 = f32::NEG_INFINITY;
                                #window
                                #out[#out_idx] = best;
                            }
                        },
                        PoolKind::Average => quote! {
                            {
                                let mut acc: f32 = 0.0f32;
                                let mut count: usize = 0;
                                #window
                                #out[#out_idx] = if count == 0 { 0.0f32 } else { acc / (count as f32) };
                            }
                        },
                    }
                })
            })
        })
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

/// `BatchNorm`: `y = (x - mean) / sqrt(var + eps) * scale + bias`, matching
/// `tpt_infer_runtime::kernels::batch_norm`'s two accepted parameter
/// layouts — per-channel (`[c]`, rank-4 NCHW input) or per-element
/// (`numel(x)`).
fn emit_batch_norm(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    epsilon: f32,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 5 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let x_id = node.inputs[0];
    let scale_id = node.inputs[1];
    let bias_id = node.inputs[2];
    let mean_id = node.inputs[3];
    let var_id = node.inputs[4];
    let x_dims = producer_dims(graph, x_id)?;
    let scale_dims = producer_dims(graph, scale_id)?;
    let bias_dims = producer_dims(graph, bias_id)?;
    let mean_dims = producer_dims(graph, mean_id)?;
    let var_dims = producer_dims(graph, var_id)?;
    let numel: usize = x_dims.iter().product();
    let plen: usize = scale_dims.iter().product();
    if bias_dims.iter().product::<usize>() != plen
        || mean_dims.iter().product::<usize>() != plen
        || var_dims.iter().product::<usize>() != plen
    {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let channel_mode = x_dims.len() == 4 && plen == x_dims[1];
    let element_mode = plen == numel;
    if !channel_mode && !element_mode {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }

    let out = var_ident(node.id);
    let x = var_ref(x_id, input_id);
    let scale = var_ref(scale_id, input_id);
    let bias = var_ref(bias_id, input_id);
    let mean = var_ref(mean_id, input_id);
    let var = var_ref(var_id, input_id);
    let len = usize_lit(numel);
    let eps_lit = float_literal(epsilon);

    let body = if channel_mode {
        let (n, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let hw = h * w;
        let hw_lit = usize_lit(hw);
        let c_lit = usize_lit(c);
        for_or_unroll("ni", n, |ni| {
            for_or_unroll("ci", c, |ci| {
                let inner = for_or_unroll("p", hw, |p| {
                    quote! {
                        {
                            let idx = ((#ni) * #c_lit + (#ci)) * #hw_lit + (#p);
                            #out[idx] = #x[idx] * f + g;
                        }
                    }
                });
                quote! {
                    {
                        let inv = 1.0f32 / (#var[(#ci)] + #eps_lit).sqrt();
                        let f = #scale[(#ci)] * inv;
                        let g = #bias[(#ci)] - #mean[(#ci)] * f;
                        #inner
                    }
                }
            })
        })
    } else {
        for_or_unroll("i", numel, |i| {
            quote! {
                {
                    let inv = 1.0f32 / (#var[(#i)] + #eps_lit).sqrt();
                    let f = #scale[(#i)] * inv;
                    #out[(#i)] = (#x[(#i)] - #mean[(#i)]) * f + #bias[(#i)];
                }
            }
        })
    };

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

/// `Concat`: streams each operand into its channel-offset region of the
/// output, matching `tpt_infer_runtime::kernels::concat_copy_one`'s
/// `outer`/`c_src`/`inner` block layout.
fn emit_concat(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    axis: i32,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() < 2 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let out_dims = node.dims();
    let rank = out_dims.len();
    if rank == 0 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let ax = if axis < 0 { axis + rank as i32 } else { axis };
    if ax < 0 || ax as usize >= rank {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let axis_us = ax as usize;
    let c_total = out_dims[axis_us];
    let inner: usize = out_dims[axis_us + 1..].iter().product();

    let out = var_ident(node.id);
    let numel: usize = out_dims.iter().product();
    let len = usize_lit(numel);
    let c_total_lit = usize_lit(c_total);
    let inner_lit = usize_lit(inner);

    let mut c_offset = 0usize;
    let mut copies = Vec::new();
    for &inp in &node.inputs {
        let dims = producer_dims(graph, inp)?;
        if dims.len() != rank {
            return Err(CompileError::ShapeMismatch { node: node.id });
        }
        for (d, &v) in dims.iter().enumerate() {
            if d != axis_us && v != out_dims[d] {
                return Err(CompileError::ShapeMismatch { node: node.id });
            }
        }
        let c_src = dims[axis_us];
        let outer: usize = dims[..axis_us].iter().product();
        let src = var_ref(inp, input_id);
        let c_src_lit = usize_lit(c_src);
        let c_offset_lit = usize_lit(c_offset);
        let copy = for_or_unroll("o", outer, |o| {
            for_or_unroll("cc", c_src, |cc| {
                for_or_unroll("p", inner, |p| {
                    quote! {
                        #out[(#o) * #c_total_lit * #inner_lit + ((#c_offset_lit) + (#cc)) * #inner_lit + (#p)]
                            = #src[(#o) * #c_src_lit * #inner_lit + (#cc) * #inner_lit + (#p)];
                    }
                })
            })
        });
        copies.push(copy);
        c_offset += c_src;
    }
    if c_offset != c_total {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #(#copies)*
    })
}

/// `Transpose`: for each output flat index, decode it into `perm`'s output
/// coordinates (via compile-time-constant strides) and permute them into
/// the source's strides, matching `tpt_infer_runtime::kernels::transpose`.
fn emit_transpose(
    graph: &ComputationGraph,
    node: &Node,
    input_id: usize,
    perm: &[usize],
    rank: usize,
) -> Result<TokenStream, CompileError> {
    if node.inputs.len() != 1 {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    let a_id = node.inputs[0];
    let x_dims = producer_dims(graph, a_id)?;
    let out_dims = node.dims();
    if x_dims.len() != rank || out_dims.len() != rank || perm.len() != rank {
        return Err(CompileError::ShapeMismatch { node: node.id });
    }
    for d in 0..rank {
        if out_dims[d] != x_dims[perm[d]] {
            return Err(CompileError::ShapeMismatch { node: node.id });
        }
    }

    let numel: usize = out_dims.iter().product();
    let x_strides = row_major_strides(x_dims);
    let out_strides = row_major_strides(out_dims);

    let out = var_ident(node.id);
    let xr = var_ref(a_id, input_id);
    let len = usize_lit(numel);

    let body = for_or_unroll("t", numel, |t| {
        let terms: Vec<TokenStream> = (0..rank)
            .map(|d| {
                let os = usize_lit(out_strides[d]);
                let od = usize_lit(out_dims[d]);
                let xs = usize_lit(x_strides[perm[d]]);
                quote! { (((#t) / #os) % #od) * #xs }
            })
            .collect();
        quote! {
            #out[(#t)] = #xr[#(#terms)+*];
        }
    });

    Ok(quote! {
        let mut #out: Vec<f32> = std::vec![0.0f32; #len];
        #body
    })
}

fn usize_lit(v: usize) -> TokenStream {
    let lit = Literal::usize_unsuffixed(v);
    quote! { #lit }
}

/// Runs `body` for each index `0..bound`, either fully unrolled (bound <=
/// `UNROLL_THRESHOLD`) with integer literals, or as a single `for #var in
/// 0..bound` loop with `var` as the index identifier.
fn for_or_unroll(
    var: &str,
    bound: usize,
    body: impl Fn(&TokenStream) -> TokenStream,
) -> TokenStream {
    if bound == 0 {
        return quote! {};
    }
    if bound <= UNROLL_THRESHOLD {
        let stmts: Vec<TokenStream> = (0..bound)
            .map(|i| {
                let lit = usize_lit(i);
                body(&lit)
            })
            .collect();
        quote! { #(#stmts)* }
    } else {
        let ident = format_ident!("{}", var);
        let bound_lit = usize_lit(bound);
        let idx_ts = quote! { #ident };
        let inner = body(&idx_ts);
        quote! {
            for #ident in 0..#bound_lit {
                #inner
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::fold_constants;
    use tpt_infer_graph::Initializer;

    #[test]
    fn row_major_strides_matches_manual_computation() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    fn generate_body(graph: &ComputationGraph, input_id: usize, output_id: usize) -> String {
        let order = graph.topological_sort().unwrap();
        let folded = fold_constants(graph, &order);
        generate(graph, &order, input_id, output_id, &folded)
            .unwrap()
            .to_string()
    }

    #[test]
    fn conv2d_rejects_wrong_bias_layout() {
        // A flat `[oc]` bias does not broadcast onto the channel axis under
        // tpt-infer-runtime's right-aligned broadcasting (only `[1, oc, 1,
        // 1]` does), so codegen must reject it rather than silently
        // generating code that disagrees with the interpreted runtime.
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 1, 3, 3]).unwrap())
            .unwrap();
        let w = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[2, 1, 2, 2])
                    .unwrap()
                    .with_name("w"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("w", &[2, 1, 2, 2], vec![1.0; 8]).unwrap());
        let bias = g
            .add_node(
                Node::new(2, Operator::Input, vec![], &[2])
                    .unwrap()
                    .with_name("bias"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("bias", &[2], vec![0.0, 0.0]).unwrap());
        let conv = g
            .add_node(
                Node::new(
                    3,
                    Operator::conv2d([1, 1], [0, 0]),
                    vec![x, w, bias],
                    &[1, 2, 2, 2],
                )
                .unwrap(),
            )
            .unwrap();
        g.mark_output(conv).unwrap();

        let order = graph_topo(&g);
        let folded = fold_constants(&g, &order);
        let err = generate(&g, &order, x, conv, &folded).unwrap_err();
        assert_eq!(
            err,
            CompileError::UnsupportedRank {
                node: bias,
                op: "Conv2d bias",
                rank: 1,
            }
        );
    }

    fn graph_topo(graph: &ComputationGraph) -> Vec<usize> {
        graph.topological_sort().unwrap()
    }

    #[test]
    fn softmax_non_last_axis_is_rejected() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(1, Operator::softmax(0), vec![x], &[2, 3]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();

        let order = graph_topo(&g);
        let folded = fold_constants(&g, &order);
        let err = generate(&g, &order, x, y, &folded).unwrap_err();
        assert!(matches!(
            err,
            CompileError::UnsupportedRank {
                node: 1,
                op: "Softmax",
                ..
            }
        ));
    }

    #[test]
    fn softmax_last_axis_generates_exp_and_division() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(1, Operator::softmax(-1), vec![x], &[1, 4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();

        let src = generate_body(&g, x, y);
        assert!(src.contains("exp"));
        assert!(src.contains("max_v"));
    }

    #[test]
    fn transpose_uses_the_active_permutation_only() {
        // Regression test: `Operator::Transpose::perm` is a `[usize; 8]`
        // array zero-padded past `rank`; codegen must slice it to
        // `perm[..rank]` rather than treating the trailing zero padding as
        // real permutation entries.
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
            .unwrap();
        let y = g
            .add_node(
                Node::new(1, Operator::transpose(&[1, 0]).unwrap(), vec![x], &[3, 2]).unwrap(),
            )
            .unwrap();
        g.mark_output(y).unwrap();

        let order = graph_topo(&g);
        let folded = fold_constants(&g, &order);
        assert!(generate(&g, &order, x, y, &folded).is_ok());
    }
}
