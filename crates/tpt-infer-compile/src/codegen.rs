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
        Operator::Reshape { .. } | Operator::Flatten { .. } => emit_reshape(graph, node, input_id),
        other => Err(CompileError::UnsupportedOperator {
            name: other.name().to_string(),
            node: node.id,
        }),
    }
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
