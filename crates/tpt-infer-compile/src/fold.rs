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
        _ => None,
    }
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
}
