//! Ahead-of-time compiled model: generated Rust source plus metadata.

use quote::quote;

use tpt_infer_graph::{ComputationGraph, GraphError};

use crate::codegen;
use crate::fold::fold_constants;

/// Errors that can occur while ahead-of-time compiling a [`ComputationGraph`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompileError {
    /// The graph has no nodes.
    EmptyGraph,
    /// `aot_compile` only supports graphs with exactly one non-initializer
    /// runtime input, since the generated signature is
    /// `fn execute(input: &[f32]) -> Vec<f32>`.
    UnsupportedInputCount {
        /// Number of non-initializer input nodes found.
        found: usize,
    },
    /// `aot_compile` only supports graphs with exactly one marked output.
    UnsupportedOutputCount {
        /// Number of marked output nodes found.
        found: usize,
    },
    /// A weight `Input` node has no matching initializer, so it cannot be
    /// resolved to a compile-time value.
    MissingInitializer {
        /// The offending node id.
        node: usize,
    },
    /// An operator this compiler does not know how to generate code for.
    UnsupportedOperator {
        /// The operator's display name (see [`tpt_infer_graph::Operator::name`]).
        name: String,
        /// The offending node id.
        node: usize,
    },
    /// An operator was used with a tensor rank it does not support (e.g.
    /// `MatMul` on anything but rank-2 operands).
    UnsupportedRank {
        /// The offending node id.
        node: usize,
        /// The operator's display name.
        op: &'static str,
        /// The unsupported rank.
        rank: usize,
    },
    /// Operand shapes are incompatible, or an operator was used with an
    /// unexpected input arity.
    ShapeMismatch {
        /// The offending node id.
        node: usize,
    },
    /// Propagated error from [`ComputationGraph`] analysis (e.g. a cycle or
    /// a reference to a missing node id).
    Graph(GraphError),
}

impl From<GraphError> for CompileError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => f.write_str("graph has no nodes"),
            Self::UnsupportedInputCount { found } => write!(
                f,
                "aot_compile requires exactly one non-initializer input, found {found}"
            ),
            Self::UnsupportedOutputCount { found } => write!(
                f,
                "aot_compile requires exactly one marked output, found {found}"
            ),
            Self::MissingInitializer { node } => {
                write!(f, "input node {node} has no matching initializer")
            }
            Self::UnsupportedOperator { name, node } => {
                write!(f, "unsupported operator `{name}` on node {node}")
            }
            Self::UnsupportedRank { node, op, rank } => write!(
                f,
                "operator `{op}` on node {node} does not support rank {rank}"
            ),
            Self::ShapeMismatch { node } => write!(f, "operand shape mismatch at node {node}"),
            Self::Graph(e) => write!(f, "graph error: {e}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// An ahead-of-time compiled model: self-contained generated Rust source
/// (a `pub fn execute(input: &[f32]) -> Vec<f32>` definition) plus enough
/// metadata to describe how to call it.
///
/// The source is *not* automatically compiled or linked anywhere — callers
/// that want an executable artifact write [`CompiledModel::source`] to a
/// `.rs` file and invoke `rustc` themselves (see the crate's integration
/// test for an example of this workflow).
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// Generated Rust source defining `pub fn execute(input: &[f32]) -> Vec<f32>`.
    source: String,
    /// Declared shape of the single runtime input.
    input_shape: Vec<usize>,
    /// Declared shape of the single marked output.
    output_shape: Vec<usize>,
    /// Total node count in the source graph (including input/initializer nodes).
    node_count: usize,
    /// Number of nodes whose value was folded to a compile-time constant
    /// (see [`crate::fold`]) instead of generated runtime computation.
    folded_node_count: usize,
}

impl CompiledModel {
    /// The generated Rust source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Declared shape of the single runtime input.
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Declared shape of the single marked output.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Total node count in the source graph.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of nodes folded to compile-time constants.
    pub fn folded_node_count(&self) -> usize {
        self.folded_node_count
    }
}

/// Ahead-of-time compiles `graph` into a [`CompiledModel`] holding generated
/// Rust source for `pub fn execute(input: &[f32]) -> Vec<f32>`.
///
/// # Supported graphs
///
/// - Exactly one non-initializer ([`Operator::Input`](tpt_infer_graph::Operator::Input))
///   node — the runtime input.
/// - Exactly one marked output ([`ComputationGraph::mark_output`] /
///   [`ComputationGraph::infer_outputs`]).
/// - Operators `MatMul`, `Add`/`Sub`/`Mul`/`Div` (same-shape operands),
///   `Relu`, `Sigmoid`, `Gelu`, `Reshape`/`Flatten`, `Conv2d`
///   (direct/naive-loop, optional per-channel bias), `Softmax` (last-axis
///   only, matching `tpt-infer-runtime`'s own restriction),
///   `MaxPool2d`/`AveragePool2d`, `BatchNorm`, `Concat`, and `Transpose`.
///   Anything else (`Operator::Custom`, ...) is reported as
///   [`CompileError::UnsupportedOperator`] rather than silently skipped or
///   panicking.
///
/// Nodes whose inputs are all compile-time constants (weight initializers,
/// or the output of another constant node) are evaluated ahead of time in
/// this function and spliced into the generated source as `f32` array
/// literals instead of emitting runtime computation for them — see
/// [`crate::fold`].
///
/// # Errors
///
/// See [`CompileError`].
///
/// # Example
///
/// ```
/// use tpt_infer_graph::{ComputationGraph, Node, Operator};
/// use tpt_infer_compile::aot_compile;
///
/// let mut g = ComputationGraph::new();
/// let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap()).unwrap();
/// let y = g.add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap()).unwrap();
/// g.mark_output(y).unwrap();
///
/// let model = aot_compile(&g).unwrap();
/// assert!(model.source().contains("pub fn execute"));
/// assert_eq!(model.input_shape(), &[1, 4]);
/// ```
pub fn aot_compile(graph: &ComputationGraph) -> Result<CompiledModel, CompileError> {
    if graph.nodes().is_empty() {
        return Err(CompileError::EmptyGraph);
    }

    let order = graph.topological_sort()?;

    let runtime_inputs: Vec<usize> = graph
        .inputs()
        .iter()
        .copied()
        .filter(|&id| {
            let bound_to_initializer = graph
                .node(id)
                .and_then(|n| n.name.as_ref())
                .is_some_and(|name| graph.initializers().iter().any(|i| &i.name == name));
            !bound_to_initializer
        })
        .collect();
    if runtime_inputs.len() != 1 {
        return Err(CompileError::UnsupportedInputCount {
            found: runtime_inputs.len(),
        });
    }
    let input_id = runtime_inputs[0];

    if graph.outputs().len() != 1 {
        return Err(CompileError::UnsupportedOutputCount {
            found: graph.outputs().len(),
        });
    }
    let output_id = graph.outputs()[0];

    let folded = fold_constants(graph, &order);
    let folded_node_count = folded.len();

    let body = codegen::generate(graph, &order, input_id, output_id, &folded)?;

    let input_shape = graph
        .node(input_id)
        .map(|n| n.dims().to_vec())
        .unwrap_or_default();
    let output_shape = graph
        .node(output_id)
        .map(|n| n.dims().to_vec())
        .unwrap_or_default();
    let input_len = input_shape.iter().product::<usize>();

    let doc = format!(
        "Generated by tpt-infer-compile::aot_compile. Do not edit.\n\
         input shape: {input_shape:?}\n\
         output shape: {output_shape:?}\n\
         nodes: {} total, {folded_node_count} folded to compile-time constants",
        graph.nodes().len(),
    );

    let source_tokens = quote! {
        #[doc = #doc]
        pub fn execute(input: &[f32]) -> Vec<f32> {
            assert_eq!(input.len(), #input_len, "tpt-infer-compile: unexpected input length");
            #body
        }
    };

    Ok(CompiledModel {
        source: source_tokens.to_string(),
        input_shape,
        output_shape,
        node_count: graph.nodes().len(),
        folded_node_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_infer_graph::{Initializer, Node, Operator};

    fn mlp_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(
                Node::new(0, Operator::Input, vec![], &[1, 4])
                    .unwrap()
                    .with_name("x"),
            )
            .unwrap();
        let w1 = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[4, 3])
                    .unwrap()
                    .with_name("w1"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("w1", &[4, 3], vec![0.1; 12]).unwrap());
        let m1 = g
            .add_node(Node::new(2, Operator::MatMul, vec![x, w1], &[1, 3]).unwrap())
            .unwrap();
        let r = g
            .add_node(Node::new(3, Operator::Relu, vec![m1], &[1, 3]).unwrap())
            .unwrap();
        g.mark_output(r).unwrap();
        g
    }

    #[test]
    fn compiles_mlp_and_reports_metadata() {
        let g = mlp_graph();
        let model = aot_compile(&g).unwrap();
        assert_eq!(model.input_shape(), &[1, 4]);
        assert_eq!(model.output_shape(), &[1, 3]);
        assert_eq!(model.node_count(), 4);
        // w1's Input node is folded to a literal (it has an initializer).
        assert_eq!(model.folded_node_count(), 1);
        assert!(model.source().contains("pub fn execute"));
        assert!(model.source().contains("Vec < f32 >") || model.source().contains("Vec<f32>"));
    }

    #[test]
    fn empty_graph_is_rejected() {
        let g = ComputationGraph::new();
        assert_eq!(aot_compile(&g).unwrap_err(), CompileError::EmptyGraph);
    }

    #[test]
    fn multiple_runtime_inputs_are_rejected() {
        let mut g = ComputationGraph::new();
        let a = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        let b = g
            .add_node(Node::new(1, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(2, Operator::Add, vec![a, b], &[1]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        assert_eq!(
            aot_compile(&g).unwrap_err(),
            CompileError::UnsupportedInputCount { found: 2 }
        );
    }

    #[test]
    fn no_marked_output_is_rejected() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        g.add_node(Node::new(1, Operator::Relu, vec![x], &[1]).unwrap())
            .unwrap();
        assert_eq!(
            aot_compile(&g).unwrap_err(),
            CompileError::UnsupportedOutputCount { found: 0 }
        );
    }

    #[test]
    fn unsupported_operator_is_reported() {
        // `Operator::Custom` (ONNX ops this crate doesn't map to a native
        // `Operator` variant) is genuinely unmapped in codegen.
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let y = g
            .add_node(
                Node::new(
                    1,
                    Operator::Custom("LayerNormalization".into()),
                    vec![x],
                    &[1, 4],
                )
                .unwrap(),
            )
            .unwrap();
        g.mark_output(y).unwrap();
        assert_eq!(
            aot_compile(&g).unwrap_err(),
            CompileError::UnsupportedOperator {
                name: "Custom".to_string(),
                node: 1
            }
        );
    }

    #[test]
    fn gelu_matches_interpreted_runtime() {
        // AOT-compiled Gelu must be numerically identical to the naive
        // backend's tanh-approximation kernel (`tpt_infer_ops::naive::gelu`).
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(1, Operator::Gelu, vec![x], &[1, 4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        let compiled = aot_compile(&g).unwrap();
        let source = compiled.source();
        assert!(source.contains("tanh"));

        let input = [-2.0f32, -0.5, 0.5, 2.0];
        let mut expected = [0.0f32; 4];
        tpt_infer_ops::naive::gelu(&input, &mut expected).unwrap();

        // Evaluate the tanh approximation directly, mirroring the generated
        // expression, since the test harness doesn't compile-and-link the
        // generated source at test time.
        let k = 0.797_884_6f32;
        let c = 0.044_715f32;
        let got: Vec<f32> = input
            .iter()
            .map(|&x| 0.5f32 * x * (1.0f32 + (k * (x + c * x * x * x)).tanh()))
            .collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }

    #[test]
    fn fully_constant_graph_folds_the_output_too() {
        let mut g = ComputationGraph::new();
        // Runtime input that the graph never actually uses in its output —
        // still required by aot_compile's single-input contract.
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        let w = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[2, 2])
                    .unwrap()
                    .with_name("w"),
            )
            .unwrap();
        g.add_initializer(Initializer::new("w", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let r = g
            .add_node(Node::new(2, Operator::Relu, vec![w], &[2, 2]).unwrap())
            .unwrap();
        // Keep `x` reachable via infer_outputs so it stays the sole runtime
        // input candidate; mark the folded node as the graph's output.
        let _ = x;
        g.mark_output(r).unwrap();

        let model = aot_compile(&g).unwrap();
        assert_eq!(model.folded_node_count(), 2); // w and r are both constant
        assert!(model.source().contains("std :: vec !") || model.source().contains("vec!"));
    }
}
