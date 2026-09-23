//! Runtime computation graph: nodes, edges, initializers, and topological sort.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use tpt_infer_core::{Shape, MAX_RANK};

use crate::operator::Operator;

/// Errors produced while building or analysing a [`ComputationGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    /// A node id referenced by an edge or input list does not exist.
    UnknownNode {
        /// The unknown node id.
        id: usize,
    },
    /// A node was added with an id other than the next sequential id.
    NodeIdMismatch {
        /// The id the graph expected (current node count).
        expected: usize,
        /// The id found on the node.
        found: usize,
    },
    /// A shape rank exceeds [`MAX_RANK`].
    RankTooLarge {
        /// The offending rank.
        rank: usize,
    },
    /// Element count does not match the product of the shape.
    SizeMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// The graph contains a cycle.
    CycleDetected,
}

impl core::fmt::Display for GraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphError::UnknownNode { id } => write!(f, "unknown node id {id}"),
            GraphError::NodeIdMismatch { expected, found } => {
                write!(f, "node id mismatch: expected {expected}, found {found}")
            }
            GraphError::RankTooLarge { rank } => {
                write!(f, "shape rank {rank} exceeds maximum {MAX_RANK}")
            }
            GraphError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch: expected {expected} elements, got {actual}"
                )
            }
            GraphError::CycleDetected => f.write_str("graph contains a cycle"),
        }
    }
}

#[cfg(any(feature = "std", test))]
impl core::error::Error for GraphError {}

/// A node in a [`ComputationGraph`].
///
/// Each node has exactly one output tensor, described by
/// [`output_shape`](Node::output_shape) and [`output_rank`](Node::output_rank),
/// and consumes the outputs of the nodes listed in [`inputs`](Node::inputs).
///
/// Node ids must be assigned sequentially: the id of the first node added to a
/// graph is `0`, and each subsequent node's id is one greater (see
/// [`ComputationGraph::add_node`]).
///
/// # Example
/// ```
/// use tpt_infer_graph::{Node, Operator};
/// let n = Node::new(0, Operator::Input, vec![], &[1, 784]).unwrap();
/// assert_eq!(n.dims(), &[1, 784]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Unique id (equal to the node's index in [`ComputationGraph::nodes`]).
    pub id: usize,
    /// Operation performed by this node.
    pub operator: Operator,
    /// Ids of nodes whose outputs feed this node's inputs (in port order).
    pub inputs: Vec<usize>,
    /// Output tensor shape, zero-padded to [`MAX_RANK`].
    pub output_shape: Shape,
    /// Number of active dimensions in [`output_shape`](Node::output_shape).
    pub output_rank: usize,
    /// Optional model-supplied name (e.g. the ONNX tensor/node name).
    pub name: Option<String>,
}

impl Node {
    /// Create a node with the given id, operator, inputs, and output `dims`.
    ///
    /// # Errors
    /// [`GraphError::RankTooLarge`] if `dims.len()` exceeds [`MAX_RANK`].
    ///
    /// # Example
    /// ```
    /// use tpt_infer_graph::{GraphError, Node, Operator};
    /// let n = Node::new(0, Operator::Relu, vec![7], &[1, 128]).unwrap();
    /// assert_eq!(n.inputs, vec![7]);
    /// let err = Node::new(0, Operator::Relu, vec![], &[1; 9]).unwrap_err();
    /// assert_eq!(err, GraphError::RankTooLarge { rank: 9 });
    /// ```
    pub fn new(
        id: usize,
        operator: Operator,
        inputs: Vec<usize>,
        dims: &[usize],
    ) -> Result<Self, GraphError> {
        if dims.len() > MAX_RANK {
            return Err(GraphError::RankTooLarge { rank: dims.len() });
        }
        let mut output_shape = [0; MAX_RANK];
        output_shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            id,
            operator,
            inputs,
            output_shape,
            output_rank: dims.len(),
            name: None,
        })
    }

    /// Active output dimensions (the first [`output_rank`](Node::output_rank)
    /// entries of [`output_shape`](Node::output_shape)).
    pub fn dims(&self) -> &[usize] {
        &self.output_shape[..self.output_rank]
    }

    /// Set the node's model-supplied name and return `self`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// A directed edge between two node ports, carrying the tensor shape that
/// flows along it.
///
/// Edges mirror the per-node [`inputs`](Node::inputs) lists with additional
/// port and shape metadata for downstream consumers (AOT compiler, runtime).
///
/// # Example
/// ```
/// use tpt_infer_graph::Edge;
/// let e = Edge::new(0, 0, 2, 1, &[1, 784]).unwrap();
/// assert_eq!(e.dims(), &[1, 784]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Source node id.
    pub from: usize,
    /// Source output port (most operators have a single output, port `0`).
    pub from_port: usize,
    /// Destination node id.
    pub to: usize,
    /// Destination input port (index into the destination's `inputs` list).
    pub to_port: usize,
    /// Shape of the tensor flowing along this edge, zero-padded to [`MAX_RANK`].
    pub shape: Shape,
    /// Number of active dimensions in [`shape`](Edge::shape).
    pub rank: usize,
}

impl Edge {
    /// Create an edge from output port `from_port` of node `from` to input
    /// port `to_port` of node `to`, carrying tensor shape `dims`.
    ///
    /// # Errors
    /// [`GraphError::RankTooLarge`] if `dims.len()` exceeds [`MAX_RANK`].
    pub fn new(
        from: usize,
        from_port: usize,
        to: usize,
        to_port: usize,
        dims: &[usize],
    ) -> Result<Self, GraphError> {
        if dims.len() > MAX_RANK {
            return Err(GraphError::RankTooLarge { rank: dims.len() });
        }
        let mut shape = [0; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            from,
            from_port,
            to,
            to_port,
            shape,
            rank: dims.len(),
        })
    }

    /// Active edge dimensions (the first [`rank`](Edge::rank) entries of
    /// [`shape`](Edge::shape)).
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }
}

/// A named constant tensor (model weight) bound to an [`Operator::Input`] node.
///
/// # Example
/// ```
/// use tpt_infer_graph::Initializer;
/// let init = Initializer::new("w1", &[2, 2], vec![0.0; 4]).unwrap();
/// assert_eq!(init.dims(), &[2, 2]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Initializer {
    /// Name binding this tensor to an input node (e.g. an ONNX initializer name).
    pub name: String,
    /// Tensor shape, zero-padded to [`MAX_RANK`].
    pub shape: Shape,
    /// Number of active dimensions in [`shape`](Initializer::shape).
    pub rank: usize,
    /// Element data (row-major, `f32` weights for the reference runtime).
    pub data: Vec<f32>,
}

impl Initializer {
    /// Create an initializer with the given name, shape, and element data.
    ///
    /// # Errors
    /// - [`GraphError::RankTooLarge`] if `dims.len()` exceeds [`MAX_RANK`]
    /// - [`GraphError::SizeMismatch`] if `data.len()` differs from the product of `dims`
    pub fn new(
        name: impl Into<String>,
        dims: &[usize],
        data: Vec<f32>,
    ) -> Result<Self, GraphError> {
        if dims.len() > MAX_RANK {
            return Err(GraphError::RankTooLarge { rank: dims.len() });
        }
        let expected: usize = dims.iter().product();
        if data.len() != expected {
            return Err(GraphError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let mut shape = [0; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            name: name.into(),
            shape,
            rank: dims.len(),
            data,
        })
    }

    /// Active dimensions (the first [`rank`](Initializer::rank) entries of
    /// [`shape`](Initializer::shape)).
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }
}

/// A dynamic computation graph (rank ≤ [`MAX_RANK`]) with shape-carrying
/// edges and Kahn-algorithm topological sorting.
///
/// This is the representation consumed by the ONNX loader, AOT compiler, and
/// runtime. For compile-time shape checking, build graphs with
/// [`GraphBuilder`](crate::GraphBuilder) instead.
///
/// # Example
/// ```
/// use tpt_infer_graph::{ComputationGraph, Node, Operator};
/// let mut g = ComputationGraph::new();
/// let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap()).unwrap();
/// let y = g.add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap()).unwrap();
/// g.mark_output(y).unwrap();
/// assert_eq!(g.topological_sort().unwrap(), vec![0, 1]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComputationGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    initializers: Vec<Initializer>,
}

impl ComputationGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// All nodes, indexed by id (`nodes[i].id == i`).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// All edges.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Ids of graph input nodes (nodes with [`Operator::Input`]), in insertion order.
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// Ids of graph output nodes, as set by [`mark_output`](Self::mark_output)
    /// or [`infer_outputs`](Self::infer_outputs).
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// Constant tensors bound to input nodes, in insertion order.
    pub fn initializers(&self) -> &[Initializer] {
        &self.initializers
    }

    /// Look up a node by id.
    pub fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Append `node`, assigning it the next sequential id.
    ///
    /// Nodes with [`Operator::Input`] are automatically registered as graph
    /// inputs.
    ///
    /// # Errors
    /// [`GraphError::NodeIdMismatch`] if `node.id` is not the next sequential
    /// id (the current node count).
    pub fn add_node(&mut self, node: Node) -> Result<usize, GraphError> {
        let expected = self.nodes.len();
        if node.id != expected {
            return Err(GraphError::NodeIdMismatch {
                expected,
                found: node.id,
            });
        }
        Ok(self.push_node(node))
    }

    /// Append `edge`.
    ///
    /// # Errors
    /// [`GraphError::UnknownNode`] if either endpoint id does not exist yet.
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), GraphError> {
        if edge.from >= self.nodes.len() {
            return Err(GraphError::UnknownNode { id: edge.from });
        }
        if edge.to >= self.nodes.len() {
            return Err(GraphError::UnknownNode { id: edge.to });
        }
        self.push_edge(edge);
        Ok(())
    }

    /// Register `id` as a graph input. Idempotent.
    ///
    /// # Errors
    /// [`GraphError::UnknownNode`] if `id` does not exist.
    pub fn mark_input(&mut self, id: usize) -> Result<(), GraphError> {
        if id >= self.nodes.len() {
            return Err(GraphError::UnknownNode { id });
        }
        if !self.inputs.contains(&id) {
            self.inputs.push(id);
        }
        Ok(())
    }

    /// Register `id` as a graph output. Idempotent.
    ///
    /// # Errors
    /// [`GraphError::UnknownNode`] if `id` does not exist.
    pub fn mark_output(&mut self, id: usize) -> Result<(), GraphError> {
        if id >= self.nodes.len() {
            return Err(GraphError::UnknownNode { id });
        }
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
        Ok(())
    }

    /// Attach a named constant tensor to the graph.
    pub fn add_initializer(&mut self, init: Initializer) {
        self.initializers.push(init);
    }

    /// Recompute [`outputs`](Self::outputs) as every non-input node that is
    /// not consumed by another node.
    pub fn infer_outputs(&mut self) {
        let mut consumed = vec![false; self.nodes.len()];
        for node in &self.nodes {
            for &inp in &node.inputs {
                if let Some(slot) = consumed.get_mut(inp) {
                    *slot = true;
                }
            }
        }
        self.outputs = self
            .nodes
            .iter()
            .filter(|n| !n.operator.is_input() && !consumed[n.id])
            .map(|n| n.id)
            .collect();
    }

    /// Kahn's algorithm over node input lists.
    ///
    /// Ties are broken by node id, so the order is deterministic.
    ///
    /// # Errors
    /// - [`GraphError::UnknownNode`] if a node references a missing input id
    /// - [`GraphError::CycleDetected`] if the graph contains a cycle
    ///
    /// # Example
    /// ```
    /// use tpt_infer_graph::{ComputationGraph, Node, Operator};
    /// let mut g = ComputationGraph::new();
    /// g.add_node(Node::new(0, Operator::Relu, vec![1], &[1]).unwrap()).unwrap();
    /// g.add_node(Node::new(1, Operator::Relu, vec![0], &[1]).unwrap()).unwrap();
    /// assert_eq!(
    ///     g.topological_sort().unwrap_err(),
    ///     tpt_infer_graph::GraphError::CycleDetected
    /// );
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<usize>, GraphError> {
        let n = self.nodes.len();
        let mut indegree = vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for node in &self.nodes {
            for &inp in &node.inputs {
                if inp >= n {
                    return Err(GraphError::UnknownNode { id: inp });
                }
                indegree[node.id] += 1;
                adjacency[inp].push(node.id);
            }
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&id| indegree[id] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for &next in &adjacency[id] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        if order.len() != n {
            return Err(GraphError::CycleDetected);
        }
        Ok(order)
    }

    /// Append a node, assigning the next sequential id and registering
    /// [`Operator::Input`] nodes as graph inputs. Infallible: the caller has
    /// already ensured the id invariant.
    pub(crate) fn push_node(&mut self, mut node: Node) -> usize {
        let id = self.nodes.len();
        node.id = id;
        let is_input = node.operator.is_input();
        self.nodes.push(node);
        if is_input {
            self.inputs.push(id);
        }
        id
    }

    /// Append an edge without endpoint validation (internal, infallible).
    pub(crate) fn push_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let h = g
            .add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(2, Operator::Softmax { axis: -1 }, vec![h], &[1, 4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        g
    }

    #[test]
    fn valid_graph_builds_and_sorts() {
        let g = chain();
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.inputs(), &[0]);
        assert_eq!(g.outputs(), &[2]);
        assert_eq!(g.topological_sort().unwrap(), vec![0, 1, 2]);
        assert_eq!(g.node(1).unwrap().dims(), &[1, 4]);
    }

    #[test]
    fn node_id_mismatch_rejected() {
        let mut g = ComputationGraph::new();
        let err = g
            .add_node(Node::new(3, Operator::Input, vec![], &[1]).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            GraphError::NodeIdMismatch {
                expected: 0,
                found: 3
            }
        );
    }

    #[test]
    fn node_rank_too_large() {
        let err = Node::new(0, Operator::Input, vec![], &[1; 9]).unwrap_err();
        assert_eq!(err, GraphError::RankTooLarge { rank: 9 });
    }

    #[test]
    fn cycle_detected() {
        let mut g = ComputationGraph::new();
        g.add_node(Node::new(0, Operator::Add, vec![1], &[2]).unwrap())
            .unwrap();
        g.add_node(Node::new(1, Operator::Add, vec![0], &[2]).unwrap())
            .unwrap();
        assert_eq!(g.topological_sort(), Err(GraphError::CycleDetected));
    }

    #[test]
    fn self_loop_is_a_cycle() {
        let mut g = ComputationGraph::new();
        g.add_node(Node::new(0, Operator::Relu, vec![0], &[4]).unwrap())
            .unwrap();
        assert_eq!(g.topological_sort(), Err(GraphError::CycleDetected));
    }

    #[test]
    fn unknown_input_detected() {
        let mut g = ComputationGraph::new();
        g.add_node(Node::new(0, Operator::Relu, vec![9], &[4]).unwrap())
            .unwrap();
        assert_eq!(g.topological_sort(), Err(GraphError::UnknownNode { id: 9 }));
    }

    #[test]
    fn edge_endpoint_validation() {
        let mut g = chain();
        let bad = Edge::new(0, 0, 99, 0, &[1, 4]).unwrap();
        assert_eq!(g.add_edge(bad), Err(GraphError::UnknownNode { id: 99 }));
        let ok = Edge::new(0, 0, 1, 0, &[1, 4]).unwrap();
        assert!(g.add_edge(ok).is_ok());
        assert_eq!(g.edges().len(), 1);
        assert_eq!(g.edges()[0].dims(), &[1, 4]);
    }

    #[test]
    fn mark_unknown_input_and_output() {
        let mut g = ComputationGraph::new();
        assert_eq!(g.mark_input(5), Err(GraphError::UnknownNode { id: 5 }));
        assert_eq!(g.mark_output(5), Err(GraphError::UnknownNode { id: 5 }));
    }

    #[test]
    fn initializer_size_checked() {
        let err = Initializer::new("w", &[2, 3], alloc::vec![0.0; 5]).unwrap_err();
        assert_eq!(
            err,
            GraphError::SizeMismatch {
                expected: 6,
                actual: 5
            }
        );
        let err = Initializer::new("w", &[1; 9], alloc::vec![0.0; 9]).unwrap_err();
        assert_eq!(err, GraphError::RankTooLarge { rank: 9 });
        let init = Initializer::new("w", &[2, 3], alloc::vec![1.0; 6]).unwrap();
        assert_eq!(init.dims(), &[2, 3]);
        assert_eq!(init.data.len(), 6);
    }

    #[test]
    fn infer_outputs_skips_inputs_and_consumed_nodes() {
        let mut g = ComputationGraph::new();
        // Input that feeds nothing: still an input, not an output.
        g.add_node(Node::new(0, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        g.add_node(Node::new(1, Operator::Input, vec![], &[1]).unwrap())
            .unwrap();
        g.add_node(Node::new(2, Operator::Add, vec![0, 1], &[1]).unwrap())
            .unwrap();
        g.add_node(Node::new(3, Operator::Relu, vec![2], &[1]).unwrap())
            .unwrap();
        g.infer_outputs();
        assert_eq!(g.outputs(), &[3]);
    }

    #[test]
    fn mark_output_idempotent() {
        let mut g = chain();
        g.mark_output(2).unwrap();
        g.mark_output(2).unwrap();
        assert_eq!(g.outputs(), &[2]);
    }

    #[test]
    fn default_is_empty() {
        let g = ComputationGraph::new();
        assert!(g.nodes().is_empty());
        assert!(g.edges().is_empty());
        assert!(g.inputs().is_empty());
        assert!(g.outputs().is_empty());
        assert!(g.initializers().is_empty());
        assert_eq!(g.topological_sort().unwrap(), Vec::new());
    }
}
