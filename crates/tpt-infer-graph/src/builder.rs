//! Type-state `GraphBuilder`: tensor shapes are tracked in the type system,
//! so shape mismatches are compile errors rather than runtime panics.
//!
//! Shapes are modelled with the rank-2 marker [`Sh<M, K>`] for dense layers
//! (matmul / add / activations). The dynamic [`ComputationGraph`](crate::ComputationGraph)
//! produced by [`GraphBuilder::into_graph`] supports arbitrary ranks up to
//! `tpt_infer_core::MAX_RANK`; operators whose output shape involves division
//! or multiplication of attribute values (e.g. `Conv2d`, rank-4 `Flatten`)
//! cannot be expressed as stable const-generic arithmetic and are only checked
//! at runtime by the graph/ONNX layers.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use tpt_infer_core::{Shape, MAX_RANK};

use crate::graph::{ComputationGraph, Edge, GraphError, Initializer, Node};
use crate::operator::Operator;

/// A tensor shape known entirely at compile time.
///
/// Implementors expose their rank, element count, and a
/// [`Shape`](tpt_infer_core::Shape) conversion as associated constants.
pub trait ShapeMarker {
    /// Number of dimensions.
    const RANK: usize;

    /// Total number of elements (product of the active dimensions).
    const NUMEL: usize;

    /// Dynamic shape, zero-padded to [`MAX_RANK`].
    const SHAPE: Shape;
}

/// Compile-time 2-D shape marker for a tensor of shape `[M, K]`.
///
/// # Example
/// ```
/// use tpt_infer_graph::{ShapeMarker, Sh};
/// assert_eq!(Sh::<1, 784>::RANK, 2);
/// assert_eq!(Sh::<1, 784>::NUMEL, 784);
/// assert_eq!(&Sh::<1, 784>::SHAPE[..2], &[1, 784]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Sh<const M: usize, const K: usize>;

impl<const M: usize, const K: usize> ShapeMarker for Sh<M, K> {
    const RANK: usize = 2;
    const NUMEL: usize = M * K;
    const SHAPE: Shape = {
        let mut shape = [0; MAX_RANK];
        shape[0] = M;
        shape[1] = K;
        shape
    };
}

/// Type-level matrix-multiply shape computation:
/// `[M, K] x [K, N] -> [M, N]`.
///
/// The impl only exists when the contracting dimensions match, so
/// `Sh<1, 784>: MatMulShape<Sh<10, 10>>` does not hold and calling
/// [`GraphBuilder::matmul`] with those shapes fails to compile.
///
/// # Example
/// ```
/// use tpt_infer_graph::{MatMulShape, Sh, ShapeMarker};
/// fn assert_output<L, R, O>()
/// where
///     L: MatMulShape<R, Output = O>,
///     R: ShapeMarker,
/// {
/// }
/// assert_output::<Sh<1, 784>, Sh<784, 10>, Sh<1, 10>>();
/// ```
pub trait MatMulShape<Rhs: ShapeMarker>: ShapeMarker {
    /// Resulting shape marker.
    type Output: ShapeMarker;
}

impl<const M: usize, const K: usize, const N: usize> MatMulShape<Sh<K, N>> for Sh<M, K> {
    type Output = Sh<M, N>;
}

/// Type-level element-wise addition shape computation.
///
/// The impl only exists for identical shapes: broadcasting is intentionally
/// rejected at compile time.
///
/// # Example
/// ```
/// use tpt_infer_graph::{AddShape, Sh, ShapeMarker};
/// fn assert_output<L, R, O>()
/// where
///     L: AddShape<R, Output = O>,
///     R: ShapeMarker,
/// {
/// }
/// assert_output::<Sh<1, 128>, Sh<1, 128>, Sh<1, 128>>();
/// ```
pub trait AddShape<Rhs: ShapeMarker>: ShapeMarker {
    /// Resulting shape marker (identical to both inputs).
    type Output: ShapeMarker;
}

impl<const A: usize, const B: usize> AddShape<Sh<A, B>> for Sh<A, B> {
    type Output = Sh<A, B>;
}

/// Type-level flatten shape computation for a fixed split axis.
///
/// Only axis `1` (a no-op on rank-2 tensors) is expressible on stable Rust;
/// other axes require dividing out generic const parameters, which needs the
/// unstable `generic_const_exprs` feature.
pub trait FlattenShape<const AXIS: usize>: ShapeMarker {
    /// Resulting shape marker.
    type Output: ShapeMarker;
}

impl<const M: usize, const K: usize> FlattenShape<1> for Sh<M, K> {
    type Output = Sh<M, K>;
}

/// Type-level transpose shape computation for a rank-2 permutation:
/// `[P0, P1]` selects the source dimension for each output axis.
///
/// Only the identity `[0, 1]` and swap `[1, 0]` permutations exist for rank-2
/// shapes.
///
/// # Example
/// ```
/// use tpt_infer_graph::{Sh, TransposeShape};
/// fn assert_output<S, const P0: usize, const P1: usize, O>()
/// where
///     S: TransposeShape<P0, P1, Output = O>,
/// {
/// }
/// assert_output::<Sh<2, 3>, 1, 0, Sh<3, 2>>();
/// ```
pub trait TransposeShape<const P0: usize, const P1: usize>: ShapeMarker {
    /// Resulting shape marker.
    type Output: ShapeMarker;
}

impl<const A: usize, const B: usize> TransposeShape<0, 1> for Sh<A, B> {
    type Output = Sh<A, B>;
}

impl<const A: usize, const B: usize> TransposeShape<1, 0> for Sh<A, B> {
    type Output = Sh<B, A>;
}

/// A typed handle to a node producing a tensor of shape `S`.
///
/// Handles are only minted by [`GraphBuilder`], so the shape in the type
/// parameter always matches the node's actual output shape.
///
/// # Example
/// ```
/// use tpt_infer_graph::{GraphBuilder, Sh};
/// let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
/// let w = b.add_input::<Sh<4, 8>>("w");
/// assert_eq!(w.id(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRef<S: ShapeMarker> {
    id: usize,
    _shape: PhantomData<S>,
}

impl<S: ShapeMarker> NodeRef<S> {
    /// Id of the referenced node in its [`ComputationGraph`].
    pub const fn id(self) -> usize {
        self.id
    }

    /// Shape of the referenced node's output tensor.
    pub const SHAPE: Shape = S::SHAPE;

    /// Rank of the referenced node's output tensor.
    pub const RANK: usize = S::RANK;
}

/// A compile-time edge: a connection whose tensor shape `S` is checked by the
/// compiler rather than at runtime.
///
/// # Example
/// ```
/// use tpt_infer_graph::{Sh, TypedEdge};
/// let e: TypedEdge<Sh<1, 784>> = TypedEdge::new(0, 2);
/// assert_eq!(e.from_id(), 0);
/// assert_eq!(e.to_id(), 2);
/// assert_eq!(&TypedEdge::<Sh<1, 784>>::SHAPE[..2], &[1, 784]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedEdge<S: ShapeMarker> {
    from: usize,
    to: usize,
    _shape: PhantomData<S>,
}

impl<S: ShapeMarker> TypedEdge<S> {
    /// Create a typed edge from node `from` to node `to`.
    pub const fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            _shape: PhantomData,
        }
    }

    /// Source node id.
    pub const fn from_id(self) -> usize {
        self.from
    }

    /// Destination node id.
    pub const fn to_id(self) -> usize {
        self.to
    }

    /// Tensor shape flowing along this edge (a compile-time constant).
    pub const SHAPE: Shape = S::SHAPE;

    /// Rank of the tensor flowing along this edge.
    pub const RANK: usize = S::RANK;
}

/// A linearly-threaded, type-state computation graph builder.
///
/// The type parameter `S` is the shape of the tensor currently at the head of
/// the chain. Shape-changing methods consume the builder and return a builder
/// for the computed output shape, so an incompatible operation (for example a
/// matmul whose contracting dimensions disagree) fails to compile because the
/// required [`MatMulShape`]/[`AddShape`] impl does not exist.
///
/// # Example
/// ```
/// use tpt_infer_graph::{GraphBuilder, Sh};
///
/// let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
/// let w1 = b.add_input::<Sh<784, 128>>("w1");
/// let mut b = b.matmul(w1).relu();
/// let w2 = b.add_input::<Sh<128, 10>>("w2");
/// let b = b.matmul(w2).softmax(-1);
///
/// let g = b.into_graph();
/// assert_eq!(g.nodes().len(), 7);
/// assert_eq!(g.node(6).unwrap().dims(), &[1, 10]);
/// ```
pub struct GraphBuilder<S: ShapeMarker> {
    graph: ComputationGraph,
    current: usize,
    _shape: PhantomData<S>,
}

impl<S: ShapeMarker> core::fmt::Debug for GraphBuilder<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphBuilder")
            .field("shape", &&S::SHAPE[..S::RANK])
            .field("current", &self.current)
            .field("nodes", &self.graph.nodes().len())
            .finish()
    }
}

impl<S: ShapeMarker> GraphBuilder<S> {
    /// Start a graph with a single [`Operator::Input`] node named `name`
    /// producing shape `S`.
    pub fn input(name: &str) -> Self {
        let mut graph = ComputationGraph::new();
        let node = Node {
            id: 0,
            operator: Operator::Input,
            inputs: Vec::new(),
            output_shape: S::SHAPE,
            output_rank: S::RANK,
            name: Some(name.into()),
        };
        let current = graph.push_node(node);
        Self {
            graph,
            current,
            _shape: PhantomData,
        }
    }

    /// Add an auxiliary [`Operator::Input`] node (e.g. a weight tensor)
    /// producing shape `Rhs`, without advancing the chain.
    pub fn add_input<Rhs: ShapeMarker>(&mut self, name: &str) -> NodeRef<Rhs> {
        let id = self.push_input(Rhs::SHAPE, Rhs::RANK, name);
        NodeRef {
            id,
            _shape: PhantomData,
        }
    }

    /// Add a weight input backed by an [`Initializer`] holding `data`.
    ///
    /// # Errors
    /// - [`GraphError::SizeMismatch`] if `data.len()` differs from `Rhs::NUMEL`
    /// - [`GraphError::RankTooLarge`] if `Rhs` exceeds [`MAX_RANK`] (cannot
    ///   happen for the built-in markers)
    pub fn add_initializer<Rhs: ShapeMarker>(
        &mut self,
        name: &str,
        data: Vec<f32>,
    ) -> Result<NodeRef<Rhs>, GraphError> {
        let dims = Rhs::SHAPE;
        // Validate data before mutating the graph so failures leave it untouched.
        let init = Initializer::new(name, &dims[..Rhs::RANK], data)?;
        let id = self.push_input(Rhs::SHAPE, Rhs::RANK, name);
        self.graph.add_initializer(init);
        Ok(NodeRef {
            id,
            _shape: PhantomData,
        })
    }

    /// Shape of the tensor currently at the head of the chain.
    pub const fn current_shape() -> Shape {
        S::SHAPE
    }

    /// Handle to the node producing the current tensor.
    pub fn current(&self) -> NodeRef<S> {
        NodeRef {
            id: self.current,
            _shape: PhantomData,
        }
    }

    /// Borrow the partially built graph.
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }

    /// Finish building: infer graph outputs and return the
    /// [`ComputationGraph`].
    pub fn into_graph(mut self) -> ComputationGraph {
        self.graph.infer_outputs();
        self.graph
    }

    /// Matrix multiply: `[S] x [Rhs] -> [MatMulShape::Output]`.
    ///
    /// Connects both the current tensor (input port 0) and `rhs` (input
    /// port 1) to the new node.
    ///
    /// Fails to compile when the contracting dimensions of `S` and `Rhs`
    /// differ.
    pub fn matmul<Rhs: ShapeMarker>(
        mut self,
        rhs: NodeRef<Rhs>,
    ) -> GraphBuilder<<S as MatMulShape<Rhs>>::Output>
    where
        S: MatMulShape<Rhs>,
    {
        let output_shape = <<S as MatMulShape<Rhs>>::Output as ShapeMarker>::SHAPE;
        let output_rank = <<S as MatMulShape<Rhs>>::Output as ShapeMarker>::RANK;
        let from = self.current;
        let id = self.push_op(
            Operator::MatMul,
            vec![from, rhs.id],
            output_shape,
            output_rank,
        );
        self.graph.push_edge(Edge {
            from,
            from_port: 0,
            to: id,
            to_port: 0,
            shape: S::SHAPE,
            rank: S::RANK,
        });
        self.graph.push_edge(Edge {
            from: rhs.id,
            from_port: 0,
            to: id,
            to_port: 1,
            shape: Rhs::SHAPE,
            rank: Rhs::RANK,
        });
        GraphBuilder {
            graph: self.graph,
            current: id,
            _shape: PhantomData,
        }
    }

    /// Element-wise add: `[S] + [Rhs] -> [AddShape::Output]`.
    ///
    /// Fails to compile unless `S` and `Rhs` have identical shapes.
    //
    // Kept as an inherent method (alongside `matmul`) rather than
    // `std::ops::Add` so both binary graph ops share one calling convention.
    #[allow(clippy::should_implement_trait)]
    pub fn add<Rhs: ShapeMarker>(
        mut self,
        rhs: NodeRef<Rhs>,
    ) -> GraphBuilder<<S as AddShape<Rhs>>::Output>
    where
        S: AddShape<Rhs>,
    {
        let output_shape = <<S as AddShape<Rhs>>::Output as ShapeMarker>::SHAPE;
        let output_rank = <<S as AddShape<Rhs>>::Output as ShapeMarker>::RANK;
        let from = self.current;
        let id = self.push_op(Operator::Add, vec![from, rhs.id], output_shape, output_rank);
        self.graph.push_edge(Edge {
            from,
            from_port: 0,
            to: id,
            to_port: 0,
            shape: S::SHAPE,
            rank: S::RANK,
        });
        self.graph.push_edge(Edge {
            from: rhs.id,
            from_port: 0,
            to: id,
            to_port: 1,
            shape: Rhs::SHAPE,
            rank: Rhs::RANK,
        });
        GraphBuilder {
            graph: self.graph,
            current: id,
            _shape: PhantomData,
        }
    }

    /// Element-wise ReLU; the shape is unchanged.
    pub fn relu(self) -> GraphBuilder<S> {
        self.unary(Operator::Relu)
    }

    /// Element-wise logistic sigmoid; the shape is unchanged.
    pub fn sigmoid(self) -> GraphBuilder<S> {
        self.unary(Operator::Sigmoid)
    }

    /// Element-wise GELU; the shape is unchanged.
    pub fn gelu(self) -> GraphBuilder<S> {
        self.unary(Operator::Gelu)
    }

    /// Softmax along `axis`; the shape is unchanged.
    pub fn softmax(self, axis: i32) -> GraphBuilder<S> {
        self.unary(Operator::softmax(axis))
    }

    /// Flatten after axis `AXIS`; only axis `1` (shape-preserving on rank-2
    /// tensors) is supported at compile time.
    ///
    /// Fails to compile for other axes.
    pub fn flatten<const AXIS: usize>(mut self) -> GraphBuilder<<S as FlattenShape<AXIS>>::Output>
    where
        S: FlattenShape<AXIS>,
    {
        let output_shape = <<S as FlattenShape<AXIS>>::Output as ShapeMarker>::SHAPE;
        let output_rank = <<S as FlattenShape<AXIS>>::Output as ShapeMarker>::RANK;
        let from = self.current;
        let id = self.push_op(
            Operator::Flatten { axis: AXIS as i32 },
            vec![from],
            output_shape,
            output_rank,
        );
        self.graph.push_edge(Edge {
            from,
            from_port: 0,
            to: id,
            to_port: 0,
            shape: S::SHAPE,
            rank: S::RANK,
        });
        GraphBuilder {
            graph: self.graph,
            current: id,
            _shape: PhantomData,
        }
    }

    /// Transpose with rank-2 permutation `[P0, P1]`.
    ///
    /// Fails to compile for permutations without an impl (see
    /// [`TransposeShape`]).
    pub fn transpose<const P0: usize, const P1: usize>(
        mut self,
    ) -> GraphBuilder<<S as TransposeShape<P0, P1>>::Output>
    where
        S: TransposeShape<P0, P1>,
    {
        let output_shape = <<S as TransposeShape<P0, P1>>::Output as ShapeMarker>::SHAPE;
        let output_rank = <<S as TransposeShape<P0, P1>>::Output as ShapeMarker>::RANK;
        let from = self.current;
        let mut perm = [0; MAX_RANK];
        perm[0] = P0;
        perm[1] = P1;
        let id = self.push_op(
            Operator::Transpose { perm, rank: 2 },
            vec![from],
            output_shape,
            output_rank,
        );
        self.graph.push_edge(Edge {
            from,
            from_port: 0,
            to: id,
            to_port: 0,
            shape: S::SHAPE,
            rank: S::RANK,
        });
        GraphBuilder {
            graph: self.graph,
            current: id,
            _shape: PhantomData,
        }
    }

    /// Append an un-named `Input` node of the given shape and return its id.
    fn push_input(&mut self, shape: Shape, rank: usize, name: &str) -> usize {
        let node = Node {
            id: 0,
            operator: Operator::Input,
            inputs: Vec::new(),
            output_shape: shape,
            output_rank: rank,
            name: Some(name.into()),
        };
        self.graph.push_node(node)
    }

    /// Append a shape-preserving unary node and advance `self.current`.
    fn unary(mut self, operator: Operator) -> GraphBuilder<S> {
        let from = self.current;
        let id = self.push_op(operator, vec![from], S::SHAPE, S::RANK);
        self.graph.push_edge(Edge {
            from,
            from_port: 0,
            to: id,
            to_port: 0,
            shape: S::SHAPE,
            rank: S::RANK,
        });
        self.current = id;
        self
    }

    /// Append a node with no name and return its id.
    fn push_op(
        &mut self,
        operator: Operator,
        inputs: Vec<usize>,
        output_shape: Shape,
        output_rank: usize,
    ) -> usize {
        self.graph.push_node(Node {
            id: 0,
            operator,
            inputs,
            output_shape,
            output_rank,
            name: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_type_state_mlp_builds() {
        let mut b = GraphBuilder::<Sh<1, 784>>::input("x");
        assert_eq!(b.current().id(), 0);
        let w1 = b.add_input::<Sh<784, 128>>("w1");
        assert_eq!(w1.id(), 1);
        let mut b = b.matmul(w1).relu();
        assert_eq!(b.current().id(), 3);
        let w2 = b.add_input::<Sh<128, 10>>("w2");
        assert_eq!(w2.id(), 4);
        let b = b.matmul(w2).softmax(-1);
        assert_eq!(b.current().id(), 6);

        let g = b.into_graph();
        assert_eq!(g.nodes().len(), 7);
        assert_eq!(g.edges().len(), 6);
        assert_eq!(g.inputs(), &[0, 1, 4]);
        assert_eq!(g.outputs(), &[6]);
        assert_eq!(g.node(2).unwrap().dims(), &[1, 128]);
        assert_eq!(g.node(5).unwrap().dims(), &[1, 10]);
        assert_eq!(g.node(6).unwrap().dims(), &[1, 10]);
        assert_eq!(g.topological_sort().unwrap().len(), 7);
    }

    #[test]
    fn matmul_shape_computed_at_type_level() {
        fn check<L, R, O>()
        where
            L: MatMulShape<R, Output = O>,
            R: ShapeMarker,
        {
        }
        check::<Sh<1, 784>, Sh<784, 10>, Sh<1, 10>>();
        check::<Sh<4, 16>, Sh<16, 32>, Sh<4, 32>>();
    }

    #[test]
    fn add_shape_requires_identical_shapes() {
        fn check<L, R, O>()
        where
            L: AddShape<R, Output = O>,
            R: ShapeMarker,
        {
        }
        check::<Sh<1, 128>, Sh<1, 128>, Sh<1, 128>>();
    }

    #[test]
    fn transpose_swaps_dimensions() {
        fn check<S, const P0: usize, const P1: usize, O>()
        where
            S: TransposeShape<P0, P1, Output = O>,
        {
        }
        check::<Sh<2, 3>, 1, 0, Sh<3, 2>>();
        check::<Sh<2, 3>, 0, 1, Sh<2, 3>>();
    }

    #[test]
    fn same_shape_add_builds() {
        let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
        let y = b.add_input::<Sh<1, 4>>("y");
        let b = b.add(y);
        let g = b.into_graph();
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.node(2).unwrap().dims(), &[1, 4]);
    }

    #[test]
    fn initializer_backed_weight() {
        let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
        let err = b
            .add_initializer::<Sh<4, 8>>("w", alloc::vec![0.0; 7])
            .unwrap_err();
        assert_eq!(
            err,
            GraphError::SizeMismatch {
                expected: 32,
                actual: 7
            }
        );
        // Failed initializer must not have added an input node.
        assert_eq!(b.graph().nodes().len(), 1);

        let w = b
            .add_initializer::<Sh<4, 8>>("w", alloc::vec![0.5; 32])
            .unwrap();
        assert_eq!(w.id(), 1);
        let g = b.into_graph();
        assert_eq!(g.initializers().len(), 1);
        assert_eq!(g.initializers()[0].name, "w");
        assert_eq!(g.initializers()[0].dims(), &[4, 8]);
        assert_eq!(g.inputs(), &[0, 1]);
    }

    #[test]
    fn flatten_axis_one_preserves_rank2_shape() {
        let b = GraphBuilder::<Sh<1, 784>>::input("x");
        let g = b.flatten::<1>().into_graph();
        match &g.node(1).unwrap().operator {
            Operator::Flatten { axis } => assert_eq!(*axis, 1),
            other => panic!("expected Flatten, got {other:?}"),
        }
        assert_eq!(g.node(1).unwrap().dims(), &[1, 784]);
    }

    #[test]
    fn transpose_builder_swaps_shape() {
        let b = GraphBuilder::<Sh<2, 3>>::input("x");
        let g = b.transpose::<1, 0>().into_graph();
        assert_eq!(g.node(1).unwrap().dims(), &[3, 2]);
        match &g.node(1).unwrap().operator {
            Operator::Transpose { perm, rank } => {
                assert_eq!(*rank, 2);
                assert_eq!(&perm[..2], &[1, 0]);
            }
            other => panic!("expected Transpose, got {other:?}"),
        }
    }

    #[test]
    fn activation_chain_preserves_shape() {
        let b = GraphBuilder::<Sh<1, 16>>::input("x");
        let g = b.relu().sigmoid().gelu().softmax(-1).into_graph();
        assert_eq!(g.nodes().len(), 5);
        assert_eq!(g.node(4).unwrap().dims(), &[1, 16]);
        assert_eq!(g.outputs(), &[4]);
    }

    #[test]
    fn typed_edge_carries_shape_constants() {
        let e = TypedEdge::<Sh<1, 784>>::new(0, 2);
        assert_eq!(e.from_id(), 0);
        assert_eq!(e.to_id(), 2);
        assert_eq!(TypedEdge::<Sh<1, 784>>::RANK, 2);
        assert_eq!(&TypedEdge::<Sh<1, 784>>::SHAPE[..2], &[1, 784]);
        assert_eq!(&NodeRef::<Sh<1, 784>>::SHAPE[..2], &[1, 784]);
    }

    #[test]
    fn builder_debug_and_current_shape() {
        let b = GraphBuilder::<Sh<1, 4>>::input("x");
        assert_eq!(&GraphBuilder::<Sh<1, 4>>::current_shape()[..2], &[1, 4]);
        let dbg = alloc::format!("{b:?}");
        assert!(dbg.contains("GraphBuilder"));
        assert!(dbg.contains("[1, 4]"));
    }
}
