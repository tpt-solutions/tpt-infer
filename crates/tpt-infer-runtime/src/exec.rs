//! Graph execution: arena planning, input binding, and the operator
//! dispatch table.
//!
//! # Pipeline
//!
//! 1. **Topological order** — Kahn's algorithm over the graph's input lists,
//!    running entirely in arena scratch space (`indegree`, CSR adjacency,
//!    ring-buffer queue, output order). No heap allocation; a cycle yields
//!    [`RuntimeError::CycleDetected`].
//! 2. **Input binding** — every [`Operator::Input`] node resolves to either
//!    a named [`Initializer`] (weight) or the next supplied input slice.
//! 3. **Activation layout** — each non-input node receives a disjoint
//!    `[offset, offset + numel)` range of one contiguous `f32` activation
//!    buffer, assigned in topological order so a node's output always sits
//!    above its inputs' ranges. The whole buffer comes from a single
//!    [`BumpArena::alloc_slice`] call.
//! 4. **Dispatch** — nodes execute in topological order. Inputs read either
//!    external memory (user slices / initializers — never copied) or earlier
//!    activation ranges; the output is the node's own range. Kernel calls go
//!    through the [`Backend`] HAL where it has coverage
//!    (`matmul`, `conv2d`, `elementwise_add`, `relu`, `softmax`, `sigmoid`,
//!    `gelu`) and through [`crate::kernels`] otherwise.
//! 5. **Materialization** — marked graph outputs are copied out of the arena
//!    into owned [`TensorVec`] values (the only heap traffic of the API,
//!    and it happens after the hot path).
//!
//! # Safety of disjoint raw slices
//!
//! The executor derives raw pointers from the single activation allocation
//! and rebuilds `&[f32]` / `&mut [f32]` slices on demand. This is sound
//! because:
//!
//! - offsets are assigned from one monotone cursor in a single pass, so no
//!   two node ranges overlap;
//! - a node never consumes its own output (the graph is acyclic), so every
//!   slice alive during one kernel call refers to a distinct range or to
//!   caller-owned external memory;
//! - external inputs and initializers live in buffers that do not alias the
//!   arena — callers must not pass slices that overlap the arena's backing
//!   buffer.
//!
//! Arena memory is bump-allocated and never freed inside one call; the
//! caller reuses or resets the arena between inferences
//! ([`BumpArena::reset`]).
//!
//! [`Initializer`]: tpt_infer_graph::Initializer

use tpt_infer_core::{BumpArena, TensorVec};
use tpt_infer_graph::{ComputationGraph, Node, Operator};
use tpt_infer_ops::{Backend, Conv2dOptions, OpError};

use crate::error::RuntimeError;
use crate::kernels::{self, BinaryOp};

/// Where a node's output lives after binding/layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bind {
    /// Activation buffer, starting at this `f32` offset.
    Act(usize),
    /// Caller-supplied input slice at this index of `inputs`.
    Runtime(usize),
    /// Named graph initializer at this index of `initializers`.
    Weight(usize),
}

/// Bytes of alignment slack added by [`required_arena_bytes`] on top of the
/// exact payload (each arena allocation may pad up to `align - 1` bytes).
const ARENA_SLACK: usize = 128;

/// Element count of a node's declared output shape.
fn node_numel(node: &Node) -> Result<usize, RuntimeError> {
    crate::kernels::numel(node.dims())
}

/// Total activation `f32` count: the sum of every non-input node's output.
fn activation_numel(graph: &ComputationGraph) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|n| !n.operator.is_input())
        .fold(0usize, |acc, n| {
            acc.saturating_add(crate::kernels::numel(n.dims()).unwrap_or(usize::MAX))
        })
}

/// Number of node-to-node input references (edges implied by input lists).
fn edge_count(graph: &ComputationGraph) -> usize {
    graph.nodes().iter().map(|n| n.inputs.len()).sum()
}

/// Minimum arena capacity, in bytes, for one [`execute_graph`] call on
/// `graph` (topological-sort scratch + activation buffer + alignment
/// slack). Caller-supplied input slices and initializers are stored outside
/// the arena and are not counted.
///
/// Keep this in sync with the allocation sequence inside
/// [`execute_graph`]: six `usize` scratch slices (`n`, `n + 1`, `m`, `n`,
/// `n`, `n` entries), one `Bind` slice (`n` entries), and one `f32`
/// activation slice (``[`activation_numel`]`` entries).
///
/// # Example
/// ```
/// # #[cfg(feature = "std")]
/// # {
/// use tpt_infer_core::BumpArena;
/// use tpt_infer_graph::{ComputationGraph, Node, Operator};
/// use tpt_infer_runtime::required_arena_bytes;
///
/// let mut g = ComputationGraph::new();
/// let x = g.add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap()).unwrap();
/// g.add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap()).unwrap();
/// let mut mem = vec![0u8; required_arena_bytes(&g)];
/// let mut arena = BumpArena::new(&mut mem);
/// assert_eq!(arena.capacity(), mem.len());
/// # }
/// ```
pub fn required_arena_bytes(graph: &ComputationGraph) -> usize {
    let n = graph.nodes().len();
    let m = edge_count(graph);
    let scratch_usizes = n // indegree
        .saturating_add(n + 1) // adjacency offsets (CSR)
        .saturating_add(m) // adjacency list
        .saturating_add(n) // fill cursor
        .saturating_add(n) // queue ring buffer
        .saturating_add(n); // topological order
    scratch_usizes
        .saturating_mul(core::mem::size_of::<usize>())
        .saturating_add(n.saturating_mul(core::mem::size_of::<Bind>()))
        .saturating_add(activation_numel(graph).saturating_mul(core::mem::size_of::<f32>()))
        .saturating_add(ARENA_SLACK)
}

/// Kahn's algorithm, scratch entirely in arena-provided slices.
///
/// Fills `order[..k]` with a valid topological order (ties broken by node
/// id, so the result is deterministic). All scratch slices must be
/// fully written before being read — arena memory is uninitialized.
///
/// # Errors
/// - [`RuntimeError::Graph`]`(`[`GraphError::UnknownNode`]`)` if an input
///   list references a missing node
/// - [`RuntimeError::CycleDetected`] if the graph contains a cycle
fn kahn(
    graph: &ComputationGraph,
    indegree: &mut [usize],
    adj_off: &mut [usize],
    adj: &mut [usize],
    cur: &mut [usize],
    queue: &mut [usize],
    order: &mut [usize],
) -> Result<usize, RuntimeError> {
    let n = graph.nodes().len();
    debug_assert_eq!(indegree.len(), n);
    debug_assert_eq!(adj_off.len(), n + 1);
    debug_assert_eq!(cur.len(), n);
    debug_assert_eq!(queue.len(), n);
    debug_assert_eq!(order.len(), n);

    for slot in indegree.iter_mut() {
        *slot = 0;
    }
    for slot in adj_off.iter_mut() {
        *slot = 0;
    }
    for node in graph.nodes() {
        for &inp in &node.inputs {
            if inp >= n {
                return Err(RuntimeError::Graph(
                    tpt_infer_graph::GraphError::UnknownNode { id: inp },
                ));
            }
            indegree[node.id] += 1;
            adj_off[inp] += 1;
        }
    }

    // Prefix-sum the per-producer counts into CSR offsets.
    let mut acc = 0usize;
    for i in 0..=n {
        let count = adj_off[i];
        adj_off[i] = acc;
        acc = acc.saturating_add(count);
    }
    debug_assert_eq!(acc, adj.len());

    // Fill cursor starts at each producer's CSR offset.
    cur.copy_from_slice(&adj_off[..n]);
    for node in graph.nodes() {
        for &inp in &node.inputs {
            let slot = cur[inp];
            adj[slot] = node.id;
            cur[inp] = slot + 1;
        }
    }

    let mut head = 0usize;
    let mut tail = 0usize;
    for id in 0..n {
        if indegree[id] == 0 {
            queue[tail] = id;
            tail += 1;
        }
    }
    let mut k = 0usize;
    while head < tail {
        let id = queue[head];
        head += 1;
        order[k] = id;
        k += 1;
        let (s, e) = (adj_off[id], adj_off[id + 1]);
        for &consumer in &adj[s..e] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                queue[tail] = consumer;
                tail += 1;
            }
        }
    }
    if k != n {
        return Err(RuntimeError::CycleDetected);
    }
    Ok(k)
}

/// Binds every [`Operator::Input`] node to a weight initializer or a
/// caller-supplied input slice, in `graph.inputs()` order.
fn bind_inputs(
    graph: &ComputationGraph,
    binds: &mut [Bind],
    inputs: &[&[f32]],
) -> Result<(), RuntimeError> {
    let mut used = 0usize;
    for &id in graph.inputs() {
        let node = graph
            .node(id)
            .ok_or(RuntimeError::Graph(tpt_infer_graph::GraphError::UnknownNode { id }))?;
        let declared = node_numel(node)?;
        let mut bound = false;
        if let Some(name) = &node.name {
            if let Some(pos) = graph
                .initializers()
                .iter()
                .position(|it| it.name.as_str() == name.as_str())
            {
                let init = &graph.initializers()[pos];
                if init.data.len() != declared {
                    return Err(RuntimeError::InitializerMismatch { node: id });
                }
                binds[id] = Bind::Weight(pos);
                bound = true;
            }
        }
        if !bound {
            let data = inputs
                .get(used)
                .copied()
                .ok_or(RuntimeError::MissingInput { node: id })?;
            if data.len() != declared {
                return Err(RuntimeError::InputDataMismatch { node: id });
            }
            binds[id] = Bind::Runtime(used);
            used += 1;
        }
    }
    if used != inputs.len() {
        return Err(RuntimeError::InputCountMismatch {
            expected: used,
            actual: inputs.len(),
        });
    }
    Ok(())
}

/// Assigns disjoint activation offsets to non-input nodes in topological
/// order; returns the total `f32` count.
fn assign_offsets(
    graph: &ComputationGraph,
    order: &[usize],
    binds: &mut [Bind],
) -> Result<usize, RuntimeError> {
    let mut total = 0usize;
    for &id in order {
        let node = graph
            .node(id)
            .ok_or(RuntimeError::Graph(tpt_infer_graph::GraphError::UnknownNode { id }))?;
        if node.operator.is_input() {
            continue;
        }
        let n = node_numel(node)?;
        binds[id] = Bind::Act(total);
        total = total.checked_add(n).ok_or(RuntimeError::OutOfMemory)?;
    }
    Ok(total)
}

/// Execution context threaded through the dispatch table.
struct Exec<'g, B> {
    graph: &'g ComputationGraph,
    binds: &'g [Bind],
    /// Base pointer of the activation buffer (`numel * 4` bytes, ranges
    /// disjoint per [`Bind::Act`]).
    acts: *mut f32,
    inputs: &'g [&'g [f32]],
    backend: &'g B,
}

impl<'g, B: Backend> Exec<'g, B> {
    /// Resolves an input port's producer to `(data, dims)`.
    ///
    /// # Safety contract
    /// Returned activation slices are fabricated at lifetime `'g` from the
    /// arena buffer, which the caller guarantees outlives the execution.
    /// Ranges are disjoint by layout construction (see module docs).
    fn resolve(&self, producer: usize) -> Result<(&'g [f32], &'g [usize]), RuntimeError> {
        let graph: &'g ComputationGraph = self.graph;
        let pnode: &'g Node = graph.node(producer).ok_or(RuntimeError::Graph(
            tpt_infer_graph::GraphError::UnknownNode { id: producer },
        ))?;
        match self.binds[producer] {
            Bind::Act(off) => {
                let len = node_numel(pnode)?;
                // SAFETY: `off..off + len` lies inside the activation buffer
                // (assigned by `assign_offsets`), does not overlap any other
                // node's range, and the buffer outlives `'g`.
                let data = unsafe { core::slice::from_raw_parts(self.acts.add(off), len) };
                Ok((data, pnode.dims()))
            }
            Bind::Weight(wi) => {
                let init = &graph.initializers()[wi];
                Ok((&init.data[..], pnode.dims()))
            }
            Bind::Runtime(ki) => Ok((self.inputs[ki], pnode.dims())),
        }
    }

    /// Fetches input `port` of `node` or reports an arity error.
    fn port(&self, node: &Node, port: usize) -> Result<(&'g [f32], &'g [usize]), RuntimeError> {
        let producer = *node
            .inputs
            .get(port)
            .ok_or(RuntimeError::Op(OpError::Invalid))?;
        self.resolve(producer)
    }

    /// The operator dispatch table: routes one node's computation to the
    /// `Backend` HAL or a `no_std` kernel, writing `out` in full.
    fn run(&self, node: &Node, out: &mut [f32]) -> Result<(), RuntimeError> {
        let op = &node.operator;
        let out_dims = node.dims();
        match op {
            // Input nodes are consumed by binding, never dispatched.
            Operator::Input => Err(RuntimeError::UnsupportedOp { name: "Input" }),

            Operator::MatMul => {
                let (a, ad) = self.port(node, 0)?;
                let (b, bd) = self.port(node, 1)?;
                if node.inputs.len() != 2 {
                    return Err(RuntimeError::Op(OpError::Invalid));
                }
                if ad.len() != 2 || bd.len() != 2 {
                    return Err(RuntimeError::UnsupportedOp { name: "MatMul" });
                }
                self.backend
                    .matmul(a, [ad[0], ad[1]], b, [bd[0], bd[1]], out)?;
                Ok(())
            }

            Operator::Conv2d { strides, padding } => {
                if node.inputs.len() != 2 && node.inputs.len() != 3 {
                    return Err(RuntimeError::Op(OpError::Invalid));
                }
                let (x, xd) = self.port(node, 0)?;
                let (w, wd) = self.port(node, 1)?;
                if xd.len() != 4 || wd.len() != 4 {
                    return Err(RuntimeError::ShapeMismatch);
                }
                let options = Conv2dOptions::with_stride_padding(
                    strides[0],
                    strides[1],
                    padding[0],
                    padding[1],
                );
                self.backend.conv2d(
                    x,
                    [xd[0], xd[1], xd[2], xd[3]],
                    w,
                    [wd[0], wd[1], wd[2], wd[3]],
                    out,
                    options,
                )?;
                if let Some(&bias_port) = node.inputs.get(2) {
                    let (b, bd) = self.resolve(bias_port)?;
                    kernels::add_inplace_broadcast(out, out_dims, b, bd)?;
                }
                Ok(())
            }

            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                if node.inputs.len() != 2 {
                    return Err(RuntimeError::Op(OpError::Invalid));
                }
                let (a, ad) = self.port(node, 0)?;
                let (b, bd) = self.port(node, 1)?;
                let equal = ad == bd && ad == out_dims;
                if matches!(op, Operator::Add) && equal {
                    self.backend.elementwise_add(a, b, out)?;
                } else {
                    let bop = match op {
                        Operator::Add => BinaryOp::Add,
                        Operator::Sub => BinaryOp::Sub,
                        Operator::Mul => BinaryOp::Mul,
                        _ => BinaryOp::Div,
                    };
                    kernels::binary(bop, a, ad, b, bd, out, out_dims)?;
                }
                Ok(())
            }

            Operator::Relu => {
                let (a, _) = self.port(node, 0)?;
                self.backend.relu(a, out)?;
                Ok(())
            }
            Operator::Sigmoid => {
                let (a, _) = self.port(node, 0)?;
                self.backend.sigmoid(a, out)?;
                Ok(())
            }
            Operator::Gelu => {
                let (a, _) = self.port(node, 0)?;
                self.backend.gelu(a, out)?;
                Ok(())
            }

            Operator::Softmax { axis } => {
                let (a, ad) = self.port(node, 0)?;
                if ad.is_empty() {
                    return Err(RuntimeError::ShapeMismatch);
                }
                let rank = ad.len() as i32;
                let normalized = if *axis < 0 { *axis + rank } else { *axis };
                if normalized != rank - 1 {
                    // The HAL softmax only normalizes rows of the last dim.
                    return Err(RuntimeError::UnsupportedOp { name: "Softmax" });
                }
                self.backend.softmax(a, out, ad[ad.len() - 1])?;
                Ok(())
            }

            Operator::Reshape { .. } | Operator::Flatten { .. } => {
                let (a, _) = self.port(node, 0)?;
                kernels::copy_elements(a, out)
            }

            Operator::MaxPool2d {
                kernel,
                strides,
                padding,
            } => {
                let (a, ad) = self.port(node, 0)?;
                kernels::max_pool2d(a, ad, out, out_dims, *kernel, *strides, *padding)
            }
            Operator::AveragePool2d {
                kernel,
                strides,
                padding,
            } => {
                let (a, ad) = self.port(node, 0)?;
                kernels::average_pool2d(a, ad, out, out_dims, *kernel, *strides, *padding)
            }

            Operator::BatchNorm { epsilon } => {
                if node.inputs.len() != 5 {
                    return Err(RuntimeError::Op(OpError::Invalid));
                }
                let (x, xd) = self.port(node, 0)?;
                let (scale, _) = self.port(node, 1)?;
                let (bias, _) = self.port(node, 2)?;
                let (mean, _) = self.port(node, 3)?;
                let (var, _) = self.port(node, 4)?;
                kernels::batch_norm(x, xd, out, [scale, bias, mean, var], *epsilon)
            }

            Operator::Concat { axis } => {
                if node.inputs.len() < 2 {
                    return Err(RuntimeError::Op(OpError::Invalid));
                }
                let rank = out_dims.len();
                if rank == 0 {
                    return Err(RuntimeError::ShapeMismatch);
                }
                let ax = if *axis < 0 {
                    *axis + rank as i32
                } else {
                    *axis
                };
                if ax < 0 || ax as usize >= rank {
                    return Err(RuntimeError::ShapeMismatch);
                }
                let axis_us = ax as usize;
                let c_total = out_dims[axis_us];
                let inner = kernels::numel(&out_dims[axis_us + 1..])?;
                let mut c_offset = 0usize;
                for i in 0..node.inputs.len() {
                    let (data, dims) = self.port(node, i)?;
                    if dims.len() != rank {
                        return Err(RuntimeError::ShapeMismatch);
                    }
                    if dims.iter().enumerate().any(|(d, &v)| d != axis_us && v != out_dims[d]) {
                        return Err(RuntimeError::ShapeMismatch);
                    }
                    let c_src = dims[axis_us];
                    kernels::concat_copy_one(out, c_total, c_offset, c_src, inner, data)?;
                    c_offset += c_src;
                }
                if c_offset != c_total {
                    return Err(RuntimeError::ShapeMismatch);
                }
                Ok(())
            }

            Operator::Transpose { perm, rank } => {
                let (a, ad) = self.port(node, 0)?;
                kernels::transpose(a, ad, out, out_dims, &perm[..*rank])
            }

            Operator::Custom(name) => Err(RuntimeError::UnsupportedCustom { name: name.clone() }),

            // Forward-compatibility catch-all for new graph-IR variants.
            other => Err(RuntimeError::UnsupportedOp {
                name: other.name(),
            }),
        }
    }
}

/// Executes `graph` with `inputs` (in `graph.inputs()` order, skipping
/// nodes bound to initializers) on `backend`, using `arena` for all
/// intermediate storage.
///
/// Returns one [`TensorVec`] per marked graph output (see
/// [`ComputationGraph::mark_output`] / [`ComputationGraph::infer_outputs`]),
/// in output order. The inference hot path performs no heap allocation;
/// the returned tensors are the only owned copies.
///
/// The arena is *not* reset by this function — reset it yourself between
/// runs to reuse the same backing buffer:
///
/// ```ignore
/// arena.reset();
/// let outputs = execute_graph(&graph, &[&input], &mut arena, &backend)?;
/// ```
///
/// Size the buffer with [`required_arena_bytes`] (plus a safety margin).
///
/// # Errors
/// - [`RuntimeError::OutOfMemory`] if the arena is too small
/// - [`RuntimeError::MissingInput`] / [`RuntimeError::InputCountMismatch`] /
///   [`RuntimeError::InputDataMismatch`] / [`RuntimeError::InitializerMismatch`]
///   for input-binding problems
/// - [`RuntimeError::CycleDetected`] for cyclic graphs
/// - [`RuntimeError::NoOutputs`] if no outputs are marked
/// - [`RuntimeError::Op`] / [`RuntimeError::ShapeMismatch`] /
///   [`RuntimeError::UnsupportedOp`] for operator failures
///
/// # Example
/// ```ignore
/// let outs = execute_graph(&graph, &[&input], &mut arena, &backend)?;
/// assert_eq!(outs[0].dims(), &[1, 1000]);
/// ```
pub fn execute_graph<B: Backend>(
    graph: &ComputationGraph,
    inputs: &[&[f32]],
    arena: &mut BumpArena<'_>,
    backend: &B,
) -> Result<Vec<TensorVec<f32>>, RuntimeError> {
    let n = graph.nodes().len();
    let m = edge_count(graph);

    // 1. Topological order in arena scratch.
    let indegree = arena.alloc_slice::<usize>(n)?;
    let adj_off = arena.alloc_slice::<usize>(n + 1)?;
    let adj = arena.alloc_slice::<usize>(m)?;
    let cur = arena.alloc_slice::<usize>(n)?;
    let queue = arena.alloc_slice::<usize>(n)?;
    let order = arena.alloc_slice::<usize>(n)?;
    let k = kahn(graph, indegree, adj_off, adj, cur, queue, order)?;
    let order = &order[..k];

    // 2. Bind input nodes (weights / caller slices).
    let binds = arena.alloc_slice::<Bind>(n)?;
    bind_inputs(graph, binds, inputs)?;

    // 3. Activation layout in topological order, then one big allocation.
    let total = assign_offsets(graph, order, binds)?;
    let acts = arena.alloc_slice::<f32>(total)?;
    let acts_base = acts.as_mut_ptr();

    // From here on everything is shared: the mutable scratch is complete.
    let binds: &[Bind] = binds;

    // 4. Dispatch each node in topological order.
    let exec = Exec {
        graph,
        binds,
        acts: acts_base,
        inputs,
        backend,
    };
    for &id in order {
        let node = &graph.nodes()[id];
        if node.operator.is_input() {
            continue;
        }
        let off = match binds[id] {
            Bind::Act(off) => off,
            _ => return Err(RuntimeError::ShapeMismatch),
        };
        let len = node_numel(node)?;
        // SAFETY: `off..off + len` is this node's private range (see module
        // docs); no input slice handed to the kernel aliases it.
        let out = unsafe { core::slice::from_raw_parts_mut(acts_base.add(off), len) };
        exec.run(node, out)?;
    }

    // 5. Copy marked outputs out of the arena into owned tensors.
    if graph.outputs().is_empty() {
        return Err(RuntimeError::NoOutputs);
    }
    let mut results = Vec::with_capacity(graph.outputs().len());
    for &oid in graph.outputs() {
        let (data, dims) = exec.resolve(oid)?;
        results.push(TensorVec::from_vec(data.to_vec(), dims)?);
    }
    Ok(results)
}

/// Single-input, single-output convenience wrapper around [`execute_graph`].
///
/// `input` is bound to the first (and only) input node without an
/// initializer; the graph must have exactly one marked output.
///
/// # Errors
/// All errors of [`execute_graph`], plus
/// [`RuntimeError::OutputCountMismatch`] when the graph does not have
/// exactly one output.
///
/// # Example
/// ```ignore
/// let out = execute(&graph, &input, &mut arena, &backend)?;
/// let top1 = tpt_infer_runtime::argmax(out.as_slice());
/// ```
pub fn execute<B: Backend>(
    graph: &ComputationGraph,
    input: &[f32],
    arena: &mut BumpArena<'_>,
    backend: &B,
) -> Result<TensorVec<f32>, RuntimeError> {
    let mut results = execute_graph(graph, &[input], arena, backend)?;
    if results.len() != 1 {
        return Err(RuntimeError::OutputCountMismatch {
            actual: results.len(),
        });
    }
    Ok(results.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_infer_ops::NaiveBackend;

    fn relu_chain() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
            .unwrap();
        let h = g
            .add_node(Node::new(1, Operator::Relu, vec![x], &[1, 4]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(2, Operator::softmax(-1), vec![h], &[1, 4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        g
    }

    #[test]
    fn relu_softmax_executes() {
        let g = relu_chain();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let input = [-1.0f32, 2.0, -3.0, 4.0];
        let outs = execute_graph(&g, &[&input], &mut arena, &NaiveBackend::new()).unwrap();
        assert_eq!(outs.len(), 1);
        let out = outs[0].as_slice();
        // relu: [0, 2, 0, 4] → softmax rows sum to 1, argmax at index 3,
        // and the two relu zeros stay equal through softmax.
        assert!((out[0] - out[2]).abs() < 1e-6);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(crate::argmax(out), 3);
    }

    #[test]
    fn matmul_matches_backend_reference() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[2, 3]).unwrap())
            .unwrap();
        let w = g
            .add_node(
                Node::new(1, Operator::Input, vec![], &[3, 4])
                    .unwrap()
                    .with_name("w"),
            )
            .unwrap();
        g.add_initializer(
            tpt_infer_graph::Initializer::new("w", &[3, 4], vec![1.0f32; 12]).unwrap(),
        );
        let y = g
            .add_node(Node::new(2, Operator::MatMul, vec![x, w], &[2, 4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();

        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let outs = execute(&g, &input, &mut arena, &NaiveBackend::new()).unwrap();
        // Each output row is the sum of the row of x (w is all ones).
        assert_eq!(outs.dims(), &[2, 4][..]);
        let row0 = input[..3].iter().sum::<f32>();
        let row1 = input[3..].iter().sum::<f32>();
        for j in 0..4 {
            assert!((outs.as_slice()[j] - row0).abs() < 1e-5);
            assert!((outs.as_slice()[4 + j] - row1).abs() < 1e-5);
        }
    }

    #[test]
    fn out_of_order_dependencies_execute_correctly() {
        // Node 1 depends on node 2, which is added later: topological order
        // must not equal insertion order here.
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[2]).unwrap())
            .unwrap();
        let a = g
            .add_node(Node::new(1, Operator::Add, vec![x, 2], &[2]).unwrap())
            .unwrap();
        let b = g
            .add_node(Node::new(2, Operator::Relu, vec![x], &[2]).unwrap())
            .unwrap();
        g.mark_output(a).unwrap();
        g.mark_output(b).unwrap();

        let input = [-1.0f32, 5.0];
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let outs = execute_graph(&g, &[&input], &mut arena, &NaiveBackend::new()).unwrap();
        assert_eq!(outs.len(), 2);
        // output a = x + relu(x) = [-1, 10]; output b = relu(x) = [0, 5]
        assert_eq!(outs[0].as_slice(), &[-1.0, 10.0]);
        assert_eq!(outs[1].as_slice(), &[0.0, 5.0]);
    }

    #[test]
    fn missing_input_is_reported() {
        let g = relu_chain();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let err = execute_graph(&g, &[], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(err, RuntimeError::MissingInput { node: 0 });
    }

    #[test]
    fn extra_input_is_reported() {
        let g = relu_chain();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let x = [0.0f32; 4];
        let err = execute_graph(&g, &[&x, &x], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::InputCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn wrong_input_length_is_reported() {
        let g = relu_chain();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let x = [0.0f32; 3];
        let err = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(err, RuntimeError::InputDataMismatch { node: 0 });
    }

    #[test]
    fn tiny_arena_runs_out_of_memory() {
        let g = relu_chain();
        let mut mem = [0u8; 8];
        let mut arena = BumpArena::new(&mut mem);
        let x = [0.0f32; 4];
        let err = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(err, RuntimeError::OutOfMemory);
    }

    #[test]
    fn cyclic_graph_is_detected() {
        let mut g = ComputationGraph::new();
        g.add_node(Node::new(0, Operator::Relu, vec![1], &[4]).unwrap())
            .unwrap();
        g.add_node(Node::new(1, Operator::Relu, vec![0], &[4]).unwrap())
            .unwrap();
        g.mark_output(1).unwrap();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let err = execute_graph(&g, &[], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(err, RuntimeError::CycleDetected);
    }

    #[test]
    fn unmarked_graph_reports_no_outputs() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[4]).unwrap())
            .unwrap();
        g.add_node(Node::new(1, Operator::Relu, vec![x], &[4]).unwrap())
            .unwrap();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let x = [0.0f32; 4];
        let err = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(err, RuntimeError::NoOutputs);
    }

    #[test]
    fn custom_op_is_unsupported() {
        let mut g = ComputationGraph::new();
        let x = g
            .add_node(Node::new(0, Operator::Input, vec![], &[4]).unwrap())
            .unwrap();
        let y = g
            .add_node(Node::new(1, Operator::custom("Erf"), vec![x], &[4]).unwrap())
            .unwrap();
        g.mark_output(y).unwrap();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let mut arena = BumpArena::new(&mut mem);
        let x = [0.0f32; 4];
        let err = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::UnsupportedCustom { name: "Erf".into() }
        );
    }

    #[test]
    fn arena_is_reusable_after_reset() {
        let g = relu_chain();
        let bytes = required_arena_bytes(&g);
        let mut mem = vec![0u8; bytes];
        let x = [-2.0f32, 1.0, 3.0, -4.0];
        let mut arena = BumpArena::new(&mut mem);
        let first = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap();
        let used = arena.used();
        assert!(used <= arena.capacity());
        arena.reset();
        let second = execute_graph(&g, &[&x], &mut arena, &NaiveBackend::new()).unwrap();
        assert_eq!(first[0].as_slice(), second[0].as_slice());
        assert_eq!(arena.used(), used);
    }
}
