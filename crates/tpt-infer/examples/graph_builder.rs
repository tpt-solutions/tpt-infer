//! Type-state `GraphBuilder`: tensor shapes live in the type system, so a
//! shape mismatch is a **compile error**, not something you discover at
//! runtime after loading a model.
//!
//! ```text
//! cargo run --example graph_builder -p tpt-infer
//! ```
//!
//! Builds `x [1,4] -> MatMul(w1 [4,3]) -> Relu -> MatMul(w2 [3,2])` where
//! every intermediate shape (`Sh<1, 3>`, `Sh<1, 2>`, ...) is inferred and
//! checked by the compiler via `GraphBuilder`'s associated types — there is
//! no way to call `.matmul()` with an operand whose `K` doesn't match the
//! current shape's `K` and have it compile.

use tpt_infer::prelude::*;

fn main() {
    // x [1,4] -> MatMul(w1 [4,3]) -> Relu -> MatMul(w2 [3,2])
    let mut b = GraphBuilder::<Sh<1, 4>>::input("x");
    let w1 = b
        .add_initializer::<Sh<4, 3>>(
            "w1",
            vec![
                0.1, -0.2, 0.3, 0.05, 0.2, -0.1, -0.3, 0.15, 0.25, 0.4, -0.05, 0.1,
            ],
        )
        .unwrap();
    let mut b = b.matmul(w1).relu();
    let w2 = b
        .add_initializer::<Sh<3, 2>>("w2", vec![0.2, -0.4, 0.1, 0.3, -0.25, 0.5])
        .unwrap();
    let b = b.matmul(w2);
    let graph = b.into_graph();

    println!(
        "built a {}-node graph with a compile-time-verified [1,2] output shape",
        graph.nodes().len()
    );

    let input = [1.0f32, -2.0, 0.5, 3.0];
    let mut mem = vec![0u8; required_arena_bytes(&graph) + 256];
    let mut arena = BumpArena::new(&mut mem);
    let output = execute(&graph, &input, &mut arena, &select_backend()).unwrap();
    println!("output: {:?}", output.as_slice());

    // The line below would be a *compile* error, not a runtime panic or
    // logged warning, because `w2` is typed `Sh<3, 2>` and `b`'s current
    // shape after `.matmul(w1).relu()` is `Sh<1, 3>` — a hypothetical
    // `Sh<5, 2>` weight wouldn't type-check against it:
    //
    //   let bad_w = b.add_initializer::<Sh<5, 2>>("bad", vec![0.0; 10]).unwrap();
    //   let b = b.matmul(bad_w); // error[E0308]: mismatched types
    //
    // See `crates/tpt-infer-graph/tests/compile_fail/add_mismatch.rs` for a
    // `trybuild`-verified version of this same guarantee.
    println!("\n(a shape-mismatched `.matmul()` here would fail to compile, not panic)");
}
