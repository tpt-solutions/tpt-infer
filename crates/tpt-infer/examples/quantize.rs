//! INT8 post-training quantization round-trip, through the facade's
//! re-exported `ptq` API.
//!
//! ```text
//! cargo run --example quantize -p tpt-infer --features quantize
//! ```
//!
//! Builds a small `x -> MatMul(w) -> y` graph, quantizes `w` to INT8
//! (symmetric, per-channel), and reports the dequantization error — the
//! same pipeline you'd run on a real ONNX model's weights via
//! `tpt_infer_onnx::load` before AOT-compiling or executing it.

use tpt_infer::prelude::*;

fn main() {
    let mut g = ComputationGraph::new();
    let x = g
        .add_node(Node::new(0, Operator::Input, vec![], &[1, 4]).unwrap())
        .unwrap();
    let w = g
        .add_node(
            Node::new(1, Operator::Input, vec![], &[4, 8])
                .unwrap()
                .with_name("w"),
        )
        .unwrap();
    // Deterministic pseudo-random weights, no `rand` dependency needed.
    let mut seed = 7u32;
    let weights: Vec<f32> = (0..32)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (seed >> 8) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * 0.5
        })
        .collect();
    g.add_initializer(Initializer::new("w", &[4, 8], weights.clone()).unwrap());
    let y = g
        .add_node(Node::new(2, Operator::MatMul, vec![x, w], &[1, 8]).unwrap())
        .unwrap();
    g.mark_output(y).unwrap();

    let quantized = ptq(&g, PtqOptions::default()).expect("PTQ should succeed on a plain MatMul");
    println!("quantized {} weight tensor(s)", quantized.len());

    let wq = &quantized.weights["w"];
    println!("dtype: {:?}, scales: {:?}", wq.dtype, wq.scales);

    let reconstructed = wq.dequantize();
    let max_abs_err = weights
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max |original - dequantized| = {max_abs_err:.6}");
    assert!(
        max_abs_err < 0.01,
        "INT8 per-channel quantization should keep error well under 0.01"
    );
}
