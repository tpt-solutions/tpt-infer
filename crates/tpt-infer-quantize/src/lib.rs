//! INT8/INT4 quantization and dequantization kernels for edge deployment.
//!
//! - [`quant`]: f32 → INT8/INT4 (symmetric & asymmetric, per-tensor & per-channel)
//! - [`dequant`]: INT8/INT4 → f32
//! - [`ptq`]: post-training quantization pipeline over a [`ComputationGraph`]
//!
//! # Example
//! ```
//! use tpt_infer_quantize::{dequantize_i8_symmetric, quantize_i8_symmetric};
//!
//! let data = [0.0f32, 0.5, -0.5, 1.0, -1.0];
//! let (q, scale) = quantize_i8_symmetric(&data);
//! let d = dequantize_i8_symmetric(&q, scale);
//! for (a, b) in data.iter().zip(&d) {
//!     assert!((a - b).abs() < 0.01, "{a} vs {b}");
//! }
//! ```

pub mod dequant;
pub mod ptq;
pub mod quant;

pub use dequant::{
    dequantize_i4_symmetric, dequantize_i4_symmetric_per_channel, dequantize_i8_asymmetric,
    dequantize_i8_symmetric, dequantize_i8_symmetric_per_channel,
};
pub use ptq::{
    ptq, quantized_matmul_i8, PtqError, PtqOptions, QuantizedGraph, QuantizedTensor,
};
pub use quant::{
    pack_i4, quantize_i4_symmetric, quantize_i4_symmetric_per_channel, quantize_i8_asymmetric,
    quantize_i8_symmetric, quantize_i8_symmetric_per_channel, unpack_i4, QuantParams,
};
