//! Dequantization kernels: INT8 / INT4 → f32.

use crate::quant::{unpack_i4, QuantParams};

/// Dequantize symmetric INT8: `x = q * scale`.
///
/// # Example
/// ```
/// use tpt_infer_quantize::dequantize_i8_symmetric;
/// let d = dequantize_i8_symmetric(&[0, 127], 1.0 / 127.0);
/// assert!((d[1] - 1.0).abs() < 1e-6);
/// ```
pub fn dequantize_i8_symmetric(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&q| q as f32 * scale).collect()
}

/// Dequantize asymmetric INT8: `x = (q - zero_point) * scale`.
pub fn dequantize_i8_asymmetric(data: &[i8], params: QuantParams) -> Vec<f32> {
    data.iter()
        .map(|&q| (q as f32 - params.zero_point as f32) * params.scale)
        .collect()
}

/// Per-channel symmetric dequantization: one scale per `scales.len()` channels
/// (equal-sized chunks of `data`).
pub fn dequantize_i8_symmetric_per_channel(data: &[i8], scales: &[f32]) -> Vec<f32> {
    assert!(!scales.is_empty());
    assert_eq!(data.len() % scales.len(), 0);
    let cols = data.len() / scales.len();
    let mut out = Vec::with_capacity(data.len());
    for (r, &s) in scales.iter().enumerate() {
        for &q in &data[r * cols..(r + 1) * cols] {
            out.push(q as f32 * s);
        }
    }
    out
}

/// Dequantize nibble-packed symmetric INT4.
///
/// # Arguments
/// - `packed`: nibble-packed bytes
/// - `n`: number of logical elements
/// - `scale`: per-tensor scale
pub fn dequantize_i4_symmetric(packed: &[u8], n: usize, scale: f32) -> Vec<f32> {
    unpack_i4(packed, n)
        .into_iter()
        .map(|q| q as f32 * scale)
        .collect()
}

/// Per-channel INT4 dequantization; `row_elems` elements per channel row.
///
/// Packed data is laid out as consecutive per-row packed blocks (each row
/// `ceil(row_elems/2)` bytes), matching [`quantize_i4_symmetric_per_channel`](crate::quantize_i4_symmetric_per_channel).
pub fn dequantize_i4_symmetric_per_channel(
    packed: &[u8],
    row_elems: usize,
    scales: &[f32],
) -> Vec<f32> {
    assert!(!scales.is_empty());
    let row_bytes = row_elems.div_ceil(2);
    let mut out = Vec::with_capacity(row_elems * scales.len());
    for (r, &s) in scales.iter().enumerate() {
        let start = r * row_bytes;
        let end = (start + row_bytes).min(packed.len());
        let q = unpack_i4(&packed[start..end], row_elems);
        out.extend(q.into_iter().map(|v| v as f32 * s));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_roundtrip() {
        let data = [0.0f32, 1.0, -1.0, 0.5];
        let (q, scale) = crate::quantize_i8_symmetric(&data);
        let d = dequantize_i8_symmetric(&q, scale);
        for (a, b) in data.iter().zip(&d) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn asymmetric_roundtrip() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let (q, p) = crate::quantize_i8_asymmetric(&data);
        let d = dequantize_i8_asymmetric(&q, p);
        for (a, b) in data.iter().zip(&d) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn per_channel_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let (q, scales) = crate::quantize_i8_symmetric_per_channel(&data, 4);
        let d = dequantize_i8_symmetric_per_channel(&q, &scales);
        assert_eq!(d.len(), data.len());
        for (a, b) in data.iter().zip(&d) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn i4_roundtrip() {
        let data: Vec<f32> = (-16..=16).map(|i| i as f32 / 16.0).collect();
        let (packed, scale) = crate::quantize_i4_symmetric(&data);
        let d = dequantize_i4_symmetric(&packed, data.len(), scale);
        assert_eq!(d.len(), data.len());
        for (a, b) in data.iter().zip(&d) {
            assert!((a - b).abs() < 0.1, "{a} vs {b}");
        }
    }

    #[test]
    fn i4_per_channel_roundtrip() {
        let mut data = vec![0.0f32; 64];
        for (i, v) in data.iter_mut().enumerate().take(32) {
            *v = i as f32 / 31.0;
        }
        for (i, v) in data.iter_mut().enumerate().skip(32) {
            *v = (i as f32 - 32.0) / 31.0 * 0.5;
        }
        let (packed, scales) = crate::quantize_i4_symmetric_per_channel(&data, 2);
        let d = dequantize_i4_symmetric_per_channel(&packed, 32, &scales);
        assert_eq!(d.len(), 64);
        for (a, b) in data.iter().zip(&d) {
            assert!((a - b).abs() < 0.1, "{a} vs {b}");
        }
    }
}
