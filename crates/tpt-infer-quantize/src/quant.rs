//! Quantization kernels: f32 → INT8 / INT4.

/// Scale and zero-point for asymmetric quantization.
///
/// Dequantization: `f32 = (q - zero_point) * scale`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Multiplicative scale.
    pub scale: f32,
    /// Zero-point (value that maps to 0.0).
    pub zero_point: i32,
}

/// INT8 symmetric quantization: `scale = max(|x|) / 127`, `zero_point = 0`.
///
/// Values are clamped to `[-127, 127]`.
///
/// # Example
/// ```
/// use tpt_infer_quantize::quantize_i8_symmetric;
/// let (q, scale) = quantize_i8_symmetric(&[-1.0, 0.0, 1.0]);
/// assert_eq!(q, [-127, 0, 127]);
/// assert!((scale - 1.0 / 127.0).abs() < 1e-6);
/// ```
pub fn quantize_i8_symmetric(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    };
    let q = data
        .iter()
        .map(|&x| {
            let v = (x / scale).round();
            v.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, scale)
}

/// INT8 asymmetric quantization with a learned/derived zero-point.
///
/// Maps `[min, max]` onto `[-128, 127]`.
pub fn quantize_i8_asymmetric(data: &[f32]) -> (Vec<i8>, QuantParams) {
    const QMIN: f32 = -128.0;
    const QMAX: f32 = 127.0;
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (min, max) = if min > max || !min.is_finite() || !max.is_finite() {
        (0.0, 0.0)
    } else {
        (min, max)
    };

    let (scale, zp) = if (max - min).abs() < f32::EPSILON {
        (1.0, 0i32)
    } else {
        let scale0 = (max - min) / (QMAX - QMIN);
        let zp0 = (QMIN - min / scale0).round();
        if zp0 < QMIN {
            // Pin zero-point at QMIN; stretch scale so max → QMAX.
            let scale = max / (QMAX - QMIN);
            let scale = if scale <= 0.0 { scale0 } else { scale };
            (scale, QMIN as i32)
        } else if zp0 > QMAX {
            // Pin zero-point at QMAX; stretch scale so min → QMIN (min < 0).
            let scale = min / (QMIN - QMAX);
            let scale = if scale <= 0.0 { scale0 } else { scale.abs() };
            (scale, QMAX as i32)
        } else {
            (scale0, zp0 as i32)
        }
    };

    let q = data
        .iter()
        .map(|&x| {
            let v = (x / scale + zp as f32).round();
            v.clamp(QMIN, QMAX) as i8
        })
        .collect();
    (
        q,
        QuantParams {
            scale,
            zero_point: zp,
        },
    )
}

/// Per-channel (along axis 0 of a 2-D `[rows, cols]` layout) INT8 symmetric.
///
/// Returns row-major quantized data and one scale per row (`rows` scales).
pub fn quantize_i8_symmetric_per_channel(data: &[f32], rows: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(rows > 0 && data.len() % rows == 0);
    let cols = data.len() / rows;
    let mut scales = Vec::with_capacity(rows);
    let mut out = Vec::with_capacity(data.len());
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let (q, s) = quantize_i8_symmetric(row);
        scales.push(s);
        out.extend_from_slice(&q);
    }
    (out, scales)
}

/// Pack signed 4-bit values two-per-byte (low nibble first).
///
/// Each input must fit in `[-8, 7]`.
///
/// # Panics
/// Panics if any value is outside `[-8, 7]`.
///
/// # Example
/// ```
/// use tpt_infer_quantize::{pack_i4, unpack_i4};
/// let packed = pack_i4(&[-8, 7, 0, 1]);
/// assert_eq!(packed.len(), 2);
/// assert_eq!(unpack_i4(&packed, 4), vec![-8, 7, 0, 1]);
/// ```
pub fn pack_i4(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(2));
    let mut i = 0;
    while i < values.len() {
        let lo = values[i];
        assert!((-8..=7).contains(&lo), "i4 out of range: {lo}");
        let hi = if i + 1 < values.len() {
            let v = values[i + 1];
            assert!((-8..=7).contains(&v), "i4 out of range: {v}");
            v
        } else {
            0
        };
        out.push(((hi as u8 & 0x0f) << 4) | (lo as u8 & 0x0f));
        i += 2;
    }
    out
}

/// Unpack `n` signed 4-bit values from nibble-packed bytes.
pub fn unpack_i4(packed: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for &byte in packed {
        if out.len() >= n {
            break;
        }
        out.push(sign_extend_i4(byte & 0x0f));
        if out.len() >= n {
            break;
        }
        out.push(sign_extend_i4((byte >> 4) & 0x0f));
    }
    out.truncate(n);
    out
}

fn sign_extend_i4(nibble: u8) -> i8 {
    let n = nibble as i8;
    if n & 0x08 != 0 {
        n | !0x0f
    } else {
        n & 0x0f
    }
}

/// INT4 symmetric quantization (per-tensor), nibble-packed.
///
/// Returns `(packed_bytes, scale)` where `packed_bytes.len() == ceil(n/2)`.
pub fn quantize_i4_symmetric(data: &[f32]) -> (Vec<u8>, f32) {
    let max_abs = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 7.0
    };
    let qs: Vec<i8> = data
        .iter()
        .map(|&x| {
            let v = (x / scale).round();
            v.clamp(-7.0, 7.0) as i8
        })
        .collect();
    (pack_i4(&qs), scale)
}

/// Per-channel INT4 symmetric over a `[rows, cols]` layout.
///
/// Returns `(packed_bytes, scales)` with one scale per row. Packed layout is
/// row-major with each row padded to a whole number of bytes.
pub fn quantize_i4_symmetric_per_channel(data: &[f32], rows: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(rows > 0 && data.len() % rows == 0);
    let cols = data.len() / rows;
    let mut scales = Vec::with_capacity(rows);
    let mut out = Vec::new();
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let (packed, s) = quantize_i4_symmetric(row);
        scales.push(s);
        out.extend_from_slice(&packed);
    }
    (out, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8_symmetric_roundtrip_unit_range() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 / 50.0).collect();
        let (q, scale) = quantize_i8_symmetric(&data);
        assert_eq!(q.len(), data.len());
        for (&qi, &x) in q.iter().zip(&data) {
            let d = qi as f32 * scale;
            assert!((d - x).abs() < 0.01, "err {} for {}", (d - x).abs(), x);
        }
    }

    #[test]
    fn i8_asymmetric_roundtrip() {
        let data = [0.0f32, 0.25, 0.5, 1.0, 2.0];
        let (q, p) = quantize_i8_asymmetric(&data);
        for (&qi, &x) in q.iter().zip(&data) {
            let d = (qi as f32 - p.zero_point as f32) * p.scale;
            assert!((d - x).abs() < 0.01, "err {} for {}", (d - x).abs(), x);
        }
    }

    #[test]
    fn per_channel_not_worse_than_per_tensor() {
        // Channel 0 large range, channel 1 small range.
        let mut data = vec![0.0f32; 64];
        for (i, v) in data.iter_mut().enumerate().take(32) {
            *v = (i as f32 / 31.0) * 10.0;
        }
        for (i, v) in data.iter_mut().enumerate().skip(32) {
            *v = (i as f32 - 32.0) / 31.0 * 0.1;
        }
        let (qt, st) = quantize_i8_symmetric(&data);
        let (qc, sc) = quantize_i8_symmetric_per_channel(&data, 2);
        assert_eq!(sc.len(), 2);
        let err_t: f32 = data
            .iter()
            .zip(&qt)
            .map(|(&x, &q)| ((q as f32 * st) - x).abs())
            .sum();
        let err_c: f32 = data
            .iter()
            .zip(&qc)
            .enumerate()
            .map(|(i, (&x, &q))| ((q as f32 * sc[i / 32]) - x).abs())
            .sum();
        assert!(
            err_c <= err_t + 1e-5,
            "per-channel err {err_c} > per-tensor {err_t}"
        );
    }

    #[test]
    fn i4_pack_unpack_roundtrip() {
        let vals: Vec<i8> = (-8..=7).collect();
        let packed = pack_i4(&vals);
        assert_eq!(packed.len(), 8);
        assert_eq!(unpack_i4(&packed, vals.len()), vals);
        // Odd length
        let odd = vec![1i8, -1, 7];
        let p = pack_i4(&odd);
        assert_eq!(p.len(), 2);
        assert_eq!(unpack_i4(&p, 3), odd);
    }

    #[test]
    fn i4_sign_extension() {
        let vals = vec![-8i8, -1, 0, 7];
        let p = pack_i4(&vals);
        assert_eq!(unpack_i4(&p, 4), vals);
    }

    #[test]
    fn i4_quant_roundtrip_loose() {
        let data: Vec<f32> = (-20..=20).map(|i| i as f32 / 20.0).collect();
        let (packed, scale) = quantize_i4_symmetric(&data);
        let q = unpack_i4(&packed, data.len());
        for (&qi, &x) in q.iter().zip(&data) {
            let d = qi as f32 * scale;
            assert!((d - x).abs() < 0.1, "err {} for {}", (d - x).abs(), x);
        }
    }

    #[test]
    #[should_panic]
    fn pack_i4_rejects_out_of_range() {
        pack_i4(&[8]);
    }
}
