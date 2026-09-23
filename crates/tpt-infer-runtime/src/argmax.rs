//! Index of the maximum element.

/// Index of the largest element in `xs` (the first index wins on ties).
///
/// Returns `0` for an empty slice. `NaN` entries are never selected while a
/// non-`NaN` value exists; if every value is `NaN`, the first index is
/// returned.
///
/// # Example
/// ```
/// use tpt_infer_runtime::argmax;
/// assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
/// assert_eq!(argmax(&[]), 0);
/// ```
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..xs.len() {
        let b = xs[best];
        let v = xs[i];
        // `v > b` is false when either operand is NaN, so also replace a NaN
        // best with any finite value; keep the earlier index on ties.
        if v > b || (b.is_nan() && !v.is_nan()) {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_largest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -2.0]), 0);
    }

    #[test]
    fn argmax_tie_returns_first() {
        assert_eq!(argmax(&[2.0, 2.0, 2.0]), 0);
    }

    #[test]
    fn argmax_empty_is_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn argmax_nan_handling() {
        assert_eq!(argmax(&[f32::NAN, 1.0, 2.0]), 2);
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    }
}
