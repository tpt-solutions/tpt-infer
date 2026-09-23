//! Scalar float helpers that work in `no_std` builds.

#[cfg(feature = "std")]
pub(crate) fn exp_f32(x: f32) -> f32 {
    x.exp()
}

#[cfg(not(feature = "std"))]
pub(crate) fn exp_f32(x: f32) -> f32 {
    exp_via_poly(x)
}

#[cfg(feature = "std")]
pub(crate) fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

#[cfg(not(feature = "std"))]
pub(crate) fn tanh_f32(x: f32) -> f32 {
    if x >= 10.0 {
        return 1.0;
    }
    if x <= -10.0 {
        return -1.0;
    }
    let e2 = exp_via_poly(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
}

/// `exp(x)` without `std`: range reduction `x = k*ln2 + r` with
/// `|r| <= ln2/2`, a degree-8 Taylor polynomial for `exp(r)` (error
/// `< 1e-9` in range), and scaling by `2^k` via the IEEE-754 exponent field.
#[cfg(not(feature = "std"))]
fn exp_via_poly(x: f32) -> f32 {
    if x >= 88.72 {
        return f32::INFINITY;
    }
    if x <= -104.0 {
        return 0.0;
    }
    const LN_2: f32 = core::f32::consts::LN_2;
    let t = x / LN_2;
    let k = if t >= 0.0 {
        (t + 0.5) as i32
    } else {
        (t - 0.5) as i32
    };
    let r = x - (k as f32) * LN_2;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r / 40320.0)))))));
    let bits = p.to_bits();
    let field = ((bits >> 23) & 0xff) as i32;
    let new_field = field + k;
    if new_field >= 255 {
        return f32::INFINITY;
    }
    if new_field <= 0 {
        return 0.0;
    }
    f32::from_bits((bits & !(0xffu32 << 23)) | ((new_field as u32) << 23))
}

#[cfg(all(test, not(feature = "std")))]
mod tests {
    #[test]
    fn fallback_exp_matches_std() {
        for i in -200..200 {
            let x = i as f32 * 0.37;
            let got = super::exp_via_poly(x);
            let want = x.exp();
            let tol = 1e-5 * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tol,
                "exp({x}): got {got}, want {want}"
            );
        }
    }

    #[test]
    fn fallback_tanh_saturates() {
        assert!((super::tanh_f32(0.0)).abs() < 1e-6);
        assert_eq!(super::tanh_f32(50.0), 1.0);
        assert_eq!(super::tanh_f32(-50.0), -1.0);
    }
}
