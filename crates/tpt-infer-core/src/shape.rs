//! Shape arithmetic helpers.

/// Maximum rank (number of dimensions) supported by dynamic shapes.
pub const MAX_RANK: usize = 8;

// Re-export name used by the crate root: `Shape` is the dynamic shape alias.

/// A dynamically-sized shape (up to [`MAX_RANK`] dimensions).
pub type Shape = [usize; MAX_RANK];

/// Number of elements in a shape (product of all dimensions).
///
/// A zero dimension yields zero elements.
pub const fn num_elements<const N: usize>(shape: [usize; N]) -> usize {
    let mut acc = 1usize;
    let mut i = 0;
    while i < N {
        acc = acc.wrapping_mul(shape[i]);
        i += 1;
    }
    acc
}

/// Row-major strides for a shape.
///
/// `strides[i]` is the number of elements to skip to advance one step along
/// dimension `i`.
pub const fn strides<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut out = [0usize; N];
    if N == 0 {
        return out;
    }
    // Compute cumulative product from the right.
    let mut acc = 1usize;
    let mut i = N;
    while i > 0 {
        i -= 1;
        out[i] = acc;
        acc = acc.wrapping_mul(shape[i]);
    }
    out
}

/// Flatten a multi-dimensional index into a row-major linear index.
pub const fn ravel_index<const N: usize>(index: [usize; N], shape: [usize; N]) -> usize {
    let st = strides(shape);
    let mut off = 0usize;
    let mut i = 0;
    while i < N {
        off += index[i] * st[i];
        i += 1;
    }
    off
}

/// Unflatten a row-major linear index into a multi-dimensional index.
pub const fn unravel_index<const N: usize>(flat: usize, shape: [usize; N]) -> [usize; N] {
    let st = strides(shape);
    let mut out = [0usize; N];
    let mut rem = flat;
    let mut i = 0;
    while i < N {
        match rem.checked_div(st[i]) {
            Some(q) if st[i] != 0 => {
                out[i] = q;
                rem %= st[i];
            }
            _ => out[i] = 0,
        }
        i += 1;
    }
    out
}

/// Convert a rank-`N` shape into the dynamic [`Shape`] representation (zero-padded).
pub fn to_dyn<const N: usize>(shape: [usize; N]) -> Shape {
    let mut out = [0; MAX_RANK];
    for (i, &d) in shape.iter().enumerate().take(MAX_RANK.min(N)) {
        out[i] = d;
    }
    out
}

/// Convert a dynamic [`Shape`] (with trailing zeros) back into a rank-`N` shape.
///
/// Returns `None` if any dimension beyond `N` (ignoring trailing zeros) is non-zero,
/// or if a dimension is zero where `N > 0` requires otherwise.
pub fn from_dyn<const N: usize>(shape: Shape) -> Option<[usize; N]> {
    let mut out = [0usize; N];
    out.copy_from_slice(&shape[..N]);
    // Remaining dims must all be zero (padding).
    if shape[N..].iter().any(|&d| d != 0) {
        return None;
    }
    Some(out)
}

/// Broadcast two shapes (numpy semantics). Returns `None` if incompatible.
pub fn broadcast<const N: usize>(a: [usize; N], b: [usize; N]) -> Option<[usize; N]> {
    let mut out = [0usize; N];
    for i in 0..N {
        out[i] = match (a[i], b[i]) {
            (x, 1) => x,
            (1, y) => y,
            (x, y) if x == y => x,
            _ => return None,
        };
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_elements_works() {
        assert_eq!(num_elements([1, 3, 224, 224]), 150528);
        assert_eq!(num_elements([0, 5]), 0);
        assert_eq!(num_elements([7]), 7);
        assert_eq!(num_elements::<0>([]), 1);
    }

    #[test]
    fn strides_row_major() {
        assert_eq!(strides([2, 3, 4]), [12, 4, 1]);
        assert_eq!(strides([5]), [1]);
    }

    #[test]
    fn ravel_unroundtrip() {
        let shape = [2, 3, 4];
        for flat in 0..24 {
            let idx = unravel_index(flat, shape);
            assert_eq!(ravel_index(idx, shape), flat);
        }
    }

    #[test]
    fn broadcast_compat() {
        assert_eq!(
            broadcast([1, 3, 1], [2, 1, 4]),
            Some([2, 3, 4])
        );
        assert_eq!(broadcast([2, 3], [4, 3]), None);
    }

    #[test]
    fn dyn_roundtrip() {
        let s = to_dyn([1, 3, 224, 224]);
        assert_eq!(from_dyn::<4>(s), Some([1, 3, 224, 224]));
        // Too-small rank fails because trailing dim is non-zero.
        assert_eq!(from_dyn::<3>(s), None);
    }
}
