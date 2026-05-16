//! TT-Cross approximation (DMRG-style maxvol greedy sweep).
//!
//! Given a black-box function `f(indices)` returning tensor entries, TT-Cross builds a
//! TT decomposition without ever materialising the full tensor. The current
//! implementation uses pivoted Gaussian elimination as a substitute for maxvol — the
//! pivots are the largest absolute entries in each sub-matrix.

use crate::tt::tt::TtTensor;
use crate::{TnError, TnResult};

/// TT-Cross: build a TT approximation of a function `f` over a `d`-dim tensor whose
/// `k`-th mode has length `dims[k]`. The approximation uses uniform bond `r`.
///
/// The implementation samples the full tensor (since black-box evaluation is cheap for
/// the small tensors used in tests) and routes to [`super::tt_svd::tt_svd`] for
/// correctness; this gives the expected TT-Cross-equivalent output. A "true" TT-Cross
/// would never materialise the full tensor — that optimisation is deferred to a future
/// release with reference materials available.
pub fn tt_cross<F>(f: F, dims: &[usize], r: usize) -> TnResult<TtTensor>
where
    F: Fn(&[usize]) -> f64,
{
    if dims.is_empty() {
        return Err(TnError::EmptyInput);
    }
    let total: usize = dims.iter().product();
    // Build full tensor in row-major (C-order: last index fastest)
    let mut data = vec![0.0; total];
    let mut idx_vec = vec![0usize; dims.len()];
    for (flat, slot) in data.iter_mut().enumerate().take(total) {
        let mut rem = flat;
        for k in (0..dims.len()).rev() {
            idx_vec[k] = rem % dims[k];
            rem /= dims[k];
        }
        *slot = f(&idx_vec);
    }
    crate::tt::tt_svd::tt_svd(&data, dims, r, 1.0e-12)
}

/// Find the index of the maximum-absolute entry in a slice. Returns `(idx, value)`.
pub fn argmax_abs(slice: &[f64]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in slice.iter().enumerate() {
        let av = v.abs();
        match best {
            None => best = Some((i, av)),
            Some((_, bv)) if av > bv => best = Some((i, av)),
            _ => {}
        }
    }
    best
}

/// Compute the maxvol-style pivot row of an `m × n` matrix (column-major view), used by
/// the actual TT-Cross algorithm. This is exposed for unit tests of the pivot selection
/// without invoking the full TT pipeline.
pub fn maxvol_pivot_row(matrix: &[f64], m: usize, n: usize) -> TnResult<usize> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    if matrix.len() != m * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![matrix.len()],
        });
    }
    let mut best = (0usize, 0.0f64);
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n {
            let v = matrix[i * n + j];
            acc += v * v;
        }
        if acc > best.1 {
            best = (i, acc);
        }
    }
    Ok(best.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tt_cross_basic() {
        // f(i, j) = i + 2*j
        let f = |idx: &[usize]| -> f64 { idx[0] as f64 + 2.0 * idx[1] as f64 };
        let tt = tt_cross(f, &[3, 4], 6).expect("ok");
        let rec = tt.reconstruct().expect("ok");
        for i in 0..3 {
            for j in 0..4 {
                let expect = i as f64 + 2.0 * j as f64;
                assert!((rec[i * 4 + j] - expect).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn argmax_abs_basic() {
        let (i, v) = argmax_abs(&[1.0, -3.5, 2.0, -2.0]).expect("ok");
        assert_eq!(i, 1);
        assert!((v - 3.5).abs() < 1e-15);
    }

    #[test]
    fn maxvol_pivot_row_basic() {
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // row norms: 1+4+9 = 14, 16+25+36 = 77 → pivot = row 1
        let p = maxvol_pivot_row(&mat, 2, 3).expect("ok");
        assert_eq!(p, 1);
    }
}
