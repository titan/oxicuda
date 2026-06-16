//! Sequence acceleration: Aitken's Δ² process and Wynn's ε-algorithm.
//!
//! Two classical nonlinear sequence transformations that accelerate the
//! convergence of a slowly-converging sequence (typically the partial sums of a
//! series).
//!
//! # Aitken's Δ² (Aitken 1926)
//!
//! Given three consecutive terms it forms
//!
//! ```text
//! x̂_n = x_n - (Δx_n)² / Δ²x_n,   Δx_n = x_{n+1} - x_n,   Δ²x_n = x_{n+2} - 2 x_{n+1} + x_n.
//! ```
//!
//! For a sequence whose error decays geometrically, `S_n = S + c r^n`, the
//! transform returns the limit `S` *exactly* from any three consecutive terms.
//!
//! # Wynn's ε-algorithm (Shanks 1955; Wynn 1956)
//!
//! A recursive scheme that evaluates the Shanks transformation (and its
//! iterates) without forming the underlying Hankel determinants:
//!
//! ```text
//! ε_{-1}^{(n)} = 0,   ε_0^{(n)} = S_n,
//! ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1 / ( ε_k^{(n+1)} - ε_k^{(n)} ).
//! ```
//!
//! The even columns `ε_{2j}^{(n)}` are the accelerated approximants; `ε_2^{(n)}`
//! reproduces the first Shanks transform `S_1(A_n)`. Odd columns are auxiliary.

use crate::error::{NumericError, NumericResult};

/// Apply a single Aitken Δ² step to three consecutive terms `(x0, x1, x2)`.
///
/// Returns the extrapolated value `x0 - (x1 - x0)² / (x2 - 2 x1 + x0)`.
///
/// # Errors
/// Returns [`NumericError::NumericalInstability`] if the second difference
/// `Δ²x = x2 - 2 x1 + x0` is too small to divide by safely (the iteration has
/// effectively stalled); the caller should then take the latest term as the
/// best estimate.
pub fn aitken_step(x0: f64, x1: f64, x2: f64) -> NumericResult<f64> {
    let d1 = x1 - x0;
    let d2 = x2 - 2.0 * x1 + x0;
    if d2.abs() < 1.0e-300 {
        return Err(NumericError::NumericalInstability(
            "Aitken Δ²x is ~0; sequence has stalled".into(),
        ));
    }
    Ok(x0 - d1 * d1 / d2)
}

/// Apply Aitken's Δ² process across a whole sequence, producing the transformed
/// sequence of length `n - 2`.
///
/// Where the local second difference is ~0 (a stalled triple) the corresponding
/// raw term `x_{n+2}` is carried over instead of dividing by zero, so the output
/// remains well-defined even on near-converged data.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] if fewer than three terms are
/// supplied.
pub fn aitken_sequence(seq: &[f64]) -> NumericResult<Vec<f64>> {
    if seq.len() < 3 {
        return Err(NumericError::InvalidParameter(
            "Aitken Δ² needs at least 3 terms".into(),
        ));
    }
    let mut out = Vec::with_capacity(seq.len() - 2);
    for w in seq.windows(3) {
        match aitken_step(w[0], w[1], w[2]) {
            Ok(v) => out.push(v),
            // Stalled triple: best available estimate is the latest term.
            Err(_) => out.push(w[2]),
        }
    }
    Ok(out)
}

/// Repeatedly apply Aitken's Δ² until the sequence collapses (length < 3) or the
/// estimate changes by less than `tol`, returning the final extrapolate.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] if fewer than three terms are
/// supplied.
pub fn aitken_accelerate(seq: &[f64], tol: f64) -> NumericResult<f64> {
    if seq.len() < 3 {
        return Err(NumericError::InvalidParameter(
            "Aitken Δ² needs at least 3 terms".into(),
        ));
    }
    let mut current = seq.to_vec();
    let mut best = *current.last().unwrap_or(&0.0);
    while current.len() >= 3 {
        let next = aitken_sequence(&current)?;
        if let Some(&last) = next.last() {
            if (last - best).abs() < tol {
                return Ok(last);
            }
            best = last;
        }
        current = next;
    }
    Ok(best)
}

/// Result of running Wynn's ε-algorithm on a sequence of partial sums.
#[derive(Debug, Clone)]
pub struct WynnEpsilon {
    /// The best (last computable) even-column approximant.
    pub estimate: f64,
    /// The highest even column index `2j` reached.
    pub order: usize,
}

/// Run Wynn's ε-algorithm on the partial sums `seq` and return the best
/// even-column extrapolate (the most-accelerated Shanks-type approximant).
///
/// The full lower-triangular ε-table is built; the most accelerated value is the
/// even-column entry from the deepest column that could be formed before a
/// division by a (near-)zero ε difference forced an early stop.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] if fewer than three partial sums
/// are supplied (the first nontrivial column `ε_2` needs three terms).
pub fn wynn_epsilon(seq: &[f64]) -> NumericResult<WynnEpsilon> {
    let n = seq.len();
    if n < 3 {
        return Err(NumericError::InvalidParameter(
            "Wynn's ε-algorithm needs at least 3 partial sums".into(),
        ));
    }

    // eps[k][n] with the standard staircase indexing. We store columns 0..=n.
    // Column 0 is the input; column -1 is implicitly 0.
    // Use a 2D table indexed [column][start-row].
    let mut prev_col = vec![0.0_f64; n + 1]; // represents ε_{-1}: all zeros.
    let mut cur_col: Vec<f64> = seq.to_vec(); // ε_0^{(n)} = S_n.

    let mut best = *cur_col.last().unwrap_or(&seq[n - 1]);
    let mut best_order = 0usize;

    // Build successive columns. Column k has length (n - k).
    for k in 1..n {
        let len = n - k;
        let mut next_col = vec![0.0_f64; len];
        for row in 0..len {
            let denom = cur_col[row + 1] - cur_col[row];
            if denom.abs() < 1.0e-300 {
                // ε difference vanished: cannot extend this entry. Use the most
                // recent good even-column estimate and stop.
                return Ok(WynnEpsilon {
                    estimate: best,
                    order: best_order,
                });
            }
            next_col[row] = prev_col[row + 1] + 1.0 / denom;
        }
        // Even columns (k even) carry the accelerated approximants.
        if k % 2 == 0 {
            if let Some(&v) = next_col.last() {
                if v.is_finite() {
                    best = v;
                    best_order = k;
                }
            }
        }
        prev_col = cur_col;
        cur_col = next_col;
    }

    Ok(WynnEpsilon {
        estimate: best,
        order: best_order,
    })
}

/// Compute the single Shanks transform `S_1(A_n) = (A_{n+1} A_{n-1} - A_n²) /
/// (A_{n+1} - 2 A_n + A_{n-1})` directly from three consecutive partial sums.
///
/// This is identical to one Aitken Δ² step (centred at `A_n`) and equals the
/// `ε_2` column of Wynn's algorithm.
///
/// # Errors
/// Returns [`NumericError::NumericalInstability`] if the denominator is ~0.
pub fn shanks(a_prev: f64, a_curr: f64, a_next: f64) -> NumericResult<f64> {
    let denom = a_next - 2.0 * a_curr + a_prev;
    if denom.abs() < 1.0e-300 {
        return Err(NumericError::NumericalInstability(
            "Shanks denominator is ~0".into(),
        ));
    }
    Ok((a_next * a_prev - a_curr * a_curr) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn aitken_exact_for_geometric() {
        // S_n = S - c r^n  ⇒  Aitken recovers S exactly from 3 terms.
        let s_limit = 4.2_f64;
        let c = 1.7_f64;
        let r = 0.6_f64;
        let term = |n: i32| s_limit - c * r.powi(n);
        let v = aitken_step(term(0), term(1), term(2)).expect("aitken");
        assert!((v - s_limit).abs() < 1.0e-12, "got {v}");

        // Also from a shifted window.
        let v2 = aitken_step(term(5), term(6), term(7)).expect("aitken");
        assert!((v2 - s_limit).abs() < 1.0e-10, "got {v2}");
    }

    #[test]
    fn aitken_geometric_series_sum() {
        // Σ_{k=0}^∞ r^k = 1/(1-r). Partial sums S_n = (1 - r^{n+1})/(1 - r).
        let r = 0.5_f64;
        let exact = 1.0 / (1.0 - r);
        let partial: Vec<f64> = (0..3).map(|n| (1.0 - r.powi(n + 1)) / (1.0 - r)).collect();
        let v = aitken_step(partial[0], partial[1], partial[2]).expect("aitken");
        assert!((v - exact).abs() < 1.0e-12, "got {v}, want {exact}");
    }

    #[test]
    fn aitken_accelerates_fixed_point() {
        // Linearly-convergent fixed point: x_{n+1} = cos(x_n) → Dottie number
        // (asymptotic rate |x'| = |sin(D)| ≈ 0.674). Aitken must reach a given
        // tolerance using strictly FEWER iterates than the raw iteration.
        let dottie = 0.739_085_133_215_160_6_f64;
        let tau = 1.0e-6_f64;

        // (1) Raw iteration count to reach `tau`.
        let mut raw_iters = 0usize;
        let mut xr = 1.0_f64;
        while (xr - dottie).abs() > tau {
            xr = xr.cos();
            raw_iters += 1;
            assert!(raw_iters <= 10_000, "raw iteration failed to converge");
        }

        // (2) Generate the raw iterate sequence and find the SMALLEST prefix
        //     length for which Aitken acceleration reaches `tau`.
        let mut iterates = vec![1.0_f64];
        let mut x = 1.0_f64;
        for _ in 0..40 {
            x = x.cos();
            iterates.push(x);
        }
        let mut aitken_terms = usize::MAX;
        for len in 3..=iterates.len() {
            let acc = aitken_accelerate(&iterates[..len], 1.0e-15).expect("aitken");
            if (acc - dottie).abs() <= tau {
                aitken_terms = len;
                break;
            }
        }
        assert!(aitken_terms != usize::MAX, "Aitken never reached tolerance");

        // The decisive property: Aitken needs fewer iterates than the raw walk.
        assert!(
            aitken_terms < raw_iters,
            "Aitken used {aitken_terms} terms, raw needed {raw_iters}"
        );

        // Sanity: at equal length, the accelerated value beats the raw term.
        let n = raw_iters.min(iterates.len() - 1);
        let acc_n = aitken_accelerate(&iterates[..=n], 1.0e-15).expect("aitken");
        let raw_err = (iterates[n] - dottie).abs();
        let acc_err = (acc_n - dottie).abs();
        assert!(
            acc_err < raw_err,
            "accelerated err {acc_err} not better than raw {raw_err}"
        );
    }

    #[test]
    fn aitken_graceful_on_zero_second_difference() {
        // Arithmetic sequence: Δ²x ≡ 0. Step must error (not divide-by-zero) and
        // the sequence variant must carry the latest term.
        let res = aitken_step(1.0, 2.0, 3.0);
        assert!(matches!(res, Err(NumericError::NumericalInstability(_))));

        let seq = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = aitken_sequence(&seq).expect("aitken seq");
        // Each window stalled → carries x2 of the window.
        assert_eq!(out, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn wynn_sums_leibniz_pi() {
        // Leibniz: π/4 = Σ_{k=0}^∞ (-1)^k / (2k+1). Painfully slow (error ~1/n).
        // Wynn's ε must beat the last partial sum by orders of magnitude.
        let n = 21usize;
        let mut partial = Vec::with_capacity(n);
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += if k % 2 == 0 {
                1.0 / (2 * k + 1) as f64
            } else {
                -1.0 / (2 * k + 1) as f64
            };
            partial.push(acc);
        }
        let target = PI / 4.0;
        let last_partial_err = (partial[n - 1] - target).abs();
        let wynn = wynn_epsilon(&partial).expect("wynn");
        let wynn_err = (wynn.estimate - target).abs();
        assert!(
            wynn_err < last_partial_err * 1.0e-4,
            "wynn err {wynn_err} not ≪ partial err {last_partial_err}"
        );
        assert!(wynn_err < 1.0e-8, "wynn err too large: {wynn_err}");
    }

    #[test]
    fn wynn_geometric_exact() {
        // Σ r^k = 1/(1-r). Wynn ε must recover it from a handful of partial sums.
        let r = 0.7_f64;
        let exact = 1.0 / (1.0 - r);
        let mut partial = Vec::new();
        let mut acc = 0.0_f64;
        let mut p = 1.0_f64;
        for _ in 0..6 {
            acc += p;
            p *= r;
            partial.push(acc);
        }
        let wynn = wynn_epsilon(&partial).expect("wynn");
        assert!(
            (wynn.estimate - exact).abs() < 1.0e-10,
            "got {}, want {exact}",
            wynn.estimate
        );
    }

    #[test]
    fn wynn_eps2_reproduces_shanks() {
        // ε_2^{(n)} (the first even column) equals the Shanks transform S_1(A_n).
        // Use a generic sequence and compare the lowest-order Wynn estimate from
        // exactly three terms against the direct Shanks formula.
        let a = [1.0_f64, 1.5, 1.75, 1.875]; // partial sums of Σ (1/2)^k
        // Shanks on the centred triple (a0,a1,a2).
        let s1 = shanks(a[0], a[1], a[2]).expect("shanks");
        // Wynn on exactly three terms returns ε_2 as its best even column.
        let wynn3 = wynn_epsilon(&a[0..3]).expect("wynn");
        assert_eq!(wynn3.order, 2, "expected ε_2 to be the best column");
        assert!(
            (wynn3.estimate - s1).abs() < 1.0e-12,
            "wynn ε_2 {} vs Shanks {s1}",
            wynn3.estimate
        );
    }

    #[test]
    fn shanks_matches_aitken() {
        // The Shanks transform centred at A_n equals one Aitken step on
        // (A_{n-1}, A_n, A_{n+1}).
        let a_prev = 2.0_f64;
        let a_curr = 2.6_f64;
        let a_next = 2.85_f64;
        let s = shanks(a_prev, a_curr, a_next).expect("shanks");
        let ai = aitken_step(a_prev, a_curr, a_next).expect("aitken");
        assert!((s - ai).abs() < 1.0e-12, "shanks {s} vs aitken {ai}");
    }

    #[test]
    fn wynn_graceful_on_constant() {
        // Constant partial sums (already converged): ε differences are zero. The
        // algorithm must stop gracefully and return the constant.
        let partial = vec![3.0_f64, 3.0, 3.0, 3.0, 3.0];
        let wynn = wynn_epsilon(&partial).expect("wynn");
        assert!(
            (wynn.estimate - 3.0).abs() < 1.0e-12,
            "got {}",
            wynn.estimate
        );
    }

    #[test]
    fn error_handling() {
        assert!(matches!(
            aitken_sequence(&[1.0, 2.0]),
            Err(NumericError::InvalidParameter(_))
        ));
        assert!(matches!(
            aitken_accelerate(&[1.0, 2.0], 1e-9),
            Err(NumericError::InvalidParameter(_))
        ));
        assert!(matches!(
            wynn_epsilon(&[1.0, 2.0]),
            Err(NumericError::InvalidParameter(_))
        ));
        assert!(matches!(
            shanks(1.0, 2.0, 3.0),
            Err(NumericError::NumericalInstability(_))
        ));
    }

    #[test]
    fn wynn_beats_aitken_on_alternating() {
        // On the slowly/alternating Leibniz series, the full Wynn table (using
        // all terms) should beat a single Aitken step.
        let n = 13usize;
        let mut partial = Vec::with_capacity(n);
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += if k % 2 == 0 {
                1.0 / (2 * k + 1) as f64
            } else {
                -1.0 / (2 * k + 1) as f64
            };
            partial.push(acc);
        }
        let target = PI / 4.0;
        let wynn = wynn_epsilon(&partial).expect("wynn");
        let aitken1 = aitken_step(partial[n - 3], partial[n - 2], partial[n - 1]).expect("aitken");
        let wynn_err = (wynn.estimate - target).abs();
        let aitken_err = (aitken1 - target).abs();
        assert!(
            wynn_err < aitken_err,
            "wynn {wynn_err} should beat single aitken {aitken_err}"
        );
    }
}
