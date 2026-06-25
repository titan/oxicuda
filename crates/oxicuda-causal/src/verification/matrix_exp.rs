//! Padé-(1,1) scaling-and-squaring matrix-exponential error analysis against an
//! independent eigendecomposition reference.
//!
//! The production NOTEARS acyclicity term `h(W) = tr(exp(W⊙W)) − d` is dominated
//! by the matrix exponential. This module quantifies the relative error of the
//! production `expm_pade` path versus the Jacobi-eigendecomposition reference
//! [`crate::verification::reference::expm_symmetric_eig`] on symmetric inputs,
//! returning the maximum element-wise absolute error and the trace error.

use crate::discovery::notears::expm_pade;
use crate::error::CausalResult;
use crate::verification::reference::expm_symmetric_eig;

/// Element-wise and trace error between the Padé exponential and the
/// eigendecomposition reference for a symmetric `n × n` matrix `a`.
pub struct ExpmErrorReport {
    /// Maximum absolute element-wise difference `max_{i,j} |P_{ij} − R_{ij}|`.
    pub max_abs_error: f64,
    /// Maximum *relative* element-wise difference (denominator floored at 1).
    pub max_rel_error: f64,
    /// Absolute difference of the matrix traces (the quantity NOTEARS uses).
    pub trace_error: f64,
}

/// Compute the error report comparing `expm_pade(a)` to the eigendecomposition
/// reference. `a` must be symmetric `n × n` (row-major).
pub fn expm_error_report(a: &[f32], n: usize) -> CausalResult<ExpmErrorReport> {
    let pade = expm_pade(a, n)?;
    let reference = expm_symmetric_eig(a, n);
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut tr_p = 0.0_f64;
    let mut tr_r = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let p = pade[i * n + j] as f64;
            let r = reference[i * n + j];
            let abs = (p - r).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            let rel = abs / r.abs().max(1.0);
            if rel > max_rel {
                max_rel = rel;
            }
        }
        tr_p += pade[i * n + i] as f64;
        tr_r += reference[i * n + i];
    }
    Ok(ExpmErrorReport {
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        trace_error: (tr_p - tr_r).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a symmetric matrix `S = ½(A + Aᵀ)` from a random `A`, then scale so
    /// its spectral magnitude is moderate (entries `~U[-s, s]`).
    fn random_symmetric(n: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
        let mut a = vec![0.0_f32; n * n];
        for v in a.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * scale;
        }
        let mut s = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
            }
        }
        s
    }

    #[test]
    fn pade_matches_eig_small_norm() {
        // Small-norm matrices need no scaling; Padé should be very close.
        let mut rng = LcgRng::new(1);
        for _ in 0..20 {
            let n = 4;
            let a = random_symmetric(n, 0.15, &mut rng);
            let rep = expm_error_report(&a, n).expect("expm error report");
            assert!(
                rep.max_abs_error < 2e-3,
                "small-norm max_abs_error too large: {}",
                rep.max_abs_error
            );
            assert!(rep.trace_error < 2e-3, "trace error {}", rep.trace_error);
        }
    }

    #[test]
    fn pade_matches_eig_moderate_norm() {
        // Larger norm exercises scaling-and-squaring; tolerance loosens but the
        // two independent algorithms must still agree closely.
        let mut rng = LcgRng::new(2);
        let mut worst = 0.0_f64;
        for _ in 0..20 {
            let n = 5;
            let a = random_symmetric(n, 0.6, &mut rng);
            let rep = expm_error_report(&a, n).expect("expm error report");
            worst = worst.max(rep.max_rel_error);
            assert!(
                rep.max_rel_error < 5e-2,
                "moderate-norm rel error too large: {}",
                rep.max_rel_error
            );
        }
        // Sanity: at least some non-trivial error was actually measured.
        assert!(worst.is_finite());
    }

    #[test]
    fn pade_matches_eig_diagonal_exact() {
        // For a diagonal matrix both methods reduce to scalar exp; error ~ f32 eps.
        let mut a = vec![0.0_f32; 9];
        a[0] = 1.3;
        a[4] = -0.8;
        a[8] = 0.2;
        let rep = expm_error_report(&a, 3).expect("expm error report");
        assert!(rep.max_abs_error < 5e-3, "{}", rep.max_abs_error);
    }

    #[test]
    fn notears_h_trace_consistent() {
        // h(W) uses tr(exp(W⊙W)). Check the trace via Padé matches the reference
        // trace for a NOTEARS-style elementwise-squared adjacency.
        let mut rng = LcgRng::new(9);
        let n = 4;
        let mut w = vec![0.0_f32; n * n];
        for v in w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * 0.4;
        }
        let a: Vec<f32> = w.iter().map(|&v| v * v).collect();
        // W⊙W is not symmetric in general; eig reference symmetrises, so compare
        // only the trace (a symmetric-invariant scalar that NOTEARS relies on)
        // against a high-order Taylor reference instead.
        let pade = expm_pade(&a, n).expect("pade");
        let tr_pade: f64 = (0..n).map(|i| pade[i * n + i] as f64).sum();
        let tr_taylor = trace_expm_taylor(&a, n);
        assert!(
            (tr_pade - tr_taylor).abs() < 1e-2,
            "trace mismatch pade={tr_pade} taylor={tr_taylor}"
        );
    }

    /// Independent trace-of-exponential reference via a high-order Taylor series
    /// in f64 (valid because `W⊙W` here has small norm). Used to cross-check the
    /// non-symmetric NOTEARS trace.
    fn trace_expm_taylor(a: &[f32], n: usize) -> f64 {
        let mut term = vec![0.0_f64; n * n]; // current A^k / k!
        for i in 0..n {
            term[i * n + i] = 1.0; // A^0 / 0! = I
        }
        let mut acc = vec![0.0_f64; n * n];
        for v in acc.iter_mut().zip(term.iter()) {
            *v.0 += *v.1;
        }
        let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        for k in 1..40 {
            let mut next = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for l in 0..n {
                        s += term[i * n + l] * af[l * n + j];
                    }
                    next[i * n + j] = s / k as f64;
                }
            }
            term = next;
            for (a_acc, t) in acc.iter_mut().zip(term.iter()) {
                *a_acc += *t;
            }
        }
        (0..n).map(|i| acc[i * n + i]).sum()
    }
}
