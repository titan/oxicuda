//! Reverse-mode (backward) pass for the selective-scan / SSM linear recurrence.
//!
//! # Forward
//!
//! The selective scan computes, per independent `(channel, state)` element, the
//! scalar linear recurrence
//!
//! ```text
//! h_t = a_t · h_{t-1} + b_t ,   t = 0 … L−1 ,   h_{-1} = h_init
//! ```
//!
//! where `a_t` is the (already discretized) decay `Ā_t` and `b_t = B̄_t · u_t`
//! is the input drive.  The forward states `h_0 … h_{L-1}` are produced by
//! [`scan_forward`] (this is exactly [`crate::ssm::parallel_scan::ssm_state_scan`]
//! generalised to a non-zero initial state).
//!
//! # Backward
//!
//! Given the upstream gradient `dL/dh_t = grad_y[t]` for every step, reverse
//! mode accumulates the parameter gradients by running the adjoint recurrence
//! backwards in time:
//!
//! ```text
//! g_t        = grad_y[t] + a_{t+1} · g_{t+1}          (g_L ≡ 0)
//! dL/db_t    = g_t                                     (∂h_t/∂b_t = 1)
//! dL/da_t    = g_t · h_{t-1}                           (∂h_t/∂a_t = h_{t-1})
//! dL/dh_init = a_0 · g_0                               (∂h_0/∂h_init = a_0)
//! ```
//!
//! Because each `h_t` is *multilinear* in the parameters (every `a_t`, `b_t`
//! and `h_init` appears to the first power), the analytic gradients above match
//! a central finite-difference to within `f32` rounding — the property the unit
//! tests check.
//!
//! All arithmetic is `f32` to match the forward kernels.

use crate::error::{MambaError, MambaResult};

// ─── Gradient bundle ─────────────────────────────────────────────────────────

/// Gradients of a scalar loss w.r.t. the inputs of a single linear scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanGrads {
    /// `dL/da_t`, length `L`.
    pub grad_a: Vec<f32>,
    /// `dL/db_t`, length `L`.
    pub grad_b: Vec<f32>,
    /// `dL/dh_init` (gradient w.r.t. the initial state `h_{-1}`).
    pub grad_h_init: f32,
}

// ─── Forward ─────────────────────────────────────────────────────────────────

/// Forward pass of the scalar linear scan `h_t = a_t · h_{t-1} + b_t`.
///
/// Returns the states `[h_0, h_1, …, h_{L-1}]` (length `L`), starting from
/// `h_{-1} = h_init`.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`]        — if `a` is empty.
/// * [`MambaError::DimensionMismatch`] — if `a.len() != b.len()`.
pub fn scan_forward(a: &[f32], b: &[f32], h_init: f32) -> MambaResult<Vec<f32>> {
    if a.is_empty() {
        return Err(MambaError::EmptyInput("a"));
    }
    if a.len() != b.len() {
        return Err(MambaError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut states = Vec::with_capacity(a.len());
    let mut prev = h_init;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let cur = ai * prev + bi;
        states.push(cur);
        prev = cur;
    }
    Ok(states)
}

// ─── Backward ────────────────────────────────────────────────────────────────

/// Backward (reverse-mode) pass of the scalar linear scan.
///
/// # Arguments
///
/// * `a`      — decay sequence `[a_0 … a_{L-1}]`.
/// * `b`      — drive sequence `[b_0 … b_{L-1}]`.
/// * `h_init` — initial state `h_{-1}`.
/// * `grad_y` — upstream gradient `dL/dh_t` for each step, length `L`.
///
/// # Returns
///
/// [`ScanGrads`] holding `dL/da`, `dL/db` and `dL/dh_init`.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`]        — if `a` is empty.
/// * [`MambaError::DimensionMismatch`] — if `a`, `b` and `grad_y` lengths differ.
pub fn scan_backward(a: &[f32], b: &[f32], h_init: f32, grad_y: &[f32]) -> MambaResult<ScanGrads> {
    if a.is_empty() {
        return Err(MambaError::EmptyInput("a"));
    }
    if a.len() != b.len() {
        return Err(MambaError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.len() != grad_y.len() {
        return Err(MambaError::DimensionMismatch {
            expected: a.len(),
            got: grad_y.len(),
        });
    }

    let l = a.len();
    // Recompute (or accept) the forward states; needed for dL/da_t = g_t·h_{t-1}.
    let states = scan_forward(a, b, h_init)?;

    let mut grad_a = vec![0.0_f32; l];
    let mut grad_b = vec![0.0_f32; l];

    // `carried` holds a_{t+1} · g_{t+1}; it is 0 at t = L-1 (no successor) and,
    // after the loop completes (t = 0 processed), equals a_0 · g_0 = dL/dh_init.
    let mut carried = 0.0_f32;
    for t in (0..l).rev() {
        let g_t = grad_y[t] + carried;
        grad_b[t] = g_t;
        let h_prev = if t == 0 { h_init } else { states[t - 1] };
        grad_a[t] = g_t * h_prev;
        carried = a[t] * g_t;
    }
    let grad_h_init = carried;

    Ok(ScanGrads {
        grad_a,
        grad_b,
        grad_h_init,
    })
}

// ─── Batched backward ────────────────────────────────────────────────────────

/// Gradients for a batch of `n_scans` independent scalar scans, each of length
/// `scan_len`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedScanGrads {
    /// `dL/da`, row-major `[n_scans × scan_len]`.
    pub grad_a: Vec<f32>,
    /// `dL/db`, row-major `[n_scans × scan_len]`.
    pub grad_b: Vec<f32>,
    /// `dL/dh_init` per scan, length `n_scans`.
    pub grad_h_init: Vec<f32>,
}

/// Backward pass over a batch of independent scalar scans.
///
/// The selective scan applies the same scalar recurrence independently to every
/// `(channel, state)` element; this convenience wrapper runs [`scan_backward`]
/// over `n_scans` rows laid out row-major (`scan_len` consecutive steps each).
///
/// # Arguments
///
/// * `a`, `b`, `grad_y` — row-major `[n_scans × scan_len]`.
/// * `h_init`           — initial state per scan, length `n_scans`.
/// * `scan_len`         — steps per scan (`> 0`).
///
/// # Errors
///
/// * [`MambaError::InvalidSeqLen`]     — if `scan_len == 0`.
/// * [`MambaError::DimensionMismatch`] — if any buffer length is inconsistent
///   with `n_scans · scan_len` (or `h_init.len() != n_scans`).
pub fn scan_backward_batched(
    a: &[f32],
    b: &[f32],
    h_init: &[f32],
    grad_y: &[f32],
    scan_len: usize,
) -> MambaResult<BatchedScanGrads> {
    if scan_len == 0 {
        return Err(MambaError::InvalidSeqLen(0));
    }
    let n_scans = h_init.len();
    let expected = n_scans * scan_len;
    for got in [a.len(), b.len(), grad_y.len()] {
        if got != expected {
            return Err(MambaError::DimensionMismatch { expected, got });
        }
    }

    let mut grad_a = vec![0.0_f32; expected];
    let mut grad_b = vec![0.0_f32; expected];
    let mut grad_h_init = vec![0.0_f32; n_scans];

    for (s, &h0) in h_init.iter().enumerate() {
        let lo = s * scan_len;
        let hi = lo + scan_len;
        let g = scan_backward(&a[lo..hi], &b[lo..hi], h0, &grad_y[lo..hi])?;
        grad_a[lo..hi].copy_from_slice(&g.grad_a);
        grad_b[lo..hi].copy_from_slice(&g.grad_b);
        grad_h_init[s] = g.grad_h_init;
    }

    Ok(BatchedScanGrads {
        grad_a,
        grad_b,
        grad_h_init,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Linear loss `L = Σ_t coef_t · h_t`; `grad_y = coef` (constant) isolates
    /// the scan gradient (`h` is still nonlinear in the `a`'s).
    fn loss(a: &[f32], b: &[f32], h_init: f32, coef: &[f32]) -> f32 {
        let h = scan_forward(a, b, h_init).expect("forward");
        h.iter().zip(coef.iter()).map(|(&hv, &cv)| hv * cv).sum()
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    #[test]
    fn forward_manual_recurrence() {
        // h_init = 0.4
        // h0 = 0.5·0.4 + 1.0 = 1.2
        // h1 = 0.5·1.2 + 2.0 = 2.6
        // h2 = 0.5·2.6 + 0.5 = 1.8
        let a = [0.5_f32, 0.5, 0.5];
        let b = [1.0_f32, 2.0, 0.5];
        let h = scan_forward(&a, &b, 0.4).expect("forward");
        assert!((h[0] - 1.2).abs() < 1e-6, "h0={}", h[0]);
        assert!((h[1] - 2.6).abs() < 1e-6, "h1={}", h[1]);
        assert!((h[2] - 1.8).abs() < 1e-6, "h2={}", h[2]);
    }

    #[test]
    fn forward_matches_zero_init_state_scan() {
        // With h_init = 0 this must equal the existing ssm_state_scan.
        use crate::ssm::parallel_scan::ssm_state_scan;
        let a = [0.9_f32, 0.8, 0.7, 0.6];
        let b = [0.1_f32, -0.2, 0.3, 0.05];
        let ours = scan_forward(&a, &b, 0.0).expect("forward");
        let theirs = ssm_state_scan(&a, &b).expect("state scan");
        for (i, (&x, &y)) in ours.iter().zip(theirs.iter()).enumerate() {
            assert!((x - y).abs() < 1e-6, "mismatch at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn forward_errors() {
        assert!(matches!(
            scan_forward(&[], &[], 0.0),
            Err(MambaError::EmptyInput(_))
        ));
        assert!(matches!(
            scan_forward(&[0.5, 0.5], &[1.0], 0.0),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    // ── Backward: analytic L = 1 case ─────────────────────────────────────────

    #[test]
    fn backward_single_step_analytic() {
        // L = 1: h0 = a0·h_init + b0, grad_y = [g].
        // dL/da0 = g·h_init ; dL/db0 = g ; dL/dh_init = a0·g.
        let a = [0.7_f32];
        let b = [0.3_f32];
        let h_init = 1.5_f32;
        let g = 2.0_f32;
        let grads = scan_backward(&a, &b, h_init, &[g]).expect("backward");
        assert!((grads.grad_a[0] - g * h_init).abs() < 1e-6);
        assert!((grads.grad_b[0] - g).abs() < 1e-6);
        assert!((grads.grad_h_init - a[0] * g).abs() < 1e-6);
    }

    // ── Backward: finite-difference check (load-bearing) ──────────────────────

    #[test]
    fn backward_matches_finite_difference() {
        let a = [0.6_f32, 0.3, 0.85, 0.5, 0.2];
        let b = [0.1_f32, -0.2, 0.3, 0.05, -0.1];
        let h_init = 0.4_f32;
        let coef = [0.7_f32, -0.3, 1.1, 0.2, -0.5]; // grad_y = coef
        let l = a.len();

        let grads = scan_backward(&a, &b, h_init, &coef).expect("backward");

        let eps = 1e-2_f32;
        let tol = 1e-3_f32;

        // dL/da_t
        for i in 0..l {
            let mut ap = a;
            let mut am = a;
            ap[i] += eps;
            am[i] -= eps;
            let num = (loss(&ap, &b, h_init, &coef) - loss(&am, &b, h_init, &coef)) / (2.0 * eps);
            assert!(
                (grads.grad_a[i] - num).abs() < tol,
                "grad_a[{i}]: analytic {} vs numeric {num}",
                grads.grad_a[i]
            );
        }

        // dL/db_t
        for i in 0..l {
            let mut bp = b;
            let mut bm = b;
            bp[i] += eps;
            bm[i] -= eps;
            let num = (loss(&a, &bp, h_init, &coef) - loss(&a, &bm, h_init, &coef)) / (2.0 * eps);
            assert!(
                (grads.grad_b[i] - num).abs() < tol,
                "grad_b[{i}]: analytic {} vs numeric {num}",
                grads.grad_b[i]
            );
        }

        // dL/dh_init
        let num_h =
            (loss(&a, &b, h_init + eps, &coef) - loss(&a, &b, h_init - eps, &coef)) / (2.0 * eps);
        assert!(
            (grads.grad_h_init - num_h).abs() < tol,
            "grad_h_init: analytic {} vs numeric {num_h}",
            grads.grad_h_init
        );
    }

    #[test]
    fn backward_random_finite_difference() {
        // Same FD check but with randomised, longer scans (stable a ∈ (0,1)).
        let mut rng = LcgRng::new(31);
        let l = 12_usize;
        let a: Vec<f32> = (0..l).map(|_| rng.next_f32() * 0.8 + 0.1).collect(); // (0.1,0.9)
        let mut b = vec![0.0_f32; l];
        rng.fill_normal(&mut b);
        let mut coef = vec![0.0_f32; l];
        rng.fill_normal(&mut coef);
        let h_init = 0.25_f32;

        let grads = scan_backward(&a, &b, h_init, &coef).expect("backward");

        let eps = 1e-2_f32;
        let tol = 5e-3_f32;
        for i in 0..l {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[i] += eps;
            am[i] -= eps;
            let num = (loss(&ap, &b, h_init, &coef) - loss(&am, &b, h_init, &coef)) / (2.0 * eps);
            assert!(
                (grads.grad_a[i] - num).abs() < tol,
                "grad_a[{i}]: {} vs {num}",
                grads.grad_a[i]
            );

            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[i] += eps;
            bm[i] -= eps;
            let num_b = (loss(&a, &bp, h_init, &coef) - loss(&a, &bm, h_init, &coef)) / (2.0 * eps);
            assert!(
                (grads.grad_b[i] - num_b).abs() < tol,
                "grad_b[{i}]: {} vs {num_b}",
                grads.grad_b[i]
            );
        }
    }

    // ── Backward: shapes & finiteness ─────────────────────────────────────────

    #[test]
    fn backward_shapes_and_finite() {
        let mut rng = LcgRng::new(5);
        let l = 8_usize;
        let a: Vec<f32> = (0..l).map(|_| rng.next_f32() * 0.9).collect();
        let mut b = vec![0.0_f32; l];
        let mut gy = vec![0.0_f32; l];
        rng.fill_normal(&mut b);
        rng.fill_normal(&mut gy);
        let grads = scan_backward(&a, &b, 0.1, &gy).expect("backward");
        assert_eq!(grads.grad_a.len(), l);
        assert_eq!(grads.grad_b.len(), l);
        assert!(grads.grad_a.iter().all(|v| v.is_finite()));
        assert!(grads.grad_b.iter().all(|v| v.is_finite()));
        assert!(grads.grad_h_init.is_finite());
    }

    #[test]
    fn backward_errors() {
        assert!(matches!(
            scan_backward(&[], &[], 0.0, &[]),
            Err(MambaError::EmptyInput(_))
        ));
        assert!(matches!(
            scan_backward(&[0.5, 0.5], &[1.0, 2.0], 0.0, &[1.0]),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    // ── Batched ───────────────────────────────────────────────────────────────

    #[test]
    fn batched_matches_per_scan() {
        let mut rng = LcgRng::new(17);
        let n_scans = 3_usize;
        let scan_len = 6_usize;
        let total = n_scans * scan_len;
        let a: Vec<f32> = (0..total).map(|_| rng.next_f32() * 0.8 + 0.1).collect();
        let mut b = vec![0.0_f32; total];
        let mut gy = vec![0.0_f32; total];
        rng.fill_normal(&mut b);
        rng.fill_normal(&mut gy);
        let h_init = vec![0.2_f32, -0.1, 0.5];

        let batched = scan_backward_batched(&a, &b, &h_init, &gy, scan_len).expect("batched");
        assert_eq!(batched.grad_a.len(), total);
        assert_eq!(batched.grad_b.len(), total);
        assert_eq!(batched.grad_h_init.len(), n_scans);

        for (s, &h0) in h_init.iter().enumerate() {
            let lo = s * scan_len;
            let hi = lo + scan_len;
            let single = scan_backward(&a[lo..hi], &b[lo..hi], h0, &gy[lo..hi]).expect("single");
            for (k, (&ba, &bb)) in single.grad_a.iter().zip(single.grad_b.iter()).enumerate() {
                assert!((batched.grad_a[lo + k] - ba).abs() < 1e-7);
                assert!((batched.grad_b[lo + k] - bb).abs() < 1e-7);
            }
            assert!((batched.grad_h_init[s] - single.grad_h_init).abs() < 1e-7);
        }
    }

    #[test]
    fn batched_errors() {
        assert!(matches!(
            scan_backward_batched(&[], &[], &[], &[], 0),
            Err(MambaError::InvalidSeqLen(0))
        ));
        // h_init implies n_scans = 2, scan_len = 3 → expected 6, but a has 5.
        assert!(matches!(
            scan_backward_batched(&[0.0; 5], &[0.0; 6], &[0.0; 2], &[0.0; 6], 3),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }
}
