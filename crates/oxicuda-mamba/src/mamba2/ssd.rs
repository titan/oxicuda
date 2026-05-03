//! State Space Duality (SSD) — the core algorithm of Mamba-2.
//!
//! # Theory (Dao & Gu, 2024)
//!
//! The SSD framework shows that selective SSMs are equivalent to structured
//! matrix-vector products.  For a length-L sequence with scalar per-element
//! decay `A[t]` and B/C vectors of dimension N, the output is:
//!
//! ```text
//! y[i] = Σ_{j ≤ i} C[i] · (Π_{k=j+1}^{i} A[k]) · B[j] · x[j]
//! ```
//!
//! where `C[i] · B[j] = Σ_n C[i,n] * B[j,n]` is the inner product in state
//! space, and the Π product of decays collapses to a scalar prefix product.
//!
//! This module provides:
//! - [`ssd_naive`] — Direct O(L²·N) computation of the structured matrix.
//! - [`ssd_recurrent`] — Equivalent O(L·N) hidden-state recurrence.
//! - [`verify_ssd_equivalence`] — Check both forms agree within tolerance.

use crate::error::{MambaError, MambaResult};

// ─── ssd_naive ───────────────────────────────────────────────────────────────

/// Compute the SSD/SSM output `y = M · x` naively in O(L²·N) time.
///
/// The output matrix `M` is lower-triangular with semi-separable structure:
///
/// ```text
/// M[i, j] = C[i] · (Π_{k=j+1..=i} A[k]) · B[j]   for i ≥ j
/// M[i, j] = 0                                        for i < j
/// ```
///
/// For each output position `i`:
/// ```text
/// y[i] = Σ_{j≤i} (C[i] · B[j]) * (Π_{k=j+1..=i} A[k]) * x[j]
/// ```
///
/// # Arguments
///
/// * `a_seq`     — Per-timestep decay scalars, length `L`.  Should lie in `(0, 1)`.
/// * `b_seq`     — B vectors `[L × N]`, row-major: `b_seq[t * N + n]`.
/// * `c_seq`     — C vectors `[L × N]`, row-major: `c_seq[t * N + n]`.
/// * `x`         — Scalar input per timestep, length `L`.
/// * `seq_len`   — Sequence length `L`.
/// * `state_dim` — State dimension `N`.
///
/// # Errors
///
/// * [`MambaError::InvalidSeqLen`]    — if `seq_len == 0`.
/// * [`MambaError::InvalidSsmOrder`]  — if `state_dim == 0`.
/// * [`MambaError::DimensionMismatch`] — if slice lengths don't match.
pub fn ssd_naive(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    seq_len: usize,
    state_dim: usize,
) -> MambaResult<Vec<f32>> {
    if seq_len == 0 {
        return Err(MambaError::InvalidSeqLen(seq_len));
    }
    if state_dim == 0 {
        return Err(MambaError::InvalidSsmOrder(state_dim));
    }
    validate_ssd_inputs(a_seq, b_seq, c_seq, x, seq_len, state_dim)?;

    let mut y = vec![0.0_f32; seq_len];

    for (i, y_i) in y.iter_mut().enumerate() {
        // Accumulate contributions from all j ≤ i.
        // Walk backward from j=i down to j=0 accumulating the decay product.
        // At j=i the decay product Π_{k=i+1}^{i} A[k] is empty → 1.0.
        let mut decay_product = 1.0_f32;
        // cb_dot = C[i] · B[j] inner product in R^N
        for j in (0..=i).rev() {
            if j < i {
                // Extend decay product: multiply by A[j+1] (the decay entering step j+1).
                // The product is Π_{k=j+1..=i} A[k].
                // Walking backward: when we drop j by 1, we include A[j+1].
                decay_product *= a_seq[j + 1];
            }
            let cb = dot_product(c_seq, b_seq, i, j, state_dim);
            *y_i += cb * decay_product * x[j];
        }
    }

    Ok(y)
}

// ─── ssd_recurrent ───────────────────────────────────────────────────────────

/// Compute the SSD/SSM output via the hidden-state recurrence in O(L·N) time.
///
/// The recurrence is:
/// ```text
/// h[t] = A[t] * h[t-1] + B[t] * x[t]   (vector of dimension N)
/// y[t] = C[t] · h[t]                     (dot product → scalar)
/// ```
///
/// This is algebraically equivalent to [`ssd_naive`] but avoids the O(L²)
/// inner loop by maintaining the running state `h[t] ∈ Rᴺ`.
///
/// # Arguments
///
/// Same as [`ssd_naive`].
///
/// # Errors
///
/// Same as [`ssd_naive`].
pub fn ssd_recurrent(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    seq_len: usize,
    state_dim: usize,
) -> MambaResult<Vec<f32>> {
    if seq_len == 0 {
        return Err(MambaError::InvalidSeqLen(seq_len));
    }
    if state_dim == 0 {
        return Err(MambaError::InvalidSsmOrder(state_dim));
    }
    validate_ssd_inputs(a_seq, b_seq, c_seq, x, seq_len, state_dim)?;

    let mut y = vec![0.0_f32; seq_len];
    // Hidden state h ∈ Rᴺ, initialised to zero (h[-1] = 0).
    let mut h = vec![0.0_f32; state_dim];

    for t in 0..seq_len {
        let a_t = a_seq[t];
        let x_t = x[t];
        let b_offset = t * state_dim;
        let c_offset = t * state_dim;

        // h[t] = A[t] * h[t-1] + B[t] * x[t]
        let mut y_t = 0.0_f32;
        for n in 0..state_dim {
            h[n] = a_t * h[n] + b_seq[b_offset + n] * x_t;
            y_t += c_seq[c_offset + n] * h[n];
        }
        y[t] = y_t;
    }

    Ok(y)
}

// ─── verify_ssd_equivalence ──────────────────────────────────────────────────

/// Verify that [`ssd_naive`] and [`ssd_recurrent`] produce identical output.
///
/// Runs both algorithms and checks that every element satisfies
/// `|naive[t] - recurrent[t]| ≤ tol`.  Returns `Ok(true)` if they agree,
/// `Ok(false)` if any element exceeds the tolerance.
///
/// # Errors
///
/// Propagates any error from [`ssd_naive`] or [`ssd_recurrent`].
pub fn verify_ssd_equivalence(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    seq_len: usize,
    state_dim: usize,
    tol: f32,
) -> MambaResult<bool> {
    let naive = ssd_naive(a_seq, b_seq, c_seq, x, seq_len, state_dim)?;
    let recurrent = ssd_recurrent(a_seq, b_seq, c_seq, x, seq_len, state_dim)?;

    let agrees = naive
        .iter()
        .zip(recurrent.iter())
        .all(|(&n, &r)| (n - r).abs() <= tol);
    Ok(agrees)
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Compute the dot product `C[i] · B[j]` in R^N from row-major flat slices.
///
/// `c_seq[i * N + n]` and `b_seq[j * N + n]` for `n = 0..N`.
#[inline]
fn dot_product(c_seq: &[f32], b_seq: &[f32], i: usize, j: usize, n: usize) -> f32 {
    let c_offset = i * n;
    let b_offset = j * n;
    let mut acc = 0.0_f32;
    for k in 0..n {
        acc += c_seq[c_offset + k] * b_seq[b_offset + k];
    }
    acc
}

/// Validate that all slice lengths match the expected shapes.
fn validate_ssd_inputs(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    seq_len: usize,
    state_dim: usize,
) -> MambaResult<()> {
    if a_seq.len() != seq_len {
        return Err(MambaError::DimensionMismatch {
            expected: seq_len,
            got: a_seq.len(),
        });
    }
    let bc_len = seq_len * state_dim;
    if b_seq.len() != bc_len {
        return Err(MambaError::DimensionMismatch {
            expected: bc_len,
            got: b_seq.len(),
        });
    }
    if c_seq.len() != bc_len {
        return Err(MambaError::DimensionMismatch {
            expected: bc_len,
            got: c_seq.len(),
        });
    }
    if x.len() != seq_len {
        return Err(MambaError::DimensionMismatch {
            expected: seq_len,
            got: x.len(),
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_stable_inputs(
        rng: &mut LcgRng,
        seq_len: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // a in (0.1, 0.9) for stability
        let a: Vec<f32> = (0..seq_len).map(|_| 0.1 + rng.next_f32() * 0.8).collect();
        let mut b = vec![0.0_f32; seq_len * state_dim];
        let mut c = vec![0.0_f32; seq_len * state_dim];
        let mut x = vec![0.0_f32; seq_len];
        rng.fill_normal(&mut b);
        rng.fill_normal(&mut c);
        rng.fill_normal(&mut x);
        (a, b, c, x)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ssd_naive tests
    // ─────────────────────────────────────────────────────────────────────────

    /// L=1, N=1: y[0] = C[0]*B[0]*x[0]  (decay product over empty range = 1).
    #[test]
    fn ssd_naive_length_1() {
        let a = vec![0.5_f32];
        let b = vec![2.0_f32]; // B[0, 0] = 2
        let c = vec![3.0_f32]; // C[0, 0] = 3
        let x = vec![4.0_f32]; // x[0]    = 4
        let y = ssd_naive(&a, &b, &c, &x, 1, 1).expect("length-1 ssd_naive");
        // y[0] = C[0]·B[0] * (empty product = 1) * x[0] = (3*2)*1*4 = 24
        assert!((y[0] - 24.0_f32).abs() < 1e-5, "y[0]={}", y[0]);
    }

    /// Output length must equal seq_len.
    #[test]
    fn ssd_naive_output_shape() {
        let mut rng = LcgRng::new(1);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 7, 3);
        let y = ssd_naive(&a, &b, &c, &x, 7, 3).expect("ssd_naive_output_shape");
        assert_eq!(y.len(), 7);
    }

    /// All outputs must be finite for stable (a ∈ (0,1)) random inputs.
    #[test]
    fn ssd_naive_output_finite() {
        let mut rng = LcgRng::new(42);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 16, 4);
        let y = ssd_naive(&a, &b, &c, &x, 16, 4).expect("ssd_naive_output_finite");
        for (t, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{t}]={v} not finite");
        }
    }

    /// Causality: swapping a future x[j] (j > i) must not change y[i].
    ///
    /// We set x[3] = 0 or 99.0 and verify y[0], y[1], y[2] are identical.
    #[test]
    fn ssd_naive_causal() {
        let seq_len = 5_usize;
        let state_dim = 2_usize;
        let mut rng = LcgRng::new(7);
        let (a, b, c, mut x) = make_stable_inputs(&mut rng, seq_len, state_dim);

        let y_orig = ssd_naive(&a, &b, &c, &x, seq_len, state_dim).expect("original");
        // Perturb future positions
        x[3] = 99.0;
        x[4] = -99.0;
        let y_perturbed = ssd_naive(&a, &b, &c, &x, seq_len, state_dim).expect("perturbed");

        // Past outputs (indices 0..3) must be unaffected
        for i in 0..3 {
            assert!(
                (y_orig[i] - y_perturbed[i]).abs() < 1e-6,
                "y[{i}] changed after perturbing future: orig={} perturbed={}",
                y_orig[i],
                y_perturbed[i]
            );
        }
    }

    /// Error on empty sequence.
    #[test]
    fn ssd_empty_seq() {
        let err = ssd_naive(&[], &[], &[], &[], 0, 1).expect_err("should fail on empty");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// Error on zero state_dim.
    #[test]
    fn ssd_naive_zero_state_dim() {
        let err = ssd_naive(&[0.5], &[], &[], &[1.0], 1, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    /// L=3, N=1, a=0: each step is independent (no coupling across time).
    #[test]
    fn ssd_naive_zero_a_independent_steps() {
        // With a = 0: M[i,j] = C[i]*B[j] for j==i, 0 otherwise (decay Π=0 for j<i).
        let a = vec![0.0_f32; 3];
        let b = vec![1.0_f32, 2.0, 3.0]; // N=1
        let c = vec![1.0_f32, 1.0, 1.0];
        let x = vec![1.0_f32, 1.0, 1.0];
        // For j < i: decay product starts at a[j+1] = 0 → entire term is 0
        // For j == i: decay = 1, so y[i] = c[i]*b[i]*x[i]
        let y = ssd_naive(&a, &b, &c, &x, 3, 1).expect("zero-a");
        assert!((y[0] - 1.0).abs() < 1e-6, "y[0]={}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-6, "y[1]={}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-6, "y[2]={}", y[2]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ssd_recurrent tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Output length must equal seq_len.
    #[test]
    fn ssd_recurrent_output_shape() {
        let mut rng = LcgRng::new(2);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 9, 2);
        let y = ssd_recurrent(&a, &b, &c, &x, 9, 2).expect("ssd_recurrent_output_shape");
        assert_eq!(y.len(), 9);
    }

    /// All outputs must be finite.
    #[test]
    fn ssd_recurrent_output_finite() {
        let mut rng = LcgRng::new(55);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 12, 3);
        let y = ssd_recurrent(&a, &b, &c, &x, 12, 3).expect("ssd_recurrent_output_finite");
        for (t, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{t}]={v} not finite");
        }
    }

    /// a = 0 everywhere: each step is independent, h[t] = B[t] * x[t].
    #[test]
    fn ssd_recurrent_zero_a() {
        let seq_len = 5_usize;
        let state_dim = 2_usize;
        let a = vec![0.0_f32; seq_len];
        // B[t, n] = t+1 for all n.  C[t, n] = 1 for all n.  x[t] = 1.
        let b: Vec<f32> = (0..seq_len)
            .flat_map(|t| vec![(t + 1) as f32; state_dim])
            .collect();
        let c = vec![1.0_f32; seq_len * state_dim];
        let x = vec![1.0_f32; seq_len];

        let y = ssd_recurrent(&a, &b, &c, &x, seq_len, state_dim).expect("zero-a recurrent");
        // h[t] = B[t] * x[t] (no prior state), y[t] = C[t]·h[t] = N*(t+1)
        for (t, &y_val) in y.iter().enumerate() {
            let expected = state_dim as f32 * (t + 1) as f32;
            assert!(
                (y_val - expected).abs() < 1e-5,
                "y[{t}]={y_val} expected {expected}",
            );
        }
    }

    /// a = 1 everywhere, N=1: h[t] = cumulative sum of B[t]*x[t].
    #[test]
    fn ssd_recurrent_unit_a() {
        let seq_len = 5_usize;
        let state_dim = 1_usize;
        let a = vec![1.0_f32; seq_len];
        let b = vec![1.0_f32; seq_len]; // B[t,0] = 1
        let c = vec![1.0_f32; seq_len]; // C[t,0] = 1
        let x: Vec<f32> = (1..=seq_len as u32).map(|v| v as f32).collect(); // 1,2,3,4,5

        let y = ssd_recurrent(&a, &b, &c, &x, seq_len, state_dim).expect("unit-a recurrent");
        // h[t] = h[t-1] + x[t], y[t] = h[t]
        let expected = [1.0_f32, 3.0, 6.0, 10.0, 15.0];
        for t in 0..seq_len {
            assert!(
                (y[t] - expected[t]).abs() < 1e-5,
                "y[{t}]={} expected {}",
                y[t],
                expected[t]
            );
        }
    }

    /// Error on empty sequence for recurrent form.
    #[test]
    fn ssd_recurrent_empty_seq() {
        let err = ssd_recurrent(&[], &[], &[], &[], 0, 1).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Equivalence tests
    // ─────────────────────────────────────────────────────────────────────────

    /// L=4, N=1: naive and recurrent agree within 1e-5.
    #[test]
    fn ssd_equivalence_l4_n1() {
        let mut rng = LcgRng::new(100);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 4, 1);
        let agrees = verify_ssd_equivalence(&a, &b, &c, &x, 4, 1, 1e-5).expect("equivalence l4_n1");
        assert!(agrees, "naive and recurrent disagree for L=4, N=1");
    }

    /// L=8, N=2: naive and recurrent agree within 1e-5.
    #[test]
    fn ssd_equivalence_l8_n2() {
        let mut rng = LcgRng::new(200);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 8, 2);
        let agrees = verify_ssd_equivalence(&a, &b, &c, &x, 8, 2, 1e-5).expect("equivalence l8_n2");
        assert!(agrees, "naive and recurrent disagree for L=8, N=2");
    }

    /// L=16, N=4: naive and recurrent agree within 1e-5.
    #[test]
    fn ssd_equivalence_l16_n4() {
        let mut rng = LcgRng::new(300);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 16, 4);
        let agrees =
            verify_ssd_equivalence(&a, &b, &c, &x, 16, 4, 1e-5).expect("equivalence l16_n4");
        assert!(agrees, "naive and recurrent disagree for L=16, N=4");
    }

    /// L=1: both forms agree on the trivial single-step case.
    #[test]
    fn ssd_equivalence_l1_n1() {
        let a = vec![0.7_f32];
        let b = vec![1.5_f32];
        let c = vec![0.8_f32];
        let x = vec![2.0_f32];
        let agrees = verify_ssd_equivalence(&a, &b, &c, &x, 1, 1, 1e-6).expect("equivalence l1_n1");
        assert!(agrees, "naive and recurrent disagree for L=1, N=1");
    }

    /// L=32, N=8: agreement for larger sequence and state.
    #[test]
    fn ssd_equivalence_l32_n8() {
        let mut rng = LcgRng::new(999);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 32, 8);
        let agrees =
            verify_ssd_equivalence(&a, &b, &c, &x, 32, 8, 2e-4).expect("equivalence l32_n8");
        assert!(agrees, "naive and recurrent disagree for L=32, N=8");
    }
}
