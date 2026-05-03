//! Diagonal Plus Low Rank (DPLR) parameterization for S4 SSMs.
//!
//! S4 parameterizes the state-transition matrix `A` as:
//!
//! ```text
//! A = diag(Lambda) − P · Q^T
//! ```
//!
//! where `Lambda`, `P`, `Q` ∈ ℝ^N are real vectors (the general complex-valued
//! case reduces to real arithmetic for HiPPO-LegS due to its symmetric
//! structure).  This rank-1 update of a diagonal matrix admits an efficient
//! evaluation of the SSM convolution kernel via the Cauchy kernel identity.
//!
//! ## SSM kernel computation
//!
//! Given ZOH discretization, the convolution kernel is:
//!
//! ```text
//! K[k] = Σ_n  C[n] · exp(λ_n · Δ)^k · B̄_n
//! ```
//!
//! where the ZOH B̄ is applied once at `k = 0`:
//!
//! ```text
//! B̄_n = (exp(λ_n · Δ) − 1) / λ_n    (L'Hôpital limit Δ when λ_n ≈ 0)
//! ```

use crate::error::{MambaError, MambaResult};
use crate::s4::hippo::hippo_nplr;

// ─── Near-zero threshold for lambda (same as ZOH in discretize.rs) ──────────
const LAMBDA_ZERO_THRESHOLD: f32 = 1e-6_f32;

// ─── Dplr struct ─────────────────────────────────────────────────────────────

/// Diagonal Plus Low Rank (DPLR) representation of an SSM A matrix.
///
/// Represents `A = diag(lambda) − P · Q^T` (rank-1 update of a diagonal).
///
/// This is the canonical parameterization used by S4.  The HiPPO-LegS matrix
/// decomposes exactly into this form via the NPLR identities.
#[derive(Debug, Clone)]
pub struct Dplr {
    /// Diagonal component, length `N`.
    pub lambda: Vec<f32>,
    /// Left low-rank vector, length `N`.
    pub p: Vec<f32>,
    /// Right low-rank vector, length `N`.
    pub q: Vec<f32>,
}

impl Dplr {
    /// Construct a `Dplr` from the HiPPO-LegS NPLR decomposition for order `n`.
    ///
    /// # Errors
    ///
    /// [`MambaError::InvalidSsmOrder`] if `n == 0`.
    pub fn from_hippo(n: usize) -> MambaResult<Self> {
        let (lambda, p, q) = hippo_nplr(n)?;
        Ok(Self { lambda, p, q })
    }

    /// Construct a `Dplr` from explicit component vectors.
    ///
    /// All three vectors must have the same non-zero length.
    ///
    /// # Errors
    ///
    /// * [`MambaError::EmptyInput`] — if any vector is empty.
    /// * [`MambaError::DimensionMismatch`] — if lengths differ.
    pub fn new(lambda: Vec<f32>, p: Vec<f32>, q: Vec<f32>) -> MambaResult<Self> {
        if lambda.is_empty() {
            return Err(MambaError::EmptyInput("lambda"));
        }
        if p.len() != lambda.len() {
            return Err(MambaError::DimensionMismatch {
                expected: lambda.len(),
                got: p.len(),
            });
        }
        if q.len() != lambda.len() {
            return Err(MambaError::DimensionMismatch {
                expected: lambda.len(),
                got: q.len(),
            });
        }
        Ok(Self { lambda, p, q })
    }

    /// Return the state dimension `N`.
    #[inline]
    pub fn order(&self) -> usize {
        self.lambda.len()
    }

    /// Reconstruct the full `N×N` A matrix (row-major) from the DPLR components.
    ///
    /// `A[i, j] = lambda[i] · δ_{ij} − p[i] · q[j]`
    ///
    /// This is primarily intended for testing and debugging; for production use
    /// the DPLR components are consumed directly by the kernel computation.
    pub fn to_dense(&self) -> Vec<f32> {
        let n = self.order();
        let mut a = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let delta_ij = if i == j { 1.0_f32 } else { 0.0_f32 };
                a[i * n + j] = self.lambda[i] * delta_ij - self.p[i] * self.q[j];
            }
        }
        a
    }

    /// Compute the SSM convolution kernel of length `seq_len`.
    ///
    /// For each time step `k ∈ [0, seq_len)`:
    ///
    /// ```text
    /// K[k] = Σ_n  c[n] · ā_n^k · b̄_n
    /// ```
    ///
    /// where:
    /// - `ā_n = exp(lambda[n] · delta)` — ZOH discrete eigenvalue
    /// - `b̄_n = (ā_n − 1) / lambda[n]` — ZOH B̄ factor (limit `delta` when `lambda[n] ≈ 0`)
    ///
    /// The kernel `K` can be convolved with the input to obtain the SSM output.
    ///
    /// # Arguments
    ///
    /// * `c`       — Output projection vector `[N]`; must match `self.order()`.
    /// * `delta`   — Positive discretization time-step `Δ > 0`.
    /// * `seq_len` — Desired kernel length `L > 0`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::NonPositiveDelta`]   — if `delta ≤ 0`.
    /// * [`MambaError::InvalidSeqLen`]      — if `seq_len == 0`.
    /// * [`MambaError::DimensionMismatch`]  — if `c.len() ≠ N`.
    pub fn ssm_kernel(&self, c: &[f32], delta: f32, seq_len: usize) -> MambaResult<Vec<f32>> {
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        let n = self.order();
        if c.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }

        // Pre-compute per-mode scalars: ā_n and b̄_n · c_n
        // These are the "mode weights" that scale the k-step contribution.
        let mut a_bar = Vec::with_capacity(n);
        let mut cb_bar = Vec::with_capacity(n); // c[n] * b_bar[n]

        for (c_val, lam) in c.iter().zip(self.lambda.iter()) {
            let ab = (lam * delta).exp();
            a_bar.push(ab);

            let b_bar = if lam.abs() < LAMBDA_ZERO_THRESHOLD {
                // L'Hôpital limit: (exp(λΔ)−1)/λ → Δ as λ→0
                delta
            } else {
                (ab - 1.0) / lam
            };
            cb_bar.push(c_val * b_bar);
        }

        // K[k] = Σ_n  cb_bar[n] · ā_n^k
        // Maintain a_bar_power[n] = ā_n^k, updated by multiplication each step.
        let mut a_bar_power: Vec<f32> = vec![1.0_f32; n]; // ā_n^0 = 1
        let mut kernel = Vec::with_capacity(seq_len);

        for _ in 0..seq_len {
            let k_val: f32 = cb_bar
                .iter()
                .zip(a_bar_power.iter())
                .map(|(&cb, &ap)| cb * ap)
                .sum();
            kernel.push(k_val);
            // Advance a_bar_power: a_bar_power[n] *= ā_n
            for i in 0..n {
                a_bar_power[i] *= a_bar[i];
            }
        }

        Ok(kernel)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── Construction ─────────────────────────────────────────────────────────

    /// from_hippo(4) produces three vectors of length 4.
    #[test]
    fn dplr_from_hippo_n4() {
        let dplr = Dplr::from_hippo(4).expect("n=4 is valid");
        assert_eq!(dplr.lambda.len(), 4);
        assert_eq!(dplr.p.len(), 4);
        assert_eq!(dplr.q.len(), 4);
        assert_eq!(dplr.order(), 4);
    }

    /// Explicit construction with matched vectors succeeds.
    #[test]
    fn dplr_new_valid() {
        let lambda = vec![-0.5_f32, -1.5, -2.5];
        let p = vec![0.707_f32, 1.225, 1.581];
        let q = p.clone();
        let dplr = Dplr::new(lambda.clone(), p.clone(), q.clone()).expect("valid");
        assert_eq!(dplr.order(), 3);
        for i in 0..3 {
            assert!((dplr.lambda[i] - lambda[i]).abs() < EPS);
            assert!((dplr.p[i] - p[i]).abs() < EPS);
            assert!((dplr.q[i] - q[i]).abs() < EPS);
        }
    }

    /// Mismatched p length returns DimensionMismatch.
    #[test]
    fn dplr_new_mismatched_len() {
        let err = Dplr::new(
            vec![-1.0_f32, -2.0],
            vec![1.0_f32], // wrong: length 1, not 2
            vec![1.0_f32, 1.0],
        )
        .expect_err("mismatched should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    /// Empty lambda returns EmptyInput.
    #[test]
    fn dplr_new_empty() {
        let err = Dplr::new(vec![], vec![], vec![]).expect_err("empty should fail");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    // ── order ─────────────────────────────────────────────────────────────────

    /// order() returns the correct N.
    #[test]
    fn dplr_order() {
        for n in [1, 4, 8, 16, 32] {
            let dplr = Dplr::from_hippo(n).expect("valid n");
            assert_eq!(dplr.order(), n, "order() should equal n={n}");
        }
    }

    // ── to_dense ──────────────────────────────────────────────────────────────

    /// n=1: A[0,0] = lambda[0] - p[0]*q[0].
    #[test]
    fn dplr_to_dense_n1() {
        let lambda = vec![-0.5_f32];
        let p = vec![0.707_f32];
        let q = vec![0.707_f32];
        let dplr = Dplr::new(lambda.clone(), p.clone(), q.clone()).expect("valid");
        let a = dplr.to_dense();
        assert_eq!(a.len(), 1);
        let expected = lambda[0] - p[0] * q[0];
        assert!(
            (a[0] - expected).abs() < EPS,
            "A[0,0]={} expected {}",
            a[0],
            expected
        );
    }

    /// HiPPO DPLR diagonal: A[i,i] ≈ lambda[i] - p[i]^2.
    #[test]
    fn dplr_to_dense_diagonal_dominant() {
        let n = 6;
        let dplr = Dplr::from_hippo(n).expect("valid n");
        let a = dplr.to_dense();
        for i in 0..n {
            let diag = a[i * n + i];
            let expected = dplr.lambda[i] - dplr.p[i] * dplr.q[i];
            assert!(
                (diag - expected).abs() < EPS,
                "diagonal mismatch at i={i}: dense={diag}, formula={expected}"
            );
        }
    }

    // ── ssm_kernel ────────────────────────────────────────────────────────────

    /// Kernel length equals seq_len.
    #[test]
    fn ssm_kernel_shape() {
        let dplr = Dplr::from_hippo(4).expect("valid n");
        let c = vec![1.0_f32; 4];
        for &seq_len in &[1_usize, 8, 16, 64] {
            let kernel = dplr.ssm_kernel(&c, 0.01, seq_len).expect("valid args");
            assert_eq!(kernel.len(), seq_len, "kernel length should be {seq_len}");
        }
    }

    /// All kernel values are finite for stable lambda (all negative).
    #[test]
    fn ssm_kernel_finite() {
        let dplr = Dplr::from_hippo(8).expect("valid n");
        let c = vec![1.0_f32; 8];
        let kernel = dplr.ssm_kernel(&c, 0.001, 32).expect("valid args");
        for (k, &v) in kernel.iter().enumerate() {
            assert!(v.is_finite(), "kernel[{k}]={v} not finite");
        }
    }

    /// For a stable system (negative lambda), the kernel must decay:
    /// |K[0]| > |K[L-1]| for reasonably long sequences.
    #[test]
    fn ssm_kernel_decays_with_stable_lambda() {
        // Use a single-mode system for clarity: lambda=-1, p=q=1, c=1.
        // K[k] = c * b_bar * a_bar^k  where a_bar = exp(-delta), a_bar < 1.
        let lambda = vec![-1.0_f32];
        let p = vec![1.0_f32];
        let q = vec![1.0_f32];
        let dplr = Dplr::new(lambda, p, q).expect("valid");
        let c = vec![1.0_f32];
        let kernel = dplr.ssm_kernel(&c, 0.1, 20).expect("valid args");
        // K[0] = c * b_bar * 1  (a_bar^0 = 1)
        // K[19] = c * b_bar * a_bar^19  which is small
        assert!(
            kernel[0].abs() > kernel[19].abs(),
            "kernel should decay: K[0]={}, K[19]={}",
            kernel[0],
            kernel[19]
        );
    }

    /// Non-positive delta returns NonPositiveDelta.
    #[test]
    fn ssm_kernel_non_positive_delta() {
        let dplr = Dplr::from_hippo(4).expect("valid n");
        let c = vec![1.0_f32; 4];
        let err = dplr.ssm_kernel(&c, 0.0, 8).expect_err("should fail");
        assert!(matches!(err, MambaError::NonPositiveDelta(_)));
    }

    /// seq_len=0 returns InvalidSeqLen.
    #[test]
    fn ssm_kernel_zero_seq_len() {
        let dplr = Dplr::from_hippo(4).expect("valid n");
        let c = vec![1.0_f32; 4];
        let err = dplr.ssm_kernel(&c, 0.01, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// c.len() != N returns DimensionMismatch.
    #[test]
    fn ssm_kernel_c_mismatch() {
        let dplr = Dplr::from_hippo(4).expect("valid n");
        let c = vec![1.0_f32; 3]; // wrong length
        let err = dplr.ssm_kernel(&c, 0.01, 8).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
