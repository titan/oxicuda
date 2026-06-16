//! MinT (Minimum Trace) reconciliation for hierarchical/grouped time series.
//!
//! Wickramasuriya, Athanasopoulos & Hyndman (2019), "Optimal Forecast
//! Reconciliation for Hierarchical and Grouped Time Series Through Trace
//! Minimization", JASA 114(526):804–819.
//!
//! Given a hierarchy described by a *summing matrix* `S` (`n_total × n_bottom`,
//! 0/1 entries, with the bottom-level series forming an identity block) and a
//! vector of incoherent *base* forecasts `ŷ` (`n_total`), the reconciled
//! forecasts are
//!
//! ```text
//! ỹ = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ ŷ
//! ```
//!
//! which minimises the trace of the reconciled error covariance. `W` is the
//! base-forecast error covariance; this implementation supports
//! [`MintMethod::Ols`] (`W = I`) and [`MintMethod::WlsDiag`] (`W = diag(w)`).
//!
//! Because `ỹ = S b̃` for the reconciled bottom forecasts `b̃`, the result is
//! **structurally coherent**: every aggregate equals the sum of its children
//! regardless of the weighting.
//!
//! Computation is performed in `f64` (Cholesky factorisation of the
//! `n_bottom × n_bottom` normal matrix) and cast back to `f32` at the boundary.

use crate::error::{TsError, TsResult};

// ── Method ───────────────────────────────────────────────────────────────────

/// Weighting scheme for the reconciliation.
#[derive(Debug, Clone, PartialEq)]
pub enum MintMethod {
    /// Ordinary least squares: `W = I`.
    Ols,
    /// Weighted least squares with a diagonal `W = diag(w)`.
    ///
    /// `w` holds the `n_total` positive base-error variances (one per series,
    /// in the same row order as `S`).
    WlsDiag(Vec<f32>),
}

// ── Reconciler ───────────────────────────────────────────────────────────────

/// Hierarchical forecast reconciler.
///
/// The expensive part — forming and factorising `Sᵀ W⁻¹ S` — is done once in
/// [`new`](Self::new); [`reconcile`](Self::reconcile) then only performs
/// triangular solves and matrix–vector products per forecast horizon.
#[derive(Debug, Clone)]
pub struct MintReconciler {
    n_total: usize,
    n_bottom: usize,
    /// Summing matrix `S` (`n_total × n_bottom`, row-major) in `f64`.
    s: Vec<f64>,
    /// Inverse diagonal of `W` (`1/w_i`), length `n_total`.
    winv: Vec<f64>,
    /// Lower Cholesky factor `L` of `Sᵀ W⁻¹ S` (`n_bottom × n_bottom`, row-major).
    chol: Vec<f64>,
}

impl MintReconciler {
    /// Build a reconciler from the summing matrix and weighting method.
    ///
    /// `s_matrix` is the row-major `n_total × n_bottom` summing matrix.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `n_total == 0` or `n_bottom == 0`.
    /// - [`TsError::ShapeMismatch`] when `n_bottom > n_total`, when
    ///   `s_matrix.len() != n_total * n_bottom`, or when the resulting normal
    ///   matrix is rank-deficient.
    /// - [`TsError::WeightShapeMismatch`] when WLS weights are mis-sized.
    /// - [`TsError::NonFinite`] when a weight is non-positive or non-finite.
    pub fn new(
        s_matrix: &[f32],
        n_total: usize,
        n_bottom: usize,
        method: MintMethod,
    ) -> TsResult<Self> {
        if n_total == 0 || n_bottom == 0 {
            return Err(TsError::EmptyInput {
                msg: "n_total and n_bottom must be > 0".to_string(),
            });
        }
        if n_bottom > n_total {
            return Err(TsError::ShapeMismatch {
                msg: format!("n_bottom={n_bottom} cannot exceed n_total={n_total}"),
            });
        }
        if s_matrix.len() != n_total * n_bottom {
            return Err(TsError::ShapeMismatch {
                msg: format!(
                    "summing matrix has {} entries, expected {}",
                    s_matrix.len(),
                    n_total * n_bottom
                ),
            });
        }

        let winv = match &method {
            MintMethod::Ols => vec![1.0_f64; n_total],
            MintMethod::WlsDiag(w) => {
                if w.len() != n_total {
                    return Err(TsError::WeightShapeMismatch {
                        msg: format!("WLS weights have {} entries, expected {}", w.len(), n_total),
                    });
                }
                let mut inv = vec![0.0_f64; n_total];
                for (slot, &wi) in inv.iter_mut().zip(w.iter()) {
                    let wf = f64::from(wi);
                    if !wf.is_finite() || wf <= 0.0 {
                        return Err(TsError::NonFinite);
                    }
                    *slot = 1.0 / wf;
                }
                inv
            }
        };

        let s: Vec<f64> = s_matrix.iter().map(|&v| f64::from(v)).collect();

        // Normal matrix M = Sᵀ W⁻¹ S (n_bottom × n_bottom, symmetric PSD).
        let mut m = vec![0.0_f64; n_bottom * n_bottom];
        for i in 0..n_total {
            let wi = winv[i];
            if wi == 0.0 {
                continue;
            }
            let row = &s[i * n_bottom..i * n_bottom + n_bottom];
            for j in 0..n_bottom {
                let sij = row[j];
                if sij == 0.0 {
                    continue;
                }
                let prefix = wi * sij;
                for k in 0..n_bottom {
                    m[j * n_bottom + k] += prefix * row[k];
                }
            }
        }

        let chol = cholesky_with_jitter(&m, n_bottom).ok_or_else(|| TsError::ShapeMismatch {
            msg: "summing matrix is rank-deficient (SᵀW⁻¹S not positive-definite)".to_string(),
        })?;

        Ok(Self {
            n_total,
            n_bottom,
            s,
            winv,
            chol,
        })
    }

    /// Number of series across all levels of the hierarchy.
    #[must_use]
    pub fn n_total(&self) -> usize {
        self.n_total
    }

    /// Number of bottom-level series.
    #[must_use]
    pub fn n_bottom(&self) -> usize {
        self.n_bottom
    }

    /// Reconcile base forecasts so that the hierarchy is coherent.
    ///
    /// `base` is either a single forecast vector of length `n_total` or a
    /// flattened `n_total × h` block (series-major: `base[i*h + c]` is series
    /// `i`, horizon `c`). The output has the same layout/length and is exactly
    /// coherent: every aggregate equals the sum of its children.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `base` is empty.
    /// - [`TsError::ShapeMismatch`] when `base.len()` is not a positive multiple
    ///   of `n_total`.
    /// - [`TsError::NonFinite`] when `base` contains a non-finite value.
    pub fn reconcile(&self, base: &[f32]) -> TsResult<Vec<f32>> {
        if base.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "base forecasts must be non-empty".to_string(),
            });
        }
        if base.len() % self.n_total != 0 {
            return Err(TsError::ShapeMismatch {
                msg: format!(
                    "base length {} is not a multiple of n_total={}",
                    base.len(),
                    self.n_total
                ),
            });
        }
        if base.iter().any(|v| !v.is_finite()) {
            return Err(TsError::NonFinite);
        }

        let h = base.len() / self.n_total;
        let nt = self.n_total;
        let nb = self.n_bottom;
        let mut out = vec![0.0_f32; base.len()];

        for c in 0..h {
            // rhs = Sᵀ W⁻¹ base_col (length n_bottom).
            let mut rhs = vec![0.0_f64; nb];
            for i in 0..nt {
                let weighted = self.winv[i] * f64::from(base[i * h + c]);
                if weighted == 0.0 {
                    continue;
                }
                let row = &self.s[i * nb..i * nb + nb];
                for (j, &sij) in row.iter().enumerate() {
                    if sij != 0.0 {
                        rhs[j] += sij * weighted;
                    }
                }
            }

            // Solve (Sᵀ W⁻¹ S) b̃ = rhs.
            let b_bottom = chol_solve(&self.chol, &rhs, nb);

            // Reconciled = S b̃ (structurally coherent).
            for i in 0..nt {
                let row = &self.s[i * nb..i * nb + nb];
                let val: f64 = row
                    .iter()
                    .zip(b_bottom.iter())
                    .map(|(&sij, &bj)| sij * bj)
                    .sum();
                out[i * h + c] = val as f32;
            }
        }

        Ok(out)
    }
}

// ── Linear algebra ─────────────────────────────────────────────────────────────

/// Lower-triangular Cholesky factor `L` (`L Lᵀ = a`) of an SPD `n × n` matrix.
///
/// Returns `None` if `a` is not positive-definite.
fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Cholesky factorisation with progressive diagonal jitter for robustness.
fn cholesky_with_jitter(m: &[f64], n: usize) -> Option<Vec<f64>> {
    if let Some(l) = cholesky_lower(m, n) {
        return Some(l);
    }
    let trace: f64 = (0..n).map(|i| m[i * n + i]).sum();
    let mut jitter = (trace / n as f64).max(1.0) * 1e-9;
    for _ in 0..8 {
        let mut a = m.to_vec();
        for i in 0..n {
            a[i * n + i] += jitter;
        }
        if let Some(l) = cholesky_lower(&a, n) {
            return Some(l);
        }
        jitter *= 10.0;
    }
    None
}

/// Solve `L Lᵀ x = b` given the lower Cholesky factor `l` (`n × n`).
fn chol_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Forward substitution: L z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * z[k];
        }
        z[i] = s / l[i * n + i];
    }
    // Back substitution: Lᵀ x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple two-level hierarchy: 1 top = sum of 2 bottoms.
    /// Rows: [top; bottom_0; bottom_1]; columns: [bottom_0, bottom_1].
    fn small_s() -> Vec<f32> {
        vec![
            1.0, 1.0, // top = b0 + b1
            1.0, 0.0, // b0
            0.0, 1.0, // b1
        ]
    }

    /// Three-bottom hierarchy with two middle aggregates and one grand total.
    /// Rows: [total; A=b0+b1; B=b2; b0; b1; b2]; columns: [b0,b1,b2].
    fn medium_s() -> Vec<f32> {
        vec![
            1.0, 1.0, 1.0, // total
            1.0, 1.0, 0.0, // A
            0.0, 0.0, 1.0, // B
            1.0, 0.0, 0.0, // b0
            0.0, 1.0, 0.0, // b1
            0.0, 0.0, 1.0, // b2
        ]
    }

    fn assert_coherent(s: &[f32], rec: &[f32], n_total: usize, n_bottom: usize) {
        // Bottom values are the last n_bottom rows.
        let bottom_start = n_total - n_bottom;
        for i in 0..n_total {
            let row = &s[i * n_bottom..i * n_bottom + n_bottom];
            let expected: f32 = (0..n_bottom).map(|j| row[j] * rec[bottom_start + j]).sum();
            assert!(
                (rec[i] - expected).abs() < 1e-4,
                "row {i} incoherent: {} vs {}",
                rec[i],
                expected
            );
        }
    }

    #[test]
    fn mint_ols_is_coherent() {
        let s = small_s();
        let r = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("new");
        // Incoherent base: top says 10, bottoms say 4 and 5 (sum 9).
        let base = [10.0_f32, 4.0, 5.0];
        let rec = r.reconcile(&base).expect("reconcile");
        assert_coherent(&s, &rec, 3, 2);
    }

    #[test]
    fn mint_coherent_input_unchanged() {
        let s = small_s();
        let r = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("new");
        // Already coherent: top = 7 = 3 + 4.
        let base = [7.0_f32, 3.0, 4.0];
        let rec = r.reconcile(&base).expect("reconcile");
        for (a, b) in base.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-4, "coherent input changed: {a} -> {b}");
        }
    }

    #[test]
    fn mint_ols_splits_discrepancy() {
        let s = small_s();
        let r = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("new");
        // top=10, bottoms 4 & 5 (sum 9). Discrepancy = 1.
        let base = [10.0_f32, 4.0, 5.0];
        let rec = r.reconcile(&base).expect("reconcile");
        assert_coherent(&s, &rec, 3, 2);
        // OLS spreads the +1 top surplus: each bottom rises by 1/3, top falls by 1/3.
        assert!((rec[1] - (4.0 + 1.0 / 3.0)).abs() < 1e-3, "b0={}", rec[1]);
        assert!((rec[2] - (5.0 + 1.0 / 3.0)).abs() < 1e-3, "b1={}", rec[2]);
        assert!((rec[0] - (10.0 - 1.0 / 3.0)).abs() < 1e-3, "top={}", rec[0]);
        // Reconciled top equals sum of reconciled bottoms.
        assert!((rec[0] - (rec[1] + rec[2])).abs() < 1e-4);
    }

    #[test]
    fn mint_wls_differs_but_coherent() {
        let s = small_s();
        let ols = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("ols");
        // Heavier weight (larger variance) on bottom 1 → it absorbs less.
        let wls =
            MintReconciler::new(&s, 3, 2, MintMethod::WlsDiag(vec![1.0, 1.0, 4.0])).expect("wls");
        let base = [10.0_f32, 4.0, 5.0];
        let rec_ols = ols.reconcile(&base).expect("ols rec");
        let rec_wls = wls.reconcile(&base).expect("wls rec");
        assert_coherent(&s, &rec_ols, 3, 2);
        assert_coherent(&s, &rec_wls, 3, 2);
        // The two solutions are genuinely different.
        let diff: f32 = rec_ols
            .iter()
            .zip(rec_wls.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "OLS and WLS should differ, diff={diff}");
    }

    #[test]
    fn mint_three_level_coherent() {
        let s = medium_s();
        let r = MintReconciler::new(&s, 6, 3, MintMethod::Ols).expect("new");
        let base = [20.0_f32, 9.0, 8.0, 3.0, 5.0, 7.0]; // incoherent everywhere
        let rec = r.reconcile(&base).expect("reconcile");
        assert_coherent(&s, &rec, 6, 3);
    }

    #[test]
    fn mint_multi_horizon() {
        let s = small_s();
        let r = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("new");
        // h = 2, series-major layout: [top_h0, top_h1, b0_h0, b0_h1, b1_h0, b1_h1].
        let base = [10.0_f32, 12.0, 4.0, 5.0, 5.0, 6.0];
        let rec = r.reconcile(&base).expect("reconcile");
        assert_eq!(rec.len(), 6);
        // Check coherence per horizon column.
        for c in 0..2 {
            assert!(
                (rec[c] - (rec[2 + c] + rec[4 + c])).abs() < 1e-4,
                "horizon {c} incoherent"
            );
        }
    }

    #[test]
    fn mint_wls_weights_recover_ols_when_equal() {
        let s = small_s();
        let ols = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("ols");
        let wls =
            MintReconciler::new(&s, 3, 2, MintMethod::WlsDiag(vec![2.0, 2.0, 2.0])).expect("wls");
        let base = [10.0_f32, 4.0, 5.0];
        let a = ols.reconcile(&base).expect("a");
        let b = wls.reconcile(&base).expect("b");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "equal weights should match OLS: {x} vs {y}"
            );
        }
    }

    #[test]
    fn mint_err_bad_s_shape() {
        let s = vec![1.0_f32, 1.0, 1.0]; // 3 entries, expect 3*2=6
        assert!(matches!(
            MintReconciler::new(&s, 3, 2, MintMethod::Ols).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn mint_err_bad_base_shape() {
        let s = small_s();
        let r = MintReconciler::new(&s, 3, 2, MintMethod::Ols).expect("new");
        let base = [1.0_f32, 2.0, 3.0, 4.0]; // 4 not a multiple of 3
        assert!(matches!(
            r.reconcile(&base).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn mint_err_bad_weights() {
        let s = small_s();
        assert!(matches!(
            MintReconciler::new(&s, 3, 2, MintMethod::WlsDiag(vec![1.0, 1.0])).unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }

    #[test]
    fn mint_err_zero_dims() {
        let s = small_s();
        assert!(matches!(
            MintReconciler::new(&s, 0, 2, MintMethod::Ols).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn mint_err_nonpositive_weight() {
        let s = small_s();
        assert!(matches!(
            MintReconciler::new(&s, 3, 2, MintMethod::WlsDiag(vec![1.0, 0.0, 1.0])).unwrap_err(),
            TsError::NonFinite
        ));
    }
}
