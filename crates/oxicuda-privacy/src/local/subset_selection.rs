//! Subset-selection mechanism for locally differentially private frequency
//! oracles.
//!
//! Reference: Ye M, Barg A (2017) "Optimal Schemes for Discrete Distribution
//! Estimation under Locally Differential Privacy", IEEE Transactions on
//! Information Theory 63(11):6957–6982 — Theorem 1 derives the asymptotically
//! optimal k-subset mechanism whose communication is `log2 C(d, k)` bits and
//! whose estimation variance dominates GRR/OUE for moderate ε.
//!
//! See also: Wang T, Blocki J, Li N, Jha S (2017) "Locally Differentially
//! Private Protocols for Frequency Estimation", USENIX Security — gives the
//! same `(p, q)` correction formulas in the OUE/SUE family.
//!
//! # Protocol
//! Each user holds `x in {0, ..., d-1}` (with d >= 2) and reports a
//! `Vec<bool>` of length `d` that has exactly `k` true entries. The protocol
//! is:
//!
//! 1. With probability `p_in = k * e^ε / (k * e^ε + d - k)`, the report
//!    contains `x` (so the other `k-1` slots are chosen uniformly at random
//!    from the other `d-1` elements of the domain).
//! 2. Otherwise (probability `1 - p_in`), the report does not contain `x` and
//!    all `k` slots are chosen uniformly at random from the other `d-1`
//!    elements.
//!
//! This achieves ε-LDP (likelihood ratio `e^ε` between any two inputs).
//!
//! # Unbiased frequency estimator
//! For domain index `j`,
//!
//! ```text
//! p = Pr[j in Y | x = j] = k * e^ε / (k * e^ε + d - k)
//! q = Pr[j in Y | x ≠ j] = p · (k-1)/(d-1) + (1-p) · k/(d-1)
//! ```
//!
//! With `n` reports and `c_j = Σ_t reports[t][j] as f64`, the unbiased
//! frequency estimate (Ye-Barg 2017 Theorem 1) is
//!
//! ```text
//! f̂_j = (c_j/n − q) / (p − q).
//! ```
//!
//! ## Degeneracy
//! At `ε = 0` we have `p = k/d = q` so the estimator is undefined: the
//! aggregate call returns `InvalidParameter` if `|p − q| < 1e-12`. By the
//! configuration validator, real `ε > 0` (strictly) which yields `p > q`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

/// Numerical tolerance for the `p − q` denominator in the unbiased estimator.
const PQ_DENOM_TOL: f64 = 1e-12;

/// Configuration for the subset-selection mechanism.
#[derive(Debug, Clone, Copy)]
pub struct SubsetSelectionConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Domain size d >= 2.
    pub d: usize,
    /// Subset size k with 1 <= k < d. Callers may use
    /// `SubsetSelection::optimal_k(ε, d)` for the variance-minimising default.
    pub k: usize,
}

impl SubsetSelectionConfig {
    /// Validate and construct.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon <= 0` or non-finite.
    /// - `InvalidParameter` if `d < 2`.
    /// - `InvalidParameter` if `k == 0` or `k >= d`.
    pub fn new(epsilon: f64, d: usize, k: usize) -> PrivacyResult<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if d < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "domain size d must be >= 2, got {d}"
            )));
        }
        if k == 0 || k >= d {
            return Err(PrivacyError::InvalidParameter(format!(
                "subset size k must satisfy 1 <= k < d (k={k}, d={d})"
            )));
        }
        Ok(Self { epsilon, d, k })
    }
}

/// Subset-selection LDP frequency oracle (Ye-Barg 2017 Theorem 1).
#[derive(Debug, Clone, Copy)]
pub struct SubsetSelection {
    /// Active configuration.
    pub cfg: SubsetSelectionConfig,
}

impl SubsetSelection {
    /// Construct after revalidating the configuration.
    ///
    /// # Errors
    /// Propagates `SubsetSelectionConfig::new` errors.
    pub fn new(cfg: SubsetSelectionConfig) -> PrivacyResult<Self> {
        let cfg = SubsetSelectionConfig::new(cfg.epsilon, cfg.d, cfg.k)?;
        Ok(Self { cfg })
    }

    /// Variance-minimising subset size `k* ≈ round(d / (e^ε + 1))`, clamped
    /// into `[1, d − 1]` (Ye-Barg 2017 §III.B).
    ///
    /// At ε = 0 this returns `round(d/2)`. As ε → ∞ the optimum drops toward
    /// 0; the clamp keeps `k >= 1` so a configuration is always producible.
    #[must_use]
    pub fn optimal_k(epsilon: f64, d: usize) -> usize {
        if d < 2 {
            return 0;
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            // Hand back the centred default; callers must validate ε for the
            // configuration anyway.
            return ((d as f64) * 0.5).round() as usize;
        }
        let raw = (d as f64) / (epsilon.exp() + 1.0);
        let rounded = raw.round() as i64;
        let max_k = (d as i64) - 1;
        if rounded < 1 {
            1
        } else if rounded > max_k {
            max_k as usize
        } else {
            rounded as usize
        }
    }

    /// Active configuration.
    #[must_use]
    pub fn config(&self) -> &SubsetSelectionConfig {
        &self.cfg
    }

    /// Pr[j in Y | x = j] = k · e^ε / (k · e^ε + d − k).
    #[must_use]
    pub fn p_in(&self) -> f64 {
        let exp_eps = self.cfg.epsilon.exp();
        let k_f = self.cfg.k as f64;
        let d_minus_k = (self.cfg.d - self.cfg.k) as f64;
        let numer = k_f * exp_eps;
        numer / (numer + d_minus_k)
    }

    /// Pr[j in Y | x ≠ j] = p · (k-1)/(d-1) + (1-p) · k/(d-1).
    #[must_use]
    pub fn p_out(&self) -> f64 {
        let p = self.p_in();
        let k_f = self.cfg.k as f64;
        let d_minus_1 = (self.cfg.d - 1) as f64;
        p * (k_f - 1.0) / d_minus_1 + (1.0 - p) * k_f / d_minus_1
    }

    /// Privatise a single input `x in [0, d)`.
    ///
    /// Returns a `Vec<bool>` of length `d` with exactly `k` true entries.
    ///
    /// # Errors
    /// - `IndexOutOfRange` if `x >= cfg.d`.
    pub fn privatise(&self, x: usize, handle: &mut PrivacyHandle) -> PrivacyResult<Vec<bool>> {
        if x >= self.cfg.d {
            return Err(PrivacyError::IndexOutOfRange(x, self.cfg.d));
        }
        let d = self.cfg.d;
        let k = self.cfg.k;
        let p = self.p_in();

        // Decide whether the report should include x or not.
        let include_x = handle.rng.next_f64() < p;

        // Build the candidate pool: {0,...,d-1} \ {x}. With d <= 2^31 in
        // practice the linear-scan construction below is fine; for very large
        // d a "skip x" Fisher-Yates variant on indices 0..d-1 is also OK.
        let mut others: Vec<usize> = Vec::with_capacity(d - 1);
        for i in 0..d {
            if i != x {
                others.push(i);
            }
        }

        // Partial Fisher-Yates shuffle: produce the first `m` distinct
        // uniformly-random elements of `others` (where m = k or k-1 depending
        // on whether x is included).
        let mut result = vec![false; d];
        let m = if include_x {
            result[x] = true;
            if k == 0 { 0 } else { k - 1 }
        } else {
            k
        };
        let pool_len = others.len();
        // Partial shuffle: for i in 0..m, swap others[i] with others[uniform i..pool_len].
        for i in 0..m {
            // uniform integer in [i, pool_len).
            let raw = handle.rng.next_u64() as u128;
            let span = (pool_len - i) as u128;
            let pick = i + (raw % span) as usize;
            others.swap(i, pick);
            result[others[i]] = true;
        }

        Ok(result)
    }

    /// Aggregate `n` privatised reports into the de-biased frequency vector.
    ///
    /// # Arguments
    /// - `reports`: each report has length `d` with exactly `k` true entries.
    /// - `d`: declared domain size.
    /// - `epsilon`, `k`: protocol parameters used to recover `p` and `q`.
    ///
    /// # Errors
    /// - `EmptyInput` if `reports` is empty.
    /// - `DimensionMismatch` if any report length != `d`.
    /// - `InvalidParameter` if `|p − q| < 1e-12` (estimator degenerate).
    /// - Configuration validation errors.
    pub fn aggregate(
        reports: &[Vec<bool>],
        d: usize,
        epsilon: f64,
        k: usize,
    ) -> PrivacyResult<Vec<f64>> {
        if reports.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let cfg = SubsetSelectionConfig::new(epsilon, d, k)?;
        let mech = SubsetSelection::new(cfg)?;
        let p = mech.p_in();
        let q = mech.p_out();
        let denom = p - q;
        if denom.abs() < PQ_DENOM_TOL {
            return Err(PrivacyError::InvalidParameter(format!(
                "estimator degenerate: |p - q| = {} < tol {PQ_DENOM_TOL}",
                denom.abs()
            )));
        }
        let n_f = reports.len() as f64;
        let mut counts = vec![0.0f64; d];
        for r in reports {
            if r.len() != d {
                return Err(PrivacyError::DimensionMismatch {
                    expected: d,
                    got: r.len(),
                });
            }
            for (j, &bit) in r.iter().enumerate() {
                if bit {
                    counts[j] += 1.0;
                }
            }
        }
        let mut out = Vec::with_capacity(d);
        for c in counts {
            out.push((c / n_f - q) / denom);
        }
        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ε <= 0 -> NonPositiveEpsilon.
    #[test]
    fn test_new_nonpositive_epsilon_errors() {
        assert!(matches!(
            SubsetSelectionConfig::new(0.0, 10, 5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(-1.0, 10, 5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(f64::NAN, 10, 5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(f64::INFINITY, 10, 5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
    }

    // 2. d < 2 -> InvalidParameter.
    #[test]
    fn test_new_d_too_small_errors() {
        assert!(matches!(
            SubsetSelectionConfig::new(1.0, 0, 1),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(1.0, 1, 1),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 3. k == 0 or k >= d -> InvalidParameter.
    #[test]
    fn test_new_bad_k_errors() {
        assert!(matches!(
            SubsetSelectionConfig::new(1.0, 5, 0),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(1.0, 5, 5),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            SubsetSelectionConfig::new(1.0, 5, 9),
            Err(PrivacyError::InvalidParameter(_))
        ));
        // Boundary case: k = d - 1 is valid.
        assert!(SubsetSelectionConfig::new(1.0, 5, 4).is_ok());
    }

    // 4. privatise with x >= d -> IndexOutOfRange.
    #[test]
    fn test_privatise_x_out_of_range_errors() {
        let cfg = SubsetSelectionConfig::new(2.0, 4, 2).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 0);
        assert!(matches!(
            m.privatise(4, &mut h),
            Err(PrivacyError::IndexOutOfRange(_, _))
        ));
        assert!(matches!(
            m.privatise(99, &mut h),
            Err(PrivacyError::IndexOutOfRange(_, _))
        ));
    }

    // 5. privatise output has exactly k true entries.
    #[test]
    fn test_privatise_output_has_exactly_k_true() {
        let cfg = SubsetSelectionConfig::new(1.5, 8, 3).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 13);
        for x in 0..8usize {
            for _ in 0..50 {
                let r = m.privatise(x, &mut h).expect("ok");
                assert_eq!(r.len(), 8);
                let true_count = r.iter().filter(|b| **b).count();
                assert_eq!(true_count, 3, "expected exactly k=3 trues");
            }
        }
    }

    // 6. optimal_k(ε=0, d=10) = 5 (round(10/2)).
    #[test]
    fn test_optimal_k_epsilon_zero() {
        // Passing 0.0 (technically not allowed for cfg) — we only test the
        // static formula. The function clamps to [1, d-1].
        assert_eq!(SubsetSelection::optimal_k(0.0, 10), 5);
        assert_eq!(SubsetSelection::optimal_k(0.0, 4), 2);
        // Edge: d = 2 → round(2/2) = 1.
        assert_eq!(SubsetSelection::optimal_k(0.0, 2), 1);
    }

    // 7. optimal_k(ε=10, d=10) is clamped to 1 (e^10 huge).
    #[test]
    fn test_optimal_k_large_epsilon_clamps_to_one() {
        assert_eq!(SubsetSelection::optimal_k(10.0, 10), 1);
        assert_eq!(SubsetSelection::optimal_k(20.0, 100), 1);
    }

    // 8. aggregate with empty reports -> EmptyInput.
    #[test]
    fn test_aggregate_empty_errors() {
        let empty: Vec<Vec<bool>> = vec![];
        let r = SubsetSelection::aggregate(&empty, 4, 1.0, 2);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
    }

    // 9. aggregate dim mismatch -> DimensionMismatch.
    #[test]
    fn test_aggregate_dim_mismatch_errors() {
        let reports = vec![vec![true, false, false, false], vec![true, false]];
        let r = SubsetSelection::aggregate(&reports, 4, 1.0, 1);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }

    // 10. Unbiased frequency estimate from 5000 reports of x = 3.
    #[test]
    fn test_aggregate_unbiased_concentrated_input() {
        let d = 5usize;
        let k = 2usize;
        let eps = 2.0f64;
        let cfg = SubsetSelectionConfig::new(eps, d, k).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 314_159);
        let n = 5_000usize;
        let reports: Vec<Vec<bool>> = (0..n)
            .map(|_| m.privatise(3, &mut h).expect("ok"))
            .collect();
        let est = SubsetSelection::aggregate(&reports, d, eps, k).expect("agg");
        // f̂_3 should be ≈ 1.0; other coords ≈ 0.
        assert!(
            (est[3] - 1.0).abs() < 0.1,
            "f̂(3) = {}, expected ≈ 1.0",
            est[3]
        );
        for (i, &v) in est.iter().enumerate() {
            if i != 3 {
                assert!(v.abs() < 0.1, "f̂({i}) = {v}, expected ≈ 0");
            }
        }
    }

    // 11. Deterministic for fixed RNG seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = SubsetSelectionConfig::new(1.0, 6, 2).expect("cfg");
        let m_a = SubsetSelection::new(cfg).expect("a");
        let m_b = SubsetSelection::new(cfg).expect("b");
        let mut h_a = PrivacyHandle::new(80, 42);
        let mut h_b = PrivacyHandle::new(80, 42);
        let inputs = [0usize, 1, 2, 3, 4, 5, 0, 5, 3, 2];
        for &x in inputs.iter() {
            let a = m_a.privatise(x, &mut h_a).expect("a");
            let b = m_b.privatise(x, &mut h_b).expect("b");
            assert_eq!(a, b, "diverged at x={x}");
        }
    }

    // 12. p > q strictly when ε > 0.
    #[test]
    fn test_p_strictly_greater_than_q_for_positive_eps() {
        for &eps in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            for &(d, k) in &[(3usize, 1usize), (5, 2), (10, 3), (20, 7)] {
                let cfg = SubsetSelectionConfig::new(eps, d, k).expect("cfg");
                let m = SubsetSelection::new(cfg).expect("mech");
                let p = m.p_in();
                let q = m.p_out();
                assert!(
                    p > q + 1e-9,
                    "expected p > q for eps={eps}, d={d}, k={k}; got p={p}, q={q}"
                );
            }
        }
    }

    // 13. Frequency of Y[x] = true across many trials matches p.
    #[test]
    fn test_marginal_inclusion_of_true_input_matches_p() {
        let d = 6usize;
        let k = 2usize;
        let eps = 1.5f64;
        let cfg = SubsetSelectionConfig::new(eps, d, k).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 2718);
        let n = 10_000usize;
        let x = 4usize;
        let mut hits = 0usize;
        for _ in 0..n {
            let r = m.privatise(x, &mut h).expect("ok");
            if r[x] {
                hits += 1;
            }
        }
        let observed = hits as f64 / n as f64;
        let p = m.p_in();
        assert!(
            (observed - p).abs() < 0.02,
            "observed inclusion {observed} should ≈ p = {p}"
        );
    }

    // 14. Aggregate sums to ≈ 1 with mass on a single user.
    #[test]
    fn test_aggregate_total_mass_near_one() {
        let d = 4usize;
        let k = 1usize;
        let eps = 2.0f64;
        let cfg = SubsetSelectionConfig::new(eps, d, k).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 9001);
        let n = 8_000usize;
        // Mixed inputs: 50% x = 0, 30% x = 1, 20% x = 2; nothing on x = 3.
        let n0 = (n as f64 * 0.5) as usize;
        let n1 = (n as f64 * 0.3) as usize;
        let n2 = (n as f64 * 0.2) as usize;
        let mut reports = Vec::with_capacity(n0 + n1 + n2);
        for _ in 0..n0 {
            reports.push(m.privatise(0, &mut h).expect("ok"));
        }
        for _ in 0..n1 {
            reports.push(m.privatise(1, &mut h).expect("ok"));
        }
        for _ in 0..n2 {
            reports.push(m.privatise(2, &mut h).expect("ok"));
        }
        let est = SubsetSelection::aggregate(&reports, d, eps, k).expect("agg");
        let total: f64 = est.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.1,
            "total mass {total} should be ≈ 1"
        );
        // f̂(3) should be ≈ 0.
        assert!(est[3].abs() < 0.1, "f̂(3) = {}, expected ≈ 0", est[3]);
    }

    // 15. optimal_k boundary clamps and reasonable mid-range values.
    #[test]
    fn test_optimal_k_midrange_examples() {
        // ε = ln 2 ≈ 0.693 makes e^ε = 2 so optimal = d/3.
        let eps = (2.0f64).ln();
        // For d = 9, raw = 9 / 3 = 3 exactly.
        assert_eq!(SubsetSelection::optimal_k(eps, 9), 3);
        // For d = 6, raw = 2 exactly.
        assert_eq!(SubsetSelection::optimal_k(eps, 6), 2);
        // ε = ln 3 makes e^ε = 3 so optimal = d/4.
        let eps = (3.0f64).ln();
        // For d = 8, raw = 2 exactly.
        assert_eq!(SubsetSelection::optimal_k(eps, 8), 2);
    }

    // 16. Privatised report never includes x when noise dominates and
    //     never excludes x when ε is huge (sanity for branch correctness).
    #[test]
    fn test_branch_extremes_consistent() {
        // ε = 10 → p ≈ 1, so x is essentially always included.
        let cfg = SubsetSelectionConfig::new(10.0, 4, 1).expect("cfg");
        let m = SubsetSelection::new(cfg).expect("mech");
        let mut h = PrivacyHandle::new(80, 7);
        let mut x_in = 0usize;
        let n = 1_000usize;
        for _ in 0..n {
            let r = m.privatise(2, &mut h).expect("ok");
            if r[2] {
                x_in += 1;
            }
        }
        assert!(
            x_in as f64 / n as f64 > 0.95,
            "ε=10 should give Y[x]=true ≈ all the time, got {x_in}/{n}"
        );
    }
}
