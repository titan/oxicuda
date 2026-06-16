//! SLOPE — Sorted L1 Penalised Estimation (Bogdan et al. 2015).
//!
//! "SLOPE — Adaptive Variable Selection via Convex Optimization."
//!
//! Minimises `½ ||y - X β||² + Σ_i λ_i |β|_{(i)}` where `|β|_{(1)} ≥ … ≥ |β|_{(p)}`
//! are the decreasing-order absolute values of β.
//!
//! Solved via proximal-gradient (ISTA) with the sorted-L1 proximal operator,
//! which is computed using PAVA isotonic regression on the absolute-value permutation.

use crate::error::{CsError, CsResult};

/// Configuration for SLOPE.
#[derive(Debug, Clone)]
pub struct SlopeConfig {
    /// Non-increasing penalty sequence λ₁ ≥ λ₂ ≥ … ≥ λ_p ≥ 0.
    pub lambdas: Vec<f64>,
    /// Maximum number of proximal-gradient iterations.
    pub max_iter: usize,
    /// Relative convergence tolerance on ‖Δβ‖/‖β‖.
    pub tol: f64,
}

/// SLOPE estimator for linear regression.
#[derive(Debug, Clone)]
pub struct Slope {
    coefficients: Vec<f64>,
    config: SlopeConfig,
    fitted: bool,
}

impl Slope {
    /// Create a new SLOPE estimator from a configuration.
    ///
    /// Returns `Err` if the lambda sequence is empty or not non-increasing.
    pub fn new(config: SlopeConfig) -> CsResult<Self> {
        if config.lambdas.is_empty() {
            return Err(CsError::InvalidParameter(
                "lambdas must be non-empty".into(),
            ));
        }
        // Validate non-increasing requirement.
        for w in config.lambdas.windows(2) {
            if w[1] > w[0] {
                return Err(CsError::InvalidParameter(
                    "lambdas must be non-increasing".into(),
                ));
            }
        }
        for &lam in &config.lambdas {
            if lam < 0.0 {
                return Err(CsError::InvalidParameter("all lambdas must be >= 0".into()));
            }
        }
        Ok(Self {
            coefficients: Vec::new(),
            config,
            fitted: false,
        })
    }

    /// Fit SLOPE on design matrix `x` of shape [n × p] row-major and response `y` of length n.
    ///
    /// Uses proximal-gradient (ISTA) with step size `1/L` where `L` is estimated via
    /// 30 power-method iterations on `X^T X`.
    pub fn fit(&mut self, x: &[f64], y: &[f64], n: usize, p: usize) -> CsResult<()> {
        if x.len() != n * p {
            return Err(CsError::ShapeMismatch {
                expected: vec![n, p],
                got: vec![x.len()],
            });
        }
        if y.len() != n {
            return Err(CsError::DimensionMismatch { a: y.len(), b: n });
        }
        // Trim / pad lambda sequence to length p.
        let lam_p = build_lambda_sequence(&self.config.lambdas, p);

        // Estimate Lipschitz constant L ≈ ‖X^T X‖_op via power method.
        let lip = estimate_lipschitz(x, n, p);
        let step = if lip > 0.0 { 1.0 / lip } else { 1.0 };

        let mut beta = vec![0.0_f64; p];
        for _ in 0..self.config.max_iter {
            // Gradient: X^T (X β - y)
            let xb = matvec_nn(x, &beta, n, p);
            let mut residual = vec![0.0_f64; n];
            for i in 0..n {
                residual[i] = xb[i] - y[i];
            }
            let grad = matvec_tn(x, &residual, n, p);

            let mut u = vec![0.0_f64; p];
            for j in 0..p {
                u[j] = beta[j] - step * grad[j];
            }
            let beta_new = sorted_l1_prox(&u, &lam_p.iter().map(|&v| v * step).collect::<Vec<_>>());

            // Convergence check.
            let delta: f64 = beta_new
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let norm_b: f64 = beta.iter().map(|v| v * v).sum::<f64>().sqrt();
            beta = beta_new;
            if delta / norm_b.max(1.0e-300) < self.config.tol {
                break;
            }
        }
        self.coefficients = beta;
        self.fitted = true;
        Ok(())
    }

    /// Predict response for new observations `x` of shape [n × p] row-major.
    ///
    /// Returns `Err` if the model has not been fitted.
    pub fn predict(&self, x: &[f64], n: usize, p: usize) -> CsResult<Vec<f64>> {
        if !self.fitted {
            return Err(CsError::InvalidParameter(
                "model has not been fitted".into(),
            ));
        }
        if x.len() != n * p {
            return Err(CsError::ShapeMismatch {
                expected: vec![n, p],
                got: vec![x.len()],
            });
        }
        if self.coefficients.len() != p {
            return Err(CsError::DimensionMismatch {
                a: self.coefficients.len(),
                b: p,
            });
        }
        Ok(matvec_nn(x, &self.coefficients, n, p))
    }

    /// Return the fitted coefficient vector.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Return whether the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Generate a Benjamini-Hochberg (BH) lambda sequence of length `p`.
    ///
    /// `q` is the FDR level (e.g. 0.05), `sigma` is the noise standard deviation.
    ///
    /// λ_i = σ * Φ^{-1}(1 - i*q / (2p)) for i = 1..=p (non-increasing).
    pub fn bh_lambdas(p: usize, q: f64, sigma: f64) -> Vec<f64> {
        if p == 0 {
            return Vec::new();
        }
        (1..=p)
            .map(|i| {
                let prob = 1.0 - (i as f64) * q / (2.0 * p as f64);
                let prob_clamped = prob.clamp(0.5, 1.0 - 1.0e-12);
                sigma * normal_quantile(prob_clamped)
            })
            .collect()
    }
}

/// Sorted-L1 proximal operator.
///
/// Given a vector `v` and a non-increasing penalty sequence `lambdas` (same length as `v`),
/// computes `prox_{‖·‖_sorted}(v)`.
///
/// Algorithm:
/// 1. Record original signs and work on absolute values sorted descending.
/// 2. Subtract lambda sequence.
/// 3. Apply isotonic regression (PAVA) to enforce decreasing order of result.
/// 4. Clip to ≥ 0.
/// 5. Reconstruct original permutation and signs.
pub fn sorted_l1_prox(v: &[f64], lambdas: &[f64]) -> Vec<f64> {
    let p = v.len();
    if p == 0 {
        return Vec::new();
    }
    let lam_len = lambdas.len().min(p);

    // Build (|v_i|, original index) sorted descending by |v_i|.
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| {
        v[b].abs()
            .partial_cmp(&v[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // u_i = |v|_{(i)} - lambda_i, clipped to ≥ 0.
    let mut u: Vec<f64> = (0..p)
        .map(|i| {
            let abs_v = v[idx[i]].abs();
            let lam = if i < lam_len { lambdas[i] } else { 0.0 };
            abs_v - lam
        })
        .collect();

    // PAVA isotonic regression (decreasing) to enforce u_{(1)} ≥ u_{(2)} ≥ … ≥ u_{(p)}.
    pava_isotonic_decreasing(&mut u);

    // Clip to ≥ 0 after pooling.
    for val in u.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }

    // Reconstruct result in original index order with original signs.
    let mut result = vec![0.0_f64; p];
    for i in 0..p {
        let orig_idx = idx[i];
        let sign = if v[orig_idx] >= 0.0 { 1.0 } else { -1.0 };
        result[orig_idx] = sign * u[i];
    }
    result
}

/// PAVA isotonic regression to enforce non-increasing order on `v` in-place.
///
/// Groups consecutive elements whose mean violates the non-increasing constraint,
/// replacing them all with their group mean.
fn pava_isotonic_decreasing(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut groups: Vec<(f64, usize)> = v.iter().map(|&x| (x, 1)).collect();
    let mut i = 0;
    while i + 1 < groups.len() {
        if groups[i].0 < groups[i + 1].0 {
            let total = groups[i].0 * groups[i].1 as f64 + groups[i + 1].0 * groups[i + 1].1 as f64;
            let count = groups[i].1 + groups[i + 1].1;
            groups[i] = (total / count as f64, count);
            groups.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    let mut idx = 0;
    for (val, cnt) in groups {
        for _ in 0..cnt {
            v[idx] = val;
            idx += 1;
        }
    }
}

/// Build the lambda sequence for `p` predictors.
/// Trims to length `p` or pads with the last value.
fn build_lambda_sequence(lambdas: &[f64], p: usize) -> Vec<f64> {
    if lambdas.len() >= p {
        lambdas[..p].to_vec()
    } else {
        let mut out = lambdas.to_vec();
        let last = *lambdas.last().unwrap_or(&0.0);
        out.resize(p, last);
        out
    }
}

/// Estimate the spectral norm of `X^T X` via 30 power-method iterations.
fn estimate_lipschitz(x: &[f64], n: usize, p: usize) -> f64 {
    if n == 0 || p == 0 {
        return 1.0;
    }
    let mut v = vec![1.0_f64 / (p as f64).sqrt(); p];
    let mut lam = 1.0_f64;
    for _ in 0..30 {
        let xv = matvec_nn(x, &v, n, p);
        let xtxv = matvec_tn(x, &xv, n, p);
        let nrm: f64 = xtxv.iter().map(|a| a * a).sum::<f64>().sqrt().max(1.0e-300);
        lam = nrm;
        for j in 0..p {
            v[j] = xtxv[j] / nrm;
        }
    }
    lam
}

/// Compute `X * v` where `X` is [n × p] row-major.
fn matvec_nn(x: &[f64], v: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += x[i * p + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Compute `X^T * v` where `X` is [n × p] row-major.
fn matvec_tn(x: &[f64], v: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p {
            out[j] += x[i * p + j] * v[i];
        }
    }
    out
}

/// Approximate normal quantile using Beasley-Springer-Moro algorithm.
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation for the normal quantile (Abramowitz & Stegun 26.2.17).
    let a = [2.515_517, 0.802_853, 0.010_328];
    let b = [1.432_788, 0.189_269, 0.001_308];
    let t = (-2.0 * (p.min(1.0 - p)).ln()).sqrt();
    let num = a[0] + a[1] * t + a[2] * t * t;
    let den = 1.0 + b[0] * t + b[1] * t * t + b[2] * t * t * t;
    let approx = t - num / den;
    if p >= 0.5 { approx } else { -approx }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_design(n: usize, p: usize) -> Vec<f64> {
        // Simple deterministic design matrix.
        (0..n * p)
            .map(|k| {
                let i = k / p;
                let j = k % p;
                ((i + 1) as f64 * (j + 1) as f64 * 0.1).sin()
            })
            .collect()
    }

    fn make_lambdas(p: usize, base: f64) -> Vec<f64> {
        (0..p)
            .map(|i| base * (1.0 - i as f64 / (p as f64 + 1.0)))
            .collect()
    }

    // Test 1
    #[test]
    fn coefficients_shape() {
        let n = 20;
        let p = 8;
        let x = make_design(n, p);
        let y = vec![1.0_f64; n];
        let lams = make_lambdas(p, 0.1);
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 100,
            tol: 1e-6,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");
        assert_eq!(slope.coefficients().len(), p);
    }

    // Test 2
    #[test]
    fn predict_shape() {
        let n = 20;
        let p = 8;
        let x = make_design(n, p);
        let y = vec![1.0_f64; n];
        let lams = make_lambdas(p, 0.05);
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 100,
            tol: 1e-6,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");
        let pred = slope.predict(&x, n, p).expect("ok");
        assert_eq!(pred.len(), n);
    }

    // Test 3
    #[test]
    fn sparse_solution() {
        let n = 30;
        let p = 10;
        let x = make_design(n, p);
        let y = vec![0.5_f64; n];
        // Large lambdas → many coefficients should be zero.
        let lams = make_lambdas(p, 5.0);
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 200,
            tol: 1e-8,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");
        let n_zero = slope
            .coefficients()
            .iter()
            .filter(|&&v| v.abs() < 1e-8)
            .count();
        assert!(
            n_zero > 0,
            "expected at least one zero coefficient with large lambdas"
        );
    }

    // Test 4
    #[test]
    fn soft_threshold_limit() {
        let n = 20;
        let p = 6;
        let x = make_design(n, p);
        let y = vec![0.3_f64; n];
        // Extremely large lambdas → all zeros.
        let lams = vec![1000.0_f64; p];
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 100,
            tol: 1e-9,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");
        for &c in slope.coefficients() {
            assert!(c.abs() < 1e-6, "expected zero, got {c}");
        }
    }

    // Test 5
    #[test]
    fn pava_decreasing_correct() {
        let mut v = vec![3.0_f64, 1.0, 2.0];
        pava_isotonic_decreasing(&mut v);
        // [3, 1, 2]: 1 < 2 → pool → [3, 1.5, 1.5]
        assert!((v[0] - 3.0).abs() < 1e-10);
        assert!((v[1] - 1.5).abs() < 1e-10);
        assert!((v[2] - 1.5).abs() < 1e-10);
    }

    // Test 6
    #[test]
    fn sorted_l1_prox_zeros_small() {
        let v = vec![0.1_f64, 0.2];
        let lams = vec![1.0_f64, 1.0];
        let result = sorted_l1_prox(&v, &lams);
        assert_eq!(result.len(), 2);
        for &r in &result {
            assert!(r.abs() < 1e-10, "expected 0.0, got {r}");
        }
    }

    // Test 7
    #[test]
    fn sorted_l1_prox_identity_large() {
        // Input much larger than lambdas: prox ≈ input - lambda.
        let v = vec![10.0_f64, 8.0];
        let lams = vec![0.5_f64, 0.3];
        let result = sorted_l1_prox(&v, &lams);
        // For positive inputs sorted descending: result ≈ [10 - 0.5, 8 - 0.3] = [9.5, 7.7].
        assert!(
            (result[0] - 9.5).abs() < 0.2,
            "expected ≈9.5, got {}",
            result[0]
        );
        assert!(
            (result[1] - 7.7).abs() < 0.2,
            "expected ≈7.7, got {}",
            result[1]
        );
    }

    // Test 8
    #[test]
    fn lambdas_not_decreasing_error() {
        let lams = vec![1.0_f64, 2.0]; // increasing → Err
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 100,
            tol: 1e-6,
        };
        let result = Slope::new(cfg);
        assert!(result.is_err(), "expected Err for non-decreasing lambdas");
    }

    // Test 9
    #[test]
    fn predict_after_fit_finite() {
        let n = 20;
        let p = 5;
        let x = make_design(n, p);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let lams = make_lambdas(p, 0.02);
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 200,
            tol: 1e-8,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");
        let pred = slope.predict(&x, n, p).expect("ok");
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "prediction has non-finite value"
        );
    }

    // Test 10
    #[test]
    fn fit_reduces_residual() {
        let n = 30;
        let p = 5;
        let x = make_design(n, p);
        // Create a response that's actually achievable.
        let true_beta = vec![1.0_f64, -0.5, 0.3, 0.0, 0.0];
        let y = matvec_nn(&x, &true_beta, n, p);

        let lams = make_lambdas(p, 0.01);
        let cfg = SlopeConfig {
            lambdas: lams,
            max_iter: 500,
            tol: 1e-9,
        };
        let mut slope = Slope::new(cfg).expect("ok");
        slope.fit(&x, &y, n, p).expect("ok");

        let pred = slope.predict(&x, n, p).expect("ok");
        let residual_sq: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (p - yi) * (p - yi))
            .sum();
        // The null predictor (all zeros) has residual = ||y||^2.
        let null_sq: f64 = y.iter().map(|v| v * v).sum();
        assert!(
            residual_sq < null_sq,
            "fitted model should reduce residual: fitted={residual_sq:.4}, null={null_sq:.4}"
        );
    }

    // Test 11
    #[test]
    fn bh_lambdas_decreasing() {
        let p = 10;
        let lams = Slope::bh_lambdas(p, 0.05, 1.0);
        assert_eq!(lams.len(), p);
        for w in lams.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-10,
                "BH lambdas should be non-increasing: {} < {}",
                w[0],
                w[1]
            );
        }
    }
}
