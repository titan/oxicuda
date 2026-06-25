//! Adaptive CFG schedule for dynamic guidance scale during inference.
//!
//! Allows the guidance scale to vary across denoising steps,
//! enabling techniques like time-varying guidance (TVG).

use crate::error::{GenError, GenResult};
use crate::guidance::cfg::{CfgConfig, CfgGuidance};

// ─── AdaptiveCfgPolicy ────────────────────────────────────────────────────────

/// Policy for adapting the CFG guidance scale over denoising steps.
#[derive(Debug, Clone)]
pub enum AdaptiveCfgPolicy {
    /// Constant scale throughout all steps.
    Constant(f32),
    /// Linear interpolation from `start` at step 0 to `end` at the final step.
    Linear { start: f32, end: f32 },
    /// Cosine annealing from `start` to `end`.
    Cosine { start: f32, end: f32 },
    /// Step-wise constant: at each listed `(step, scale)` pair, the scale
    /// applies from that step until the next one. Pairs must be sorted by step.
    StepWise { steps: Vec<(usize, f32)> },
}

// ─── AdaptiveCfgScheduler ─────────────────────────────────────────────────────

/// Adaptive CFG scheduler that varies the guidance scale across denoising steps.
///
/// Useful for techniques like time-varying guidance (TVG) where early steps
/// use a higher scale for structure and later steps use a lower scale for detail.
#[derive(Debug, Clone)]
pub struct AdaptiveCfgScheduler {
    policy: AdaptiveCfgPolicy,
    total_steps: usize,
}

impl AdaptiveCfgScheduler {
    /// Create a new adaptive scheduler.
    ///
    /// # Arguments
    /// - `policy`: The scale schedule policy.
    /// - `total_steps`: Total number of denoising steps.
    pub fn new(policy: AdaptiveCfgPolicy, total_steps: usize) -> Self {
        Self {
            policy,
            total_steps: total_steps.max(1),
        }
    }

    /// Compute the guidance scale at the given denoising step index.
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step >= total_steps`
    /// - `InvalidGuidanceScale` if the computed scale is < 0 (for step-wise policy)
    pub fn scale_at(&self, step: usize) -> GenResult<f32> {
        if step >= self.total_steps {
            return Err(GenError::InvalidTimestep {
                t: step,
                max_t: self.total_steps,
            });
        }
        let scale = match &self.policy {
            AdaptiveCfgPolicy::Constant(s) => *s,
            AdaptiveCfgPolicy::Linear { start, end } => {
                let t = if self.total_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.total_steps - 1) as f32
                };
                start + t * (end - start)
            }
            AdaptiveCfgPolicy::Cosine { start, end } => {
                let t = if self.total_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.total_steps - 1) as f32
                };
                let cos_t = (t * std::f32::consts::PI).cos();
                end + (start - end) * (cos_t + 1.0) * 0.5
            }
            AdaptiveCfgPolicy::StepWise { steps } => {
                // Find the last (step_threshold, scale) pair where threshold <= step
                let mut result = 1.0_f32;
                for &(threshold, s) in steps {
                    if step >= threshold {
                        result = s;
                    }
                }
                result
            }
        };
        Ok(scale.max(1.0)) // clamp to minimum valid guidance scale
    }

    /// Apply CFG at the given step with the adaptive scale.
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step >= total_steps`
    /// - `InvalidGuidanceScale` if computed scale is invalid
    /// - All errors from `CfgGuidance::apply`
    pub fn apply_at_step(&self, cond: &[f32], uncond: &[f32], step: usize) -> GenResult<Vec<f32>> {
        let scale = self.scale_at(step)?;
        let config = CfgConfig::new(scale)?;
        let guide = CfgGuidance::new(config);
        guide.apply(cond, uncond)
    }

    /// Return all scales for all steps.
    ///
    /// Useful for visualisation and debugging.
    pub fn all_scales(&self) -> GenResult<Vec<f32>> {
        (0..self.total_steps).map(|s| self.scale_at(s)).collect()
    }

    /// Return the total number of steps.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Return a reference to the policy.
    pub fn policy(&self) -> &AdaptiveCfgPolicy {
        &self.policy
    }

    /// Sample the guidance-scale curve at an explicit set of step indices.
    ///
    /// Returns one scale per requested index, using the same per-step rule as
    /// [`Self::scale_at`] (so the sample matches the scheduler exactly at every
    /// grid point). This is the curve-sampling helper used for plotting and as
    /// the input to least-squares fitting.
    ///
    /// # Errors
    /// - `InvalidTimestep` if any index is `>= total_steps`.
    pub fn sample_on_grid(&self, steps: &[usize]) -> GenResult<Vec<f32>> {
        steps.iter().map(|&s| self.scale_at(s)).collect()
    }

    /// Sample the curve on a uniform grid of `n` step indices spanning
    /// `0 ..= total_steps - 1`.
    ///
    /// The grid endpoints are always the first and last step. With `n == 1`
    /// only the first step is sampled. Returns `(steps, scales)`.
    ///
    /// # Errors
    /// - `EmptyInput` if `n == 0`.
    pub fn sample_uniform_grid(&self, n: usize) -> GenResult<(Vec<usize>, Vec<f32>)> {
        if n == 0 {
            return Err(GenError::EmptyInput("grid size n must be > 0"));
        }
        let last = self.total_steps - 1;
        let steps: Vec<usize> = if n == 1 {
            vec![0]
        } else {
            (0..n)
                .map(|i| {
                    // Round-to-nearest mapping of i/(n-1) onto [0, last].
                    let frac = i as f32 / (n - 1) as f32;
                    (frac * last as f32).round() as usize
                })
                .collect()
        };
        let scales = self.sample_on_grid(&steps)?;
        Ok((steps, scales))
    }

    /// Least-squares-fit a polynomial of the given `degree` to this scheduler's
    /// own guidance-scale curve, sampled over all `total_steps` steps.
    ///
    /// The independent variable is the normalised step `x = step / (total_steps
    /// - 1) ∈ [0, 1]` (or `0` when there is a single step). Returns a
    /// [`PolynomialFit`] whose `eval` reproduces the curve.
    ///
    /// # Errors
    /// - `EmptyInput` if `degree + 1 > total_steps` (under-determined fit).
    /// - Propagates errors from [`Self::all_scales`].
    pub fn fit_polynomial(&self, degree: usize) -> GenResult<PolynomialFit> {
        let scales = self.all_scales()?;
        let denom = if self.total_steps <= 1 {
            1.0
        } else {
            (self.total_steps - 1) as f32
        };
        let xs: Vec<f32> = (0..self.total_steps).map(|s| s as f32 / denom).collect();
        PolynomialFit::fit(&xs, &scales, degree)
    }
}

// ─── PolynomialFit ────────────────────────────────────────────────────────────

/// A least-squares polynomial fit `y ≈ Σ c_k x^k`.
///
/// Fitting solves the normal equations `(VᵀV) c = Vᵀ y` for the Vandermonde
/// design matrix `V`, via Gaussian elimination with partial pivoting — pure
/// Rust, no external linear-algebra dependency.
#[derive(Debug, Clone)]
pub struct PolynomialFit {
    /// Coefficients `c[k]` for the `x^k` term, length `degree + 1`.
    coeffs: Vec<f32>,
}

impl PolynomialFit {
    /// Fit a polynomial of the given `degree` to the points `(xs[i], ys[i])`.
    ///
    /// # Errors
    /// - `EmptyInput` if `xs`/`ys` is empty or has fewer points than
    ///   `degree + 1` (the fit would be under-determined).
    /// - `DimensionMismatch` if `xs.len() != ys.len()`.
    /// - `Internal` if the normal-equation matrix is singular.
    pub fn fit(xs: &[f32], ys: &[f32], degree: usize) -> GenResult<Self> {
        if xs.is_empty() || ys.is_empty() {
            return Err(GenError::EmptyInput("fit points must be non-empty"));
        }
        if xs.len() != ys.len() {
            return Err(GenError::DimensionMismatch {
                expected: xs.len(),
                got: ys.len(),
            });
        }
        let ncoeff = degree + 1;
        if xs.len() < ncoeff {
            return Err(GenError::EmptyInput(
                "need at least degree+1 points for a determined fit",
            ));
        }
        // Build the Vandermonde matrix V: [n_points × ncoeff], V[i][k] = x_i^k.
        let n = xs.len();
        let mut vander = vec![0.0_f64; n * ncoeff];
        for (i, &x) in xs.iter().enumerate() {
            let mut p = 1.0_f64;
            for k in 0..ncoeff {
                vander[i * ncoeff + k] = p;
                p *= x as f64;
            }
        }
        // Normal equations: A = VᵀV  [ncoeff × ncoeff],  rhs = Vᵀy  [ncoeff].
        let mut a = vec![0.0_f64; ncoeff * ncoeff];
        let mut rhs = vec![0.0_f64; ncoeff];
        for r in 0..ncoeff {
            for c in 0..ncoeff {
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += vander[i * ncoeff + r] * vander[i * ncoeff + c];
                }
                a[r * ncoeff + c] = acc;
            }
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += vander[i * ncoeff + r] * ys[i] as f64;
            }
            rhs[r] = acc;
        }
        let coeffs_f64 = solve_linear_system(&mut a, &mut rhs, ncoeff)?;
        let coeffs = coeffs_f64.into_iter().map(|v| v as f32).collect();
        Ok(Self { coeffs })
    }

    /// Evaluate the fitted polynomial at `x` via Horner's method.
    pub fn eval(&self, x: f32) -> f32 {
        let mut acc = 0.0_f32;
        for &c in self.coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Return the fitted coefficients (`c[k]` for the `x^k` term).
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Polynomial degree (one less than the coefficient count).
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Root-mean-square residual of the fit against the supplied points.
    ///
    /// # Errors
    /// - `EmptyInput` if `xs`/`ys` is empty.
    /// - `DimensionMismatch` if `xs.len() != ys.len()`.
    pub fn rmse(&self, xs: &[f32], ys: &[f32]) -> GenResult<f32> {
        if xs.is_empty() {
            return Err(GenError::EmptyInput("rmse points must be non-empty"));
        }
        if xs.len() != ys.len() {
            return Err(GenError::DimensionMismatch {
                expected: xs.len(),
                got: ys.len(),
            });
        }
        let mut sse = 0.0_f64;
        for (&x, &y) in xs.iter().zip(ys) {
            let d = (self.eval(x) - y) as f64;
            sse += d * d;
        }
        Ok((sse / xs.len() as f64).sqrt() as f32)
    }
}

/// Solve the dense linear system `A x = b` (both mutated) of size `n × n` via
/// Gaussian elimination with partial pivoting, in `f64` for numerical headroom.
///
/// # Errors
/// - `Internal` if a near-singular pivot is encountered.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> GenResult<Vec<f64>> {
    for col in 0..n {
        // Partial pivot: find the row (>= col) with the largest |A[row][col]|.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return Err(GenError::Internal(
                "singular normal-equation matrix in polynomial fit".to_string(),
            ));
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        // Eliminate below the pivot.
        let diag = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / diag;
            if factor != 0.0 {
                for k in col..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            acc -= a[row * n + k] * x[k];
        }
        x[row] = acc / a[row * n + row];
    }
    Ok(x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn constant_policy_same_scale() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(5.0), 10);
        for i in 0..10 {
            let s = sched
                .scale_at(i)
                .expect("scale_at should succeed for valid step index in range 0..total_steps");
            assert!((s - 5.0).abs() < EPS, "step {i}: expected 5.0, got {s}");
        }
    }

    #[test]
    fn linear_policy_boundary_values() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 7.0,
                end: 3.0,
            },
            10,
        );
        let s0 = sched
            .scale_at(0)
            .expect("scale_at step 0 should succeed for linear policy boundary check");
        let s9 = sched.scale_at(9).expect(
            "scale_at final step 9 should succeed for 10-step linear policy boundary check",
        );
        assert!((s0 - 7.0).abs() < EPS, "start: {s0}");
        assert!((s9 - 3.0).abs() < EPS, "end: {s9}");
    }

    #[test]
    fn linear_policy_monotone_decreasing() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 7.0,
                end: 3.0,
            },
            10,
        );
        let scales: Vec<f32> = (0..10)
            .map(|i| sched.scale_at(i).expect("scale_at should succeed"))
            .collect();
        for w in scales.windows(2) {
            assert!(
                w[1] <= w[0] + EPS,
                "scale should decrease: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn cosine_policy_boundary_values() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Cosine {
                start: 8.0,
                end: 2.0,
            },
            100,
        );
        let s0 = sched.scale_at(0).expect("scale_at should succeed");
        let s99 = sched.scale_at(99).expect("scale_at should succeed");
        assert!((s0 - 8.0).abs() < EPS, "cosine start: {s0}");
        assert!((s99 - 2.0).abs() < EPS, "cosine end: {s99}");
    }

    #[test]
    fn stepwise_policy_correct_segments() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::StepWise {
                steps: vec![(0, 7.0), (5, 3.0), (8, 1.5)],
            },
            10,
        );
        assert!((sched.scale_at(0).expect("scale_at should succeed") - 7.0).abs() < EPS);
        assert!((sched.scale_at(4).expect("scale_at should succeed") - 7.0).abs() < EPS);
        assert!((sched.scale_at(5).expect("scale_at should succeed") - 3.0).abs() < EPS);
        assert!((sched.scale_at(7).expect("scale_at should succeed") - 3.0).abs() < EPS);
        assert!((sched.scale_at(8).expect("scale_at should succeed") - 1.5).abs() < EPS);
        assert!((sched.scale_at(9).expect("scale_at should succeed") - 1.5).abs() < EPS);
    }

    #[test]
    fn invalid_step_rejected() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(5.0), 10);
        assert!(matches!(
            sched.scale_at(10),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn apply_at_step_output_shape() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(3.0), 10);
        let cond = vec![1.0_f32; 32];
        let uncond = vec![0.0_f32; 32];
        let out = sched
            .apply_at_step(&cond, &uncond, 5)
            .expect("apply_at_step should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn all_scales_count() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(2.0), 20);
        let scales = sched.all_scales().expect("all_scales should succeed");
        assert_eq!(scales.len(), 20);
    }

    #[test]
    fn scale_minimum_clamped_to_one() {
        // Even if policy would give < 1.0, clamp to 1.0
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 2.0,
                end: 0.5,
            },
            10,
        );
        for i in 0..10 {
            let s = sched.scale_at(i).expect("scale_at should succeed");
            assert!(s >= 1.0, "scale below 1.0 at step {i}: {s}");
        }
    }

    // ── Curve sampling + least-squares fitting ──────────────────────────────

    #[test]
    fn sample_on_grid_matches_scheduler_at_grid_points() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Cosine {
                start: 9.0,
                end: 2.0,
            },
            40,
        );
        let grid = [0_usize, 3, 7, 13, 21, 34, 39];
        let sampled = sched.sample_on_grid(&grid).expect("sample_on_grid");
        assert_eq!(sampled.len(), grid.len());
        for (&step, &s) in grid.iter().zip(&sampled) {
            let direct = sched.scale_at(step).expect("scale_at");
            assert!(
                (s - direct).abs() < EPS,
                "grid sample at step {step} ({s}) must equal scheduler ({direct})"
            );
        }
    }

    #[test]
    fn sample_uniform_grid_endpoints_and_match() {
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 8.0,
                end: 3.0,
            },
            50,
        );
        let (steps, scales) = sched.sample_uniform_grid(6).expect("uniform grid");
        assert_eq!(steps.len(), 6);
        assert_eq!(scales.len(), 6);
        assert_eq!(steps[0], 0, "first grid point is step 0");
        assert_eq!(
            *steps.last().expect("nonempty"),
            49,
            "last grid point is final step"
        );
        // Every sampled value must coincide with the scheduler at that step.
        for (&step, &s) in steps.iter().zip(&scales) {
            let direct = sched.scale_at(step).expect("scale_at");
            assert!((s - direct).abs() < EPS, "mismatch at {step}");
        }
    }

    #[test]
    fn sample_uniform_grid_rejects_zero() {
        let sched = AdaptiveCfgScheduler::new(AdaptiveCfgPolicy::Constant(5.0), 10);
        assert!(matches!(
            sched.sample_uniform_grid(0),
            Err(GenError::InvalidTimestep { .. }) | Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn fit_reproduces_known_quadratic_within_tolerance() {
        // Construct a known quadratic y = 1 + 2x + 3x² over x∈[0,1], sample it,
        // fit degree 2, and confirm the coefficients are recovered.
        let n = 25;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
        let fit = PolynomialFit::fit(&xs, &ys, 2).expect("quadratic fit");
        assert_eq!(fit.degree(), 2);
        let c = fit.coeffs();
        assert!((c[0] - 1.0).abs() < 1e-3, "c0={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-3, "c1={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-3, "c2={}", c[2]);
        // Residual must be essentially zero for an exact polynomial.
        let rmse = fit.rmse(&xs, &ys).expect("rmse");
        assert!(rmse < 1e-3, "rmse too large: {rmse}");
        // Evaluation reproduces the curve off the sample grid as well.
        let x_mid = 0.5_f32;
        let expected = 1.0 + 2.0 * x_mid + 3.0 * x_mid * x_mid;
        assert!((fit.eval(x_mid) - expected).abs() < 1e-3);
    }

    #[test]
    fn fit_linear_policy_is_exact_degree_one() {
        // The linear policy curve is affine in normalised step, so a degree-1
        // fit should reproduce it essentially exactly (away from the >=1 clamp).
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Linear {
                start: 9.0,
                end: 4.0,
            },
            32,
        );
        let fit = sched.fit_polynomial(1).expect("linear fit");
        let scales = sched.all_scales().expect("all_scales");
        let denom = (sched.total_steps() - 1) as f32;
        for (step, &target) in scales.iter().enumerate() {
            let x = step as f32 / denom;
            assert!(
                (fit.eval(x) - target).abs() < 1e-2,
                "degree-1 fit should track linear curve at step {step}: {} vs {target}",
                fit.eval(x)
            );
        }
    }

    #[test]
    fn fit_cosine_policy_low_residual() {
        // A degree-4 fit of the cosine schedule should achieve a small RMSE.
        let sched = AdaptiveCfgScheduler::new(
            AdaptiveCfgPolicy::Cosine {
                start: 10.0,
                end: 2.0,
            },
            64,
        );
        let fit = sched.fit_polynomial(4).expect("cosine fit");
        let scales = sched.all_scales().expect("all_scales");
        let denom = (sched.total_steps() - 1) as f32;
        let xs: Vec<f32> = (0..sched.total_steps()).map(|s| s as f32 / denom).collect();
        let rmse = fit.rmse(&xs, &scales).expect("rmse");
        assert!(rmse < 0.1, "degree-4 cosine fit RMSE too high: {rmse}");
    }

    #[test]
    fn fit_rejects_underdetermined() {
        // 3 points cannot determine a degree-3 (4-coefficient) polynomial.
        let xs = [0.0_f32, 0.5, 1.0];
        let ys = [1.0_f32, 2.0, 1.5];
        assert!(matches!(
            PolynomialFit::fit(&xs, &ys, 3),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn fit_rejects_mismatched_lengths() {
        let xs = [0.0_f32, 1.0, 2.0, 3.0];
        let ys = [1.0_f32, 2.0];
        assert!(matches!(
            PolynomialFit::fit(&xs, &ys, 1),
            Err(GenError::DimensionMismatch { .. })
        ));
    }
}
