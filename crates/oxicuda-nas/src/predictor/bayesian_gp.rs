//! Gaussian-Process accuracy predictor for sample-efficient Bayesian-optimisation NAS.
//!
//! A Gaussian Process (GP) is the surrogate of choice for *sample-efficient*
//! neural architecture search: each architecture evaluation is expensive (a full
//! train-then-validate cycle), so the search must propose the next architecture
//! that maximises the information / improvement gained per evaluation. Unlike the
//! point-estimate predictors in [`crate::predictor::accuracy`], a GP returns a
//! full posterior — a **mean** *and* a **variance** — at every candidate, and the
//! variance is what drives exploration.
//!
//! This module implements an exact GP regressor over architecture feature vectors:
//!
//! - **Prior**: zero-mean GP with a stationary kernel (RBF or Matérn-5/2). Targets
//!   are internally centred by their empirical mean so the zero-mean prior is
//!   appropriate; the mean is added back at prediction time.
//! - **Conditioning**: form the kernel Gram matrix `K`, add observation noise
//!   `σ_n²` to the diagonal, and factorise `K + σ_n² I = L Lᵀ` by **Cholesky**
//!   decomposition. The posterior weights `α = (K + σ_n² I)⁻¹ (y − ȳ)` are
//!   obtained by a forward then back triangular solve — never by an explicit
//!   inverse.
//! - **Posterior mean** at a test point `x*`:
//!   `μ(x*) = ȳ + k*ᵀ α` with `k*_i = k(x*, x_i)`.
//! - **Posterior variance**:
//!   `σ²(x*) = k(x*, x*) − vᵀ v`, `v = L⁻¹ k*` (a forward solve), clamped at 0.
//!
//! On top of the posterior the GP exposes two **acquisition functions** for
//! deciding where to evaluate next (both *maximised* over candidates, since the
//! objective here — validation accuracy — is maximised):
//!
//! - **UCB** (upper confidence bound): `μ(x*) + β · σ(x*)`.
//! - **EI** (expected improvement) over the best observation `f⁺`:
//!   `EI = (μ − f⁺ − ξ) Φ(z) + σ φ(z)`, `z = (μ − f⁺ − ξ)/σ`, and `EI = 0` when
//!   `σ = 0` (a fully-known point offers no improvement).
//!
//! Everything is pure CPU `f32` linear algebra; no external BLAS / LAPACK.

use crate::error::{NasError, NasResult};
use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};

// ─── Kernel ────────────────────────────────────────────────────────────────────

/// Stationary covariance kernel for the GP.
///
/// Both kernels are functions of the Euclidean distance `r = ‖x − x'‖` only:
/// - [`Kernel::Rbf`] — squared-exponential `σ_f² exp(−r² / (2ℓ²))`, infinitely
///   smooth.
/// - [`Kernel::Matern52`] — Matérn-ν=5/2 `σ_f² (1 + √5 r/ℓ + 5r²/(3ℓ²))
///   exp(−√5 r/ℓ)`, twice mean-square differentiable (a more robust default for
///   noisy real-world objectives).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Radial basis (squared-exponential) kernel with length-scale `ℓ` and
    /// signal variance `σ_f²`.
    Rbf {
        /// Length-scale `ℓ` (> 0): larger ⇒ smoother / longer-range correlation.
        length_scale: f32,
        /// Signal variance `σ_f²` (> 0): the prior variance of the function.
        signal_var: f32,
    },
    /// Matérn-5/2 kernel with length-scale `ℓ` and signal variance `σ_f²`.
    Matern52 {
        /// Length-scale `ℓ` (> 0).
        length_scale: f32,
        /// Signal variance `σ_f²` (> 0).
        signal_var: f32,
    },
}

impl Kernel {
    /// RBF kernel with unit signal variance.
    #[must_use]
    pub fn rbf(length_scale: f32) -> Self {
        Kernel::Rbf {
            length_scale,
            signal_var: 1.0,
        }
    }

    /// Matérn-5/2 kernel with unit signal variance.
    #[must_use]
    pub fn matern52(length_scale: f32) -> Self {
        Kernel::Matern52 {
            length_scale,
            signal_var: 1.0,
        }
    }

    /// The prior variance `k(x, x) = σ_f²` of the kernel (the `r = 0` value).
    #[must_use]
    pub fn variance(&self) -> f32 {
        match *self {
            Kernel::Rbf { signal_var, .. } | Kernel::Matern52 { signal_var, .. } => signal_var,
        }
    }

    fn validate(&self) -> NasResult<()> {
        let (ls, sv) = match *self {
            Kernel::Rbf {
                length_scale,
                signal_var,
            }
            | Kernel::Matern52 {
                length_scale,
                signal_var,
            } => (length_scale, signal_var),
        };
        if !(ls.is_finite() && ls > 0.0 && sv.is_finite() && sv > 0.0) {
            return Err(NasError::NanInArchParams);
        }
        Ok(())
    }

    /// Evaluate the kernel given a squared distance `r²`.
    #[inline]
    fn eval_sq(&self, r2: f32) -> f32 {
        let r2 = r2.max(0.0);
        match *self {
            Kernel::Rbf {
                length_scale,
                signal_var,
            } => signal_var * (-r2 / (2.0 * length_scale * length_scale)).exp(),
            Kernel::Matern52 {
                length_scale,
                signal_var,
            } => {
                let r = r2.sqrt();
                let sqrt5 = 5.0_f32.sqrt();
                let a = sqrt5 * r / length_scale;
                signal_var
                    * (1.0 + a + (5.0 * r2) / (3.0 * length_scale * length_scale))
                    * (-a).exp()
            }
        }
    }

    /// Evaluate the kernel between two feature vectors of equal length.
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let r2 = sq_distance(a, b);
        self.eval_sq(r2)
    }
}

#[inline]
fn sq_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─── Acquisition ───────────────────────────────────────────────────────────────

/// Acquisition function used to score unobserved candidates (higher = evaluate
/// next). Both are written for a *maximised* objective (validation accuracy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Acquisition {
    /// Upper confidence bound `μ + β·σ`. `beta` trades exploitation (small) for
    /// exploration (large); a common schedule uses `β ≈ 2`.
    Ucb {
        /// Exploration weight `β ≥ 0`.
        beta: f32,
    },
    /// Expected improvement over the incumbent best with exploration margin `ξ`.
    ExpectedImprovement {
        /// Margin `ξ ≥ 0` that discourages improvements smaller than `ξ`.
        xi: f32,
    },
}

impl Acquisition {
    /// Score a candidate from its posterior `(mean, variance)` and the current
    /// best observed objective value `f_best` (used by EI; ignored by UCB).
    #[must_use]
    pub fn score(&self, mean: f32, variance: f32, f_best: f32) -> f32 {
        let sigma = variance.max(0.0).sqrt();
        match *self {
            Acquisition::Ucb { beta } => mean + beta * sigma,
            Acquisition::ExpectedImprovement { xi } => {
                if sigma <= 1e-12 {
                    // No uncertainty ⇒ no expected improvement.
                    return 0.0;
                }
                let improve = mean - f_best - xi;
                let z = improve / sigma;
                improve * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)
            }
        }
    }
}

/// Standard-normal probability density `φ(z) = exp(−z²/2)/√(2π)`.
#[inline]
fn standard_normal_pdf(z: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.398_942_28; // 1/√(2π)
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Standard-normal cumulative distribution `Φ(z) = ½(1 + erf(z/√2))`.
#[inline]
fn standard_normal_cdf(z: f32) -> f32 {
    0.5 * (1.0 + erf(z * std::f32::consts::FRAC_1_SQRT_2))
}

/// Error function via Abramowitz & Stegun 7.1.26 (max abs error ≈ 1.5e-7).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // Constants.
    const A1: f32 = 0.254_829_592;
    const A2: f32 = -0.284_496_736;
    const A3: f32 = 1.421_413_741;
    const A4: f32 = -1.453_152_027;
    const A5: f32 = 1.061_405_429;
    const P: f32 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

// ─── GaussianProcess ─────────────────────────────────────────────────────────────

/// Exact Gaussian-Process regressor over architecture feature vectors.
///
/// Fit once on `(features, accuracy)` pairs; thereafter `posterior` /
/// `posterior_mean` / `acquisition` are O(n) per query (and O(n²) for the
/// variance, dominated by the `L⁻¹ k*` solve).
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Training feature vectors (the inducing points).
    train_x: Vec<Vec<f32>>,
    /// Posterior weight vector `α = (K + σ_n² I)⁻¹ (y − ȳ)`.
    alpha: Vec<f32>,
    /// Lower-triangular Cholesky factor `L` of `K + σ_n² I`, row-major `[n × n]`.
    chol: Vec<f32>,
    /// Empirical mean of the training targets (the centring constant `ȳ`).
    y_mean: f32,
    /// Best (maximum) *observed* target value, for EI's incumbent.
    f_best: f32,
    /// Covariance kernel.
    kernel: Kernel,
    /// Observation-noise variance `σ_n²` added to the kernel diagonal.
    noise_var: f32,
    /// Feature dimensionality.
    dim: usize,
}

impl GaussianProcess {
    /// Fit a GP to `(features, accuracy)` samples.
    ///
    /// `noise_var` is the i.i.d. observation noise `σ_n²` added to the kernel
    /// diagonal; a small positive value (e.g. `1e-6`) both models measurement
    /// noise and regularises the Cholesky factorisation. For interpolation tests
    /// keep it tiny.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `samples` is empty.
    /// - [`NasError::DimensionMismatch`] if feature lengths are inconsistent.
    /// - [`NasError::NanInArchParams`] if a target is non-finite, `noise_var` is
    ///   negative / non-finite, or the kernel hyper-parameters are invalid.
    /// - [`NasError::Internal`] if `K + σ_n² I` is not positive-definite (Cholesky
    ///   breakdown) — raise `noise_var`.
    pub fn fit(samples: &[(Vec<f32>, f32)], kernel: Kernel, noise_var: f32) -> NasResult<Self> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        kernel.validate()?;
        if !(noise_var.is_finite() && noise_var >= 0.0) {
            return Err(NasError::NanInArchParams);
        }
        let n = samples.len();
        let dim = samples[0].0.len();
        if dim == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        for (x, t) in samples {
            if x.len() != dim {
                return Err(NasError::DimensionMismatch {
                    expected: dim,
                    got: x.len(),
                });
            }
            if !t.is_finite() {
                return Err(NasError::NanInArchParams);
            }
        }

        // Centre the targets so the zero-mean prior is appropriate.
        let y_mean = samples.iter().map(|(_, t)| *t).sum::<f32>() / n as f32;
        let y_centered: Vec<f32> = samples.iter().map(|(_, t)| *t - y_mean).collect();
        let f_best = samples
            .iter()
            .map(|(_, t)| *t)
            .fold(f32::NEG_INFINITY, f32::max);

        let train_x: Vec<Vec<f32>> = samples.iter().map(|(x, _)| x.clone()).collect();

        // Gram matrix K + σ_n² I (row-major, symmetric).
        let mut k_mat = vec![0.0_f32; n * n];
        for i in 0..n {
            // Diagonal: k(x_i, x_i) = σ_f², plus noise.
            k_mat[i * n + i] = kernel.variance() + noise_var;
            for j in (i + 1)..n {
                let kij = kernel.eval(&train_x[i], &train_x[j]);
                k_mat[i * n + j] = kij;
                k_mat[j * n + i] = kij;
            }
        }

        // Cholesky factorise: K = L Lᵀ.
        let chol = cholesky(&k_mat, n)?;
        // Solve L Lᵀ α = y_centered  ⇒  forward solve L z = y, back solve Lᵀ α = z.
        let z = forward_substitution(&chol, &y_centered, n);
        let alpha = back_substitution_transpose(&chol, &z, n);

        Ok(Self {
            train_x,
            alpha,
            chol,
            y_mean,
            f_best,
            kernel,
            noise_var,
            dim,
        })
    }

    /// Number of training (inducing) points.
    #[must_use]
    pub fn n_train(&self) -> usize {
        self.train_x.len()
    }

    /// Feature dimensionality the GP was fitted on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Best observed target value (EI incumbent `f⁺`).
    #[must_use]
    pub fn best_observed(&self) -> f32 {
        self.f_best
    }

    /// Observation-noise variance `σ_n²` the GP was fitted with.
    #[must_use]
    pub fn noise_var(&self) -> f32 {
        self.noise_var
    }

    /// The covariance kernel the GP was fitted with.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Posterior mean *and* variance at a raw feature vector `x*`.
    ///
    /// Returns `(mean, variance)`; `variance` is clamped to be non-negative.
    ///
    /// # Errors
    /// [`NasError::DimensionMismatch`] if `x.len()` ≠ the training dimension.
    pub fn posterior(&self, x: &[f32]) -> NasResult<(f32, f32)> {
        if x.len() != self.dim {
            return Err(NasError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let n = self.train_x.len();
        // Cross-covariance k* between x and each training point.
        let mut k_star = vec![0.0_f32; n];
        for (i, xi) in self.train_x.iter().enumerate() {
            k_star[i] = self.kernel.eval(x, xi);
        }
        // Mean: ȳ + k*ᵀ α.
        let mut mean = self.y_mean;
        for (ks, a) in k_star.iter().zip(self.alpha.iter()) {
            mean += ks * a;
        }
        // Variance: k(x,x) − vᵀv with v = L⁻¹ k*.
        let v = forward_substitution(&self.chol, &k_star, n);
        let vtv: f32 = v.iter().map(|&vi| vi * vi).sum();
        let prior_var = self.kernel.variance();
        let variance = (prior_var - vtv).max(0.0);
        Ok((mean, variance))
    }

    /// Posterior mean *and* variance for a candidate architecture.
    ///
    /// # Errors
    /// - propagates [`ArchFeatures::from_layers`] errors.
    /// - [`NasError::DimensionMismatch`] if the feature dimension disagrees.
    pub fn posterior_arch(&self, layers: &[LayerSpec]) -> NasResult<(f32, f32)> {
        let f = ArchFeatures::from_layers(layers)?;
        self.posterior(&f.data)
    }

    /// Posterior mean accuracy for a candidate architecture, clamped to `[0, 1]`.
    ///
    /// # Errors
    /// As [`GaussianProcess::posterior_arch`].
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        let (mean, _) = self.posterior_arch(layers)?;
        Ok(mean.clamp(0.0, 1.0))
    }

    /// Acquisition score of a raw feature vector under `acq` (higher ⇒ evaluate
    /// next).
    ///
    /// # Errors
    /// [`NasError::DimensionMismatch`] if `x.len()` ≠ the training dimension.
    pub fn acquisition(&self, x: &[f32], acq: Acquisition) -> NasResult<f32> {
        let (mean, var) = self.posterior(x)?;
        Ok(acq.score(mean, var, self.f_best))
    }

    /// Acquisition score of a candidate architecture under `acq`.
    ///
    /// # Errors
    /// As [`GaussianProcess::posterior_arch`].
    pub fn acquisition_arch(&self, layers: &[LayerSpec], acq: Acquisition) -> NasResult<f32> {
        let (mean, var) = self.posterior_arch(layers)?;
        Ok(acq.score(mean, var, self.f_best))
    }

    /// Pick the index of the candidate that *maximises* the acquisition function.
    ///
    /// `candidates` are raw feature vectors of the training dimension. Returns the
    /// chosen index and its acquisition score.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `candidates` is empty.
    /// - [`NasError::DimensionMismatch`] if any candidate has the wrong length.
    pub fn propose(&self, candidates: &[Vec<f32>], acq: Acquisition) -> NasResult<(usize, f32)> {
        if candidates.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, c) in candidates.iter().enumerate() {
            let s = self.acquisition(c, acq)?;
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        Ok((best_idx, best_score))
    }
}

// ─── Linear algebra (Cholesky + triangular solves) ──────────────────────────────

/// Cholesky factorisation of a symmetric positive-definite row-major `[n × n]`
/// matrix `A`, returning the lower-triangular `L` (also row-major, with zeros in
/// the strict upper triangle) such that `A = L Lᵀ`.
///
/// # Errors
/// [`NasError::Internal`] if a non-positive pivot is encountered (the matrix is
/// not positive-definite within `f32` precision).
fn cholesky(a: &[f32], n: usize) -> NasResult<Vec<f32>> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            // sum_{k<j} L[i,k] L[j,k]
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(NasError::Internal(
                        "GP kernel matrix not positive-definite (raise noise_var)".into(),
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let ljj = l[j * n + j];
                // ljj > 0 here because the diagonal is processed first (j < i).
                l[i * n + j] = sum / ljj;
            }
        }
    }
    Ok(l)
}

/// Solve `L z = b` for `z` by forward substitution, `L` lower-triangular
/// row-major `[n × n]`.
fn forward_substitution(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut z = vec![0.0_f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * z[k];
        }
        z[i] = sum / l[i * n + i];
    }
    z
}

/// Solve `Lᵀ x = z` for `x` by back substitution, `L` lower-triangular row-major
/// `[n × n]` (so `Lᵀ` is upper-triangular and `Lᵀ[i,k] = L[k,i]`).
fn back_substitution_transpose(l: &[f32], z: &[f32], n: usize) -> Vec<f32> {
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            // Lᵀ[i,k] = L[k,i]
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::OpKind;

    // ── Linear-algebra primitives ──────────────────────────────────────────────

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        // A = [[4,2,2],[2,5,3],[2,3,6]] is SPD.
        let n = 3;
        let a = vec![4.0, 2.0, 2.0, 2.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let l = cholesky(&a, n).expect("cholesky");
        // Reconstruct L Lᵀ and compare.
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for k in 0..n {
                    acc += l[i * n + k] * l[j * n + k];
                }
                assert!((acc - a[i * n + j]).abs() < 1e-4, "entry ({i},{j})");
            }
        }
        // Strict upper triangle is zero.
        assert_eq!(l[1], 0.0); // (0, 1)
        assert_eq!(l[2], 0.0); // (0, 2)
        assert_eq!(l[n + 2], 0.0); // (1, 2)
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        // Indefinite matrix.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&a, 2).is_err());
    }

    #[test]
    fn triangular_solves_invert_cholesky() {
        // Solve A x = b via two triangular solves; verify A x ≈ b.
        let n = 3;
        let a = vec![4.0, 2.0, 2.0, 2.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let b = vec![1.0_f32, -2.0, 0.5];
        let l = cholesky(&a, n).expect("cholesky");
        let z = forward_substitution(&l, &b, n);
        let x = back_substitution_transpose(&l, &z, n);
        for i in 0..n {
            let mut acc = 0.0_f32;
            for j in 0..n {
                acc += a[i * n + j] * x[j];
            }
            assert!((acc - b[i]).abs() < 1e-4, "row {i}: {acc} vs {}", b[i]);
        }
    }

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-4);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-4);
        // Φ(0) = 0.5
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
        // Φ(1.96) ≈ 0.975
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 1e-3);
    }

    // ── GP posterior properties ─────────────────────────────────────────────────

    /// 1-D smooth synthetic objective: f(x) = sin(3x) + 0.3x on a grid.
    fn synthetic_1d() -> Vec<(Vec<f32>, f32)> {
        let xs = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        xs.iter()
            .map(|&x| (vec![x], (3.0 * x).sin() + 0.3 * x))
            .collect()
    }

    #[test]
    fn gp_interpolates_training_points() {
        let data = synthetic_1d();
        // Tiny noise ⇒ (near-)interpolation.
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.5), 1e-8).expect("fit should succeed");
        for (x, y) in &data {
            let (mean, var) = gp.posterior(x).expect("posterior");
            assert!(
                (mean - y).abs() < 1e-3,
                "mean {mean} should interpolate target {y}"
            );
            // Variance at an observed point is ~0.
            assert!(var < 1e-3, "variance {var} should be ~0 at observed point");
        }
    }

    #[test]
    fn gp_variance_grows_away_from_data() {
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.4), 1e-6).expect("fit");
        // At a training point variance is ~0.
        let (_, var_at) = gp.posterior(&[1.0]).expect("posterior");
        // Far outside the [0,3] training range, variance approaches the prior σ_f².
        let (_, var_far) = gp.posterior(&[8.0]).expect("posterior");
        assert!(
            var_far > var_at + 0.1,
            "variance should grow away from data: near={var_at} far={var_far}"
        );
        // Far-away variance approaches the prior variance (1.0 here).
        assert!(
            var_far > 0.8,
            "far variance {var_far} should approach prior"
        );
    }

    #[test]
    fn gp_matern52_also_interpolates() {
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::matern52(0.6), 1e-8).expect("fit");
        for (x, y) in &data {
            let (mean, _) = gp.posterior(x).expect("posterior");
            assert!((mean - y).abs() < 2e-3, "matern mean {mean} vs {y}");
        }
    }

    #[test]
    fn gp_mean_interpolates_between_points() {
        // Between two close training points the posterior mean stays near the
        // local function values (smooth interpolation, not wild extrapolation).
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.6), 1e-6).expect("fit");
        let (mean, _) = gp.posterior(&[0.25]).expect("posterior");
        let f_lo = (3.0_f32 * 0.0).sin() + 0.3 * 0.0; // x=0
        let f_hi = (3.0_f32 * 0.5).sin() + 0.3 * 0.5; // x=0.5
        let (lo, hi) = (f_lo.min(f_hi), f_lo.max(f_hi));
        assert!(
            mean >= lo - 0.3 && mean <= hi + 0.3,
            "interpolated mean {mean} outside neighbourhood [{lo},{hi}]"
        );
    }

    // ── Acquisition properties ──────────────────────────────────────────────────

    /// Same smooth objective `f(x) = sin(3x) + 0.3x`, but sampled with a genuine
    /// hole left in `[1.2, 2.6]` so the posterior is *uncertain* in that gap.
    /// The objective's interior peak (near x≈2.07) lives inside the hole, so the
    /// gap is both under-explored and promising — the case acquisition functions
    /// exist to exploit.
    fn synthetic_1d_with_gap() -> Vec<(Vec<f32>, f32)> {
        let xs = [0.0_f32, 0.4, 0.8, 1.2, 2.6, 3.0];
        xs.iter()
            .map(|&x| (vec![x], (3.0 * x).sin() + 0.3 * x))
            .collect()
    }

    #[test]
    fn ucb_highest_in_promising_unobserved_region() {
        // With a real hole in [1.2, 2.6], UCB must score a point inside the gap
        // (high posterior variance) above a thoroughly-known observed point.
        let data = synthetic_1d_with_gap();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.6), 1e-6).expect("fit");
        let acq = Acquisition::Ucb { beta: 2.0 };
        // x=0.4 is an observed (known) point; x=2.07 is deep inside the gap.
        let known = gp.acquisition(&[0.4], acq).expect("acq known");
        let gap = gp.acquisition(&[2.07], acq).expect("acq gap");
        assert!(
            gap > known,
            "UCB should favour the uncertain promising gap: gap={gap} known={known}"
        );
    }

    #[test]
    fn ei_zero_at_observed_points_positive_elsewhere() {
        let data = synthetic_1d_with_gap();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.6), 1e-8).expect("fit");
        let acq = Acquisition::ExpectedImprovement { xi: 0.0 };
        // At an observed point variance ~0 ⇒ EI ~0.
        let ei_obs = gp.acquisition(&[0.8], acq).expect("acq");
        assert!(ei_obs < 1e-3, "EI at observed point should be ~0: {ei_obs}");
        // Inside the uncertain gap, EI must be strictly positive (the posterior
        // assigns non-zero probability to beating the incumbent there).
        let ei_gap = gp.acquisition(&[2.07], acq).expect("acq");
        assert!(
            ei_gap > 1e-4,
            "EI in unobserved gap should be > 0: {ei_gap}"
        );
    }

    #[test]
    fn ei_proposes_into_the_uncertain_gap() {
        // With a hole in [1.2, 2.6], a grid-wide EI search must propose a point
        // *inside* the gap — that is where the posterior is most uncertain and the
        // objective's interior peak (x≈2.07) hides.
        let data = synthetic_1d_with_gap();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.6), 1e-6).expect("fit");
        let acq = Acquisition::ExpectedImprovement { xi: 0.01 };
        // Candidate set spanning the domain (raw 1-D features).
        let cands: Vec<Vec<f32>> = (0..=60).map(|i| vec![i as f32 * 0.05]).collect();
        let (idx, score) = gp.propose(&cands, acq).expect("propose");
        let chosen_x = cands[idx][0];
        assert!(score > 0.0, "best EI score should be positive: {score}");
        assert!(
            (1.2..=2.6).contains(&chosen_x),
            "EI proposal x={chosen_x} should target the uncertain gap [1.2, 2.6]"
        );
    }

    #[test]
    fn propose_picks_acquisition_argmax() {
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.4), 1e-6).expect("fit");
        let acq = Acquisition::Ucb { beta: 1.5 };
        let cands: Vec<Vec<f32>> = vec![vec![0.5], vec![1.0], vec![1.75], vec![6.0]];
        let (idx, score) = gp.propose(&cands, acq).expect("propose");
        // Verify it is genuinely the argmax by recomputing each score.
        let mut best = f32::NEG_INFINITY;
        let mut best_i = 0;
        for (i, c) in cands.iter().enumerate() {
            let s = gp.acquisition(c, acq).expect("acq");
            if s > best {
                best = s;
                best_i = i;
            }
        }
        assert_eq!(idx, best_i);
        assert!((score - best).abs() < 1e-6);
    }

    // ── Architecture-level API + error paths ───────────────────────────────────

    #[test]
    fn gp_arch_round_trip_constant_target() {
        let layers = [
            LayerSpec::new(OpKind::SepConv3x3, 3, 16, 32, 32),
            LayerSpec::new(OpKind::SepConv5x5, 16, 16, 32, 32),
        ];
        let f = ArchFeatures::from_layers(&layers).expect("features");
        // A few constant-accuracy samples (with jitter via feature scale).
        let samples = vec![(f.data.clone(), 0.8_f32); 3];
        let gp = GaussianProcess::fit(&samples, Kernel::rbf(50.0), 1e-6).expect("fit");
        let pred = gp.predict(&layers).expect("predict");
        assert!((pred - 0.8).abs() < 1e-2, "predict {pred} vs 0.8");
        // predict() clamps to [0,1].
        assert!((0.0..=1.0).contains(&pred));
    }

    #[test]
    fn gp_rejects_empty_and_bad_dims() {
        assert_eq!(
            GaussianProcess::fit(&[], Kernel::rbf(1.0), 1e-6).unwrap_err(),
            NasError::EmptySearchSpace
        );
        let bad = vec![(vec![0.0_f32, 1.0], 0.5), (vec![0.0_f32], 0.6)];
        assert!(matches!(
            GaussianProcess::fit(&bad, Kernel::rbf(1.0), 1e-6),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gp_rejects_bad_hyperparams() {
        let data = vec![(vec![0.0_f32], 0.5)];
        // Non-positive length scale.
        assert!(GaussianProcess::fit(&data, Kernel::rbf(0.0), 1e-6).is_err());
        // Negative noise.
        assert!(GaussianProcess::fit(&data, Kernel::rbf(1.0), -1.0).is_err());
        // Non-finite target.
        let nan = vec![(vec![0.0_f32], f32::NAN)];
        assert!(GaussianProcess::fit(&nan, Kernel::rbf(1.0), 1e-6).is_err());
    }

    #[test]
    fn posterior_rejects_dim_mismatch() {
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.5), 1e-6).expect("fit");
        assert!(matches!(
            gp.posterior(&[1.0, 2.0]),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gp_fit_is_deterministic() {
        // The GP fit is fully deterministic (no RNG); two fits agree bit-for-bit.
        let data = synthetic_1d();
        let a = GaussianProcess::fit(&data, Kernel::rbf(0.5), 1e-6).expect("a");
        let b = GaussianProcess::fit(&data, Kernel::rbf(0.5), 1e-6).expect("b");
        let (ma, va) = a.posterior(&[1.3]).expect("a");
        let (mb, vb) = b.posterior(&[1.3]).expect("b");
        assert_eq!(ma.to_bits(), mb.to_bits());
        assert_eq!(va.to_bits(), vb.to_bits());
    }

    #[test]
    fn best_observed_is_training_max() {
        let data = synthetic_1d();
        let gp = GaussianProcess::fit(&data, Kernel::rbf(0.5), 1e-6).expect("fit");
        let expected = data
            .iter()
            .map(|(_, t)| *t)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((gp.best_observed() - expected).abs() < 1e-6);
    }
}
