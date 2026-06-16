//! Multivariate Gaussian copula (Sklar 1959; Li 2000).
//!
//! This is the *d-dimensional* Gaussian copula characterised by a full
//! correlation matrix Σ. It is distinct from the bivariate scalar-ρ
//! [`crate::copula::copulas::CopulaFamily::Gaussian`] family (which models a
//! single pair via method-of-moments on Kendall's τ): here we model the joint
//! dependence of `dim` margins through the correlation matrix of the normal
//! scores.
//!
//! # Workflow
//! 1. **Pseudo-observations** — each column is mapped to `(0, 1)` by the
//!    rank-based empirical CDF `u_{ij} = rank(x_{ij}) / (n + 1)` (average ranks
//!    for ties). The `n + 1` denominator keeps the scores strictly inside the
//!    open unit interval so the Normal PPF stays finite.
//! 2. **Normal scores** — `z_{ij} = Φ⁻¹(u_{ij})`.
//! 3. **Fit Σ** — the maximum-likelihood / method-of-moments estimate of the
//!    Gaussian-copula correlation is the sample (Pearson) correlation matrix of
//!    the normal scores; this guarantees a symmetric matrix with unit diagonal.
//! 4. **Density** — `c(u) = |Σ|^{-1/2} · exp(-½ zᵀ (Σ⁻¹ − I) z)` with
//!    `z_j = Φ⁻¹(u_j)`.
//! 5. **Sampling** — Cholesky `Σ = L Lᵀ`, draw `g ~ N(0, I)`, set `w = L g`,
//!    `u_j = Φ(w_j)`.
//!
//! # References
//! - Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs
//!   marges*. Publ. Inst. Statist. Univ. Paris 8: 229-231.
//! - Li, D. X. (2000). *On Default Correlation: A Copula Function Approach*.
//!   J. Fixed Income 9(4): 43-54.

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;
use std::cmp::Ordering;

/// A fitted multivariate Gaussian copula.
///
/// The dependence structure is fully described by the `dim × dim` correlation
/// matrix `corr` (row-major, symmetric, unit diagonal).
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    /// Number of margins (dimension).
    pub dim: usize,
    /// Flattened row-major `dim × dim` correlation matrix Σ.
    pub corr: Vec<f64>,
    /// Number of observations used to fit the copula.
    pub n_samples: usize,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_matrix(data: &[f64], n: usize, dim: usize) -> StatsResult<()> {
    if dim == 0 {
        return Err(StatsError::InvalidParameter {
            name: "dim".to_owned(),
            reason: "dimension must be ≥ 1".to_owned(),
        });
    }
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if data.len() != n * dim {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n * dim],
            got: vec![data.len()],
        });
    }
    for (i, &x) in data.iter().enumerate() {
        if !x.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Average ranks (1-based, ties averaged)
// ---------------------------------------------------------------------------

fn average_ranks(col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && col[idx[j + 1]] == col[idx[i]] {
            j += 1;
        }
        // Mean of the 1-based ranks (i+1)..=(j+1) for the tied block.
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &pos in &idx[i..=j] {
            ranks[pos] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Compute rank-based empirical-CDF pseudo-observations for a data matrix.
///
/// `data` is row-major `n × dim`. Each column `j` is transformed to
/// `u_{ij} = rank(x_{ij}) / (n + 1)`, which lies strictly inside `(0, 1)`.
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if `dim == 0`.
/// - [`StatsError::InsufficientSampleSize`] if `n < 2`.
/// - [`StatsError::ShapeMismatch`] if `data.len() != n * dim`.
/// - [`StatsError::NonFiniteValue`] if any entry is non-finite.
pub fn pseudo_observations(data: &[f64], n: usize, dim: usize) -> StatsResult<Vec<f64>> {
    validate_matrix(data, n, dim)?;
    let denom = n as f64 + 1.0;
    let mut u = vec![0.0; n * dim];
    let mut col = vec![0.0; n];
    for j in 0..dim {
        for (i, c) in col.iter_mut().enumerate() {
            *c = data[i * dim + j];
        }
        let ranks = average_ranks(&col);
        for i in 0..n {
            u[i * dim + j] = ranks[i] / denom;
        }
    }
    Ok(u)
}

// ---------------------------------------------------------------------------
// Correlation matrix of the normal scores
// ---------------------------------------------------------------------------

fn correlation_matrix(z: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut mean = vec![0.0; dim];
    for i in 0..n {
        for (j, m) in mean.iter_mut().enumerate() {
            *m += z[i * dim + j];
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }
    let mut cov = vec![0.0; dim * dim];
    for i in 0..n {
        for a in 0..dim {
            let za = z[i * dim + a] - mean[a];
            for b in 0..dim {
                let zb = z[i * dim + b] - mean[b];
                cov[a * dim + b] += za * zb;
            }
        }
    }
    let mut corr = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            if a == b {
                corr[a * dim + b] = 1.0;
            } else {
                let denom = (cov[a * dim + a] * cov[b * dim + b]).sqrt();
                corr[a * dim + b] = if denom > 0.0 {
                    (cov[a * dim + b] / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
            }
        }
    }
    corr
}

// ---------------------------------------------------------------------------
// Cholesky helpers for the (SPD) correlation matrix
// ---------------------------------------------------------------------------

fn cholesky_lower(a: &[f64], d: usize) -> StatsResult<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(StatsError::SingularMatrix(
                        "Gaussian copula correlation matrix is not positive definite".to_owned(),
                    ));
                }
                l[i * d + j] = sum.sqrt();
            } else {
                l[i * d + j] = sum / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// Solve `L y = b` for lower-triangular `L`.
fn forward_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * d + k] * y[k];
        }
        y[i] = sum / l[i * d + i];
    }
    y
}

/// Solve `Lᵀ x = y` for lower-triangular `L`.
fn backward_solve_lt(l: &[f64], y: &[f64], d: usize) -> Vec<f64> {
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in (i + 1)..d {
            sum -= l[k * d + i] * x[k];
        }
        x[i] = sum / l[i * d + i];
    }
    x
}

/// Solve `Σ x = b` given the Cholesky factor `L` of `Σ`.
fn chol_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    backward_solve_lt(l, &forward_solve(l, b, d), d)
}

impl GaussianCopula {
    /// Fit a Gaussian copula to a row-major `n × dim` data matrix.
    ///
    /// The margins are transformed to pseudo-observations via their empirical
    /// CDFs and the correlation matrix Σ is estimated as the sample correlation
    /// of the resulting normal scores.
    ///
    /// # Errors
    /// Propagates validation errors from [`pseudo_observations`].
    pub fn fit(data: &[f64], n: usize, dim: usize) -> StatsResult<Self> {
        let u = pseudo_observations(data, n, dim)?;
        let std = Normal::standard();
        let mut z = vec![0.0; n * dim];
        for (zi, &ui) in z.iter_mut().zip(u.iter()) {
            *zi = std.ppf(ui)?;
        }
        let corr = correlation_matrix(&z, n, dim);
        Ok(Self {
            dim,
            corr,
            n_samples: n,
        })
    }

    /// Construct a Gaussian copula directly from a correlation matrix.
    ///
    /// `corr` must be a row-major `dim × dim` symmetric matrix with unit
    /// diagonal and entries in `[-1, 1]`.
    ///
    /// # Errors
    /// - [`StatsError::InvalidParameter`] if `dim == 0`.
    /// - [`StatsError::ShapeMismatch`] if `corr.len() != dim * dim`.
    pub fn from_correlation(corr: Vec<f64>, dim: usize) -> StatsResult<Self> {
        if dim == 0 {
            return Err(StatsError::InvalidParameter {
                name: "dim".to_owned(),
                reason: "dimension must be ≥ 1".to_owned(),
            });
        }
        if corr.len() != dim * dim {
            return Err(StatsError::ShapeMismatch {
                expected: vec![dim * dim],
                got: vec![corr.len()],
            });
        }
        Ok(Self {
            dim,
            corr,
            n_samples: 0,
        })
    }

    /// Map a point `u ∈ (0, 1)^dim` to its normal scores `z = Φ⁻¹(u)`.
    fn scores(&self, u: &[f64]) -> StatsResult<Vec<f64>> {
        if u.len() != self.dim {
            return Err(StatsError::DimensionMismatch {
                a: u.len(),
                b: self.dim,
            });
        }
        let std = Normal::standard();
        let mut z = vec![0.0; self.dim];
        for (zi, &ui) in z.iter_mut().zip(u.iter()) {
            if !(ui > 0.0 && ui < 1.0) {
                return Err(StatsError::InvalidParameter {
                    name: "u".to_owned(),
                    reason: format!("copula argument must be in (0, 1), got {ui}"),
                });
            }
            *zi = std.ppf(ui)?;
        }
        Ok(z)
    }

    /// Log copula density `ln c(u)` at a single interior point `u ∈ (0, 1)^dim`.
    ///
    /// # Errors
    /// - [`StatsError::DimensionMismatch`] if `u.len() != dim`.
    /// - [`StatsError::InvalidParameter`] if any `u_j ∉ (0, 1)`.
    /// - [`StatsError::SingularMatrix`] if Σ is not positive definite.
    pub fn log_density(&self, u: &[f64]) -> StatsResult<f64> {
        let z = self.scores(u)?;
        let l = cholesky_lower(&self.corr, self.dim)?;
        let log_det: f64 = (0..self.dim).map(|i| l[i * self.dim + i].ln()).sum::<f64>() * 2.0;
        // zᵀ Σ⁻¹ z = ‖L⁻¹ z‖²
        let y = forward_solve(&l, &z, self.dim);
        let quad_inv: f64 = y.iter().map(|v| v * v).sum();
        let quad_z: f64 = z.iter().map(|v| v * v).sum();
        Ok(-0.5 * log_det - 0.5 * (quad_inv - quad_z))
    }

    /// Copula density `c(u)` at a single interior point.
    ///
    /// # Errors
    /// See [`GaussianCopula::log_density`].
    pub fn density(&self, u: &[f64]) -> StatsResult<f64> {
        Ok(self.log_density(u)?.exp())
    }

    /// Total log-likelihood of a row-major `n × dim` matrix of
    /// pseudo-observations under the fitted copula.
    ///
    /// # Errors
    /// - [`StatsError::ShapeMismatch`] / [`StatsError::InvalidParameter`] on bad
    ///   shapes or arguments outside `(0, 1)`.
    /// - [`StatsError::SingularMatrix`] if Σ is not positive definite.
    pub fn log_likelihood(&self, u: &[f64], n: usize) -> StatsResult<f64> {
        if n == 0 {
            return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
        }
        if u.len() != n * self.dim {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n * self.dim],
                got: vec![u.len()],
            });
        }
        let l = cholesky_lower(&self.corr, self.dim)?;
        let log_det: f64 = (0..self.dim).map(|i| l[i * self.dim + i].ln()).sum::<f64>() * 2.0;
        let std = Normal::standard();
        let mut total = 0.0;
        let mut z = vec![0.0; self.dim];
        for row in 0..n {
            for (j, zj) in z.iter_mut().enumerate() {
                let uij = u[row * self.dim + j];
                if !(uij > 0.0 && uij < 1.0) {
                    return Err(StatsError::InvalidParameter {
                        name: "u".to_owned(),
                        reason: format!("copula argument must be in (0, 1), got {uij}"),
                    });
                }
                *zj = std.ppf(uij)?;
            }
            let y = forward_solve(&l, &z, self.dim);
            let quad_inv: f64 = y.iter().map(|v| v * v).sum();
            let quad_z: f64 = z.iter().map(|v| v * v).sum();
            total += -0.5 * log_det - 0.5 * (quad_inv - quad_z);
        }
        Ok(total)
    }

    /// Draw `n` samples from the copula, returned as a row-major `n × dim`
    /// vector of values in `(0, 1)`.
    ///
    /// # Errors
    /// - [`StatsError::InsufficientSampleSize`] if `n == 0`.
    /// - [`StatsError::SingularMatrix`] if Σ is not positive definite.
    pub fn sample(&self, n: usize, rng: &mut LcgRng) -> StatsResult<Vec<f64>> {
        if n == 0 {
            return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
        }
        let l = cholesky_lower(&self.corr, self.dim)?;
        let std = Normal::standard();
        let mut out = vec![0.0; n * self.dim];
        let mut g = vec![0.0; self.dim];
        for row in 0..n {
            for gi in g.iter_mut() {
                *gi = rng.next_normal();
            }
            for i in 0..self.dim {
                let mut w = 0.0;
                for k in 0..=i {
                    w += l[i * self.dim + k] * g[k];
                }
                out[row * self.dim + i] = std.cdf(w).clamp(1e-15, 1.0 - 1e-15);
            }
        }
        Ok(out)
    }

    /// Conditional normal parameters of margin `target` given fixed values of
    /// other margins (on the copula scale, i.e. `u`-values in `(0, 1)`).
    ///
    /// Returns the conditional mean and variance of the *normal score*
    /// `z_target = Φ⁻¹(u_target)`. With no conditioning information the result
    /// is the standard-normal marginal `(0, 1)`.
    ///
    /// # Errors
    /// - [`StatsError::IndexOutOfBounds`] if any index ≥ `dim`.
    /// - [`StatsError::InvalidParameter`] if `target` is also conditioned on, or
    ///   a conditioning `u`-value lies outside `(0, 1)`.
    /// - [`StatsError::SingularMatrix`] if the conditioning sub-block is singular.
    pub fn conditional_normal(
        &self,
        target: usize,
        given: &[(usize, f64)],
    ) -> StatsResult<(f64, f64)> {
        if target >= self.dim {
            return Err(StatsError::IndexOutOfBounds {
                index: target,
                len: self.dim,
            });
        }
        if given.is_empty() {
            return Ok((0.0, 1.0));
        }
        let m = given.len();
        let std = Normal::standard();
        let mut zg = vec![0.0; m];
        for (slot, &(idx, uval)) in zg.iter_mut().zip(given.iter()) {
            if idx >= self.dim {
                return Err(StatsError::IndexOutOfBounds {
                    index: idx,
                    len: self.dim,
                });
            }
            if idx == target {
                return Err(StatsError::InvalidParameter {
                    name: "given".to_owned(),
                    reason: "target index cannot appear in the conditioning set".to_owned(),
                });
            }
            if !(uval > 0.0 && uval < 1.0) {
                return Err(StatsError::InvalidParameter {
                    name: "given".to_owned(),
                    reason: format!("conditioning value must be in (0, 1), got {uval}"),
                });
            }
            *slot = std.ppf(uval)?;
        }
        // Σ_gg (m × m) and Σ_tg (length m).
        let mut sigma_gg = vec![0.0; m * m];
        let mut sigma_tg = vec![0.0; m];
        for (a, &(ia, _)) in given.iter().enumerate() {
            sigma_tg[a] = self.corr[target * self.dim + ia];
            for (b, &(ib, _)) in given.iter().enumerate() {
                sigma_gg[a * m + b] = self.corr[ia * self.dim + ib];
            }
        }
        let l = cholesky_lower(&sigma_gg, m)?;
        // w = Σ_gg⁻¹ z_g ; s = Σ_gg⁻¹ Σ_gt
        let w = chol_solve(&l, &zg, m);
        let s = chol_solve(&l, &sigma_tg, m);
        let cond_mean: f64 = sigma_tg.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let cond_var: f64 = 1.0
            - sigma_tg
                .iter()
                .zip(s.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        Ok((cond_mean, cond_var.max(0.0)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a row-major `n × dim` matrix of independent uniforms.
    fn independent_matrix(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_f64()).collect()
    }

    #[test]
    fn pseudo_obs_in_unit_interval_and_rank_based() {
        // Distinct, deliberately unsorted column.
        let data = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.0]; // n = 6, dim = 1
        let u = pseudo_observations(&data, 6, 1).expect("ok");
        for &ui in &u {
            assert!(ui > 0.0 && ui < 1.0, "pseudo-obs {ui} not in (0,1)");
        }
        // Rank-based: order of pseudo-obs matches order of data.
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(
                    data[i].partial_cmp(&data[j]),
                    u[i].partial_cmp(&u[j]),
                    "pseudo-observations must be a monotone (rank) transform"
                );
            }
        }
    }

    #[test]
    fn fit_independent_off_diagonals_near_zero() {
        let n = 400;
        let dim = 3;
        let data = independent_matrix(n, dim, 0xC0FFEE);
        let gc = GaussianCopula::fit(&data, n, dim).expect("fit ok");
        for a in 0..dim {
            for b in 0..dim {
                if a != b {
                    let r = gc.corr[a * dim + b];
                    assert!(
                        r.abs() < 0.2,
                        "independent margins should give ρ≈0, got ρ[{a},{b}]={r}"
                    );
                }
            }
        }
    }

    #[test]
    fn fit_comonotone_correlation_near_one() {
        // Column 1 is a strictly increasing transform of column 0 ⇒ identical
        // ranks ⇒ fitted correlation is exactly 1.
        let n = 50;
        let dim = 2;
        let mut rng = LcgRng::new(7);
        let mut data = vec![0.0; n * dim];
        for i in 0..n {
            let x = rng.next_f64();
            data[i * dim] = x;
            data[i * dim + 1] = 3.0 * x + 1.0; // monotone increasing
        }
        let gc = GaussianCopula::fit(&data, n, dim).expect("fit ok");
        let r = gc.corr[1];
        assert!((r - 1.0).abs() < 1e-9, "comonotone ρ should be ≈1, got {r}");
    }

    #[test]
    fn correlation_matrix_symmetric_unit_diagonal() {
        let n = 120;
        let dim = 4;
        let data = independent_matrix(n, dim, 0xABCDE);
        let gc = GaussianCopula::fit(&data, n, dim).expect("fit ok");
        for a in 0..dim {
            assert!((gc.corr[a * dim + a] - 1.0).abs() < 1e-12, "unit diagonal");
            for b in 0..dim {
                let above = gc.corr[a * dim + b];
                let below = gc.corr[b * dim + a];
                assert!((above - below).abs() < 1e-12, "symmetry");
            }
        }
    }

    #[test]
    fn density_identity_correlation_is_one() {
        // Σ = I ⇒ c(u) ≡ 1.
        let gc = GaussianCopula::from_correlation(vec![1.0, 0.0, 0.0, 1.0], 2).expect("ok");
        let d = gc.density(&[0.3, 0.8]).expect("ok");
        assert!(
            (d - 1.0).abs() < 1e-12,
            "identity density should be 1, got {d}"
        );
    }

    #[test]
    fn density_matches_closed_form_bivariate() {
        // For d=2 at u=v=0.5 (z=0): c = 1/√(1-ρ²).
        let rho = 0.5;
        let gc = GaussianCopula::from_correlation(vec![1.0, rho, rho, 1.0], 2).expect("ok");
        let d = gc.density(&[0.5, 0.5]).expect("ok");
        let expected = 1.0 / (1.0 - rho * rho).sqrt();
        assert!(
            (d - expected).abs() < 1e-9,
            "density {d} should equal {expected}"
        );
    }

    #[test]
    fn log_likelihood_finite() {
        let n = 200;
        let dim = 2;
        let data = independent_matrix(n, dim, 0x1234);
        let gc = GaussianCopula::fit(&data, n, dim).expect("fit ok");
        let u = pseudo_observations(&data, n, dim).expect("ok");
        let ll = gc.log_likelihood(&u, n).expect("ok");
        assert!(ll.is_finite(), "log-likelihood should be finite, got {ll}");
    }

    #[test]
    fn sample_in_unit_interval_and_length() {
        let gc = GaussianCopula::from_correlation(vec![1.0, 0.6, 0.6, 1.0], 2).expect("ok");
        let mut rng = LcgRng::new(99);
        let s = gc.sample(150, &mut rng).expect("ok");
        assert_eq!(s.len(), 300);
        for &x in &s {
            assert!(x > 0.0 && x < 1.0, "sample {x} not in (0,1)");
        }
    }

    #[test]
    fn sampled_dependence_matches_sign() {
        // Strong positive correlation ⇒ refit recovers a positive off-diagonal.
        let gc = GaussianCopula::from_correlation(vec![1.0, 0.8, 0.8, 1.0], 2).expect("ok");
        let mut rng = LcgRng::new(2024);
        let s = gc.sample(600, &mut rng).expect("ok");
        let refit = GaussianCopula::fit(&s, 600, 2).expect("ok");
        assert!(
            refit.corr[1] > 0.5,
            "resampled correlation should stay strongly positive, got {}",
            refit.corr[1]
        );
    }

    #[test]
    fn conditional_independence_gives_standard_normal() {
        let gc = GaussianCopula::from_correlation(vec![1.0, 0.0, 0.0, 1.0], 2).expect("ok");
        let (mean, var) = gc.conditional_normal(0, &[(1, 0.95)]).expect("ok");
        assert!(
            mean.abs() < 1e-12,
            "independent conditional mean should be 0"
        );
        assert!(
            (var - 1.0).abs() < 1e-12,
            "independent conditional var should be 1"
        );
    }

    #[test]
    fn conditional_correlated_shrinks_variance() {
        // For ρ: conditional variance = 1 - ρ², mean = ρ·z_given.
        let rho = 0.6;
        let gc = GaussianCopula::from_correlation(vec![1.0, rho, rho, 1.0], 2).expect("ok");
        let u_given = 0.8;
        let (mean, var) = gc.conditional_normal(0, &[(1, u_given)]).expect("ok");
        let z_given = Normal::standard().ppf(u_given).expect("ok");
        assert!((var - (1.0 - rho * rho)).abs() < 1e-9, "cond var = 1-ρ²");
        assert!((mean - rho * z_given).abs() < 1e-9, "cond mean = ρ·z");
    }

    #[test]
    fn shape_and_dimension_errors() {
        // data length mismatch.
        assert!(matches!(
            GaussianCopula::fit(&[0.1, 0.2, 0.3], 2, 2),
            Err(StatsError::ShapeMismatch { .. })
        ));
        // dim = 0.
        assert!(matches!(
            GaussianCopula::fit(&[], 2, 0),
            Err(StatsError::InvalidParameter { .. })
        ));
        // n < 2.
        assert!(matches!(
            GaussianCopula::fit(&[0.5, 0.6], 1, 2),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
        // non-finite.
        assert!(matches!(
            pseudo_observations(&[0.1, f64::NAN, 0.3, 0.4], 2, 2),
            Err(StatsError::NonFiniteValue(_))
        ));
    }

    #[test]
    fn density_argument_out_of_range_errors() {
        let gc = GaussianCopula::from_correlation(vec![1.0, 0.0, 0.0, 1.0], 2).expect("ok");
        assert!(matches!(
            gc.density(&[1.5, 0.5]),
            Err(StatsError::InvalidParameter { .. })
        ));
        assert!(matches!(
            gc.density(&[0.5]),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }
}
