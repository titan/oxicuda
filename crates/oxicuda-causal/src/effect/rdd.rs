//! RDD — Sharp Regression Discontinuity Design with the IK optimal bandwidth.
//!
//! Reference: Imbens, G. W. & Kalyanaraman, K. (2012). "Optimal Bandwidth
//! Choice for the Regression Discontinuity Estimator." *Review of Economic
//! Studies*, 79(3):933-959.  See also Hahn, J., Todd, P. & Van der Klaauw, W.
//! (2001). "Identification and Estimation of Treatment Effects with a
//! Regression-Discontinuity Design." *Econometrica*, 69(1):201-209.
//!
//! # Algorithm
//!
//! For a *sharp* RDD with running variable `R` and outcome `Y`, treatment
//! `T = 1[R ≥ c]` is a deterministic function of `R` at the cutoff `c`.
//! The local average treatment effect at the cutoff is
//!
//! ```text
//!   τ_RD = lim_{r ↓ c} E[Y | R = r] − lim_{r ↑ c} E[Y | R = r].
//! ```
//!
//! We estimate the two one-sided limits by **kernel-weighted local linear
//! regression** on either side of `c` within bandwidth `h`:
//!
//! ```text
//!   min_{α, β}  Σ_i  w_i · (y_i − α − β · (r_i − c))²,
//!   w_i = K((r_i − c)/h).
//! ```
//!
//! The 2×2 weighted normal equations are
//!
//! ```text
//!   ⎡ Σ w_i      Σ w_i u_i  ⎤ ⎡α⎤   ⎡ Σ w_i y_i      ⎤
//!   ⎣ Σ w_i u_i  Σ w_i u_i² ⎦ ⎣β⎦ = ⎣ Σ w_i u_i y_i  ⎦
//! ```
//!
//! with `u_i = r_i − c`.  The point estimate is `τ̂ = α_R − α_L`.
//!
//! ## Bandwidth selection (IK plug-in)
//!
//! When no bandwidth is supplied, we use the Imbens-Kalyanaraman optimal
//! plug-in:
//!
//! ```text
//!   h_IK = C_K · ( (σ²(c⁻) + σ²(c⁺)) / (f(c) · (m''(c⁺) − m''(c⁻))²) )^(1/5)
//!          · n^(−1/5)
//! ```
//!
//! - Pilot bandwidth `h₁ = 1.84 · σ_r · n^(−1/5)` (Silverman).
//! - Density `f̂(c)` from a boxcar count within `h₁`.
//! - Curvatures `m''(c±)` from a local-quadratic fit within `h₁` on either
//!   side.
//! - Kernel constant `C_K` (triangular ≈ 3.4375, uniform ≈ 5.4000,
//!   Epanechnikov ≈ 3.5400 — see IK 2012 Table 1).
//!
//! ## Standard error
//!
//! Per side, the weighted residual variance is
//! `σ²_s = Σ w_i (y_i − α_s − β_s u_i)² / Σ w_i` and
//! `Var(α_s) = σ²_s · Σ w_i² / (Σ w_i)²`.  The asymptotic SE of `τ̂` is
//! `SE = sqrt(Var(α_R) + Var(α_L))`.

use crate::error::{CausalError, CausalResult};

/// Kernel used to down-weight observations away from the cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RddKernel {
    /// `K(u) = max(1 − |u|, 0)`.
    Triangular,
    /// `K(u) = 1·[|u| ≤ 1]`.
    Uniform,
    /// `K(u) = (3/4) · max(1 − u², 0)`.
    Epanechnikov,
}

impl RddKernel {
    /// Evaluate the kernel at `u`.
    #[inline]
    pub fn weight(self, u: f64) -> f64 {
        match self {
            RddKernel::Triangular => (1.0 - u.abs()).max(0.0),
            RddKernel::Uniform => {
                if u.abs() <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            RddKernel::Epanechnikov => 0.75 * (1.0 - u * u).max(0.0),
        }
    }

    /// IK 2012 Table 1 plug-in constant `C_K` for the optimal bandwidth.
    #[inline]
    pub fn ik_constant(self) -> f64 {
        match self {
            RddKernel::Triangular => 3.4375,
            RddKernel::Uniform => 5.4000,
            RddKernel::Epanechnikov => 3.5400,
        }
    }
}

/// Configuration for [`Rdd::estimate`].
#[derive(Debug, Clone)]
pub struct RddConfig {
    /// Cutoff value `c` on the running variable.
    pub cutoff: f64,
    /// Bandwidth `h`.  `None` triggers IK plug-in selection.
    pub bandwidth: Option<f64>,
    /// Kernel for local-linear weighting.
    pub kernel: RddKernel,
}

impl Default for RddConfig {
    fn default() -> Self {
        Self {
            cutoff: 0.0,
            bandwidth: None,
            kernel: RddKernel::Triangular,
        }
    }
}

/// Result of [`Rdd::estimate`].
#[derive(Debug, Clone)]
pub struct RddResult {
    /// Estimated treatment effect `τ̂ = α_R − α_L`.
    pub tau: f64,
    /// Asymptotic SE of `τ̂`.
    pub se: f64,
    /// Bandwidth actually used (= supplied value or IK plug-in).
    pub bandwidth_used: f64,
    /// Number of observations on the left side within the bandwidth.
    pub n_left: usize,
    /// Number of observations on the right side within the bandwidth.
    pub n_right: usize,
}

/// Zero-sized handle exposing the RDD entry points.
pub struct Rdd;

impl Rdd {
    /// Estimate the sharp-RDD treatment effect at the cutoff.
    ///
    /// # Parameters
    /// - `y`: outcomes, length `n`.
    /// - `r`: running variable, length `n`.
    /// - `cfg`: see [`RddConfig`].
    ///
    /// # Errors
    /// - [`CausalError::DimensionMismatch`] for empty data or `y.len() != r.len()`.
    /// - [`CausalError::IncompatibleData`] for `bandwidth ≤ 0`, cutoff
    ///   outside `[min(r), max(r)]`, or no observations on either side of
    ///   the cutoff within the bandwidth.
    /// - [`CausalError::MatrixSingular`] if the 2×2 weighted normal
    ///   equations are degenerate (insufficient variation in `r`).
    pub fn estimate(y: &[f64], r: &[f64], cfg: &RddConfig) -> CausalResult<RddResult> {
        let n = y.len();
        if n == 0 || r.is_empty() {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: 0,
            });
        }
        if r.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: r.len(),
            });
        }
        let (rmin, rmax) = min_max(r);
        if cfg.cutoff < rmin || cfg.cutoff > rmax {
            return Err(CausalError::IncompatibleData);
        }
        if let Some(h) = cfg.bandwidth
            && (h <= 0.0 || !h.is_finite())
        {
            return Err(CausalError::IncompatibleData);
        }

        let h = match cfg.bandwidth {
            Some(h) => h,
            None => Self::optimal_bandwidth_ik_with(y, r, cfg.cutoff, cfg.kernel)?,
        };

        // Collect (r, y, w) triples per side.
        let mut left: Vec<(f64, f64, f64)> = Vec::new();
        let mut right: Vec<(f64, f64, f64)> = Vec::new();
        for i in 0..n {
            let u = r[i] - cfg.cutoff;
            let w = cfg.kernel.weight(u / h);
            if w <= 0.0 {
                continue;
            }
            if r[i] < cfg.cutoff {
                left.push((u, y[i], w));
            } else {
                right.push((u, y[i], w));
            }
        }
        if left.is_empty() || right.is_empty() {
            return Err(CausalError::IncompatibleData);
        }

        let (alpha_l, var_l) = weighted_local_linear(&left)?;
        let (alpha_r, var_r) = weighted_local_linear(&right)?;
        let tau = alpha_r - alpha_l;
        let se = (var_l + var_r).max(0.0).sqrt();

        Ok(RddResult {
            tau,
            se,
            bandwidth_used: h,
            n_left: left.len(),
            n_right: right.len(),
        })
    }

    /// Imbens-Kalyanaraman plug-in bandwidth with a triangular kernel.
    ///
    /// Public convenience wrapper around the kernel-aware helper.
    pub fn optimal_bandwidth_ik(y: &[f64], r: &[f64], cutoff: f64) -> CausalResult<f64> {
        Self::optimal_bandwidth_ik_with(y, r, cutoff, RddKernel::Triangular)
    }

    fn optimal_bandwidth_ik_with(
        y: &[f64],
        r: &[f64],
        cutoff: f64,
        kernel: RddKernel,
    ) -> CausalResult<f64> {
        let n = y.len();
        if n == 0 || r.is_empty() {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: 0,
            });
        }
        if r.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: r.len(),
            });
        }
        let (rmin, rmax) = min_max(r);
        if cutoff < rmin || cutoff > rmax {
            return Err(CausalError::IncompatibleData);
        }

        // ---- pilot bandwidth h₁ = 1.84 · σ_r · n^(−1/5) ----------------
        let mean_r = r.iter().sum::<f64>() / n as f64;
        let var_r = r.iter().map(|&v| (v - mean_r) * (v - mean_r)).sum::<f64>() / n as f64;
        let sigma_r = var_r.max(0.0).sqrt().max(1e-12);
        let n_f = n as f64;
        let h1 = 1.84 * sigma_r * n_f.powf(-0.2);
        let h1 = h1.max(1e-9);

        // ---- density f̂(c) from boxcar count within h₁ -----------------
        let mut cnt = 0_usize;
        for &ri in r {
            if (ri - cutoff).abs() <= h1 {
                cnt += 1;
            }
        }
        let f_c = (cnt as f64 / (2.0 * n_f * h1)).max(1e-12);

        // ---- local-quadratic fit per side within h₁ for m''(c±) --------
        let mut lq_left: Vec<(f64, f64)> = Vec::new();
        let mut lq_right: Vec<(f64, f64)> = Vec::new();
        for i in 0..n {
            let u = r[i] - cutoff;
            if u.abs() > h1 {
                continue;
            }
            if r[i] < cutoff {
                lq_left.push((u, y[i]));
            } else {
                lq_right.push((u, y[i]));
            }
        }
        let m2_left = local_quadratic_curvature(&lq_left).unwrap_or(0.0);
        let m2_right = local_quadratic_curvature(&lq_right).unwrap_or(0.0);

        // ---- σ²(c±) — unweighted residual variance on the same window --
        let s2_left = side_variance(&lq_left).unwrap_or(0.0);
        let s2_right = side_variance(&lq_right).unwrap_or(0.0);

        // ---- IK plug-in formula ----------------------------------------
        let curvature_diff = (m2_right - m2_left).powi(2);
        let denom = (f_c * curvature_diff).max(1e-12);
        let numer = (s2_left + s2_right).max(1e-12);
        let ck = kernel.ik_constant();
        let h_ik = ck * (numer / denom).powf(0.2) * n_f.powf(-0.2);
        if !h_ik.is_finite() || h_ik <= 0.0 {
            // Fallback to the pilot bandwidth — still a legitimate choice.
            return Ok(h1);
        }
        Ok(h_ik)
    }
}

// =====================================================================
// helpers
// =====================================================================

#[inline]
fn min_max(r: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in r {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Weighted local-linear fit on triples `(u_i, y_i, w_i)`.
/// Returns `(α, Var(α))` where `α` is the intercept at `u = 0`.
fn weighted_local_linear(data: &[(f64, f64, f64)]) -> CausalResult<(f64, f64)> {
    let mut sw = 0.0_f64;
    let mut swu = 0.0_f64;
    let mut swuu = 0.0_f64;
    let mut swy = 0.0_f64;
    let mut swuy = 0.0_f64;
    let mut sww = 0.0_f64;
    for &(u, y, w) in data {
        sw += w;
        swu += w * u;
        swuu += w * u * u;
        swy += w * y;
        swuy += w * u * y;
        sww += w * w;
    }
    let det = sw * swuu - swu * swu;
    if det.abs() < 1e-15 {
        return Err(CausalError::MatrixSingular);
    }
    let alpha = (swuu * swy - swu * swuy) / det;
    let beta = (sw * swuy - swu * swy) / det;
    // Residual variance σ² = Σ w (y − ŷ)² / Σ w.
    let mut ssr = 0.0_f64;
    for &(u, y, w) in data {
        let yhat = alpha + beta * u;
        ssr += w * (y - yhat) * (y - yhat);
    }
    let sigma2 = if sw > 0.0 { ssr / sw } else { 0.0 };
    let var_alpha = sigma2 * sww / (sw * sw);
    Ok((alpha, var_alpha))
}

/// Unweighted local quadratic fit `y = a + b u + (c/2) u²`; returns the
/// second derivative `m''(0) = c`.  Returns `None` if fewer than three
/// distinct points are available or the design is singular.
fn local_quadratic_curvature(data: &[(f64, f64)]) -> Option<f64> {
    if data.len() < 3 {
        return None;
    }
    // Normal equations on [1, u, u²/2] columns.
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut s4 = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut suy = 0.0_f64;
    let mut su2y = 0.0_f64;
    for &(u, y) in data {
        let u2 = u * u;
        s0 += 1.0;
        s1 += u;
        s2 += u2;
        s3 += u2 * u;
        s4 += u2 * u2;
        sy += y;
        suy += u * y;
        su2y += u2 * y;
    }
    // Solve the 3x3 system A · [a, b, c_half]^T = [sy, suy, su2y] where
    // the columns are [1, u, u²].
    let a = [s0, s1, s2, s1, s2, s3, s2, s3, s4];
    let b = [sy, suy, su2y];
    let coef = solve_3x3(&a, &b)?;
    // The coefficient of u² is `coef[2]`; the second derivative is 2 · coef[2].
    Some(2.0 * coef[2])
}

/// Solve a 3×3 linear system using Cramer's rule.  Returns `None` if the
/// determinant is too small.
fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if det.abs() < 1e-15 {
        return None;
    }
    let inv_det = 1.0 / det;
    let d1 = b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2])
        + a[2] * (b[1] * a[7] - a[4] * b[2]);
    let d2 = a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * b[2] - b[1] * a[6]);
    let d3 = a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6])
        + b[0] * (a[3] * a[7] - a[4] * a[6]);
    Some([d1 * inv_det, d2 * inv_det, d3 * inv_det])
}

/// Unweighted residual variance from a local-linear fit (no kernel).
fn side_variance(data: &[(f64, f64)]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }
    let n = data.len() as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for &(u, y) in data {
        sx += u;
        sy += y;
        sxx += u * u;
        sxy += u * y;
    }
    let mean_x = sx / n;
    let mean_y = sy / n;
    let denom = sxx - n * mean_x * mean_x;
    if denom.abs() < 1e-15 {
        // Fall back to plain Var(y).
        let mut s2 = 0.0_f64;
        for &(_, y) in data {
            s2 += (y - mean_y) * (y - mean_y);
        }
        return Some(s2 / n);
    }
    let slope = (sxy - n * mean_x * mean_y) / denom;
    let intercept = mean_y - slope * mean_x;
    let mut ssr = 0.0_f64;
    for &(u, y) in data {
        let yhat = intercept + slope * u;
        ssr += (y - yhat) * (y - yhat);
    }
    Some(ssr / n)
}

// tests live in `rdd_tests.rs` (registered from `effect/mod.rs`).
