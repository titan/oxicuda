//! Kernel density estimation (Silverman, 1986).
//!
//! Implements the classical 1-D kernel density estimator
//!
//! ```text
//! f̂(x) = (1 / (n·h)) Σ_{i=1}^{n} K((x − x_i) / h)
//! ```
//!
//! with the **Gaussian** and **Epanechnikov** kernels, automatic bandwidth
//! selection via **Silverman's** and **Scott's** rules of thumb, and an optional
//! 2-D product-kernel estimator.
//!
//! Reference: B. W. Silverman, *Density Estimation for Statistics and Data
//! Analysis*, Chapman & Hall, 1986 (§3.4 for the bandwidth rules, §2.4 for the
//! product kernel).

use crate::descriptive::summary::sample_std;
use crate::error::{StatsError, StatsResult};

/// `1 / sqrt(2π)`, the normalising constant of the standard-normal density.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `3/4`, the normalising constant of the Epanechnikov kernel.
const EPANECHNIKOV_NORM: f64 = 0.75;

/// Smoothing kernel `K(u)`. Every kernel here is a symmetric probability density
/// with unit integral and unit variance (Gaussian) or compact support
/// (Epanechnikov), as required for the rule-of-thumb bandwidths to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Gaussian kernel `K(u) = (1/√(2π)) exp(−u²/2)` (infinite support).
    Gaussian,
    /// Epanechnikov kernel `K(u) = ¾ (1 − u²)` for `|u| ≤ 1`, else `0`.
    Epanechnikov,
}

impl Kernel {
    /// Evaluate the kernel at `u`.
    #[must_use]
    pub fn eval(self, u: f64) -> f64 {
        match self {
            Self::Gaussian => INV_SQRT_2PI * (-0.5 * u * u).exp(),
            Self::Epanechnikov => {
                if u.abs() <= 1.0 {
                    EPANECHNIKOV_NORM * (1.0 - u * u)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Bandwidth-selection rule of thumb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthRule {
    /// Silverman's rule: `h = 1.06 · σ · n^(−1/5)`.
    Silverman,
    /// Scott's rule: `h = σ · n^(−1/5)`.
    Scott,
}

/// Silverman's rule-of-thumb bandwidth `h = 1.06 · σ · n^(−1/5)`.
///
/// `σ` is the sample standard deviation (divisor `n − 1`).
///
/// # Errors
/// Returns [`StatsError::InsufficientSampleSize`] for fewer than two points and
/// [`StatsError::NumericalInstability`] when the data have zero spread.
pub fn silverman_bandwidth(data: &[f64]) -> StatsResult<f64> {
    let sigma = sample_std(data)?;
    bandwidth_from_sigma(sigma, data.len(), 1.06)
}

/// Scott's rule-of-thumb bandwidth `h = σ · n^(−1/5)`.
///
/// # Errors
/// Returns [`StatsError::InsufficientSampleSize`] for fewer than two points and
/// [`StatsError::NumericalInstability`] when the data have zero spread.
pub fn scott_bandwidth(data: &[f64]) -> StatsResult<f64> {
    let sigma = sample_std(data)?;
    bandwidth_from_sigma(sigma, data.len(), 1.0)
}

/// Bandwidth from a precomputed `σ`, count `n`, and a leading `factor`.
fn bandwidth_from_sigma(sigma: f64, n: usize, factor: f64) -> StatsResult<f64> {
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(StatsError::NumericalInstability(
            "zero-variance data: bandwidth undefined".to_string(),
        ));
    }
    let h = factor * sigma * (n as f64).powf(-0.2);
    Ok(h)
}

/// 1-D kernel density estimator.
///
/// Holds the data sample, the chosen kernel, and a (positive) bandwidth `h`.
#[derive(Debug, Clone)]
pub struct KernelDensity {
    data: Vec<f64>,
    kernel: Kernel,
    bandwidth: f64,
}

impl KernelDensity {
    /// Build a KDE with an explicit, user-supplied bandwidth.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyInput`] for empty data,
    /// [`StatsError::NonFiniteValue`] when a sample is not finite, and
    /// [`StatsError::InvalidParameter`] when `bandwidth ≤ 0`.
    pub fn new(data: &[f64], kernel: Kernel, bandwidth: f64) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        for (i, &v) in data.iter().enumerate() {
            if !v.is_finite() {
                return Err(StatsError::NonFiniteValue(i));
            }
        }
        if !(bandwidth.is_finite() && bandwidth > 0.0) {
            return Err(StatsError::InvalidParameter {
                name: "bandwidth".to_string(),
                reason: format!("must be > 0, got {bandwidth}"),
            });
        }
        Ok(Self {
            data: data.to_vec(),
            kernel,
            bandwidth,
        })
    }

    /// Build a KDE selecting the bandwidth by a rule of thumb.
    ///
    /// # Errors
    /// Propagates the errors of [`silverman_bandwidth`] / [`scott_bandwidth`] and
    /// of [`KernelDensity::new`].
    pub fn with_rule(data: &[f64], kernel: Kernel, rule: BandwidthRule) -> StatsResult<Self> {
        let bandwidth = match rule {
            BandwidthRule::Silverman => silverman_bandwidth(data)?,
            BandwidthRule::Scott => scott_bandwidth(data)?,
        };
        Self::new(data, kernel, bandwidth)
    }

    /// The bandwidth `h` in use.
    #[must_use]
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The kernel in use.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the sample is empty (never true for a constructed estimator).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Evaluate the density estimate `f̂(x)` at a single query point.
    ///
    /// `f̂(x) = (1 / (n·h)) Σ_i K((x − x_i) / h)`.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let inv_h = 1.0 / self.bandwidth;
        let acc: f64 = self
            .data
            .iter()
            .map(|&xi| self.kernel.eval((x - xi) * inv_h))
            .sum();
        acc * inv_h / n
    }

    /// Evaluate the density estimate at many query points.
    ///
    /// # Errors
    /// Returns [`StatsError::NonFiniteValue`] when a query point is not finite.
    pub fn pdf_many(&self, queries: &[f64]) -> StatsResult<Vec<f64>> {
        let mut out = Vec::with_capacity(queries.len());
        for (i, &q) in queries.iter().enumerate() {
            if !q.is_finite() {
                return Err(StatsError::NonFiniteValue(i));
            }
            out.push(self.pdf(q));
        }
        Ok(out)
    }

    /// Approximate the integral of `f̂` over `[lo, hi]` by the composite
    /// trapezoidal rule with `n_grid` panels (`n_grid + 1` nodes).
    ///
    /// Over a wide grid this approaches `1`.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidParameter`] when `hi ≤ lo` or `n_grid == 0`.
    pub fn integrate(&self, lo: f64, hi: f64, n_grid: usize) -> StatsResult<f64> {
        if hi <= lo {
            return Err(StatsError::InvalidParameter {
                name: "range".to_string(),
                reason: format!("require hi > lo, got [{lo}, {hi}]"),
            });
        }
        if n_grid == 0 {
            return Err(StatsError::InvalidParameter {
                name: "n_grid".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        let step = (hi - lo) / n_grid as f64;
        let mut acc = 0.5 * (self.pdf(lo) + self.pdf(hi));
        for i in 1..n_grid {
            acc += self.pdf(lo + step * i as f64);
        }
        Ok(acc * step)
    }
}

/// 2-D kernel density estimator with a product kernel.
///
/// ```text
/// f̂(x, y) = (1 / (n · h_x · h_y)) Σ_i K((x − x_i)/h_x) · K((y − y_i)/h_y)
/// ```
///
/// The same 1-D `kernel` is applied independently along each axis.
#[derive(Debug, Clone)]
pub struct KernelDensity2d {
    xs: Vec<f64>,
    ys: Vec<f64>,
    kernel: Kernel,
    h_x: f64,
    h_y: f64,
}

impl KernelDensity2d {
    /// Build a 2-D KDE with explicit per-axis bandwidths.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyInput`] for empty data,
    /// [`StatsError::DimensionMismatch`] when `xs.len() != ys.len()`,
    /// [`StatsError::NonFiniteValue`] for a non-finite coordinate, and
    /// [`StatsError::InvalidParameter`] when a bandwidth is `≤ 0`.
    pub fn new(xs: &[f64], ys: &[f64], kernel: Kernel, h_x: f64, h_y: f64) -> StatsResult<Self> {
        if xs.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        if xs.len() != ys.len() {
            return Err(StatsError::DimensionMismatch {
                a: xs.len(),
                b: ys.len(),
            });
        }
        for (i, (&xi, &yi)) in xs.iter().zip(ys.iter()).enumerate() {
            if !xi.is_finite() || !yi.is_finite() {
                return Err(StatsError::NonFiniteValue(i));
            }
        }
        for (name, h) in [("h_x", h_x), ("h_y", h_y)] {
            if !(h.is_finite() && h > 0.0) {
                return Err(StatsError::InvalidParameter {
                    name: name.to_string(),
                    reason: format!("must be > 0, got {h}"),
                });
            }
        }
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            kernel,
            h_x,
            h_y,
        })
    }

    /// Build a 2-D KDE selecting each axis' bandwidth by a rule of thumb applied
    /// to that axis' marginal.
    ///
    /// # Errors
    /// Propagates bandwidth-selection and constructor errors.
    pub fn with_rule(
        xs: &[f64],
        ys: &[f64],
        kernel: Kernel,
        rule: BandwidthRule,
    ) -> StatsResult<Self> {
        if xs.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        if xs.len() != ys.len() {
            return Err(StatsError::DimensionMismatch {
                a: xs.len(),
                b: ys.len(),
            });
        }
        let select = |d: &[f64]| match rule {
            BandwidthRule::Silverman => silverman_bandwidth(d),
            BandwidthRule::Scott => scott_bandwidth(d),
        };
        let h_x = select(xs)?;
        let h_y = select(ys)?;
        Self::new(xs, ys, kernel, h_x, h_y)
    }

    /// Per-axis bandwidths `(h_x, h_y)`.
    #[must_use]
    pub fn bandwidths(&self) -> (f64, f64) {
        (self.h_x, self.h_y)
    }

    /// Evaluate the 2-D density estimate at `(x, y)`.
    #[must_use]
    pub fn pdf(&self, x: f64, y: f64) -> f64 {
        let n = self.xs.len() as f64;
        let inv_hx = 1.0 / self.h_x;
        let inv_hy = 1.0 / self.h_y;
        let acc: f64 = self
            .xs
            .iter()
            .zip(self.ys.iter())
            .map(|(&xi, &yi)| {
                self.kernel.eval((x - xi) * inv_hx) * self.kernel.eval((y - yi) * inv_hy)
            })
            .sum();
        acc * inv_hx * inv_hy / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::normal::Normal;
    use crate::handle::LcgRng;

    /// Trapezoidal integral of `f` over `[lo, hi]` with `n` panels.
    fn trapz(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
        let step = (hi - lo) / n as f64;
        let mut acc = 0.5 * (f(lo) + f(hi));
        for i in 1..n {
            acc += f(lo + step * i as f64);
        }
        acc * step
    }

    // ----------------------------------------------------------------------
    // (f) the kernel is symmetric and integrates to 1.
    // ----------------------------------------------------------------------

    #[test]
    fn gaussian_kernel_symmetric() {
        for &u in &[0.3_f64, 1.0, 2.5, 4.0] {
            assert!((Kernel::Gaussian.eval(u) - Kernel::Gaussian.eval(-u)).abs() < 1e-15);
        }
    }

    #[test]
    fn epanechnikov_kernel_symmetric_and_compact() {
        for &u in &[0.2_f64, 0.5, 0.9] {
            assert!((Kernel::Epanechnikov.eval(u) - Kernel::Epanechnikov.eval(-u)).abs() < 1e-15);
        }
        // Compact support: zero beyond |u| = 1.
        assert_eq!(Kernel::Epanechnikov.eval(1.0001), 0.0);
        assert_eq!(Kernel::Epanechnikov.eval(-2.0), 0.0);
    }

    #[test]
    fn gaussian_kernel_integrates_to_one() {
        let area = trapz(|u| Kernel::Gaussian.eval(u), -10.0, 10.0, 20_000);
        assert!((area - 1.0).abs() < 1e-6, "gaussian kernel area {area}");
    }

    #[test]
    fn epanechnikov_kernel_integrates_to_one() {
        let area = trapz(|u| Kernel::Epanechnikov.eval(u), -1.5, 1.5, 20_000);
        assert!((area - 1.0).abs() < 1e-5, "epanechnikov kernel area {area}");
    }

    // ----------------------------------------------------------------------
    // (c) Silverman bandwidth == 1.06·σ·n^(−1/5).
    // ----------------------------------------------------------------------

    #[test]
    fn silverman_bandwidth_formula() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sigma = sample_std(&data).expect("sample_std should succeed");
        let n = data.len() as f64;
        let expected = 1.06 * sigma * n.powf(-0.2);
        let got = silverman_bandwidth(&data).expect("silverman_bandwidth should succeed");
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn scott_bandwidth_formula() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sigma = sample_std(&data).expect("sample_std should succeed");
        let n = data.len() as f64;
        let expected = sigma * n.powf(-0.2);
        let got = scott_bandwidth(&data).expect("scott_bandwidth should succeed");
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn zero_variance_bandwidth_errors() {
        let data = [3.0_f64; 10];
        assert!(silverman_bandwidth(&data).is_err());
    }

    // ----------------------------------------------------------------------
    // (a) the KDE integrates to ≈ 1 over a wide grid.
    // ----------------------------------------------------------------------

    #[test]
    fn kde_integrates_to_one_gaussian() {
        let mut rng = LcgRng::new(42);
        let data: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let kde = KernelDensity::with_rule(&data, Kernel::Gaussian, BandwidthRule::Silverman)
            .expect("with_rule should succeed");
        let area = kde
            .integrate(-12.0, 12.0, 6000)
            .expect("integrate should succeed");
        assert!((area - 1.0).abs() < 1e-3, "KDE integral {area}");
    }

    #[test]
    fn kde_integrates_to_one_epanechnikov() {
        let mut rng = LcgRng::new(7);
        let data: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let kde = KernelDensity::with_rule(&data, Kernel::Epanechnikov, BandwidthRule::Silverman)
            .expect("value should be present");
        let area = kde
            .integrate(-12.0, 12.0, 6000)
            .expect("integrate should succeed");
        assert!((area - 1.0).abs() < 1e-3, "KDE integral {area}");
    }

    // ----------------------------------------------------------------------
    // (b) on N(0,1) data the KDE ≈ the standard-normal pdf at several points.
    // ----------------------------------------------------------------------

    #[test]
    fn kde_approximates_standard_normal() {
        let mut rng = LcgRng::new(2024);
        let data: Vec<f64> = (0..4000).map(|_| rng.next_normal()).collect();
        let kde = KernelDensity::with_rule(&data, Kernel::Gaussian, BandwidthRule::Silverman)
            .expect("with_rule should succeed");
        let normal = Normal::standard();
        for &x in &[-1.5_f64, -0.5, 0.0, 0.5, 1.5] {
            let approx = kde.pdf(x);
            let truth = normal.pdf(x);
            assert!(
                (approx - truth).abs() < 0.04,
                "x={x}: kde={approx}, normal={truth}"
            );
        }
    }

    // ----------------------------------------------------------------------
    // (d) the density is everywhere ≥ 0.
    // ----------------------------------------------------------------------

    #[test]
    fn kde_is_non_negative() {
        let data = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        for kernel in [Kernel::Gaussian, Kernel::Epanechnikov] {
            let kde = KernelDensity::new(&data, kernel, 0.5).expect("new should succeed");
            let mut x = -10.0;
            while x <= 10.0 {
                assert!(kde.pdf(x) >= 0.0, "negative density at x={x}");
                x += 0.05;
            }
        }
    }

    // ----------------------------------------------------------------------
    // (e) larger bandwidth ⇒ smoother (smaller peak).
    // ----------------------------------------------------------------------

    #[test]
    fn larger_bandwidth_lowers_peak() {
        // Symmetric data centred at 0; the density peak sits at 0.
        let data = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let narrow = KernelDensity::new(&data, Kernel::Gaussian, 0.3).expect("new should succeed");
        let wide = KernelDensity::new(&data, Kernel::Gaussian, 1.2).expect("new should succeed");
        let peak_narrow = narrow.pdf(0.0);
        let peak_wide = wide.pdf(0.0);
        assert!(
            peak_wide < peak_narrow,
            "wide peak {peak_wide} should be < narrow peak {peak_narrow}"
        );
    }

    #[test]
    fn larger_bandwidth_reduces_curvature() {
        // "Smoother" = smaller roughness functional R(f̂) = ∫ (f̂'')² dx, the
        // standard measure of wiggliness. A larger bandwidth must reduce it.
        let data = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let roughness = |h: f64| {
            let kde = KernelDensity::new(&data, Kernel::Gaussian, h).expect("new should succeed");
            let dx = 0.01_f64;
            let mut x = -8.0_f64;
            let mut acc = 0.0_f64;
            while x <= 8.0 {
                let f2 = (kde.pdf(x + dx) - 2.0 * kde.pdf(x) + kde.pdf(x - dx)) / (dx * dx);
                acc += f2 * f2 * dx;
                x += dx;
            }
            acc
        };
        let rough_narrow = roughness(0.3);
        let rough_wide = roughness(1.2);
        assert!(
            rough_wide < rough_narrow,
            "wide roughness {rough_wide} should be < narrow roughness {rough_narrow}"
        );
    }

    // ----------------------------------------------------------------------
    // (g) far from all data the density ≈ 0.
    // ----------------------------------------------------------------------

    #[test]
    fn density_far_from_data_is_near_zero() {
        let data = [0.0, 0.1, -0.1, 0.2, -0.2];
        // Gaussian: exponentially small but non-zero far away.
        let g = KernelDensity::new(&data, Kernel::Gaussian, 0.3).expect("new should succeed");
        assert!(g.pdf(50.0) < 1e-12, "gaussian tail {}", g.pdf(50.0));
        // Epanechnikov: exactly zero beyond the support.
        let e = KernelDensity::new(&data, Kernel::Epanechnikov, 0.3).expect("new should succeed");
        assert_eq!(e.pdf(50.0), 0.0);
    }

    // ----------------------------------------------------------------------
    // pdf_many and error paths.
    // ----------------------------------------------------------------------

    #[test]
    fn pdf_many_matches_pointwise() {
        let data = [0.0, 1.0, 2.0];
        let kde = KernelDensity::new(&data, Kernel::Gaussian, 0.5).expect("new should succeed");
        let qs = [-1.0, 0.0, 1.0, 2.0, 3.0];
        let many = kde.pdf_many(&qs).expect("pdf_many should succeed");
        for (i, &q) in qs.iter().enumerate() {
            assert!((many[i] - kde.pdf(q)).abs() < 1e-15);
        }
    }

    #[test]
    fn new_rejects_non_positive_bandwidth() {
        let data = [0.0, 1.0];
        assert!(KernelDensity::new(&data, Kernel::Gaussian, 0.0).is_err());
        assert!(KernelDensity::new(&data, Kernel::Gaussian, -1.0).is_err());
    }

    #[test]
    fn new_rejects_empty() {
        assert!(KernelDensity::new(&[], Kernel::Gaussian, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite() {
        let data = [0.0, f64::NAN, 1.0];
        assert!(KernelDensity::new(&data, Kernel::Gaussian, 1.0).is_err());
    }

    // ----------------------------------------------------------------------
    // 2-D product KDE.
    // ----------------------------------------------------------------------

    #[test]
    fn kde_2d_integrates_to_one() {
        let mut rng = LcgRng::new(11);
        let n = 300;
        let xs: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let kde = KernelDensity2d::with_rule(&xs, &ys, Kernel::Gaussian, BandwidthRule::Silverman)
            .expect("value should be present");
        // 2-D trapezoid over a wide square grid.
        let lo = -10.0_f64;
        let hi = 10.0_f64;
        let m = 240_usize;
        let step = (hi - lo) / m as f64;
        let weight = |i: usize| if i == 0 || i == m { 0.5 } else { 1.0 };
        let mut area = 0.0_f64;
        for i in 0..=m {
            let x = lo + step * i as f64;
            for j in 0..=m {
                let y = lo + step * j as f64;
                area += weight(i) * weight(j) * kde.pdf(x, y);
            }
        }
        area *= step * step;
        assert!((area - 1.0).abs() < 5e-3, "2-D KDE integral {area}");
    }

    #[test]
    fn kde_2d_non_negative_and_dim_check() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0];
        assert!(KernelDensity2d::new(&xs, &ys, Kernel::Gaussian, 0.5, 0.5).is_err());

        let ys = [0.0, 1.0, 2.0];
        let kde = KernelDensity2d::new(&xs, &ys, Kernel::Epanechnikov, 0.5, 0.5)
            .expect("new should succeed");
        assert!(kde.pdf(0.5, 0.5) >= 0.0);
        assert_eq!(kde.pdf(100.0, 100.0), 0.0);
        let (hx, hy) = kde.bandwidths();
        assert!((hx - 0.5).abs() < 1e-15 && (hy - 0.5).abs() < 1e-15);
    }
}
