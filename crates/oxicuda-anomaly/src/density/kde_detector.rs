//! Kernel Density Estimation (KDE / Parzen-window) anomaly detector.
//!
//! A non-parametric density model: the probability density at a query point `x` is
//! the average of kernels centred on every training point,
//!
//! ```text
//! p̂(x) = (1 / (n · h^d)) Σ_i K( ‖x − x_i‖ / h ),
//! ```
//!
//! where `K` is a radial kernel, `h` the bandwidth, `d` the feature dimension, and
//! `n` the number of training samples. Regions of **low density** are anomalous, so the
//! score is `−log p̂(x)` (higher ⟹ more anomalous), with a small floor to keep the
//! logarithm finite.
//!
//! # Kernels
//!
//! | Kernel | `K(u)` (radial argument `u = r/h ≥ 0`) | Support |
//! |---|---|---|
//! | Gaussian | `(2π)^{−d/2} exp(−u²/2)` | ℝ⁺ |
//! | Epanechnikov | `c_d (1 − u²)` for `u ≤ 1` | `[0, 1]` |
//! | Tophat | `c_d` for `u ≤ 1` | `[0, 1]` |
//!
//! The Epanechnikov / Tophat normalisers use the volume of the unit `d`-ball.
//!
//! # Bandwidth
//!
//! Either a user-supplied `h`, or a data-driven rule:
//! - **Scott**: `h = σ̄ · n^{−1/(d+4)}`,
//! - **Silverman**: `h = σ̄ · (4 / (d + 2))^{1/(d+4)} · n^{−1/(d+4)}`,
//!
//! where `σ̄` is the mean per-feature standard deviation of the training data.
//!
//! # References
//!
//! - E. Parzen (1962), "On Estimation of a Probability Density Function and Mode",
//!   Ann. Math. Statist. 33(3):1065-1076.
//! - D. W. Scott (1992), "Multivariate Density Estimation", Wiley.
//! - B. W. Silverman (1986), "Density Estimation for Statistics and Data Analysis".

use crate::error::{AnomalyError, AnomalyResult};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Density floor to keep `−log p̂` finite.
const DENSITY_FLOOR: f32 = 1e-30;

// ─── Kernel & bandwidth selection ─────────────────────────────────────────────

/// Radial kernel choice for the KDE detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdeKernel {
    /// Gaussian kernel (infinite support, smooth density).
    Gaussian,
    /// Epanechnikov kernel (compact support, MISE-optimal).
    Epanechnikov,
    /// Tophat / uniform kernel (compact support, piecewise-constant density).
    Tophat,
}

/// Bandwidth-selection strategy.
#[derive(Debug, Clone, Copy)]
pub enum Bandwidth {
    /// Use the supplied bandwidth `h > 0` directly.
    Fixed(f32),
    /// Scott's rule `h = σ̄ n^{−1/(d+4)}`.
    Scott,
    /// Silverman's rule.
    Silverman,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`KdeDetector`].
#[derive(Debug, Clone)]
pub struct KdeConfig {
    /// Kernel (default Gaussian).
    pub kernel: KdeKernel,
    /// Bandwidth rule (default Scott).
    pub bandwidth: Bandwidth,
}

impl Default for KdeConfig {
    fn default() -> Self {
        Self {
            kernel: KdeKernel::Gaussian,
            bandwidth: Bandwidth::Scott,
        }
    }
}

// ─── KdeDetector ──────────────────────────────────────────────────────────────

/// Kernel-density-estimation anomaly detector.
///
/// Call [`KdeDetector::new`] → [`KdeDetector::fit`] → [`KdeDetector::score`].
#[derive(Debug, Clone)]
pub struct KdeDetector {
    config: KdeConfig,
    /// Training data, row-major `[n_samples × n_features]`.
    train: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    /// Resolved bandwidth `h > 0`.
    h: f32,
    /// Kernel normalising constant (depends on kernel, `h`, and `d`).
    norm: f32,
    fitted: bool,
}

impl Default for KdeDetector {
    fn default() -> Self {
        Self::new(KdeConfig::default())
    }
}

impl KdeDetector {
    /// Create an unfitted detector with `config`.
    #[must_use]
    pub fn new(config: KdeConfig) -> Self {
        Self {
            config,
            train: Vec::new(),
            n_samples: 0,
            n_features: 0,
            h: 1.0,
            norm: 1.0,
            fitted: false,
        }
    }

    /// Resolved bandwidth after [`fit`](Self::fit).
    #[must_use]
    pub fn bandwidth(&self) -> f32 {
        self.h
    }

    /// Fit on `data` (row-major, `n_samples × n_features`).
    ///
    /// # Errors
    /// * [`AnomalyError::EmptyInput`] for `n_samples == 0`.
    /// * [`AnomalyError::InvalidFeatureCount`] for `n_features == 0`.
    /// * [`AnomalyError::DimensionMismatch`] if `data.len() != n_samples * n_features`.
    /// * [`AnomalyError::Internal`] for a non-positive fixed bandwidth.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        // Resolve bandwidth.
        let h = match self.config.bandwidth {
            Bandwidth::Fixed(h) => {
                if h <= 0.0 || !h.is_finite() {
                    return Err(AnomalyError::Internal {
                        msg: format!("KDE bandwidth must be > 0, got {h}"),
                    });
                }
                h
            }
            Bandwidth::Scott => {
                let sigma = mean_std(data, n_samples, n_features);
                let exponent = -1.0 / (n_features as f32 + 4.0);
                (sigma * (n_samples as f32).powf(exponent)).max(1e-6)
            }
            Bandwidth::Silverman => {
                let sigma = mean_std(data, n_samples, n_features);
                let d = n_features as f32;
                let factor = (4.0 / (d + 2.0)).powf(1.0 / (d + 4.0));
                let exponent = -1.0 / (d + 4.0);
                (sigma * factor * (n_samples as f32).powf(exponent)).max(1e-6)
            }
        };

        let norm = kernel_norm(self.config.kernel, n_features, h);

        self.train = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;
        self.h = h;
        self.norm = norm;
        self.fitted = true;
        Ok(())
    }

    /// Estimated density `p̂(x)` at a single query point.
    ///
    /// # Errors
    /// * [`AnomalyError::NotFitted`] if [`fit`](Self::fit) has not been called.
    /// * [`AnomalyError::FeatureCountMismatch`] if `x.len() != n_features`.
    pub fn density(&self, x: &[f32]) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let inv_h = 1.0 / self.h;
        let mut acc = 0.0_f32;
        for i in 0..self.n_samples {
            let row = &self.train[i * self.n_features..(i + 1) * self.n_features];
            let mut r_sq = 0.0_f32;
            for (a, b) in x.iter().zip(row.iter()) {
                let d = a - b;
                r_sq += d * d;
            }
            let u = r_sq.sqrt() * inv_h;
            acc += kernel_value(self.config.kernel, u, self.n_features);
        }
        Ok(self.norm * acc / self.n_samples as f32)
    }

    /// Anomaly score `−log p̂(x)` for a single query (higher ⟹ more anomalous).
    ///
    /// # Errors
    /// As [`density`](Self::density).
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let p = self.density(x)?;
        Ok(-(p.max(DENSITY_FLOOR)).ln())
    }

    /// Score a batch of `n` query points (row-major `[n × n_features]`).
    ///
    /// # Errors
    /// * [`AnomalyError::NotFitted`] / [`AnomalyError::DimensionMismatch`].
    pub fn score_batch(&self, data: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if data.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: data.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.score(&data[i * self.n_features..(i + 1) * self.n_features])?);
        }
        Ok(out)
    }
}

// ─── Kernel maths ─────────────────────────────────────────────────────────────

/// Mean per-feature standard deviation of row-major data.
fn mean_std(data: &[f32], n_samples: usize, n_features: usize) -> f32 {
    let inv_n = 1.0 / n_samples as f32;
    let mut total = 0.0_f32;
    for j in 0..n_features {
        let mut mean = 0.0_f32;
        for i in 0..n_samples {
            mean += data[i * n_features + j];
        }
        mean *= inv_n;
        let mut var = 0.0_f32;
        for i in 0..n_samples {
            let d = data[i * n_features + j] - mean;
            var += d * d;
        }
        var *= inv_n;
        total += var.max(0.0).sqrt();
    }
    (total / n_features as f32).max(1e-6)
}

/// Unnormalised radial kernel value `K̃(u)` (the normaliser is folded into [`kernel_norm`]
/// for the constant-coefficient part; Gaussian keeps its exp here).
fn kernel_value(kernel: KdeKernel, u: f32, _d: usize) -> f32 {
    match kernel {
        KdeKernel::Gaussian => (-0.5 * u * u).exp(),
        KdeKernel::Epanechnikov => {
            if u < 1.0 {
                1.0 - u * u
            } else {
                0.0
            }
        }
        KdeKernel::Tophat => {
            if u < 1.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Full normalising constant so that the kernel integrates to 1 over ℝ^d:
/// `norm = c_kernel / h^d`.
fn kernel_norm(kernel: KdeKernel, d: usize, h: f32) -> f32 {
    let df = d as f32;
    let h_pow = h.powi(d as i32);
    match kernel {
        KdeKernel::Gaussian => {
            // (2π)^{−d/2} / h^d.
            let c = (std::f32::consts::TAU).powf(-df / 2.0);
            c / h_pow
        }
        KdeKernel::Epanechnikov => {
            // K(u) = (1 − u²) for u ≤ 1; ∫ (1 − u²) over unit d-ball = V_d · 2/(d+2).
            let v_d = unit_ball_volume(d);
            let integral = v_d * 2.0 / (df + 2.0);
            1.0 / (integral * h_pow)
        }
        KdeKernel::Tophat => {
            // K(u) = 1 for u ≤ 1; ∫ = V_d.
            let v_d = unit_ball_volume(d);
            1.0 / (v_d * h_pow)
        }
    }
}

/// Volume of the unit `d`-ball `V_d = π^{d/2} / Γ(d/2 + 1)`.
fn unit_ball_volume(d: usize) -> f32 {
    let df = d as f32;
    let pi = std::f32::consts::PI;
    (pi.powf(df / 2.0) / gamma_half_integer(df / 2.0 + 1.0)).max(1e-30)
}

/// `Γ(x)` for the half-integer / integer arguments arising in ball volumes, via the
/// Lanczos approximation (adequate for the small `d` used in practice).
fn gamma_half_integer(x: f32) -> f32 {
    // Lanczos approximation (g = 7, n = 9). Computed in f64 for accuracy.
    let x = x as f64;
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        let r = std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_f64(1.0 - x));
        return r as f32;
    }
    gamma_f64_lanczos(x, &C, G) as f32
}

fn gamma_f64(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI
            / ((std::f64::consts::PI * x).sin() * gamma_f64_lanczos(1.0 - x, &C, G))
    } else {
        gamma_f64_lanczos(x, &C, G)
    }
}

fn gamma_f64_lanczos(x: f64, c: &[f64; 9], g: f64) -> f64 {
    let x = x - 1.0;
    let mut a = c[0];
    let t = x + g + 0.5;
    for (i, ci) in c.iter().enumerate().skip(1) {
        a += ci / (x + i as f64);
    }
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster(n: usize, d: usize, center: f32, spread: f32, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random scatter around `center`.
        let mut s = seed;
        let mut out = Vec::with_capacity(n * d);
        for _ in 0..n * d {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0; // ≈ U[-1,1]
            out.push(center + spread * u);
        }
        out
    }

    #[test]
    fn fit_then_score_runs() {
        let data = cluster(40, 2, 0.0, 1.0, 1);
        let mut kde = KdeDetector::default();
        kde.fit(&data, 40, 2).expect("fit");
        let s = kde.score(&[0.0, 0.0]).expect("score");
        assert!(s.is_finite());
    }

    #[test]
    fn inlier_scores_below_outlier() {
        // Tight cluster around the origin; a far point must score higher (more anomalous).
        let data = cluster(80, 2, 0.0, 0.5, 2);
        let mut kde = KdeDetector::default();
        kde.fit(&data, 80, 2).expect("fit");
        let inlier = kde.score(&[0.0, 0.0]).expect("score");
        let outlier = kde.score(&[20.0, 20.0]).expect("score");
        assert!(
            outlier > inlier,
            "outlier {outlier} should exceed inlier {inlier}"
        );
    }

    #[test]
    fn density_higher_at_center() {
        let data = cluster(80, 1, 5.0, 0.5, 3);
        let mut kde = KdeDetector::default();
        kde.fit(&data, 80, 1).expect("fit");
        let p_center = kde.density(&[5.0]).expect("d");
        let p_far = kde.density(&[50.0]).expect("d");
        assert!(p_center > p_far, "center {p_center} far {p_far}");
        assert!(p_center >= 0.0 && p_far >= 0.0);
    }

    #[test]
    fn epanechnikov_compact_support() {
        // A point well outside every kernel's support has exactly zero density.
        let data = cluster(30, 1, 0.0, 0.1, 4);
        let cfg = KdeConfig {
            kernel: KdeKernel::Epanechnikov,
            bandwidth: Bandwidth::Fixed(0.5),
        };
        let mut kde = KdeDetector::new(cfg);
        kde.fit(&data, 30, 1).expect("fit");
        let p_far = kde.density(&[100.0]).expect("d");
        assert_eq!(p_far, 0.0, "compact support ⇒ zero density far away");
        // Score saturates to the floor.
        let s_far = kde.score(&[100.0]).expect("s");
        assert!((s_far - (-(DENSITY_FLOOR).ln())).abs() < 1e-3);
    }

    #[test]
    fn tophat_density_constant_inside() {
        let data = vec![0.0_f32];
        let cfg = KdeConfig {
            kernel: KdeKernel::Tophat,
            bandwidth: Bandwidth::Fixed(1.0),
        };
        let mut kde = KdeDetector::new(cfg);
        kde.fit(&data, 1, 1).expect("fit");
        // Inside [-1, 1] density is constant 1/(2h) = 0.5 for d = 1.
        let p0 = kde.density(&[0.0]).expect("d");
        let p_half = kde.density(&[0.5]).expect("d");
        assert!((p0 - 0.5).abs() < 1e-5, "p0 = {p0}");
        assert!((p_half - 0.5).abs() < 1e-5, "p_half = {p_half}");
    }

    #[test]
    fn gaussian_density_integrates_to_one_1d() {
        // Single training point at 0, Gaussian h=1: density is a unit Gaussian.
        // Riemann-sum the density over a wide grid; should be ≈ 1.
        let data = vec![0.0_f32];
        let cfg = KdeConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth: Bandwidth::Fixed(1.0),
        };
        let mut kde = KdeDetector::new(cfg);
        kde.fit(&data, 1, 1).expect("fit");
        let dx = 0.01_f32;
        let mut integral = 0.0_f32;
        let mut x = -8.0_f32;
        while x <= 8.0 {
            integral += kde.density(&[x]).expect("d") * dx;
            x += dx;
        }
        assert!((integral - 1.0).abs() < 0.02, "∫ p = {integral}");
    }

    #[test]
    fn scott_and_silverman_bandwidths_positive() {
        let data = cluster(50, 3, 0.0, 2.0, 5);
        for rule in [Bandwidth::Scott, Bandwidth::Silverman] {
            let mut kde = KdeDetector::new(KdeConfig {
                kernel: KdeKernel::Gaussian,
                bandwidth: rule,
            });
            kde.fit(&data, 50, 3).expect("fit");
            assert!(kde.bandwidth() > 0.0 && kde.bandwidth().is_finite());
        }
    }

    #[test]
    fn score_batch_matches_single() {
        let data = cluster(40, 2, 0.0, 1.0, 6);
        let mut kde = KdeDetector::default();
        kde.fit(&data, 40, 2).expect("fit");
        let queries = vec![0.0_f32, 0.0, 3.0, 3.0, -2.0, 1.0];
        let batch = kde.score_batch(&queries, 3).expect("batch");
        for i in 0..3 {
            let single = kde.score(&queries[i * 2..i * 2 + 2]).expect("single");
            assert!((batch[i] - single).abs() < 1e-5);
        }
    }

    #[test]
    fn score_before_fit_errors() {
        let kde = KdeDetector::default();
        assert!(matches!(
            kde.score(&[0.0, 0.0]),
            Err(AnomalyError::NotFitted)
        ));
        assert!(matches!(kde.density(&[0.0]), Err(AnomalyError::NotFitted)));
    }

    #[test]
    fn feature_mismatch_errors() {
        let data = cluster(10, 2, 0.0, 1.0, 7);
        let mut kde = KdeDetector::default();
        kde.fit(&data, 10, 2).expect("fit");
        assert!(matches!(
            kde.score(&[1.0, 2.0, 3.0]),
            Err(AnomalyError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn fit_rejects_bad_shapes() {
        let mut kde = KdeDetector::default();
        assert!(matches!(kde.fit(&[], 0, 2), Err(AnomalyError::EmptyInput)));
        assert!(matches!(
            kde.fit(&[1.0, 2.0], 1, 0),
            Err(AnomalyError::InvalidFeatureCount { .. })
        ));
        assert!(matches!(
            kde.fit(&[1.0, 2.0, 3.0], 2, 2),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
        // Bad fixed bandwidth.
        let mut kde2 = KdeDetector::new(KdeConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth: Bandwidth::Fixed(-1.0),
        });
        assert!(matches!(
            kde2.fit(&[0.0, 1.0], 2, 1),
            Err(AnomalyError::Internal { .. })
        ));
    }

    #[test]
    fn unit_ball_volume_known_values() {
        // V_1 = 2, V_2 = π, V_3 = 4π/3.
        assert!(
            (unit_ball_volume(1) - 2.0).abs() < 1e-3,
            "V1 = {}",
            unit_ball_volume(1)
        );
        assert!((unit_ball_volume(2) - std::f32::consts::PI).abs() < 1e-3);
        assert!(
            (unit_ball_volume(3) - 4.0 * std::f32::consts::PI / 3.0).abs() < 1e-2,
            "V3 = {}",
            unit_ball_volume(3)
        );
    }
}
