//! Radial basis function (RBF) input embedding for spectral-bias mitigation.
//!
//! Plain coordinate MLPs exhibit *spectral bias*: they learn the low-frequency
//! content of a target field long before the high-frequency content (Rahaman et
//! al. 2019; Wang, Yu & Perdikaris 2021 on the same pathology in PINNs). Mapping
//! the input coordinates through a bank of localised basis functions before the
//! network supplies a high-frequency, well-conditioned feature space that lets the
//! downstream model resolve sharp features — the RBF analogue of Fourier-feature
//! embeddings (and the classical radial basis function network of Broomhead &
//! Lowe 1988; Park & Sandberg 1991).
//!
//! ## Feature map
//! Given centers `c_k` and a shape parameter `σ`, the embedding of a point `x` is
//! ```text
//! φ_k(x) = ρ( ‖x − c_k‖² ; σ ),
//! ```
//! optionally normalised to a partition of unity `φ_k / Σ_m φ_m` (the *normalised
//! RBF network*). Three radial profiles are provided:
//!
//! | kind                  | `ρ(r²)`                | at `r=0` | as `r→∞` |
//! |-----------------------|------------------------|----------|----------|
//! | Gaussian              | `exp(−r²/2σ²)`         | `1`      | `0`      |
//! | Multiquadric          | `√(1 + r²/σ²)`         | `1`      | `∞`      |
//! | Inverse-multiquadric  | `1/√(1 + r²/σ²)`       | `1`      | `0`      |
//!
//! Closed-form input gradients `∂φ_k/∂x_j` are exposed for use inside PDE
//! residuals. [`RbfFeatureNetwork`] couples the embedding with a ridge-regularised
//! linear readout, giving a classical RBF network trainable in closed form.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// True uniform sample in `[0, 1)` (this crate's `next_u32` spans `[0, 2³¹)`).
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    (rng.next_u32() as f32) / 4_294_967_296.0_f32
}

/// Radial profile of an RBF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbfKind {
    /// `exp(−r²/2σ²)` — compact, decaying, peak `1` at the center.
    Gaussian,
    /// `√(1 + r²/σ²)` — globally supported, increasing with distance.
    Multiquadric,
    /// `1/√(1 + r²/σ²)` — globally supported, decaying with distance.
    InverseMultiquadric,
}

impl RbfKind {
    /// Evaluate `(ρ(r²), ∂ρ/∂(r²))` for a squared radius `r2` and bandwidth `σ`.
    #[inline]
    fn profile(self, r2: f32, sigma: f32) -> (f32, f32) {
        let s2 = sigma * sigma;
        match self {
            RbfKind::Gaussian => {
                let phi = (-r2 / (2.0 * s2)).exp();
                (phi, -phi / (2.0 * s2))
            }
            RbfKind::Multiquadric => {
                let s = 1.0 + r2 / s2;
                let phi = s.sqrt();
                // dρ/dr² = 1/(2σ²) · s^{-1/2}
                (phi, 1.0 / (2.0 * s2 * phi))
            }
            RbfKind::InverseMultiquadric => {
                let s = 1.0 + r2 / s2;
                let phi = 1.0 / s.sqrt();
                // dρ/dr² = −1/(2σ²) · s^{-3/2} = −φ³/(2σ²)
                (phi, -(phi * phi * phi) / (2.0 * s2))
            }
        }
    }
}

/// Configuration for an [`RbfFeatures`] embedding.
#[derive(Debug, Clone)]
pub struct RbfFeatureConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Number of RBF centers (the embedding dimension).
    pub n_centers: usize,
    /// Bandwidth / shape parameter `σ > 0`.
    pub bandwidth: f32,
    /// Radial profile.
    pub kind: RbfKind,
    /// Whether to normalise the features to a partition of unity.
    pub normalize: bool,
    /// Lower bound of the (per-dimension) domain used for center placement.
    pub domain_lo: f32,
    /// Upper bound of the (per-dimension) domain used for center placement.
    pub domain_hi: f32,
}

impl RbfFeatureConfig {
    /// Default Gaussian, non-normalised embedding on `[0, 1]`.
    #[must_use]
    pub fn new(input_dim: usize, n_centers: usize, bandwidth: f32) -> Self {
        Self {
            input_dim,
            n_centers,
            bandwidth,
            kind: RbfKind::Gaussian,
            normalize: false,
            domain_lo: 0.0,
            domain_hi: 1.0,
        }
    }
}

/// A bank of radial basis functions used as an input feature map.
#[derive(Debug, Clone)]
pub struct RbfFeatures {
    /// Center coordinates, row-major `[n_centers × input_dim]`.
    centers: Vec<f32>,
    config: RbfFeatureConfig,
}

impl RbfFeatures {
    /// Construct an embedding, placing centers automatically.
    ///
    /// For `input_dim == 1` the centers are evenly spaced (a grid) across
    /// `[domain_lo, domain_hi]`; for higher dimensions they are sampled uniformly
    /// in the box `[domain_lo, domain_hi]^{input_dim}`.
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `input_dim == 0`.
    /// - [`PinnError::InvalidLayerWidth`] if `n_centers == 0`.
    /// - [`PinnError::InvalidPdeCoefficient`] if `bandwidth <= 0` or the domain
    ///   width is non-positive.
    pub fn new(config: RbfFeatureConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        Self::validate(&config)?;
        let d = config.input_dim;
        let n = config.n_centers;
        let lo = config.domain_lo;
        let hi = config.domain_hi;

        let centers: Vec<f32> = if d == 1 {
            if n == 1 {
                vec![0.5 * (lo + hi)]
            } else {
                (0..n)
                    .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
                    .collect()
            }
        } else {
            (0..n * d)
                .map(|_| lo + (hi - lo) * unit_uniform(rng))
                .collect()
        };

        Ok(Self { centers, config })
    }

    /// Construct an embedding with explicit centers (row-major `[n_centers × d]`).
    ///
    /// # Errors
    /// - As [`RbfFeatures::new`] for the configuration.
    /// - [`PinnError::DimensionMismatch`] if `centers.len() != n_centers·input_dim`.
    pub fn with_centers(centers: Vec<f32>, config: RbfFeatureConfig) -> PinnResult<Self> {
        Self::validate(&config)?;
        let expected = config.n_centers * config.input_dim;
        if centers.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: centers.len(),
            });
        }
        Ok(Self { centers, config })
    }

    fn validate(config: &RbfFeatureConfig) -> PinnResult<()> {
        if config.input_dim == 0 {
            return Err(PinnError::EmptyInput);
        }
        if config.n_centers == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if !config.bandwidth.is_finite() || config.bandwidth <= 0.0 {
            return Err(PinnError::InvalidPdeCoefficient {
                name: "bandwidth",
                value: config.bandwidth,
            });
        }
        let width = config.domain_hi - config.domain_lo;
        if !width.is_finite() || width <= 0.0 {
            return Err(PinnError::InvalidPdeCoefficient {
                name: "domain_width",
                value: width,
            });
        }
        Ok(())
    }

    /// Embedding dimension (number of features = number of centers).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_centers
    }

    /// Input dimensionality.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Center coordinates, row-major `[n_centers × input_dim]`.
    #[must_use]
    pub fn centers(&self) -> &[f32] {
        &self.centers
    }

    /// Squared distance from `x` to center `k`.
    fn squared_distance(&self, x: &[f32], k: usize) -> f32 {
        let d = self.config.input_dim;
        x.iter()
            .zip(self.centers[k * d..(k + 1) * d].iter())
            .map(|(&xi, &ci)| (xi - ci) * (xi - ci))
            .sum()
    }

    /// Compute the (un-normalised) radial feature values `ρ(‖x − c_k‖²)`.
    fn raw_features(&self, x: &[f32]) -> Vec<f32> {
        (0..self.config.n_centers)
            .map(|k| {
                let r2 = self.squared_distance(x, k);
                self.config.kind.profile(r2, self.config.bandwidth).0
            })
            .collect()
    }

    /// Embed `x` into the feature space `[n_centers]`.
    ///
    /// When `normalize` is set, features sum to `1` (a guard returns a uniform
    /// distribution if the total underflows to zero).
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn encode(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }
        let mut phi = self.raw_features(x);
        if self.config.normalize {
            let total: f32 = phi.iter().sum();
            if total > 0.0 {
                for v in &mut phi {
                    *v /= total;
                }
            } else {
                phi.fill(1.0 / self.config.n_centers as f32);
            }
        }
        Ok(phi)
    }

    /// Jacobian of the embedding, `∂φ_k/∂x_j`, row-major `[n_centers × input_dim]`.
    ///
    /// Honours the `normalize` flag via the quotient rule.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn encode_grad(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }
        let d = self.config.input_dim;
        let n = self.config.n_centers;
        let sigma = self.config.bandwidth;
        let kind = self.config.kind;

        let mut phi = vec![0.0_f32; n];
        let mut jac = vec![0.0_f32; n * d];
        for k in 0..n {
            let r2 = self.squared_distance(x, k);
            let (pk, dpk_dr2) = kind.profile(r2, sigma);
            phi[k] = pk;
            // ∂φ_k/∂x_j = dρ/dr² · 2 (x_j − c_kj)
            for j in 0..d {
                let diff = x[j] - self.centers[k * d + j];
                jac[k * d + j] = dpk_dr2 * 2.0 * diff;
            }
        }

        if !self.config.normalize {
            return Ok(jac);
        }

        // Quotient rule for n_k = φ_k / S, S = Σ_m φ_m:
        // ∂n_k/∂x_j = (∂φ_k/∂x_j · S − φ_k · Σ_m ∂φ_m/∂x_j) / S².
        let total: f32 = phi.iter().sum();
        if total <= 0.0 {
            return Ok(vec![0.0_f32; n * d]);
        }
        let mut ds_dx = vec![0.0_f32; d];
        for k in 0..n {
            for j in 0..d {
                ds_dx[j] += jac[k * d + j];
            }
        }
        let inv_s2 = 1.0 / (total * total);
        let mut out = vec![0.0_f32; n * d];
        for k in 0..n {
            for j in 0..d {
                out[k * d + j] = (jac[k * d + j] * total - phi[k] * ds_dx[j]) * inv_s2;
            }
        }
        Ok(out)
    }
}

/// A classical radial basis function network: an [`RbfFeatures`] embedding with a
/// ridge-regularised linear readout, fittable in closed form.
#[derive(Debug, Clone)]
pub struct RbfFeatureNetwork {
    features: RbfFeatures,
    out_dim: usize,
    /// Linear readout, row-major `[out_dim × (n_centers + 1)]`; the trailing
    /// column multiplies the bias term `1`. Zero until fitted.
    weights: Vec<f32>,
    ridge_lambda: f32,
}

impl RbfFeatureNetwork {
    /// Construct an RBF network with a zero-initialised readout.
    ///
    /// # Errors
    /// - As [`RbfFeatures::new`].
    /// - [`PinnError::EmptyInput`] if `out_dim == 0`.
    /// - [`PinnError::InvalidWeight`] if `ridge_lambda < 0` or non-finite.
    pub fn new(
        config: RbfFeatureConfig,
        out_dim: usize,
        ridge_lambda: f32,
        rng: &mut LcgRng,
    ) -> PinnResult<Self> {
        if out_dim == 0 {
            return Err(PinnError::EmptyInput);
        }
        if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: ridge_lambda,
            });
        }
        let n_centers = config.n_centers;
        let features = RbfFeatures::new(config, rng)?;
        Ok(Self {
            features,
            out_dim,
            weights: vec![0.0_f32; out_dim * (n_centers + 1)],
            ridge_lambda,
        })
    }

    /// Access the underlying feature embedding.
    #[must_use]
    pub fn features(&self) -> &RbfFeatures {
        &self.features
    }

    /// Trained readout weights, row-major `[out_dim × (n_centers + 1)]`.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Fit the linear readout by ridge regression on `n_samples` examples.
    ///
    /// `xs` is row-major `[n_samples × input_dim]`, `ys` is `[n_samples × out_dim]`.
    /// Solves `(ΦᵀΦ + λI)·W = ΦᵀY` where `Φ` is the augmented design matrix
    /// `[encode(xᵢ); 1]`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if the sample shapes are inconsistent.
    /// - [`PinnError::SolverDivergence`] if the normal-equation system is singular.
    pub fn fit(&mut self, xs: &[f32], ys: &[f32], n_samples: usize) -> PinnResult<()> {
        let d = self.features.input_dim();
        let c = self.features.n_features();
        let m = c + 1;
        if xs.len() != n_samples * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_samples * d,
                got: xs.len(),
            });
        }
        if ys.len() != n_samples * self.out_dim {
            return Err(PinnError::DimensionMismatch {
                expected: n_samples * self.out_dim,
                got: ys.len(),
            });
        }
        if n_samples == 0 {
            return Err(PinnError::EmptyCollocationSet);
        }

        let mut a = vec![0.0_f32; m * m];
        let mut bmat = vec![0.0_f32; m * self.out_dim];
        for s in 0..n_samples {
            let mut row = self.features.encode(&xs[s * d..(s + 1) * d])?;
            row.push(1.0); // bias feature
            let target = &ys[s * self.out_dim..(s + 1) * self.out_dim];
            for (i, &ri) in row.iter().enumerate() {
                for (j, &rj) in row.iter().enumerate() {
                    a[i * m + j] += ri * rj;
                }
                for (o, &yo) in target.iter().enumerate() {
                    bmat[i * self.out_dim + o] += ri * yo;
                }
            }
        }
        for i in 0..m {
            a[i * m + i] += self.ridge_lambda;
        }

        let theta = solve_linear_multi(&mut a, &mut bmat, m, self.out_dim)?;
        let mut weights = vec![0.0_f32; self.out_dim * m];
        for k in 0..m {
            for o in 0..self.out_dim {
                weights[o * m + k] = theta[k * self.out_dim + o];
            }
        }
        self.weights = weights;
        Ok(())
    }

    /// Predict `W·[encode(x); 1]`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn predict(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        let mut feat = self.features.encode(x)?;
        feat.push(1.0);
        let m = feat.len();
        let out = self
            .weights
            .chunks_exact(m)
            .map(|row| row.iter().zip(feat.iter()).map(|(&w, &f)| w * f).sum())
            .collect();
        Ok(out)
    }
}

/// Solve `A·X = B` (`n×n`, `m` RHS) by Gaussian elimination with partial pivoting.
///
/// `a` is row-major `n×n`, `rhs` is row-major `n×m`; both are overwritten. Returns
/// `X` row-major `n×m`.
///
/// # Errors
/// - [`PinnError::SolverDivergence`] if `A` is (numerically) singular.
/// - [`PinnError::NanEncountered`] if the solution is not finite.
fn solve_linear_multi(a: &mut [f32], rhs: &mut [f32], n: usize, m: usize) -> PinnResult<Vec<f32>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let val = a[r * n + col].abs();
            if val > best {
                best = val;
                pivot = r;
            }
        }
        if best <= 1e-20 {
            return Err(PinnError::SolverDivergence {
                reason: "singular matrix in RBF ridge solve",
            });
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            for c in 0..m {
                rhs.swap(col * m + c, pivot * m + c);
            }
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            for c in 0..m {
                rhs[r * m + c] -= factor * rhs[col * m + c];
            }
        }
    }
    let mut x = vec![0.0_f32; n * m];
    for col in (0..n).rev() {
        let diag = a[col * n + col];
        for c in 0..m {
            let mut s = rhs[col * m + c];
            for k in (col + 1)..n {
                s -= a[col * n + k] * x[k * m + c];
            }
            x[col * m + c] = s / diag;
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "rbf_solve_linear_multi",
        });
    }
    Ok(x)
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    fn make_features(kind: RbfKind, normalize: bool) -> RbfFeatures {
        let mut rng = LcgRng::new(1);
        let mut cfg = RbfFeatureConfig::new(1, 8, 0.15);
        cfg.kind = kind;
        cfg.normalize = normalize;
        RbfFeatures::new(cfg, &mut rng)
            .expect("RbfFeatures construction with valid config should succeed")
    }

    // ── construction / validation ─────────────────────────────────────────────

    #[test]
    fn rbf_centers_grid_1d() {
        let feats = make_features(RbfKind::Gaussian, false);
        let c = feats.centers();
        assert_eq!(c.len(), 8);
        assert!((c[0] - 0.0).abs() < 1e-6, "first center at domain_lo");
        assert!((c[7] - 1.0).abs() < 1e-6, "last center at domain_hi");
        // Strictly increasing grid.
        assert!(c.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn rbf_zero_centers_error() {
        let mut rng = LcgRng::new(1);
        let cfg = RbfFeatureConfig::new(1, 0, 0.1);
        assert!(matches!(
            RbfFeatures::new(cfg, &mut rng),
            Err(PinnError::InvalidLayerWidth)
        ));
    }

    #[test]
    fn rbf_bad_bandwidth_error() {
        let mut rng = LcgRng::new(1);
        let cfg = RbfFeatureConfig::new(1, 4, 0.0);
        assert!(matches!(
            RbfFeatures::new(cfg, &mut rng),
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn rbf_bad_domain_error() {
        let mut rng = LcgRng::new(1);
        let mut cfg = RbfFeatureConfig::new(1, 4, 0.1);
        cfg.domain_lo = 1.0;
        cfg.domain_hi = 0.0;
        assert!(matches!(
            RbfFeatures::new(cfg, &mut rng),
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn rbf_with_centers_wrong_len() {
        let cfg = RbfFeatureConfig::new(1, 4, 0.1);
        assert!(matches!(
            RbfFeatures::with_centers(vec![0.0, 0.5], cfg),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    // ── feature values ────────────────────────────────────────────────────────

    #[test]
    fn rbf_feature_one_at_own_center_all_kinds() {
        for kind in [
            RbfKind::Gaussian,
            RbfKind::Multiquadric,
            RbfKind::InverseMultiquadric,
        ] {
            let cfg = {
                let mut c = RbfFeatureConfig::new(1, 3, 0.2);
                c.kind = kind;
                c
            };
            let feats = RbfFeatures::with_centers(vec![0.0, 0.5, 1.0], cfg)
                .expect("RbfFeatures construction with valid config should succeed");
            let enc = feats
                .encode(&[0.5])
                .expect("encode should succeed for valid input"); // exactly at center index 1
            assert!(
                (enc[1] - 1.0).abs() < 1e-5,
                "{kind:?}: feature at own center should be 1, got {}",
                enc[1]
            );
        }
    }

    #[test]
    fn rbf_gaussian_decays_with_distance() {
        let feats = RbfFeatures::with_centers(vec![0.0], RbfFeatureConfig::new(1, 1, 0.3))
            .expect("RbfFeatures construction with valid config should succeed");
        let near = feats
            .encode(&[0.1])
            .expect("encode should succeed for valid input")[0];
        let far = feats
            .encode(&[0.9])
            .expect("encode should succeed for valid input")[0];
        assert!(near > far, "Gaussian must decay: near={near} far={far}");
        assert!(far > 0.0, "Gaussian stays positive");
    }

    #[test]
    fn rbf_multiquadric_increases_with_distance() {
        let cfg = {
            let mut c = RbfFeatureConfig::new(1, 1, 0.3);
            c.kind = RbfKind::Multiquadric;
            c
        };
        let feats = RbfFeatures::with_centers(vec![0.0], cfg)
            .expect("RbfFeatures construction with valid config should succeed");
        let near = feats
            .encode(&[0.1])
            .expect("encode should succeed for valid input")[0];
        let far = feats
            .encode(&[0.9])
            .expect("encode should succeed for valid input")[0];
        assert!(
            far > near,
            "Multiquadric grows with distance: near={near} far={far}"
        );
    }

    #[test]
    fn rbf_normalized_sums_to_one() {
        let feats = make_features(RbfKind::Gaussian, true);
        for &x in &[0.0_f32, 0.27, 0.5, 0.83, 1.0] {
            let enc = feats
                .encode(&[x])
                .expect("encode should succeed for valid input");
            let total: f32 = enc.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "normalised features sum to 1 at x={x}"
            );
        }
    }

    #[test]
    fn rbf_encode_length_and_dim_error() {
        let feats = make_features(RbfKind::Gaussian, false);
        assert_eq!(
            feats
                .encode(&[0.3])
                .expect("encode should succeed for valid input")
                .len(),
            8
        );
        assert!(matches!(
            feats.encode(&[0.3, 0.4]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rbf_all_features_finite_2d() {
        let mut rng = LcgRng::new(3);
        let cfg = RbfFeatureConfig::new(2, 10, 0.4);
        let feats = RbfFeatures::new(cfg, &mut rng)
            .expect("RbfFeatures construction with valid config should succeed");
        let enc = feats
            .encode(&[0.3, 0.7])
            .expect("encode should succeed for valid input");
        assert_eq!(enc.len(), 10);
        assert!(enc.iter().all(|v| v.is_finite()));
    }

    // ── gradients ─────────────────────────────────────────────────────────────

    #[test]
    fn rbf_gradient_zero_at_gaussian_center() {
        let feats = RbfFeatures::with_centers(vec![0.4], RbfFeatureConfig::new(1, 1, 0.25))
            .expect("RbfFeatures construction with valid config should succeed");
        let g = feats
            .encode_grad(&[0.4])
            .expect("encode_grad should succeed for valid input");
        assert!(
            g[0].abs() < 1e-5,
            "Gaussian gradient vanishes at its peak: {}",
            g[0]
        );
    }

    #[test]
    fn rbf_gradient_matches_finite_difference() {
        for normalize in [false, true] {
            let feats = make_features(RbfKind::Gaussian, normalize);
            let x = 0.37_f32;
            let analytic = feats
                .encode_grad(&[x])
                .expect("encode_grad should succeed for valid input");
            let h = 1e-3_f32;
            let fp = feats
                .encode(&[x + h])
                .expect("encode should succeed for valid input");
            let fm = feats
                .encode(&[x - h])
                .expect("encode should succeed for valid input");
            for k in 0..feats.n_features() {
                let numeric = (fp[k] - fm[k]) / (2.0 * h);
                assert!(
                    (analytic[k] - numeric).abs() < 1e-2,
                    "grad mismatch (normalize={normalize}) k={k}: {} vs {}",
                    analytic[k],
                    numeric
                );
            }
        }
    }

    #[test]
    fn rbf_imq_gradient_matches_finite_difference() {
        let feats = make_features(RbfKind::InverseMultiquadric, false);
        let x = 0.62_f32;
        let analytic = feats
            .encode_grad(&[x])
            .expect("encode_grad should succeed for valid input");
        let h = 1e-3_f32;
        let fp = feats
            .encode(&[x + h])
            .expect("encode should succeed for valid input");
        let fm = feats
            .encode(&[x - h])
            .expect("encode should succeed for valid input");
        for k in 0..feats.n_features() {
            let numeric = (fp[k] - fm[k]) / (2.0 * h);
            assert!(
                (analytic[k] - numeric).abs() < 1e-2,
                "IMQ grad mismatch at k={k}"
            );
        }
    }

    // ── RBF network (interpolation) ───────────────────────────────────────────

    #[test]
    fn rbf_network_reconstructs_sine() {
        // Fit sin(πx) on [0,1] with a Gaussian RBF network; check accuracy on the
        // training points and at a held-out point.
        let mut rng = LcgRng::new(4);
        let mut cfg = RbfFeatureConfig::new(1, 12, 0.12);
        cfg.kind = RbfKind::Gaussian;
        let mut net = RbfFeatureNetwork::new(cfg, 1, 1e-8, &mut rng)
            .expect("RbfFeatureNetwork construction with valid config should succeed");

        let n = 25;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let ys: Vec<f32> = xs
            .iter()
            .map(|&x| (std::f32::consts::PI * x).sin())
            .collect();
        net.fit(&xs, &ys, n)
            .expect("fit should succeed for valid training data");

        let mut max_train_err = 0.0_f32;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let p = net
                .predict(&[x])
                .expect("prediction should succeed for valid input")[0];
            max_train_err = max_train_err.max((p - y).abs());
        }
        assert!(
            max_train_err < 0.05,
            "train interpolation error: {max_train_err}"
        );

        let xt = 0.43_f32;
        let pt = net
            .predict(&[xt])
            .expect("prediction should succeed for valid input")[0];
        let yt = (std::f32::consts::PI * xt).sin();
        assert!((pt - yt).abs() < 0.1, "held-out error: {} vs {}", pt, yt);
    }

    #[test]
    fn rbf_network_ridge_shrinks_weights() {
        let n = 25;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| (3.0 * x).sin()).collect();

        let mut rng_a = LcgRng::new(5);
        let mut net_small =
            RbfFeatureNetwork::new(RbfFeatureConfig::new(1, 12, 0.12), 1, 1e-8, &mut rng_a)
                .expect("RbfFeatureNetwork construction with valid config should succeed");
        net_small
            .fit(&xs, &ys, n)
            .expect("fit should succeed for valid training data");

        let mut rng_b = LcgRng::new(5);
        let mut net_big =
            RbfFeatureNetwork::new(RbfFeatureConfig::new(1, 12, 0.12), 1, 10.0, &mut rng_b)
                .expect("RbfFeatureNetwork construction with valid config should succeed");
        net_big
            .fit(&xs, &ys, n)
            .expect("fit should succeed for valid training data");

        assert!(
            l2(net_big.weights()) < l2(net_small.weights()),
            "stronger ridge should shrink readout weights"
        );
    }

    #[test]
    fn rbf_network_out_dim_error() {
        let mut rng = LcgRng::new(6);
        assert!(matches!(
            RbfFeatureNetwork::new(RbfFeatureConfig::new(1, 4, 0.1), 0, 1e-6, &mut rng),
            Err(PinnError::EmptyInput)
        ));
    }

    #[test]
    fn rbf_network_fit_shape_error() {
        let mut rng = LcgRng::new(7);
        let mut net = RbfFeatureNetwork::new(RbfFeatureConfig::new(1, 4, 0.2), 1, 1e-6, &mut rng)
            .expect("RbfFeatureNetwork construction with valid config should succeed");
        let xs = vec![0.1_f32, 0.2, 0.3];
        let ys = vec![0.0_f32, 1.0]; // mismatched length for n=3
        assert!(matches!(
            net.fit(&xs, &ys, 3),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }
}
