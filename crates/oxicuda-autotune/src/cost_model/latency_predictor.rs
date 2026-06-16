//! Feature-based latency predictor using closed-form ridge regression.
//!
//! Where the [roofline model](crate::cost_model::roofline) is a *first-principles*
//! analytical bound, this predictor is a *learned* surrogate.  It extracts a
//! fixed-length feature vector from a kernel/loop-nest descriptor and fits a
//! linear model that maps features to measured latency.  The fit is the
//! standard ridge (Tikhonov-regularized) least-squares solution
//!
//! ```text
//! w = (XᵀX + λI)⁻¹ Xᵀ y
//! ```
//!
//! where `X` is the design matrix (one row per training sample, with a leading
//! bias column of ones), `y` the observed latencies, and `λ ≥ 0` the
//! regularization strength.  Regularization is *not* applied to the bias term,
//! so as `λ → ∞` the weights shrink toward zero and the prediction collapses to
//! the mean of the training targets.
//!
//! Predicted latency is passed through a softplus so the output is always a
//! finite, non-negative number (latency cannot be negative):
//!
//! ```text
//! softplus(z) = ln(1 + exp(z))
//! ```
//!
//! The normal equations are solved with an inline Gauss–Jordan inversion (the
//! system is tiny: `(F+1) × (F+1)` where `F` is the feature count), avoiding any
//! external linear-algebra dependency.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::cost_model::latency_predictor::{
//!     KernelDescriptor, LatencyPredictor,
//! };
//!
//! let mut predictor = LatencyPredictor::new(1e-3);
//! // Train on a few (descriptor, latency) pairs.
//! for m in [256u64, 512, 1024] {
//!     let d = KernelDescriptor {
//!         loop_extents: vec![m, m, m],
//!         total_flops: 2.0 * (m * m * m) as f64,
//!         bytes_moved: 3.0 * (m * m) as f64 * 4.0,
//!         vector_width: 4,
//!         parallel_degree: 32,
//!     };
//!     let latency = (m * m * m) as f64 * 1e-9; // synthetic
//!     predictor.add_sample(d, latency);
//! }
//! predictor.fit().expect("ridge fit");
//!
//! let query = KernelDescriptor {
//!     loop_extents: vec![768, 768, 768],
//!     total_flops: 2.0 * 768f64.powi(3),
//!     bytes_moved: 3.0 * 768f64.powi(2) * 4.0,
//!     vector_width: 4,
//!     parallel_degree: 32,
//! };
//! let pred = predictor.predict(&query).expect("fitted");
//! assert!(pred >= 0.0 && pred.is_finite());
//! ```

use crate::error::AutotuneError;

/// Number of engineered features extracted from a [`KernelDescriptor`].
///
/// The vector is, in order:
/// 0. `log1p(total_iterations)` — product of loop extents,
/// 1. `log1p(total_flops)`,
/// 2. `log1p(bytes_moved)`,
/// 3. `arithmetic_intensity = total_flops / bytes_moved`,
/// 4. `1 / vector_width` (inverse vector utilization),
/// 5. `1 / parallel_degree` (inverse parallel speedup),
/// 6. `log1p(max_loop_extent)`.
pub const NUM_FEATURES: usize = 7;

/// A compact descriptor of a kernel / loop nest from which features are
/// derived.  This is deliberately decoupled from any particular IR so it can be
/// populated from a [`Config`](crate::config::Config), a Halide-style nest, or
/// hand-written estimates.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDescriptor {
    /// Extents (trip counts) of each loop in the nest.
    pub loop_extents: Vec<u64>,
    /// Total floating-point operations performed by the kernel.
    pub total_flops: f64,
    /// Total bytes moved through memory (loads + stores, in bytes).
    pub bytes_moved: f64,
    /// SIMD / vector width used by the innermost loop (lanes).
    pub vector_width: u32,
    /// Degree of parallelism (e.g. number of concurrent threads/warps).
    pub parallel_degree: u32,
}

impl KernelDescriptor {
    /// Product of all loop extents — the total number of iteration points.
    #[must_use]
    pub fn total_iterations(&self) -> f64 {
        self.loop_extents
            .iter()
            .fold(1.0_f64, |acc, &e| acc * e as f64)
    }

    /// Arithmetic intensity `FLOPs / bytes`.  Returns `0` when no bytes move.
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.bytes_moved > 0.0 {
            self.total_flops / self.bytes_moved
        } else {
            0.0
        }
    }

    /// Extracts the fixed-length feature vector for this descriptor.
    ///
    /// The returned vector always has length [`NUM_FEATURES`], regardless of how
    /// many loops the nest has, so design matrices stay rectangular.
    #[must_use]
    pub fn feature_vector(&self) -> [f64; NUM_FEATURES] {
        let max_extent = self.loop_extents.iter().copied().max().unwrap_or(0) as f64;
        let vector_width = f64::from(self.vector_width.max(1));
        let parallel_degree = f64::from(self.parallel_degree.max(1));
        [
            self.total_iterations().ln_1p(),
            self.total_flops.max(0.0).ln_1p(),
            self.bytes_moved.max(0.0).ln_1p(),
            self.arithmetic_intensity(),
            1.0 / vector_width,
            1.0 / parallel_degree,
            max_extent.ln_1p(),
        ]
    }
}

/// Numerically stable softplus: `ln(1 + e^z)`.
///
/// For large `z` this returns `z` directly to avoid overflow; for very negative
/// `z` it returns `exp(z)` (the limiting behaviour), keeping the output finite
/// and strictly non-negative.
fn softplus(z: f64) -> f64 {
    if z > 30.0 {
        z
    } else if z < -30.0 {
        z.exp()
    } else {
        z.exp().ln_1p()
    }
}

/// A ridge-regression latency predictor over [`KernelDescriptor`] features.
#[derive(Debug, Clone)]
pub struct LatencyPredictor {
    /// Stored feature rows (each [`NUM_FEATURES`] long).
    samples: Vec<[f64; NUM_FEATURES]>,
    /// Observed latencies, aligned with `samples`.
    targets: Vec<f64>,
    /// Regularization strength `λ ≥ 0`.
    lambda: f64,
    /// Fitted weight vector of length `NUM_FEATURES + 1` (index 0 is bias),
    /// or `None` until [`LatencyPredictor::fit`] has been called.
    weights: Option<Vec<f64>>,
    /// Mean of the training targets, used as the fallback / shrinkage limit.
    target_mean: f64,
}

impl LatencyPredictor {
    /// Creates an empty predictor with regularization strength `lambda`.
    ///
    /// `lambda` is clamped to be non-negative.
    #[must_use]
    pub fn new(lambda: f64) -> Self {
        Self {
            samples: Vec::new(),
            targets: Vec::new(),
            lambda: if lambda.is_finite() && lambda > 0.0 {
                lambda
            } else {
                0.0
            },
            weights: None,
            target_mean: 0.0,
        }
    }

    /// Adds a training pair `(descriptor, latency)`.  Invalidates any prior fit.
    pub fn add_sample(&mut self, descriptor: KernelDescriptor, latency: f64) {
        self.samples.push(descriptor.feature_vector());
        self.targets.push(latency);
        self.weights = None;
    }

    /// Adds a pre-extracted feature row directly.  Invalidates any prior fit.
    pub fn add_feature_row(&mut self, features: [f64; NUM_FEATURES], latency: f64) {
        self.samples.push(features);
        self.targets.push(latency);
        self.weights = None;
    }

    /// Number of stored training samples.
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Whether the predictor has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.weights.is_some()
    }

    /// Fits the ridge model by solving the normal equations
    /// `(XᵀX + λI) w = Xᵀ y` via Gauss–Jordan inversion.
    ///
    /// A leading bias column of ones is prepended to `X`, and the bias term is
    /// excluded from regularization.  Adding `λ` to the diagonal guarantees the
    /// system is solvable even when features are collinear.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if there are no samples or if
    /// the regularized normal-equation matrix is singular (which, for `λ > 0`,
    /// should not occur).
    pub fn fit(&mut self) -> Result<(), AutotuneError> {
        let n = self.samples.len();
        if n == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "cannot fit latency predictor with zero samples".to_string(),
            ));
        }

        // Record the target mean (used both as a shrinkage limit and a check).
        let sum: f64 = self.targets.iter().sum();
        self.target_mean = sum / n as f64;

        // Design width: bias + NUM_FEATURES.
        let d = NUM_FEATURES + 1;

        // Build A = XᵀX + λI' where I' is identity except the bias diagonal is 0.
        let mut a = vec![0.0_f64; d * d];
        let mut xty = vec![0.0_f64; d];

        for (row, &y) in self.samples.iter().zip(self.targets.iter()) {
            // Augmented row: [1, f0, f1, ...].
            let mut aug = [0.0_f64; NUM_FEATURES + 1];
            aug[0] = 1.0;
            aug[1..].copy_from_slice(row);

            for i in 0..d {
                xty[i] += aug[i] * y;
                for j in 0..d {
                    a[i * d + j] += aug[i] * aug[j];
                }
            }
        }
        // Tikhonov regularization on every non-bias diagonal entry.
        for i in 1..d {
            a[i * d + i] += self.lambda;
        }

        let inv = invert_matrix(&a, d).ok_or_else(|| {
            AutotuneError::BenchmarkFailed(
                "normal-equation matrix is singular; increase lambda".to_string(),
            )
        })?;

        // w = A⁻¹ (Xᵀ y).
        let mut w = vec![0.0_f64; d];
        for i in 0..d {
            let mut acc = 0.0_f64;
            for j in 0..d {
                acc += inv[i * d + j] * xty[j];
            }
            w[i] = acc;
        }

        self.weights = Some(w);
        Ok(())
    }

    /// Predicts the latency for `descriptor`, clamped to be finite and `≥ 0` via
    /// softplus.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if the model has not been
    /// fitted.
    pub fn predict(&self, descriptor: &KernelDescriptor) -> Result<f64, AutotuneError> {
        self.predict_features(&descriptor.feature_vector())
    }

    /// Predicts the latency directly from a feature row.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if the model has not been
    /// fitted.
    pub fn predict_features(&self, features: &[f64; NUM_FEATURES]) -> Result<f64, AutotuneError> {
        let w = self.weights.as_ref().ok_or_else(|| {
            AutotuneError::BenchmarkFailed("latency predictor is not fitted".to_string())
        })?;
        // Linear response: bias + Σ w_i f_i.
        let mut z = w[0];
        for (i, &f) in features.iter().enumerate() {
            z += w[i + 1] * f;
        }
        // Softplus keeps the prediction finite and non-negative.
        Ok(softplus(z))
    }

    /// The mean of the training targets recorded at the last fit.
    #[must_use]
    pub fn target_mean(&self) -> f64 {
        self.target_mean
    }

    /// The fitted weight vector (`bias` at index 0), or `None` if unfitted.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Mean-squared error of the current fit over the training set.  Returns
    /// `None` if unfitted.
    #[must_use]
    pub fn train_mse(&self) -> Option<f64> {
        let _w = self.weights.as_ref()?;
        let mut acc = 0.0_f64;
        for (row, &y) in self.samples.iter().zip(self.targets.iter()) {
            // Safe: weights present, so predict_features returns Ok.
            let pred = self.predict_features(row).unwrap_or(0.0);
            let err = pred - y;
            acc += err * err;
        }
        Some(acc / self.samples.len() as f64)
    }
}

/// Inverts a `d × d` row-major matrix via Gauss–Jordan elimination with partial
/// pivoting.  Returns `None` if the matrix is singular.
fn invert_matrix(mat: &[f64], d: usize) -> Option<Vec<f64>> {
    // Augmented [A | I].
    let mut aug = vec![0.0_f64; d * 2 * d];
    for i in 0..d {
        for j in 0..d {
            aug[i * 2 * d + j] = mat[i * d + j];
        }
        aug[i * 2 * d + (d + i)] = 1.0;
    }

    for col in 0..d {
        // Partial pivot: find the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut best = aug[col * 2 * d + col].abs();
        for r in (col + 1)..d {
            let v = aug[r * 2 * d + col].abs();
            if v > best {
                best = v;
                pivot_row = r;
            }
        }
        if best < 1e-12 {
            return None; // Singular.
        }
        // Swap the pivot row into place.
        if pivot_row != col {
            for k in 0..(2 * d) {
                aug.swap(col * 2 * d + k, pivot_row * 2 * d + k);
            }
        }
        // Normalize the pivot row.
        let pivot = aug[col * 2 * d + col];
        for k in 0..(2 * d) {
            aug[col * 2 * d + k] /= pivot;
        }
        // Eliminate the column from all other rows.
        for r in 0..d {
            if r == col {
                continue;
            }
            let factor = aug[r * 2 * d + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..(2 * d) {
                let sub = factor * aug[col * 2 * d + k];
                aug[r * 2 * d + k] -= sub;
            }
        }
    }

    // Extract the right half — the inverse.
    let mut inv = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..d {
            inv[i * d + j] = aug[i * 2 * d + (d + j)];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(m: u64) -> KernelDescriptor {
        KernelDescriptor {
            loop_extents: vec![m, m, m],
            total_flops: 2.0 * (m * m * m) as f64,
            bytes_moved: 3.0 * (m * m) as f64 * 4.0,
            vector_width: 4,
            parallel_degree: 32,
        }
    }

    #[test]
    fn feature_vector_has_stable_length() {
        let f_small = descriptor(8).feature_vector();
        let f_large = KernelDescriptor {
            loop_extents: vec![4, 4, 4, 4, 4, 4],
            total_flops: 1.0e6,
            bytes_moved: 1.0e4,
            vector_width: 8,
            parallel_degree: 64,
        }
        .feature_vector();
        assert_eq!(f_small.len(), NUM_FEATURES);
        assert_eq!(f_large.len(), NUM_FEATURES);
    }

    #[test]
    fn arithmetic_intensity_and_iterations() {
        let d = descriptor(10);
        assert!((d.total_iterations() - 1000.0).abs() < 1e-9);
        // 2*1000 FLOPs / (3*100*4 bytes) = 2000 / 1200.
        assert!((d.arithmetic_intensity() - 2000.0 / 1200.0).abs() < 1e-9);
        // Zero bytes => intensity 0, no div-by-zero.
        let z = KernelDescriptor {
            loop_extents: vec![1],
            total_flops: 5.0,
            bytes_moved: 0.0,
            vector_width: 1,
            parallel_degree: 1,
        };
        assert_eq!(z.arithmetic_intensity(), 0.0);
    }

    /// A descriptor whose features are all non-degenerate (vector width and
    /// parallel degree vary with the size index, so no feature column is
    /// constant across the training set).
    fn varied_descriptor(m: u64, idx: usize) -> KernelDescriptor {
        KernelDescriptor {
            loop_extents: vec![m, m, m],
            total_flops: 2.0 * (m * m * m) as f64,
            bytes_moved: 3.0 * (m * m) as f64 * 4.0,
            vector_width: 2u32 + (idx as u32 % 6), // 2..=7
            parallel_degree: 8u32 + (idx as u32 % 4) * 8, // 8,16,24,32,...
        }
    }

    // Ridge fit recovers a known linear relation on synthetic data: the fitted
    // model reproduces the (linear, strictly-positive) targets with very low
    // error.  We assert on prediction accuracy rather than exact weight values,
    // since collinear feature columns admit many weight vectors with the same
    // fit — the *predictions* are what the model exists to produce.
    #[test]
    fn recovers_known_linear_relation() {
        // Ground-truth linear model: bias + per-feature weights.  Chosen so the
        // response stays comfortably positive (softplus is ~identity there), so
        // the recovered latency matches the target in the latency domain.
        let true_w = [50.0_f64, 0.3, 0.05, 0.02, 4.0, 1.0, 0.2, 1.5];
        let mut predictor = LatencyPredictor::new(1e-8);

        let sizes = [16u64, 32, 64, 128, 256, 384, 512, 640, 768, 1024];
        let mut samples = Vec::new();
        for (idx, &m) in sizes.iter().enumerate() {
            let feats = varied_descriptor(m, idx).feature_vector();
            let mut y = true_w[0];
            for (i, &f) in feats.iter().enumerate() {
                y += true_w[i + 1] * f;
            }
            assert!(y > 5.0, "target must stay positive for softplus identity");
            predictor.add_feature_row(feats, y);
            samples.push((feats, y));
        }
        predictor.fit().expect("fit ok");

        // Train MSE must be tiny: the model reproduces the linear targets.
        let mse = predictor.train_mse().expect("fitted");
        assert!(
            mse < 1e-3,
            "ridge should fit a linear relation tightly, mse={mse}"
        );

        // Spot-check individual predictions against the known targets.
        for (feats, y) in &samples {
            let pred = predictor.predict_features(feats).expect("fitted");
            assert!(
                (pred - y).abs() < 1e-1,
                "prediction {pred} should match target {y}"
            );
        }
    }

    // λ → ∞ shrinks weights → 0 and the prediction collapses to the mean.
    #[test]
    fn large_lambda_shrinks_to_mean() {
        let mut predictor = LatencyPredictor::new(1e12);
        let targets = [3.0_f64, 7.0, 11.0, 5.0];
        for (k, &y) in targets.iter().enumerate() {
            let m = 64 * (k as u64 + 1);
            predictor.add_sample(descriptor(m), y);
        }
        predictor.fit().expect("fit ok");

        let w = predictor.weights().expect("weights");
        // Non-bias weights are essentially zero under heavy regularization.
        for (i, &wi) in w.iter().enumerate().skip(1) {
            assert!(wi.abs() < 1e-3, "weight {i} should shrink to ~0, got {wi}");
        }
        // Bias absorbs the target mean.
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        assert!(
            (w[0] - mean).abs() < 1e-2,
            "bias should approach target mean {mean}, got {}",
            w[0]
        );
        // Prediction is the softplus of (approximately) the mean, finite & >= 0.
        let pred = predictor.predict(&descriptor(999)).expect("fitted");
        assert!(pred.is_finite() && pred >= 0.0);
        assert!(
            (pred - softplus(mean)).abs() < 1e-2,
            "prediction should collapse toward softplus(mean): got {pred}"
        );
    }

    // Prediction is always finite and non-negative via softplus.
    #[test]
    fn prediction_is_finite_and_nonnegative() {
        let mut predictor = LatencyPredictor::new(1e-3);
        // Deliberately train so the linear response can go negative.
        predictor.add_feature_row([1.0; NUM_FEATURES], -100.0);
        predictor.add_feature_row([2.0; NUM_FEATURES], -50.0);
        predictor.add_feature_row([3.0; NUM_FEATURES], -10.0);
        predictor.fit().expect("fit ok");

        for m in [1u64, 50, 1000, 100_000] {
            let p = predictor.predict(&descriptor(m)).expect("fitted");
            assert!(p.is_finite(), "prediction must be finite for m={m}");
            assert!(p >= 0.0, "prediction must be >= 0 for m={m}, got {p}");
        }
    }

    // Adding λ improves conditioning: solvable even with perfectly collinear
    // features (which would make XᵀX singular at λ = 0).
    #[test]
    fn lambda_makes_collinear_features_solvable() {
        // All-identical feature rows => rank-deficient design.
        let collinear = [1.0_f64; NUM_FEATURES];

        // Without regularization the normal equations are singular.
        let mut unreg = LatencyPredictor::new(0.0);
        unreg.add_feature_row(collinear, 1.0);
        unreg.add_feature_row(collinear, 1.0);
        unreg.add_feature_row(collinear, 1.0);
        assert!(
            unreg.fit().is_err(),
            "unregularized fit on collinear data should fail (singular)"
        );

        // With λ > 0 the system becomes positive-definite and solvable.
        let mut reg = LatencyPredictor::new(1.0);
        reg.add_feature_row(collinear, 1.0);
        reg.add_feature_row(collinear, 1.0);
        reg.add_feature_row(collinear, 1.0);
        assert!(reg.fit().is_ok(), "regularized fit should succeed");
        let pred = reg.predict_features(&collinear).expect("fitted");
        assert!(pred.is_finite() && pred >= 0.0);
    }

    #[test]
    fn fit_without_samples_errors() {
        let mut predictor = LatencyPredictor::new(1.0);
        assert!(predictor.fit().is_err());
    }

    #[test]
    fn predict_before_fit_errors() {
        let predictor = LatencyPredictor::new(1.0);
        assert!(predictor.predict(&descriptor(64)).is_err());
    }

    #[test]
    fn matrix_inversion_roundtrip() {
        // A simple invertible 3x3.
        let a = vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let inv = invert_matrix(&a, 3).expect("invertible");
        // A * inv should be identity.
        let mut prod = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[i * 3 + k] * inv[k * 3 + j];
                }
                prod[i * 3 + j] = s;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i * 3 + j] - expected).abs() < 1e-9);
            }
        }
        // Singular matrix returns None.
        let singular = vec![1.0, 2.0, 2.0, 4.0];
        assert!(invert_matrix(&singular, 2).is_none());
    }
}
