//! SimSiam — Chen & He 2021 — "Exploring Simple Siamese Representation Learning".
//!
//! SimSiam is BYOL without a momentum encoder. Both branches share the same
//! network weights; only one branch applies a stop-gradient (`sg`) on the
//! projection before computing the cosine similarity loss. In Rust (no autograd)
//! the stop-gradient is a **caller convention**: `z1` and `z2` are the
//! projections whose gradients must NOT flow through during backprop.
//!
//! ## Loss
//! ```text
//!     L = -(1/2) [ D(p1, sg(z2)) + D(p2, sg(z1)) ]
//!     D(p, z) = (p̂ · ẑ)   where  p̂ = p/‖p‖,  ẑ = z/‖z‖
//! ```
//! Negative cosine similarity is minimised → −1 is perfect alignment, 0 is
//! orthogonal, +1 is anti-parallel (collapse of the opposite kind).
//!
//! ## Collapse diagnostic
//! [`is_collapsed`] measures the standard deviation of L2-normalised
//! projections column-wise: near-zero std indicates all representations have
//! collapsed to a constant vector.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;
use crate::head::predictor::PredictorHead;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for a SimSiam head pair.
///
/// `d_proj` is the output dimension of the projector (= input to predictor).
/// `d_pred` is the hidden dimension of the predictor MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct SimSiamConfig {
    /// Projector output dimension (= predictor input & output dimension).
    pub d_proj: usize,
    /// Predictor hidden dimension.
    pub d_pred: usize,
}

impl Default for SimSiamConfig {
    fn default() -> Self {
        Self {
            d_proj: 128,
            d_pred: 64,
        }
    }
}

// ─── Loss helpers ─────────────────────────────────────────────────────────────

/// Compute D(p, z) = -(p̂ · ẑ) for one branch, mean over the batch.
///
/// Both `p` and `z` are `[N, D]` row-major flat slices. `z` is treated as a
/// stop-gradient target (the caller must not backprop through `z`).
///
/// Returns the mean of `-cos(p_i, z_i)` over `i ∈ [0, N)`.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] when `p.len()` or `z.len()` ≠ `n * d`.
pub fn simsiam_loss(p: &[f32], z: &[f32], n: usize, d: usize) -> SslResult<f32> {
    validate_batch(p, z, n, d)?;
    Ok(neg_cosine_mean(p, z, n, d))
}

/// Full symmetric SimSiam loss.
///
/// Computes `(D(p1, sg(z2)) + D(p2, sg(z1))) / 2` averaged over the batch.
/// This is the loss as defined in Chen & He 2021, Algorithm 1.
///
/// # Arguments
/// * `p1` — predictions from view 1, `[N, D]` row-major.
/// * `z2` — projections from view 2 (stop-gradient target), `[N, D]`.
/// * `p2` — predictions from view 2, `[N, D]`.
/// * `z1` — projections from view 1 (stop-gradient target), `[N, D]`.
/// * `n`  — batch size.
/// * `d`  — feature dimension.
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] for any shape mismatch.
pub fn simsiam_loss_batch(
    p1: &[f32],
    z2: &[f32],
    p2: &[f32],
    z1: &[f32],
    n: usize,
    d: usize,
) -> SslResult<f32> {
    validate_batch(p1, z2, n, d)?;
    validate_batch(p2, z1, n, d)?;
    let d1 = neg_cosine_mean(p1, z2, n, d);
    let d2 = neg_cosine_mean(p2, z1, n, d);
    Ok((d1 + d2) * 0.5)
}

// ─── Collapse diagnostic ──────────────────────────────────────────────────────

/// Detect representational collapse by measuring column-wise std of the
/// L2-normalised projection matrix.
///
/// Each row of `z` (`[N, D]`) is L2-normalised. The column-wise variance is
/// then computed; the mean of per-column standard deviations gives a scalar
/// diversity measure. If this measure is below `threshold` the network is
/// considered collapsed.
///
/// A threshold of `0.1` is a practical starting point (rows that are almost
/// identical give values close to 0).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] when `z.len() != n * d`.
pub fn is_collapsed(z: &[f32], n: usize, d: usize, threshold: f32) -> SslResult<bool> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }

    // Build the matrix of L2-normalised rows: `normed[i, j]`.
    let mut normed = vec![0.0_f64; n * d];
    for i in 0..n {
        let row = &z[i * d..(i + 1) * d];
        let norm = row
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt()
            .max(1e-12_f64);
        for j in 0..d {
            normed[i * d + j] = (row[j] as f64) / norm;
        }
    }

    // Column-wise variance: Var[col j] = E[x²] - (E[x])².
    let mut mean_std = 0.0_f64;
    let n_f = n as f64;
    for j in 0..d {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let v = normed[i * d + j];
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n_f;
        let var = (sum_sq / n_f - mean * mean).max(0.0_f64);
        mean_std += var.sqrt();
    }
    mean_std /= d as f64;

    Ok(mean_std < threshold as f64)
}

// ─── SimSiamPredictor ─────────────────────────────────────────────────────────

/// SimSiam predictor head wrapper.
///
/// Wraps a [`PredictorHead`] for ergonomic use in a SimSiam training loop:
/// ```text
///     p1 = predictor(z1_online);
///     p2 = predictor(z2_online);
///     loss = simsiam_loss_batch(&p1, &z2, &p2, &z1, n, d);
/// ```
/// Note: `z1` and `z2` are **stop-gradient** targets; in pure Rust this
/// means the caller must not update the network through them.
#[derive(Debug, Clone)]
pub struct SimSiamPredictor {
    /// Underlying predictor MLP.
    pub predictor: PredictorHead,
}

impl SimSiamPredictor {
    /// Create a new [`SimSiamPredictor`] from an existing [`PredictorHead`].
    #[must_use]
    pub fn new(predictor: PredictorHead) -> Self {
        Self { predictor }
    }

    /// Convenience constructor that allocates a [`PredictorHead`] from
    /// a [`SimSiamConfig`], using `d_proj` as both input and output dims
    /// and `d_pred` as the hidden dim.
    ///
    /// # Errors
    /// Propagates [`PredictorHead::new`] errors.
    pub fn from_config(cfg: &SimSiamConfig, rng: &mut LcgRng) -> SslResult<Self> {
        let predictor = PredictorHead::new(cfg.d_proj, cfg.d_pred, cfg.d_proj, rng)?;
        Ok(Self { predictor })
    }

    /// Apply the predictor to a single feature vector.
    ///
    /// # Errors
    /// Propagates [`PredictorHead::forward`] errors.
    pub fn forward(&self, z: &[f32]) -> SslResult<Vec<f32>> {
        self.predictor.forward(z)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that `p` and `z` both have length `n * d` and `n, d > 0`.
#[inline]
fn validate_batch(p: &[f32], z: &[f32], n: usize, d: usize) -> SslResult<()> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if p.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: p.len(),
        });
    }
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }
    Ok(())
}

/// Mean of `-(p̂ · ẑ)` over `N` rows.
///
/// Uses `f64` accumulators for numerical stability.
#[inline]
fn neg_cosine_mean(p: &[f32], z: &[f32], n: usize, d: usize) -> f32 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let p_row = &p[i * d..(i + 1) * d];
        let z_row = &z[i * d..(i + 1) * d];

        let p_norm = p_row
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt()
            .max(1e-12_f64);
        let z_norm = z_row
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt()
            .max(1e-12_f64);

        let dot: f64 = p_row
            .iter()
            .zip(z_row.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        let cos = dot / (p_norm * z_norm);
        total -= cos; // negative cosine similarity
    }
    (total / n as f64) as f32
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── basic loss properties ─────────────────────────────────────────────────

    /// Aligned vectors (identical direction) → cos = 1 → loss = −1.
    #[test]
    fn simsiam_loss_aligned_gives_minus_one() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let l = simsiam_loss(&v, &v, 1, 4).expect("simsiam_loss should succeed");
        assert!((l + 1.0).abs() < 1e-5, "loss = {l}");
    }

    /// Orthogonal vectors → cos = 0 → loss = 0.
    #[test]
    fn simsiam_loss_orthogonal_gives_zero() {
        let p = vec![1.0_f32, 0.0];
        let z = vec![0.0_f32, 1.0];
        let l = simsiam_loss(&p, &z, 1, 2).expect("simsiam_loss should succeed");
        assert!(l.abs() < 1e-5, "loss = {l}");
    }

    /// Anti-parallel vectors → cos = −1 → loss = +1.
    #[test]
    fn simsiam_loss_antiparallel_gives_plus_one() {
        let p = vec![1.0_f32, 0.0];
        let z = vec![-1.0_f32, 0.0];
        let l = simsiam_loss(&p, &z, 1, 2).expect("simsiam_loss should succeed");
        assert!((l - 1.0).abs() < 1e-5, "loss = {l}");
    }

    /// Full symmetric batch loss must equal mean of two branches.
    #[test]
    fn simsiam_loss_batch_symmetric() {
        // Choose p1 ∥ z2 and p2 ⊥ z1.
        // D(p1,z2) = −1,  D(p2,z1) = 0  → symmetric = −0.5
        let p1 = vec![1.0_f32, 0.0]; // ∥ z2
        let z2 = vec![1.0_f32, 0.0];
        let p2 = vec![0.0_f32, 1.0]; // ⊥ z1
        let z1 = vec![1.0_f32, 0.0];

        let sym = simsiam_loss_batch(&p1, &z2, &p2, &z1, 1, 2)
            .expect("simsiam_loss_batch should succeed");
        let expected = (-1.0_f32 + 0.0_f32) * 0.5;
        assert!((sym - expected).abs() < 1e-5, "sym = {sym}");
    }

    /// batch loss equals plain loss when both branches are identical.
    #[test]
    fn simsiam_loss_batch_equals_single_when_symmetric_inputs() {
        let p: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 + 0.5).collect();
        let z: Vec<f32> = (0..12).map(|i| (12 - i) as f32 * 0.1 + 0.3).collect();
        let single = simsiam_loss(&p, &z, 3, 4).expect("simsiam_loss should succeed");
        let batch =
            simsiam_loss_batch(&p, &z, &p, &z, 3, 4).expect("simsiam_loss_batch should succeed");
        assert!(
            (single - batch).abs() < 1e-5,
            "single={single} batch={batch}"
        );
    }

    // ── predictor ─────────────────────────────────────────────────────────────

    /// Predictor forward produces a vector of the correct shape.
    #[test]
    fn simsiam_predictor_forward_shape() {
        let mut rng = LcgRng::new(42);
        let cfg = SimSiamConfig {
            d_proj: 16,
            d_pred: 8,
        };
        let pred =
            SimSiamPredictor::from_config(&cfg, &mut rng).expect("from_config should succeed");
        let z = vec![0.5_f32; 16];
        let p = pred.forward(&z).expect("forward should succeed");
        assert_eq!(p.len(), 16, "output dim must equal d_proj");
    }

    // ── collapse detection ────────────────────────────────────────────────────

    /// All rows identical → std = 0 → collapsed.
    #[test]
    fn collapse_detection_constant_projections_collapsed() {
        let n = 8;
        let d = 4;
        // Every row is the same unit vector along dim 0.
        let z: Vec<f32> = (0..n * d)
            .map(|idx| if idx % d == 0 { 1.0_f32 } else { 0.0_f32 })
            .collect();
        let collapsed = is_collapsed(&z, n, d, 0.1).expect("is_collapsed should succeed");
        assert!(
            collapsed,
            "constant projections must be detected as collapsed"
        );
    }

    /// Diverse (orthogonal basis) projections → high std → not collapsed.
    #[test]
    fn collapse_detection_diverse_projections_not_collapsed() {
        let n = 4;
        let d = 4;
        // Four one-hot basis vectors — maximally diverse after normalisation.
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            z[i * d + i] = 1.0;
        }
        let collapsed = is_collapsed(&z, n, d, 0.1).expect("is_collapsed should succeed");
        assert!(
            !collapsed,
            "orthogonal projections must not be detected as collapsed"
        );
    }

    // ── error handling ────────────────────────────────────────────────────────

    /// Empty input (n=0, d=0) must return an error.
    #[test]
    fn empty_input_returns_error() {
        assert!(simsiam_loss(&[], &[], 0, 0).is_err());
        assert!(simsiam_loss_batch(&[], &[], &[], &[], 0, 0).is_err());
        assert!(is_collapsed(&[], 0, 0, 0.1).is_err());
    }

    /// Mismatched slice length must return DimensionMismatch.
    #[test]
    fn dimension_mismatch_returns_error() {
        let p = vec![1.0_f32, 0.0, 0.0]; // len=3
        let z = vec![1.0_f32, 0.0]; // len=2
        let err = simsiam_loss(&p, &z, 1, 2);
        assert!(
            matches!(err, Err(SslError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {err:?}"
        );
    }

    /// A single sample batch (n=1) must work without error.
    #[test]
    fn single_sample_valid() {
        let p = vec![0.6_f32, 0.8]; // already unit
        let z = vec![0.8_f32, 0.6];
        let l = simsiam_loss(&p, &z, 1, 2).expect("simsiam_loss should succeed");
        assert!(l.is_finite(), "loss must be finite for n=1");
    }
}
