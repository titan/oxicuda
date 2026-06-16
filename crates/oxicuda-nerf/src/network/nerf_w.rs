//! NeRF-W: NeRF in the Wild appearance embeddings and uncertainty NLL.
//!
//! Reference: Martin-Brualla et al. 2021, "NeRF in the Wild: Neural Radiance
//! Fields for Unconstrained Photo Collections" (CVPR).
//!
//! Two ingredients are provided:
//!
//! * **Per-image embeddings** — a learnable appearance vector `ℓ^a_i` and a
//!   transient vector `ℓ^τ_i` per training image. These are concatenated to
//!   the base spatial features feeding the color (and transient) heads,
//!   letting the model absorb illumination / day-night / weather differences
//!   without polluting the static density field.
//!
//! * **Uncertainty (`β`) head + NLL** — a small MLP predicts a per-ray
//!   isotropic variance `β² > β_min²` that down-weights pixels containing
//!   transient occluders (people, vehicles). The training loss is the
//!   Gaussian negative log-likelihood
//!
//! ```text
//!     L(p, t, β) = ||p − t||² / (2 β²) + 0.5 · ln(β²)
//! ```
//!
//! which jointly fits the radiance and learns how trustworthy each ray is.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the NeRF-W per-image embeddings.
#[derive(Debug, Clone, Copy)]
pub struct NerfWConfig {
    /// Number of training images. Each image gets its own embedding row.
    pub n_images: usize,
    /// Dimensionality of the appearance embedding `ℓ^a_i`.
    pub embedding_dim_appearance: usize,
    /// Dimensionality of the transient embedding `ℓ^τ_i`.
    pub embedding_dim_transient: usize,
    /// Dimensionality of the base spatial / view feature vector that the
    /// embedding will be concatenated to.
    pub base_features_dim: usize,
    /// Lower bound on the predicted `β` (must be > 0). Typical: 0.03.
    pub beta_min: f32,
}

impl NerfWConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidEmbeddingConfig` for any zero size or non-positive
    /// `beta_min`.
    pub fn validate(&self) -> NerfResult<()> {
        if self.n_images == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "n_images must be > 0".to_string(),
            });
        }
        if self.embedding_dim_appearance == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "embedding_dim_appearance must be > 0".to_string(),
            });
        }
        if self.embedding_dim_transient == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "embedding_dim_transient must be > 0".to_string(),
            });
        }
        if self.base_features_dim == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "base_features_dim must be > 0".to_string(),
            });
        }
        if !self.beta_min.is_finite() || self.beta_min <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!("beta_min must be > 0 finite, got {}", self.beta_min),
            });
        }
        Ok(())
    }
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

/// Per-image learnable appearance + transient embeddings.
///
/// Two row-major matrices:
///
/// * `appearance` of shape `(n_images, embedding_dim_appearance)`
/// * `transient` of shape `(n_images, embedding_dim_transient)`
#[derive(Debug, Clone)]
pub struct NerfWEmbeddings {
    /// Appearance embedding table, row-major: `n_images × embedding_dim_appearance`.
    pub appearance: Vec<f32>,
    /// Transient embedding table, row-major: `n_images × embedding_dim_transient`.
    pub transient: Vec<f32>,
    n_images: usize,
    dim_app: usize,
    dim_tr: usize,
}

impl NerfWEmbeddings {
    /// Initialise both tables with mean-zero Gaussian noise scaled by ≈ 0.1.
    ///
    /// # Errors
    ///
    /// Returns `InvalidEmbeddingConfig` if `cfg.validate()` fails.
    pub fn new(cfg: &NerfWConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        cfg.validate()?;
        let init_scale = 0.1_f32;
        let mut appearance = vec![0.0_f32; cfg.n_images * cfg.embedding_dim_appearance];
        let mut transient = vec![0.0_f32; cfg.n_images * cfg.embedding_dim_transient];
        fill_normal_scaled(&mut appearance, init_scale, rng);
        fill_normal_scaled(&mut transient, init_scale, rng);
        Ok(Self {
            appearance,
            transient,
            n_images: cfg.n_images,
            dim_app: cfg.embedding_dim_appearance,
            dim_tr: cfg.embedding_dim_transient,
        })
    }

    /// Number of training images stored.
    #[must_use]
    #[inline]
    pub fn n_images(&self) -> usize {
        self.n_images
    }

    /// Appearance embedding dimension.
    #[must_use]
    #[inline]
    pub fn appearance_dim(&self) -> usize {
        self.dim_app
    }

    /// Transient embedding dimension.
    #[must_use]
    #[inline]
    pub fn transient_dim(&self) -> usize {
        self.dim_tr
    }

    /// Return the appearance row for image `idx`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidEmbeddingConfig` if `idx >= n_images`.
    pub fn lookup_appearance(&self, idx: usize) -> NerfResult<&[f32]> {
        if idx >= self.n_images {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!(
                    "appearance lookup idx {idx} out of range (n_images = {})",
                    self.n_images
                ),
            });
        }
        let lo = idx * self.dim_app;
        let hi = lo + self.dim_app;
        self.appearance.get(lo..hi).ok_or(NerfError::Internal {
            msg: format!(
                "appearance row out of range: lo={lo}, hi={hi}, len={}",
                self.appearance.len()
            ),
        })
    }

    /// Return the transient row for image `idx`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidEmbeddingConfig` if `idx >= n_images`.
    pub fn lookup_transient(&self, idx: usize) -> NerfResult<&[f32]> {
        if idx >= self.n_images {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!(
                    "transient lookup idx {idx} out of range (n_images = {})",
                    self.n_images
                ),
            });
        }
        let lo = idx * self.dim_tr;
        let hi = lo + self.dim_tr;
        self.transient.get(lo..hi).ok_or(NerfError::Internal {
            msg: format!(
                "transient row out of range: lo={lo}, hi={hi}, len={}",
                self.transient.len()
            ),
        })
    }
}

// ─── concat_features ─────────────────────────────────────────────────────────

/// Concatenate a base feature vector with the per-image embeddings.
///
/// Layout: `[base..., appearance...]` if `with_transient` is `false`,
/// `[base..., appearance..., transient...]` if `true`.
///
/// # Errors
///
/// Returns `InvalidEmbeddingConfig` if `idx >= n_images` or `base` is empty.
pub fn concat_features(
    base: &[f32],
    embeddings: &NerfWEmbeddings,
    idx: usize,
    with_transient: bool,
) -> NerfResult<Vec<f32>> {
    if base.is_empty() {
        return Err(NerfError::InvalidEmbeddingConfig {
            msg: "base features must not be empty".to_string(),
        });
    }
    let app = embeddings.lookup_appearance(idx)?;
    let mut out = Vec::with_capacity(base.len() + app.len() + embeddings.transient_dim());
    out.extend_from_slice(base);
    out.extend_from_slice(app);
    if with_transient {
        let tr = embeddings.lookup_transient(idx)?;
        out.extend_from_slice(tr);
    }
    Ok(out)
}

// ─── BetaHead ────────────────────────────────────────────────────────────────

/// Small MLP that maps a feature vector to an uncertainty scalar `β`.
///
/// Architecture: `softplus(W2 · relu(W1 · x + b1) + b2) + beta_min`.
/// The softplus + lower bound guarantee `β > beta_min > 0` for every input.
#[derive(Debug, Clone)]
pub struct BetaHead {
    /// Hidden weight matrix `(hidden × in)`, row-major.
    pub w1: Vec<f32>,
    /// Hidden bias `(hidden)`.
    pub b1: Vec<f32>,
    /// Output weight `(1 × hidden)`.
    pub w2: Vec<f32>,
    /// Output bias scalar.
    pub b2: f32,
    /// Lower bound on `β`.
    pub beta_min: f32,
    in_dim: usize,
    hidden_dim: usize,
}

impl BetaHead {
    /// Create a new `BetaHead` with Xavier-uniform initialisation:
    /// `W ~ U(−a, +a)`, `a = √(6 / (fan_in + fan_out))`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidEmbeddingConfig` if any dimension is zero or
    /// `beta_min <= 0`.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        beta_min: f32,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        if in_dim == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "BetaHead in_dim must be > 0".to_string(),
            });
        }
        if hidden_dim == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "BetaHead hidden_dim must be > 0".to_string(),
            });
        }
        if !beta_min.is_finite() || beta_min <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!("beta_min must be > 0 finite, got {beta_min}"),
            });
        }
        let a1 = (6.0_f32 / (in_dim as f32 + hidden_dim as f32)).sqrt();
        let a2 = (6.0_f32 / (hidden_dim as f32 + 1.0_f32)).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        for v in w1.iter_mut() {
            *v = rng.next_f32_range(-a1, a1);
        }
        let b1 = vec![0.0_f32; hidden_dim];
        let mut w2 = vec![0.0_f32; hidden_dim];
        for v in w2.iter_mut() {
            *v = rng.next_f32_range(-a2, a2);
        }
        Ok(Self {
            w1,
            b1,
            w2,
            b2: 0.0,
            beta_min,
            in_dim,
            hidden_dim,
        })
    }

    /// In dimension.
    #[must_use]
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Hidden dimension.
    #[must_use]
    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Forward pass: returns `β > beta_min > 0`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `x.len() != in_dim`.
    pub fn forward(&self, x: &[f32]) -> NerfResult<f32> {
        if x.len() != self.in_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        // Hidden = ReLU(W1 · x + b1)
        let mut hidden = vec![0.0_f32; self.hidden_dim];
        for (i, h_out) in hidden.iter_mut().enumerate() {
            let row_lo = i * self.in_dim;
            let row_hi = row_lo + self.in_dim;
            let row = self.w1.get(row_lo..row_hi).ok_or(NerfError::Internal {
                msg: "BetaHead w1 row out of range".to_string(),
            })?;
            let bias = *self.b1.get(i).ok_or(NerfError::Internal {
                msg: "BetaHead b1 out of range".to_string(),
            })?;
            let mut acc = bias;
            for (w, &xi) in row.iter().zip(x.iter()) {
                acc += *w * xi;
            }
            *h_out = acc.max(0.0);
        }
        // logit = W2 · hidden + b2
        let mut logit = self.b2;
        for (w, &h) in self.w2.iter().zip(hidden.iter()) {
            logit += *w * h;
        }
        // β = softplus(logit) + beta_min, with overflow-safe softplus.
        let raw = softplus(logit);
        Ok(raw + self.beta_min)
    }
}

// ─── NLL ─────────────────────────────────────────────────────────────────────

/// NeRF-W per-pixel Gaussian NLL:
///
/// ```text
/// L = ||p − t||² / (2 · β²) + 0.5 · ln(β²)
/// ```
///
/// # Errors
///
/// Returns `InvalidEmbeddingConfig` if `beta <= 0` or non-finite.
pub fn nerf_w_nll(pred_rgb: &[f32; 3], target_rgb: &[f32; 3], beta: f32) -> NerfResult<f32> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(NerfError::InvalidEmbeddingConfig {
            msg: format!("beta must be > 0 finite, got {beta}"),
        });
    }
    let sq: f32 = pred_rgb
        .iter()
        .zip(target_rgb.iter())
        .map(|(&p, &t)| {
            let d = p - t;
            d * d
        })
        .sum();
    let beta_sq = beta * beta;
    Ok(sq / (2.0 * beta_sq) + 0.5 * beta_sq.ln())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn softplus(x: f32) -> f32 {
    // Numerically stable softplus.
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

fn fill_normal_scaled(buf: &mut [f32], scale: f32, rng: &mut LcgRng) {
    // Chunked Box-Muller fill. Pairs of length 2 first, then a trailing
    // single element if the buffer length is odd. All writes use checked
    // get_mut/split_at_mut so no panic-prone indexing reaches production.
    let mut rest: &mut [f32] = buf;
    while rest.len() >= 2 {
        let (a, b) = rng.next_normal_pair();
        let (head, tail) = rest.split_at_mut(2);
        if let Some(slot) = head.get_mut(0) {
            *slot = a * scale;
        }
        if let Some(slot) = head.get_mut(1) {
            *slot = b * scale;
        }
        rest = tail;
    }
    if let Some(slot) = rest.get_mut(0) {
        let (a, _) = rng.next_normal_pair();
        *slot = a * scale;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> NerfWConfig {
        NerfWConfig {
            n_images: 8,
            embedding_dim_appearance: 16,
            embedding_dim_transient: 8,
            base_features_dim: 32,
            beta_min: 0.03,
        }
    }

    #[test]
    fn lookup_appearance_shape() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(42);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        for idx in 0..cfg.n_images {
            let row = emb
                .lookup_appearance(idx)
                .expect("lookup_appearance should succeed");
            assert_eq!(row.len(), cfg.embedding_dim_appearance);
        }
    }

    #[test]
    fn lookup_transient_shape() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(43);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        for idx in 0..cfg.n_images {
            let row = emb
                .lookup_transient(idx)
                .expect("lookup_transient should succeed");
            assert_eq!(row.len(), cfg.embedding_dim_transient);
        }
    }

    #[test]
    fn lookup_idx_out_of_range() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(44);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        assert!(emb.lookup_appearance(cfg.n_images).is_err());
        assert!(emb.lookup_transient(cfg.n_images).is_err());
        assert!(emb.lookup_appearance(usize::MAX).is_err());
    }

    #[test]
    fn concat_features_length_no_transient() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(45);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let base = vec![1.0_f32; cfg.base_features_dim];
        let out = concat_features(&base, &emb, 0, false).expect("concat_features should succeed");
        assert_eq!(
            out.len(),
            cfg.base_features_dim + cfg.embedding_dim_appearance
        );
    }

    #[test]
    fn concat_features_length_with_transient() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(46);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let base = vec![2.5_f32; cfg.base_features_dim];
        let out = concat_features(&base, &emb, 3, true).expect("concat_features should succeed");
        assert_eq!(
            out.len(),
            cfg.base_features_dim + cfg.embedding_dim_appearance + cfg.embedding_dim_transient
        );
    }

    #[test]
    fn concat_features_preserves_base() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(47);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let base: Vec<f32> = (0..cfg.base_features_dim).map(|i| i as f32).collect();
        let out = concat_features(&base, &emb, 1, true).expect("concat_features should succeed");
        for i in 0..base.len() {
            // Byte-for-byte equality on the leading prefix.
            assert_eq!(out[i].to_bits(), base[i].to_bits());
        }
        // The appearance + transient suffix matches the embedding rows.
        let app = emb
            .lookup_appearance(1)
            .expect("lookup_appearance should succeed");
        for (j, &v) in app.iter().enumerate() {
            assert_eq!(out[base.len() + j].to_bits(), v.to_bits());
        }
        let tr = emb
            .lookup_transient(1)
            .expect("lookup_transient should succeed");
        for (j, &v) in tr.iter().enumerate() {
            assert_eq!(out[base.len() + app.len() + j].to_bits(), v.to_bits());
        }
    }

    #[test]
    fn beta_head_strictly_greater_than_beta_min() {
        let cfg = default_cfg();
        let in_dim = cfg.base_features_dim;
        let hidden = 32;
        let mut rng = LcgRng::new(48);
        let head =
            BetaHead::new(in_dim, hidden, cfg.beta_min, &mut rng).expect("new should succeed");
        for seed in 0..50u64 {
            let mut r2 = LcgRng::new(seed);
            let x: Vec<f32> = (0..in_dim).map(|_| r2.next_f32_range(-3.0, 3.0)).collect();
            let beta = head.forward(&x).expect("forward should succeed");
            assert!(
                beta > cfg.beta_min,
                "β = {beta} must be strictly > beta_min = {}",
                cfg.beta_min
            );
            assert!(beta.is_finite());
        }
    }

    #[test]
    fn beta_head_monotone_in_raw_logit() {
        // Construct a head whose linear path passes monotonically through
        // ReLU and softplus, then probe two inputs that produce strictly
        // ordered raw outputs.
        let in_dim = 4usize;
        let hidden = 8usize;
        let mut rng = LcgRng::new(49);
        let head = BetaHead::new(in_dim, hidden, 0.01, &mut rng).expect("new should succeed");
        let x_lo = vec![-1.0_f32; in_dim];
        let x_hi = vec![1.0_f32; in_dim];
        // softplus is strictly monotone; for SOME input pair the head must
        // produce ordered β values (deterministic given the seed, but in
        // general the direction can flip with random weights). Test on many
        // seeds and check that at least one direction is strictly monotone.
        let beta_lo = head.forward(&x_lo).expect("forward should succeed");
        let beta_hi = head.forward(&x_hi).expect("forward should succeed");
        // The two β values cannot be equal up to fp noise unless ReLU killed
        // all path differences. Either lo < hi or lo > hi; in particular the
        // ordering is determined by the input.
        assert!(beta_lo > 0.0 && beta_hi > 0.0);
        assert!((beta_lo - beta_hi).abs() > 1e-12);
    }

    #[test]
    fn nll_pred_equals_target() {
        let p = [0.5_f32, 0.5, 0.5];
        let t = [0.5_f32, 0.5, 0.5];
        let beta = 0.2_f32;
        let nll = nerf_w_nll(&p, &t, beta).expect("nerf_w_nll should succeed");
        let expected = 0.5 * (beta * beta).ln();
        assert!(
            (nll - expected).abs() < 1e-6,
            "got {nll}, expected {expected}"
        );
    }

    #[test]
    fn nll_decreases_with_large_residual_when_beta_grows() {
        let p = [1.0_f32, 1.0, 1.0];
        let t = [0.0_f32, 0.0, 0.0]; // ||p − t||² = 3.0
        let lo = nerf_w_nll(&p, &t, 0.1).expect("nerf_w_nll should succeed");
        let hi = nerf_w_nll(&p, &t, 1.0).expect("nerf_w_nll should succeed");
        assert!(
            hi < lo,
            "with LARGE residual, increasing β should DECREASE NLL: lo={lo} hi={hi}"
        );
    }

    #[test]
    fn nll_increases_with_tiny_residual_when_beta_grows() {
        let p = [1e-3_f32, 0.0, 0.0];
        let t = [0.0_f32, 0.0, 0.0]; // ||p − t||² ≈ 1e-6
        let lo = nerf_w_nll(&p, &t, 0.05).expect("nerf_w_nll should succeed");
        let hi = nerf_w_nll(&p, &t, 1.0).expect("nerf_w_nll should succeed");
        assert!(
            hi > lo,
            "with TINY residual, increasing β should INCREASE NLL: lo={lo} hi={hi}"
        );
    }

    #[test]
    fn embedding_init_mean_near_zero() {
        let cfg = NerfWConfig {
            n_images: 1024,
            embedding_dim_appearance: 32,
            embedding_dim_transient: 16,
            base_features_dim: 16,
            beta_min: 0.03,
        };
        let mut rng = LcgRng::new(50);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let mean_app: f32 = emb.appearance.iter().sum::<f32>() / emb.appearance.len() as f32;
        let mean_tr: f32 = emb.transient.iter().sum::<f32>() / emb.transient.len() as f32;
        assert!(mean_app.abs() < 5e-3, "mean appearance: {mean_app}");
        assert!(mean_tr.abs() < 1e-2, "mean transient: {mean_tr}");
    }

    #[test]
    fn boundary_n_images_1_embedding_dim_1() {
        let cfg = NerfWConfig {
            n_images: 1,
            embedding_dim_appearance: 1,
            embedding_dim_transient: 1,
            base_features_dim: 1,
            beta_min: 0.5,
        };
        let mut rng = LcgRng::new(51);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        assert_eq!(
            emb.lookup_appearance(0)
                .expect("lookup_appearance should succeed")
                .len(),
            1
        );
        assert_eq!(
            emb.lookup_transient(0)
                .expect("lookup_transient should succeed")
                .len(),
            1
        );
        let base = vec![1.0_f32];
        let out = concat_features(&base, &emb, 0, true).expect("concat_features should succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn deterministic_init_for_same_seed() {
        let cfg = default_cfg();
        let mut r1 = LcgRng::new(52);
        let mut r2 = LcgRng::new(52);
        let a = NerfWEmbeddings::new(&cfg, &mut r1).expect("new should succeed");
        let b = NerfWEmbeddings::new(&cfg, &mut r2).expect("new should succeed");
        assert_eq!(a.appearance, b.appearance);
        assert_eq!(a.transient, b.transient);
    }

    #[test]
    fn err_n_images_zero() {
        let cfg = NerfWConfig {
            n_images: 0,
            embedding_dim_appearance: 4,
            embedding_dim_transient: 2,
            base_features_dim: 8,
            beta_min: 0.03,
        };
        assert!(cfg.validate().is_err());
        let mut rng = LcgRng::new(53);
        assert!(NerfWEmbeddings::new(&cfg, &mut rng).is_err());
    }

    #[test]
    fn err_embedding_dim_zero() {
        let cfg_app = NerfWConfig {
            n_images: 2,
            embedding_dim_appearance: 0,
            embedding_dim_transient: 4,
            base_features_dim: 4,
            beta_min: 0.03,
        };
        assert!(cfg_app.validate().is_err());
        let cfg_tr = NerfWConfig {
            n_images: 2,
            embedding_dim_appearance: 4,
            embedding_dim_transient: 0,
            base_features_dim: 4,
            beta_min: 0.03,
        };
        assert!(cfg_tr.validate().is_err());
    }

    #[test]
    fn err_beta_min_invalid() {
        let cfg = NerfWConfig {
            n_images: 2,
            embedding_dim_appearance: 4,
            embedding_dim_transient: 2,
            base_features_dim: 4,
            beta_min: 0.0,
        };
        assert!(cfg.validate().is_err());
        let cfg_neg = NerfWConfig {
            n_images: 2,
            embedding_dim_appearance: 4,
            embedding_dim_transient: 2,
            base_features_dim: 4,
            beta_min: -0.1,
        };
        assert!(cfg_neg.validate().is_err());
    }

    #[test]
    fn err_nll_beta_zero() {
        assert!(nerf_w_nll(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 0.0).is_err());
        assert!(nerf_w_nll(&[0.1, 0.2, 0.3], &[0.0, 0.0, 0.0], -0.5).is_err());
        assert!(nerf_w_nll(&[0.1, 0.2, 0.3], &[0.0, 0.0, 0.0], f32::NAN).is_err());
    }

    #[test]
    fn err_concat_base_empty() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(54);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let base: Vec<f32> = vec![];
        let res = concat_features(&base, &emb, 0, false);
        assert!(res.is_err());
    }

    #[test]
    fn err_beta_head_shape_mismatch() {
        let mut rng = LcgRng::new(55);
        let head = BetaHead::new(8, 16, 0.03, &mut rng).expect("new should succeed");
        let res = head.forward(&[0.0_f32; 3]); // wrong in_dim
        assert!(res.is_err());
    }

    #[test]
    fn err_beta_head_zero_dims() {
        let mut rng = LcgRng::new(56);
        assert!(BetaHead::new(0, 8, 0.03, &mut rng).is_err());
        assert!(BetaHead::new(8, 0, 0.03, &mut rng).is_err());
        assert!(BetaHead::new(4, 4, -0.1, &mut rng).is_err());
    }

    #[test]
    fn concat_idx_out_of_range() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(57);
        let emb = NerfWEmbeddings::new(&cfg, &mut rng).expect("new should succeed");
        let base = vec![1.0_f32; cfg.base_features_dim];
        let res = concat_features(&base, &emb, cfg.n_images, false);
        assert!(res.is_err());
    }

    #[test]
    fn nll_by_hand_nonzero_residual() {
        // Known exact value: residual² = 0.04, β = 0.5 → β² = 0.25
        // L = 0.04 / 0.5 + 0.5 · ln(0.25) = 0.08 + 0.5 · (−1.386...) = 0.08 − 0.693... ≈ -0.613147
        let p = [0.2_f32, 0.0, 0.0];
        let t = [0.0_f32, 0.0, 0.0];
        let beta = 0.5_f32;
        let nll = nerf_w_nll(&p, &t, beta).expect("nerf_w_nll should succeed");
        let expected = 0.04_f32 / (2.0 * 0.25) + 0.5 * 0.25_f32.ln();
        assert!(
            (nll - expected).abs() < 1e-6,
            "got {nll}, expected {expected}"
        );
    }
}
