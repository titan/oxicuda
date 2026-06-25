//! Self-supervised pretraining objectives for tabular data.
//!
//! Two complementary, label-free pretext tasks are provided as CPU forward
//! engines with explicit gradient-free corruption / target generation:
//!
//! # VIME (Yoon et al. 2020) — Value Imputation & Mask Estimation
//!
//! Given a clean row `x`, a binary mask `m ~ Bernoulli(p_m)^d` selects features
//! to be corrupted.  The corrupted row is
//!
//! ```text
//! x̃_j = m_j · x̄_j + (1 − m_j) · x_j
//! ```
//!
//! where `x̄_j` is sampled from the empirical marginal of feature `j` (i.e. the
//! value of feature `j` taken from another randomly chosen training row).  A
//! shared encoder `z = enc(x̃)` feeds two heads:
//!
//! * **mask estimator** `m̂ = σ(W_m z + b_m)` trained with binary cross-entropy
//!   against the true mask `m`;
//! * **feature estimator** `x̂ = W_x z + b_x` trained with mean-squared error
//!   against the clean `x`.
//!
//! # SCARF (Bahri et al. 2022) — contrastive corruption
//!
//! Each anchor row is corrupted by resampling a random feature subset (fraction
//! `corruption_rate`) from the empirical marginal, producing a positive view.
//! An encoder + projection head map both views to an embedding; the InfoNCE
//! contrastive loss pulls the two views of the same row together and pushes
//! different rows apart (temperature `τ`).
//!
//! Both objectives operate on row-major `[n_samples × n_features]` arrays.  The
//! marginals are taken directly from the supplied data matrix, so the corruption
//! distribution is exact (no parametric assumptions).

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::nn::Dense;

// ─── VIME ───────────────────────────────────────────────────────────────────

/// Configuration for the VIME self-supervised pretraining model.
#[derive(Debug, Clone)]
pub struct VimeConfig {
    /// Number of input features (`d`).
    pub n_features: usize,
    /// Encoder hidden / latent dimension (`|z|`).
    pub latent_dim: usize,
    /// Per-feature corruption probability `p_m` for the Bernoulli mask.
    pub mask_prob: f32,
    /// Weight on the mask-estimation BCE term relative to the reconstruction MSE.
    pub alpha_mask: f32,
}

impl Default for VimeConfig {
    fn default() -> Self {
        Self {
            n_features: 16,
            latent_dim: 32,
            mask_prob: 0.3,
            alpha_mask: 1.0,
        }
    }
}

/// VIME pretraining model: a shared ReLU encoder with a mask-estimation head and
/// a feature-reconstruction head.
#[derive(Debug, Clone)]
pub struct VimeModel {
    config: VimeConfig,
    encoder: Dense,
    mask_head: Dense,
    feat_head: Dense,
}

/// One corrupted training instance produced by [`VimeModel::corrupt`].
#[derive(Debug, Clone)]
pub struct VimeSample {
    /// Corrupted feature row, length `n_features`.
    pub corrupted: Vec<f32>,
    /// Ground-truth binary mask (1 = feature was replaced), length `n_features`.
    pub mask: Vec<f32>,
}

impl VimeModel {
    /// Construct a VIME model with Glorot-initialised dense layers.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidParameter`] when `n_features`,
    /// `latent_dim` is zero, or `mask_prob` is outside `[0, 1]`.
    pub fn new(config: VimeConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.n_features == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_features".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.latent_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "latent_dim".into(),
                msg: "must be > 0".into(),
            });
        }
        if !(0.0..=1.0).contains(&config.mask_prob) {
            return Err(TabularError::InvalidParameter {
                name: "mask_prob".into(),
                msg: "must lie in [0, 1]".into(),
            });
        }
        let encoder = Dense::new(config.n_features, config.latent_dim, rng);
        let mask_head = Dense::new(config.latent_dim, config.n_features, rng);
        let feat_head = Dense::new(config.latent_dim, config.n_features, rng);
        Ok(Self {
            config,
            encoder,
            mask_head,
            feat_head,
        })
    }

    /// Number of input features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_features
    }

    /// Latent dimension.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    /// Produce a corrupted view of `row` by masking each feature with
    /// probability `mask_prob` and replacing masked entries with a value drawn
    /// from the empirical marginal of that feature (a value sampled from another
    /// random row of `data`).
    ///
    /// # Errors
    /// Returns an error if `row.len() != n_features` or the data matrix shape is
    /// inconsistent.
    pub fn corrupt(
        &self,
        row: &[f32],
        data: &[f32],
        n_samples: usize,
        rng: &mut LcgRng,
    ) -> TabularResult<VimeSample> {
        let d = self.config.n_features;
        if row.len() != d {
            return Err(TabularError::DimensionMismatch {
                expected: d,
                got: row.len(),
            });
        }
        if n_samples == 0 {
            return Err(TabularError::EmptyInput);
        }
        if data.len() != n_samples * d {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * d,
                got: data.len(),
            });
        }
        let mut corrupted = row.to_vec();
        let mut mask = vec![0.0_f32; d];
        for (j, (cj, mj)) in corrupted.iter_mut().zip(mask.iter_mut()).enumerate() {
            // Full-range unit uniform draw.
            let u = f64::from(rng.next_u32()) / 2f64.powi(32);
            if (u as f32) < self.config.mask_prob {
                *mj = 1.0;
                let src = rng.next_usize(n_samples);
                *cj = data[src * d + j];
            }
        }
        Ok(VimeSample { corrupted, mask })
    }

    /// Encode a (possibly corrupted) row into the latent space with a ReLU
    /// non-linearity.
    ///
    /// # Errors
    /// Returns an error if `x.len() != n_features`.
    pub fn encode(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.config.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.n_features,
                got: x.len(),
            });
        }
        let mut z = self.encoder.forward(x);
        for v in &mut z {
            *v = v.max(0.0);
        }
        Ok(z)
    }

    /// Predict the mask logits `m̂` (pre-sigmoid) and reconstructed features `x̂`
    /// from a corrupted row.
    ///
    /// Returns `(mask_logits, recon)`, each of length `n_features`.
    ///
    /// # Errors
    /// Returns an error if `corrupted.len() != n_features`.
    pub fn forward(&self, corrupted: &[f32]) -> TabularResult<(Vec<f32>, Vec<f32>)> {
        let z = self.encode(corrupted)?;
        let mask_logits = self.mask_head.forward(&z);
        let recon = self.feat_head.forward(&z);
        Ok((mask_logits, recon))
    }

    /// Combined VIME pretext loss for one corrupted sample:
    /// `alpha_mask · BCE(σ(m̂), m) + MSE(x̂, x_clean)`.
    ///
    /// # Errors
    /// Returns an error on any shape mismatch between the arguments.
    pub fn loss(&self, sample: &VimeSample, clean: &[f32]) -> TabularResult<f32> {
        let d = self.config.n_features;
        if clean.len() != d {
            return Err(TabularError::DimensionMismatch {
                expected: d,
                got: clean.len(),
            });
        }
        let (mask_logits, recon) = self.forward(&sample.corrupted)?;
        let mut bce = 0.0_f32;
        for (&logit, &m) in mask_logits.iter().zip(sample.mask.iter()) {
            // Numerically-stable BCE-with-logits: max(z,0) − z·m + log(1+e^{−|z|}).
            let p = sigmoid(logit);
            let p = p.clamp(1e-7, 1.0 - 1e-7);
            bce += -(m * p.ln() + (1.0 - m) * (1.0 - p).ln());
        }
        bce /= d as f32;
        let mut mse = 0.0_f32;
        for (&xr, &xc) in recon.iter().zip(clean.iter()) {
            let diff = xr - xc;
            mse += diff * diff;
        }
        mse /= d as f32;
        Ok(self.config.alpha_mask * bce + mse)
    }
}

// ─── SCARF ────────────────────────────────────────────────────────────────────

/// Configuration for the SCARF contrastive pretraining model.
#[derive(Debug, Clone)]
pub struct ScarfConfig {
    /// Number of input features (`d`).
    pub n_features: usize,
    /// Encoder hidden / embedding dimension.
    pub embed_dim: usize,
    /// Projection-head output dimension used for the contrastive loss.
    pub proj_dim: usize,
    /// Fraction of features resampled from the marginal for the positive view.
    pub corruption_rate: f32,
    /// InfoNCE temperature `τ`.
    pub temperature: f32,
}

impl Default for ScarfConfig {
    fn default() -> Self {
        Self {
            n_features: 16,
            embed_dim: 32,
            proj_dim: 16,
            corruption_rate: 0.6,
            temperature: 1.0,
        }
    }
}

/// SCARF contrastive pretraining model: a ReLU encoder followed by a projection
/// head producing the embedding used in the InfoNCE objective.
#[derive(Debug, Clone)]
pub struct ScarfModel {
    config: ScarfConfig,
    encoder: Dense,
    projector: Dense,
}

impl ScarfModel {
    /// Construct a SCARF model with Glorot-initialised layers.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidParameter`] when any dimension is zero or
    /// `corruption_rate` / `temperature` are out of range.
    pub fn new(config: ScarfConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.n_features == 0 || config.embed_dim == 0 || config.proj_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "dimensions".into(),
                msg: "n_features, embed_dim, proj_dim must all be > 0".into(),
            });
        }
        if !(0.0..=1.0).contains(&config.corruption_rate) {
            return Err(TabularError::InvalidParameter {
                name: "corruption_rate".into(),
                msg: "must lie in [0, 1]".into(),
            });
        }
        if config.temperature <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "temperature".into(),
                msg: "must be > 0".into(),
            });
        }
        let encoder = Dense::new(config.n_features, config.embed_dim, rng);
        let projector = Dense::new(config.embed_dim, config.proj_dim, rng);
        Ok(Self {
            config,
            encoder,
            projector,
        })
    }

    /// Number of input features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_features
    }

    /// Corrupt `row` by replacing a random `corruption_rate` fraction of its
    /// features with values drawn from the empirical marginals (other rows).
    ///
    /// # Errors
    /// Returns an error on shape mismatch.
    pub fn corrupt(
        &self,
        row: &[f32],
        data: &[f32],
        n_samples: usize,
        rng: &mut LcgRng,
    ) -> TabularResult<Vec<f32>> {
        let d = self.config.n_features;
        if row.len() != d {
            return Err(TabularError::DimensionMismatch {
                expected: d,
                got: row.len(),
            });
        }
        if n_samples == 0 {
            return Err(TabularError::EmptyInput);
        }
        if data.len() != n_samples * d {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * d,
                got: data.len(),
            });
        }
        let mut view = row.to_vec();
        for (j, vj) in view.iter_mut().enumerate() {
            let u = f64::from(rng.next_u32()) / 2f64.powi(32);
            if (u as f32) < self.config.corruption_rate {
                let src = rng.next_usize(n_samples);
                *vj = data[src * d + j];
            }
        }
        Ok(view)
    }

    /// Map a row through encoder + projection head, returning the L2-normalised
    /// embedding used for the contrastive comparison.
    ///
    /// # Errors
    /// Returns an error if `x.len() != n_features`.
    pub fn embed(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.config.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.n_features,
                got: x.len(),
            });
        }
        let mut h = self.encoder.forward(x);
        for v in &mut h {
            *v = v.max(0.0);
        }
        let z = self.projector.forward(&h);
        Ok(l2_normalize(&z))
    }

    /// InfoNCE contrastive loss over a batch.  `anchors` and `positives` are
    /// each `[batch × n_features]` row-major; `positives[i]` is a corrupted view
    /// of `anchors[i]`.  Negatives are the other in-batch positives.
    ///
    /// Returns the mean loss across the batch.
    ///
    /// # Errors
    /// Returns an error on shape mismatch or empty batch.
    pub fn info_nce_loss(
        &self,
        anchors: &[f32],
        positives: &[f32],
        batch: usize,
    ) -> TabularResult<f32> {
        let d = self.config.n_features;
        if batch == 0 {
            return Err(TabularError::EmptyInput);
        }
        if anchors.len() != batch * d || positives.len() != batch * d {
            return Err(TabularError::DimensionMismatch {
                expected: batch * d,
                got: anchors.len().min(positives.len()),
            });
        }
        // Pre-compute all anchor and positive embeddings.
        let mut anc_emb = Vec::with_capacity(batch);
        let mut pos_emb = Vec::with_capacity(batch);
        for i in 0..batch {
            anc_emb.push(self.embed(&anchors[i * d..(i + 1) * d])?);
            pos_emb.push(self.embed(&positives[i * d..(i + 1) * d])?);
        }
        let tau = self.config.temperature;
        let mut total = 0.0_f32;
        for i in 0..batch {
            // Logits = sim(anchor_i, pos_j) / τ over all j; target is j == i.
            let logits: Vec<f32> = pos_emb
                .iter()
                .map(|pj| dot(&anc_emb[i], pj) / tau)
                .collect();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
            let log_denom = max + denom.max(1e-30).ln();
            total += log_denom - logits[i];
        }
        Ok(total / batch as f32)
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = dot(v, v).sqrt().max(1e-12);
    v.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vime_corrupt_mask_is_binary_and_replaces() {
        let mut rng = LcgRng::new(1);
        let cfg = VimeConfig {
            n_features: 4,
            latent_dim: 8,
            mask_prob: 1.0, // mask everything → fully resampled
            alpha_mask: 1.0,
        };
        let model = VimeModel::new(cfg, &mut rng).expect("new");
        let data = vec![
            10.0, 20.0, 30.0, 40.0, //
            11.0, 21.0, 31.0, 41.0, //
        ];
        let sample = model
            .corrupt(&data[0..4], &data, 2, &mut rng)
            .expect("corrupt");
        assert!(sample.mask.iter().all(|&m| m == 0.0 || m == 1.0));
        // With mask_prob = 1 every feature is masked.
        assert!(sample.mask.iter().all(|&m| m == 1.0));
        // Every corrupted value must come from column-consistent marginals.
        for (j, &c) in sample.corrupted.iter().enumerate() {
            assert!(c == data[j] || c == data[4 + j]);
        }
    }

    #[test]
    fn vime_no_mask_keeps_row() {
        let mut rng = LcgRng::new(2);
        let cfg = VimeConfig {
            n_features: 4,
            latent_dim: 8,
            mask_prob: 0.0,
            alpha_mask: 1.0,
        };
        let model = VimeModel::new(cfg, &mut rng).expect("new");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sample = model
            .corrupt(&data[0..4], &data, 2, &mut rng)
            .expect("corrupt");
        assert!(sample.mask.iter().all(|&m| m == 0.0));
        assert_eq!(sample.corrupted, data[0..4].to_vec());
    }

    #[test]
    fn vime_forward_shapes_and_finite() {
        let mut rng = LcgRng::new(3);
        let cfg = VimeConfig {
            n_features: 6,
            latent_dim: 10,
            mask_prob: 0.5,
            alpha_mask: 2.0,
        };
        let model = VimeModel::new(cfg, &mut rng).expect("new");
        let row = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let (mh, recon) = model.forward(&row).expect("forward");
        assert_eq!(mh.len(), 6);
        assert_eq!(recon.len(), 6);
        assert!(mh.iter().chain(recon.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn vime_loss_is_finite_and_nonnegative() {
        let mut rng = LcgRng::new(4);
        let cfg = VimeConfig::default();
        let model = VimeModel::new(cfg.clone(), &mut rng).expect("new");
        let mut data = vec![0.0_f32; 8 * cfg.n_features];
        rng.fill_normal(&mut data);
        let clean = data[0..cfg.n_features].to_vec();
        let sample = model.corrupt(&clean, &data, 8, &mut rng).expect("corrupt");
        let l = model.loss(&sample, &clean).expect("loss");
        assert!(l.is_finite() && l >= 0.0, "loss = {l}");
    }

    #[test]
    fn vime_determinism() {
        let cfg = VimeConfig::default();
        let mut r1 = LcgRng::new(99);
        let mut r2 = LcgRng::new(99);
        let m1 = VimeModel::new(cfg.clone(), &mut r1).expect("new");
        let m2 = VimeModel::new(cfg.clone(), &mut r2).expect("new");
        let row = vec![0.25_f32; cfg.n_features];
        let a = m1.forward(&row).expect("fwd");
        let b = m2.forward(&row).expect("fwd");
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn scarf_embed_is_unit_norm() {
        let mut rng = LcgRng::new(5);
        let cfg = ScarfConfig {
            n_features: 8,
            embed_dim: 16,
            proj_dim: 8,
            corruption_rate: 0.5,
            temperature: 0.5,
        };
        let model = ScarfModel::new(cfg, &mut rng).expect("new");
        let row = [0.3, 0.1, -0.2, 0.7, 0.0, -0.4, 0.9, 0.2];
        let z = model.embed(&row).expect("embed");
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        // Either unit norm, or exactly zero if ReLU killed all activations.
        assert!((norm - 1.0).abs() < 1e-4 || norm < 1e-6, "norm = {norm}");
    }

    #[test]
    fn scarf_corrupt_respects_rate_bounds() {
        let mut rng = LcgRng::new(6);
        let cfg = ScarfConfig {
            n_features: 5,
            embed_dim: 8,
            proj_dim: 4,
            corruption_rate: 0.0, // never corrupt
            temperature: 1.0,
        };
        let model = ScarfModel::new(cfg, &mut rng).expect("new");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let v = model
            .corrupt(&data[0..5], &data, 2, &mut rng)
            .expect("corrupt");
        assert_eq!(v, data[0..5].to_vec());
    }

    #[test]
    fn scarf_info_nce_positive_and_finite() {
        let mut rng = LcgRng::new(7);
        let cfg = ScarfConfig::default();
        let model = ScarfModel::new(cfg.clone(), &mut rng).expect("new");
        let batch = 4;
        let d = cfg.n_features;
        let mut data = vec![0.0_f32; (batch + 2) * d];
        rng.fill_normal(&mut data);
        let mut anchors = vec![0.0_f32; batch * d];
        let mut positives = vec![0.0_f32; batch * d];
        for i in 0..batch {
            let row = &data[i * d..(i + 1) * d];
            anchors[i * d..(i + 1) * d].copy_from_slice(row);
            let pv = model
                .corrupt(row, &data, batch + 2, &mut rng)
                .expect("corrupt");
            positives[i * d..(i + 1) * d].copy_from_slice(&pv);
        }
        let loss = model
            .info_nce_loss(&anchors, &positives, batch)
            .expect("loss");
        assert!(loss.is_finite() && loss >= 0.0, "loss = {loss}");
    }

    #[test]
    fn scarf_identical_views_lower_loss_than_random() {
        // When positives == anchors (perfect alignment), InfoNCE should be small.
        let mut rng = LcgRng::new(8);
        let cfg = ScarfConfig {
            n_features: 6,
            embed_dim: 12,
            proj_dim: 6,
            corruption_rate: 0.5,
            temperature: 0.1,
        };
        let model = ScarfModel::new(cfg.clone(), &mut rng).expect("new");
        let batch = 5;
        let d = cfg.n_features;
        // Distinct, well-separated anchors.
        let mut anchors = vec![0.0_f32; batch * d];
        for i in 0..batch {
            for j in 0..d {
                anchors[i * d + j] = (i as f32) * 3.0 + (j as f32) * 0.1;
            }
        }
        let aligned = model
            .info_nce_loss(&anchors, &anchors, batch)
            .expect("loss");
        // Shuffle positives (row 0 paired with last row's view, etc.).
        let mut shuffled = anchors.clone();
        shuffled.rotate_right(d);
        let misaligned = model
            .info_nce_loss(&anchors, &shuffled, batch)
            .expect("loss");
        assert!(
            aligned <= misaligned + 1e-4,
            "aligned {aligned} should be <= misaligned {misaligned}"
        );
    }
}
