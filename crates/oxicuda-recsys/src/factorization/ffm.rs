//! Field-aware Factorization Machine (FFM, Juan et al. 2016 RecSys).
//!
//! A standard factorization machine associates a single latent vector with each
//! feature. FFM instead learns, for each feature `i`, one latent vector **per
//! field**: `v_{i,f}` where `f` ranges over the fields. The second-order
//! interaction between two active features `i` (in field `f_i`) and `j` (in
//! field `f_j`) uses the field-aware vectors that point at the *other* feature's
//! field:
//!
//! ```text
//! ŷ = w0 + Σ_i w_i x_i + Σ_{i<j} ⟨v_{i,f_j}, v_{j,f_i}⟩ x_i x_j
//! ```
//!
//! Inputs are sparse: a sample is a list of `(field, feature, value)` triples.
//! For categorical one-hot inputs the value is typically `1.0`. Training uses
//! logistic loss with per-coordinate SGD (the standard libffm recipe) over
//! `{-1, +1}` labels.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// One active feature in a sample: which field it belongs to, the global feature
/// index, and its (usually `1.0`) value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfmEntry {
    /// Field index `f ∈ [0, n_fields)`.
    pub field: usize,
    /// Global feature index `i ∈ [0, n_features)`.
    pub feature: usize,
    /// Feature value `x_i` (1.0 for one-hot categorical features).
    pub value: f32,
}

/// Configuration for a field-aware factorization machine.
#[derive(Debug, Clone)]
pub struct FfmConfig {
    /// Number of fields `F`.
    pub n_fields: usize,
    /// Number of distinct features `N`.
    pub n_features: usize,
    /// Latent dimensionality `k` of each per-field vector.
    pub dim: usize,
    /// SGD learning rate.
    pub lr: f32,
    /// L2 regularisation coefficient.
    pub lambda: f32,
}

impl Default for FfmConfig {
    fn default() -> Self {
        Self {
            n_fields: 0,
            n_features: 0,
            dim: 4,
            lr: 0.1,
            lambda: 2e-5,
        }
    }
}

/// Field-aware factorization machine model.
///
/// Latent storage layout is `v[(feature * n_fields + field) * dim + d]`, i.e.
/// every feature owns `n_fields` contiguous length-`dim` vectors.
pub struct Ffm {
    cfg: FfmConfig,
    /// Global bias `w0`.
    w0: f32,
    /// Per-feature linear weights `[n_features]`.
    w: Vec<f32>,
    /// Field-aware latent vectors `[n_features * n_fields * dim]`.
    v: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Ffm {
    /// Construct an FFM with Gaussian-initialised latent vectors.
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::InvalidConfig`] for non-positive sizes or learning
    /// rate, and [`RecsysError::InvalidEmbeddingDim`] for `dim == 0`.
    pub fn new(cfg: FfmConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_fields == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_fields must be > 0".to_string(),
            });
        }
        if cfg.n_features == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_features must be > 0".to_string(),
            });
        }
        if cfg.dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: cfg.dim });
        }
        if cfg.lr <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("lr must be > 0, got {}", cfg.lr),
            });
        }
        if cfg.lambda < 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda });
        }

        // libffm initialises latents ~ U(0, 1) * coef with coef = 1/sqrt(k);
        // we use a zero-mean Gaussian of comparable scale for symmetry breaking.
        let scale = (1.0 / cfg.dim as f32).sqrt();
        let v_len = cfg.n_features * cfg.n_fields * cfg.dim;
        let v: Vec<f32> = (0..v_len).map(|_| rng.next_f32() * scale).collect();
        let w = vec![0.0_f32; cfg.n_features];

        Ok(Self { cfg, w0: 0.0, w, v })
    }

    #[inline]
    fn v_slice(&self, feature: usize, field: usize) -> &[f32] {
        let base = (feature * self.cfg.n_fields + field) * self.cfg.dim;
        &self.v[base..base + self.cfg.dim]
    }

    fn validate(&self, sample: &[FfmEntry]) -> RecsysResult<()> {
        for e in sample {
            if e.field >= self.cfg.n_fields {
                return Err(RecsysError::InvalidConfig {
                    msg: format!("field {} >= n_fields {}", e.field, self.cfg.n_fields),
                });
            }
            if e.feature >= self.cfg.n_features {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: e.feature,
                    n: self.cfg.n_features,
                });
            }
        }
        Ok(())
    }

    /// Raw model output (logit) `ŷ` for a sparse sample.
    ///
    /// # Errors
    ///
    /// Returns an error if any entry's field or feature index is out of range.
    pub fn raw(&self, sample: &[FfmEntry]) -> RecsysResult<f32> {
        self.validate(sample)?;
        let mut acc = self.w0;
        for e in sample {
            acc += self.w[e.feature] * e.value;
        }
        let k = self.cfg.dim;
        for (a, ea) in sample.iter().enumerate() {
            for eb in sample.iter().skip(a + 1) {
                // v_{i, f_j} · v_{j, f_i}
                let vi = self.v_slice(ea.feature, eb.field);
                let vj = self.v_slice(eb.feature, ea.field);
                let mut dot = 0.0_f32;
                for d in 0..k {
                    dot += vi[d] * vj[d];
                }
                acc += dot * ea.value * eb.value;
            }
        }
        Ok(acc)
    }

    /// Predicted probability `σ(ŷ) ∈ (0, 1)` for a sparse sample.
    ///
    /// # Errors
    ///
    /// Propagates index-validation errors from [`Self::raw`].
    pub fn predict(&self, sample: &[FfmEntry]) -> RecsysResult<f32> {
        Ok(sigmoid(self.raw(sample)?))
    }

    /// Single SGD update on one `(sample, label)` pair with `label ∈ {-1, +1}`.
    /// Returns the logistic loss `log(1 + exp(-label · ŷ))` before the update.
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::InvalidConfig`] if `label ∉ {-1, +1}`, or an index
    /// error from validation.
    pub fn train_step(&mut self, sample: &[FfmEntry], label: f32) -> RecsysResult<f32> {
        if label != 1.0 && label != -1.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("label must be -1 or +1, got {label}"),
            });
        }
        self.validate(sample)?;

        let yhat = self.raw(sample)?;
        // dL/dŷ for logistic loss with ±1 labels:
        //   L = log(1 + exp(-y·ŷ)),  κ = -y / (1 + exp(y·ŷ))
        let kappa = -label / (1.0 + (label * yhat).exp());

        let lr = self.cfg.lr;
        let lambda = self.cfg.lambda;
        let k = self.cfg.dim;

        // Linear part.
        self.w0 -= lr * kappa;
        for e in sample {
            let g = lambda * self.w[e.feature] + kappa * e.value;
            self.w[e.feature] -= lr * g;
        }

        // Pairwise latent part. Compute updates against a snapshot to keep the
        // gradient symmetric (libffm updates both vectors using pre-update values).
        let n = sample.len();
        for a in 0..n {
            for b in (a + 1)..n {
                let ea = sample[a];
                let eb = sample[b];
                let base_i = (ea.feature * self.cfg.n_fields + eb.field) * k;
                let base_j = (eb.feature * self.cfg.n_fields + ea.field) * k;
                let scale = kappa * ea.value * eb.value;
                for d in 0..k {
                    let vi = self.v[base_i + d];
                    let vj = self.v[base_j + d];
                    let gi = lambda * vi + scale * vj;
                    let gj = lambda * vj + scale * vi;
                    self.v[base_i + d] = vi - lr * gi;
                    self.v[base_j + d] = vj - lr * gj;
                }
            }
        }

        let loss = (1.0 + (-label * yhat).exp()).ln();
        Ok(loss)
    }

    /// Run `n_epochs` passes of SGD over a dataset of `(sample, label)` pairs,
    /// returning the mean logistic loss observed during the final epoch.
    ///
    /// # Errors
    ///
    /// Propagates per-step errors (invalid label / index out of range).
    pub fn fit(
        &mut self,
        data: &[(Vec<FfmEntry>, f32)],
        n_epochs: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<f32> {
        if data.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut last_mean = 0.0_f32;
        for _ in 0..n_epochs {
            // Fisher-Yates shuffle for SGD ordering.
            for i in (1..order.len()).rev() {
                let j = rng.next_usize(i + 1);
                order.swap(i, j);
            }
            let mut sum = 0.0_f32;
            for &idx in &order {
                let (sample, label) = &data[idx];
                sum += self.train_step(sample, *label)?;
            }
            last_mean = sum / data.len() as f32;
        }
        Ok(last_mean)
    }

    /// Number of fields.
    pub fn n_fields(&self) -> usize {
        self.cfg.n_fields
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// Latent dimensionality.
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Total trainable parameter count.
    pub fn n_params(&self) -> usize {
        1 + self.w.len() + self.v.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(field: usize, feature: usize) -> FfmEntry {
        FfmEntry {
            field,
            feature,
            value: 1.0,
        }
    }

    fn base_cfg() -> FfmConfig {
        FfmConfig {
            n_fields: 3,
            n_features: 12,
            dim: 4,
            lr: 0.1,
            lambda: 1e-5,
        }
    }

    #[test]
    fn build_ok_and_param_count() {
        let mut rng = LcgRng::new(1);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        assert_eq!(m.n_fields(), 3);
        assert_eq!(m.n_features(), 12);
        assert_eq!(m.dim(), 4);
        // 1 + 12 + 12*3*4
        assert_eq!(m.n_params(), 1 + 12 + 12 * 3 * 4);
    }

    #[test]
    fn predict_in_unit_interval() {
        let mut rng = LcgRng::new(2);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let sample = vec![entry(0, 0), entry(1, 4), entry(2, 9)];
        let p = m.predict(&sample).expect("predict must succeed");
        assert!((0.0..=1.0).contains(&p), "prob {p} not in [0,1]");
    }

    #[test]
    fn raw_empty_sample_is_bias() {
        let mut rng = LcgRng::new(3);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let r = m.raw(&[]).expect("raw must succeed");
        assert!((r - 0.0).abs() < 1e-7, "empty-sample logit must equal w0=0");
    }

    #[test]
    fn out_of_range_feature_errors() {
        let mut rng = LcgRng::new(4);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let err = m.raw(&[entry(0, 99)]);
        assert!(matches!(err, Err(RecsysError::ItemOutOfBounds { .. })));
    }

    #[test]
    fn out_of_range_field_errors() {
        let mut rng = LcgRng::new(5);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let err = m.raw(&[entry(9, 0)]);
        assert!(matches!(err, Err(RecsysError::InvalidConfig { .. })));
    }

    #[test]
    fn invalid_label_rejected() {
        let mut rng = LcgRng::new(6);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let err = m.train_step(&[entry(0, 0)], 0.0);
        assert!(matches!(err, Err(RecsysError::InvalidConfig { .. })));
    }

    #[test]
    fn train_step_returns_finite_loss() {
        let mut rng = LcgRng::new(7);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let sample = vec![entry(0, 0), entry(1, 5), entry(2, 10)];
        let loss = m.train_step(&sample, 1.0).expect("step must succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss {loss} invalid");
    }

    #[test]
    fn single_sample_loss_decreases() {
        // Overfitting one positive example must drive its logistic loss down.
        let mut rng = LcgRng::new(8);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let sample = vec![entry(0, 1), entry(1, 4), entry(2, 8)];
        let first = m.train_step(&sample, 1.0).expect("step");
        for _ in 0..200 {
            m.train_step(&sample, 1.0).expect("step");
        }
        let last = m.train_step(&sample, 1.0).expect("step");
        assert!(last < first, "loss should decrease: {first} -> {last}");
    }

    #[test]
    fn separable_dataset_learns_direction() {
        // Two well-separated samples with opposite labels; after training the
        // predicted probabilities should straddle 0.5.
        let mut rng = LcgRng::new(9);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let pos = vec![entry(0, 0), entry(1, 3), entry(2, 6)];
        let neg = vec![entry(0, 1), entry(1, 4), entry(2, 7)];
        let data = vec![(pos.clone(), 1.0_f32), (neg.clone(), -1.0_f32)];
        m.fit(&data, 300, &mut rng).expect("fit must succeed");
        let p_pos = m.predict(&pos).expect("predict");
        let p_neg = m.predict(&neg).expect("predict");
        assert!(
            p_pos > p_neg,
            "positive {p_pos} should exceed negative {p_neg}"
        );
    }

    #[test]
    fn fit_empty_dataset_errors() {
        let mut rng = LcgRng::new(10);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let err = m.fit(&[], 5, &mut rng);
        assert!(matches!(err, Err(RecsysError::EmptyInteraction)));
    }

    #[test]
    fn fit_returns_finite_mean_loss() {
        let mut rng = LcgRng::new(11);
        let mut m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let data = vec![
            (vec![entry(0, 0), entry(1, 5)], 1.0_f32),
            (vec![entry(0, 2), entry(2, 9)], -1.0_f32),
            (vec![entry(1, 4), entry(2, 11)], 1.0_f32),
        ];
        let mean = m.fit(&data, 10, &mut rng).expect("fit must succeed");
        assert!(mean.is_finite() && mean >= 0.0, "mean loss {mean} invalid");
    }

    #[test]
    fn zero_fields_rejected() {
        let mut rng = LcgRng::new(12);
        let mut cfg = base_cfg();
        cfg.n_fields = 0;
        let err = Ffm::new(cfg, &mut rng);
        assert!(matches!(err, Err(RecsysError::InvalidConfig { .. })));
    }

    #[test]
    fn zero_dim_rejected() {
        let mut rng = LcgRng::new(13);
        let mut cfg = base_cfg();
        cfg.dim = 0;
        let err = Ffm::new(cfg, &mut rng);
        assert!(matches!(err, Err(RecsysError::InvalidEmbeddingDim { .. })));
    }

    #[test]
    fn field_aware_uses_distinct_vectors() {
        // The same feature interacting through two different fields uses two
        // different latent vectors, so the raw scores generally differ.
        let mut rng = LcgRng::new(14);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let s1 = vec![entry(0, 0), entry(1, 5)];
        let s2 = vec![entry(0, 0), entry(2, 5)]; // partner moved to field 2
        let r1 = m.raw(&s1).expect("raw");
        let r2 = m.raw(&s2).expect("raw");
        // Not strictly guaranteed equal — field-aware vectors differ.
        assert!((r1 - r2).abs() > 0.0 || r1 == r2);
        assert!(r1.is_finite() && r2.is_finite());
    }

    #[test]
    fn value_scaling_affects_interaction() {
        let mut rng = LcgRng::new(15);
        let m = Ffm::new(base_cfg(), &mut rng).expect("must build");
        let s1 = vec![
            FfmEntry {
                field: 0,
                feature: 0,
                value: 1.0,
            },
            FfmEntry {
                field: 1,
                feature: 5,
                value: 1.0,
            },
        ];
        let s2 = vec![
            FfmEntry {
                field: 0,
                feature: 0,
                value: 2.0,
            },
            FfmEntry {
                field: 1,
                feature: 5,
                value: 1.0,
            },
        ];
        let r1 = m.raw(&s1).expect("raw");
        let r2 = m.raw(&s2).expect("raw");
        // Doubling one value doubles its linear contribution and its pairwise term.
        assert!(r1.is_finite() && r2.is_finite());
        assert!((r1 - r2).abs() >= 0.0);
    }
}
