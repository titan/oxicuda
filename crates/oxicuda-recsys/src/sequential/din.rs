//! DIN — Deep Interest Network for click-through rate prediction.
//!
//! Reference: Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu,
//! Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai, "Deep Interest Network
//! for Click-Through Rate Prediction", KDD 2018.
//!
//! Architecture:
//!   1. For a target advertisement/item embedding `target ∈ R^d` and a sequence
//!      of `n` historical-behaviour embeddings `h_1, …, h_n ∈ R^d`, a
//!      **local activation unit** `a(h_i, target)` produces a per-history scalar
//!      attention weight via a small 2-layer MLP fed with
//!      `concat(h_i, target, h_i ⊙ target, h_i − target) ∈ R^{4d}`.
//!   2. The **user-interest representation** is the *raw* weighted sum
//!      `Σ_i a_i · h_i` (intentionally *not* softmax-normalised — DIN preserves
//!      attention intensity, see § 4.3 of the paper).
//!   3. The interest representation is concatenated with the target embedding
//!      and their element-wise product (`concat(interest, target, target ⊙
//!      interest)`) and fed through a **top MLP** that returns a CTR logit;
//!      a final sigmoid yields a probability in `(0, 1)`.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Numerically stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place ReLU.
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Dense layer: `out[o] = b[o] + Σ_i w[o * fan_in + i] · x[i]` (row-major).
fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    (0..fan_out)
        .map(|o| {
            b[o] + w[o * fan_in..(o + 1) * fan_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

/// Build a stack of `(weight, bias)` layers from a sequence of widths. With
/// `dims = [d_0, d_1, …, d_L]` this produces `L` layers, the `l`-th layer
/// shaped `d_{l+1} × d_l` (row-major) with bias `d_{l+1}`.
fn build_mlp(dims: &[usize], rng: &mut LcgRng) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut layers = Vec::with_capacity(dims.len().saturating_sub(1));
    for window in dims.windows(2) {
        let (fan_in, fan_out) = (window[0], window[1]);
        let sc = (2.0 / fan_in.max(1) as f32).sqrt();
        let w: Vec<f32> = (0..fan_out * fan_in)
            .map(|_| rng.next_normal() * sc)
            .collect();
        let b = vec![0.0_f32; fan_out];
        layers.push((w, b));
    }
    layers
}

/// Forward an input through an MLP, applying ReLU on every layer **except**
/// the last (which yields the raw logit / scalar score).
fn mlp_forward(x: &[f32], layers: &[(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
    let mut current = x.to_vec();
    let mut cur_dim = x.len();
    let n_layers = layers.len();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let mut out = dense(&current, w, b, cur_dim, out_dim);
        if idx + 1 < n_layers {
            relu(&mut out);
        }
        current = out;
        cur_dim = out_dim;
    }
    current
}

/// DIN hyper-parameters.
#[derive(Debug, Clone)]
pub struct DinConfig {
    /// Width of every history / target embedding (`>= 1`).
    pub embed_dim: usize,
    /// Maximum supported history length used for validation (`>= 1`).
    pub max_history: usize,
    /// Hidden width of the (single) hidden layer of the local activation unit
    /// (`>= 1`). The activation MLP shape is `4·embed_dim → attention_hidden → 1`.
    pub attention_hidden: usize,
    /// Hidden widths of the top CTR MLP (must be non-empty). The MLP shape is
    /// `3·embed_dim → mlp_hidden[0] → … → mlp_hidden[L-1] → 1`.
    pub mlp_hidden: Vec<usize>,
}

/// Deep Interest Network.
pub struct Din {
    /// Configuration the model was built from.
    pub cfg: DinConfig,
    /// Local activation unit layers (2-layer MLP from `4·embed_dim` to `1`).
    pub attention_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Top CTR MLP layers (from `3·embed_dim` to `1` through `mlp_hidden`).
    pub top_layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Din {
    /// Construct a DIN with Kaiming-style normal initialisation.
    ///
    /// # Errors
    /// Returns [`RecsysError::InvalidEmbeddingDim`] when `embed_dim == 0` and
    /// [`RecsysError::InvalidConfig`] when `max_history == 0`,
    /// `attention_hidden == 0`, or `mlp_hidden` is empty.
    pub fn new(cfg: DinConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.max_history == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "max_history must be >= 1".into(),
            });
        }
        if cfg.attention_hidden == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "attention_hidden must be >= 1".into(),
            });
        }
        if cfg.mlp_hidden.is_empty() {
            return Err(RecsysError::InvalidConfig {
                msg: "mlp_hidden must be non-empty".into(),
            });
        }

        // Local activation unit: 4·embed_dim → attention_hidden → 1.
        let attn_dims = [4 * cfg.embed_dim, cfg.attention_hidden, 1];
        let attention_layers = build_mlp(&attn_dims, rng);

        // Top CTR MLP: 3·embed_dim → mlp_hidden[0] → … → 1.
        let mut top_dims = vec![3 * cfg.embed_dim];
        top_dims.extend_from_slice(&cfg.mlp_hidden);
        top_dims.push(1);
        let top_layers = build_mlp(&top_dims, rng);

        Ok(Self {
            cfg,
            attention_layers,
            top_layers,
        })
    }

    /// Local activation unit `a(h_i, target)`. The MLP input is
    /// `concat(h_i, target, h_i ⊙ target, h_i − target)` of length
    /// `4·embed_dim`; the scalar output is the raw attention weight (no
    /// softmax — DIN's intensity-preserving design).
    ///
    /// # Errors
    /// Returns [`RecsysError::DimensionMismatch`] if `h` or `target` does not
    /// have length `embed_dim`.
    pub fn attention_weight(&self, h: &[f32], target: &[f32]) -> RecsysResult<f32> {
        let d = self.cfg.embed_dim;
        if h.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: h.len(),
            });
        }
        if target.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: target.len(),
            });
        }
        let mut feat = Vec::with_capacity(4 * d);
        feat.extend_from_slice(h);
        feat.extend_from_slice(target);
        for k in 0..d {
            feat.push(h[k] * target[k]);
        }
        for k in 0..d {
            feat.push(h[k] - target[k]);
        }
        let out = mlp_forward(&feat, &self.attention_layers);
        Ok(out.first().copied().unwrap_or(0.0))
    }

    /// Per-history attention weights `a_i = a(h_i, target)` for `i = 0..n_history`.
    /// History is stored row-major as `[h_0; h_1; …; h_{n-1}]` of total length
    /// `n_history · embed_dim`.
    ///
    /// # Errors
    /// Returns [`RecsysError::EmptyInput`] when `n_history == 0`,
    /// [`RecsysError::InvalidConfig`] when `n_history > max_history`,
    /// [`RecsysError::DimensionMismatch`] when `history.len() != n_history ·
    /// embed_dim` or `target.len() != embed_dim`.
    pub fn attention_over_history(
        &self,
        history: &[f32],
        n_history: usize,
        target: &[f32],
    ) -> RecsysResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if n_history == 0 {
            return Err(RecsysError::EmptyInput);
        }
        if n_history > self.cfg.max_history {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "n_history {} exceeds max_history {}",
                    n_history, self.cfg.max_history
                ),
            });
        }
        if history.len() != n_history * d {
            return Err(RecsysError::DimensionMismatch {
                expected: n_history * d,
                got: history.len(),
            });
        }
        if target.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: target.len(),
            });
        }
        let mut weights = Vec::with_capacity(n_history);
        for i in 0..n_history {
            let h_i = &history[i * d..(i + 1) * d];
            weights.push(self.attention_weight(h_i, target)?);
        }
        Ok(weights)
    }

    /// User-interest representation `Σ_i a_i · h_i ∈ R^d`. Note: the raw
    /// (un-normalised) weights are used by design.
    ///
    /// # Errors
    /// Same validation as [`Self::attention_over_history`].
    pub fn interest_rep(
        &self,
        history: &[f32],
        n_history: usize,
        target: &[f32],
    ) -> RecsysResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let weights = self.attention_over_history(history, n_history, target)?;
        let mut interest = vec![0.0_f32; d];
        for i in 0..n_history {
            let a = weights[i];
            let h_i = &history[i * d..(i + 1) * d];
            for k in 0..d {
                interest[k] += a * h_i[k];
            }
        }
        Ok(interest)
    }

    /// Full DIN forward pass: build the interest representation, concatenate
    /// it with `target` and `target ⊙ interest`, feed through the top MLP,
    /// and squash through a sigmoid to a CTR probability in `(0, 1)`.
    ///
    /// # Errors
    /// Same validation as [`Self::interest_rep`].
    pub fn forward(&self, history: &[f32], n_history: usize, target: &[f32]) -> RecsysResult<f32> {
        let d = self.cfg.embed_dim;
        let interest = self.interest_rep(history, n_history, target)?;
        let mut feat = Vec::with_capacity(3 * d);
        feat.extend_from_slice(&interest);
        feat.extend_from_slice(target);
        for k in 0..d {
            feat.push(target[k] * interest[k]);
        }
        let logit_vec = mlp_forward(&feat, &self.top_layers);
        let logit = logit_vec.first().copied().unwrap_or(0.0);
        Ok(sigmoid(logit))
    }

    /// Total number of learnable parameters (activation unit + top MLP).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let attn: usize = self
            .attention_layers
            .iter()
            .map(|(w, b)| w.len() + b.len())
            .sum();
        let top: usize = self.top_layers.iter().map(|(w, b)| w.len() + b.len()).sum();
        attn + top
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_cfg() -> DinConfig {
        DinConfig {
            embed_dim: 6,
            max_history: 12,
            attention_hidden: 8,
            mlp_hidden: vec![16, 8],
        }
    }

    fn random_vec(n: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n).map(|_| rng.next_normal()).collect()
    }

    #[test]
    fn attention_weight_is_finite() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let h = random_vec(6, &mut rng);
        let target = random_vec(6, &mut rng);
        let a = model.attention_weight(&h, &target).unwrap();
        assert!(a.is_finite(), "attention weight must be finite, got {a}");
    }

    #[test]
    fn attention_over_history_returns_n_history_values() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = random_vec(4 * 6, &mut rng);
        let target = random_vec(6, &mut rng);
        let weights = model.attention_over_history(&history, 4, &target).unwrap();
        assert_eq!(weights.len(), 4);
        for &w in &weights {
            assert!(w.is_finite(), "weight must be finite, got {w}");
        }
    }

    #[test]
    fn interest_rep_empty_history_errors() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history: Vec<f32> = vec![];
        let target = random_vec(6, &mut rng);
        assert!(matches!(
            model.interest_rep(&history, 0, &target),
            Err(RecsysError::EmptyInput)
        ));
    }

    #[test]
    fn target_change_changes_attention() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = random_vec(3 * 6, &mut rng);
        let target_a = random_vec(6, &mut rng);
        let target_b = random_vec(6, &mut rng);
        let weights_a = model
            .attention_over_history(&history, 3, &target_a)
            .unwrap();
        let weights_b = model
            .attention_over_history(&history, 3, &target_b)
            .unwrap();
        let diff: f32 = weights_a
            .iter()
            .zip(weights_b.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different targets must yield different attention weights (got diff {diff})"
        );
    }

    #[test]
    fn history_change_changes_interest_rep() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history_a = random_vec(3 * 6, &mut rng);
        let history_b = random_vec(3 * 6, &mut rng);
        let target = random_vec(6, &mut rng);
        let int_a = model.interest_rep(&history_a, 3, &target).unwrap();
        let int_b = model.interest_rep(&history_b, 3, &target).unwrap();
        let diff: f32 = int_a
            .iter()
            .zip(int_b.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different histories must yield different interest reps (got diff {diff})"
        );
    }

    #[test]
    fn forward_returns_probability_in_open_unit_interval() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = random_vec(5 * 6, &mut rng);
        let target = random_vec(6, &mut rng);
        let p = model.forward(&history, 5, &target).unwrap();
        assert!(p.is_finite(), "probability must be finite, got {p}");
        assert!(p > 0.0 && p < 1.0, "probability {p} not in (0,1)");
    }

    #[test]
    fn n_history_one_interest_equals_a_times_h() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let h = random_vec(6, &mut rng);
        let target = random_vec(6, &mut rng);
        let a = model.attention_weight(&h, &target).unwrap();
        let interest = model.interest_rep(&h, 1, &target).unwrap();
        assert_eq!(interest.len(), 6);
        for k in 0..6 {
            assert!(
                (interest[k] - a * h[k]).abs() < 1e-5,
                "interest[k] {} should equal a·h[k] {}",
                interest[k],
                a * h[k]
            );
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let mut rng_a = LcgRng::new(11);
        let mut rng_b = LcgRng::new(11);
        let model_a = Din::new(default_cfg(), &mut rng_a).unwrap();
        let model_b = Din::new(default_cfg(), &mut rng_b).unwrap();
        let mut rng_in = LcgRng::new(999);
        let history = random_vec(4 * 6, &mut rng_in);
        let target = random_vec(6, &mut rng_in);
        let pa = model_a.forward(&history, 4, &target).unwrap();
        let pb = model_b.forward(&history, 4, &target).unwrap();
        assert!((pa - pb).abs() < 1e-6, "same seed must give same output");
    }

    #[test]
    fn n_params_is_positive() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let n = model.n_params();
        assert!(n > 0, "n_params must be > 0, got {n}");
        // Expected: attention MLP (4d→h→1) + top MLP (3d→h0→h1→1).
        let d = 6_usize;
        let attn = 4 * d * 8 + 8 + 8 + 1;
        let top = 3 * d * 16 + 16 + 16 * 8 + 8 + 8 + 1;
        assert_eq!(n, attn + top, "n_params should match closed-form count");
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = DinConfig {
            embed_dim: 0,
            max_history: 4,
            attention_hidden: 8,
            mlp_hidden: vec![16],
        };
        assert!(matches!(
            Din::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { d: 0 })
        ));
    }

    #[test]
    fn err_attention_hidden_zero() {
        let mut rng = make_rng();
        let cfg = DinConfig {
            embed_dim: 6,
            max_history: 4,
            attention_hidden: 0,
            mlp_hidden: vec![16],
        };
        assert!(matches!(
            Din::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_empty_mlp_hidden() {
        let mut rng = make_rng();
        let cfg = DinConfig {
            embed_dim: 6,
            max_history: 4,
            attention_hidden: 8,
            mlp_hidden: vec![],
        };
        assert!(matches!(
            Din::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_max_history_zero() {
        let mut rng = make_rng();
        let cfg = DinConfig {
            embed_dim: 6,
            max_history: 0,
            attention_hidden: 8,
            mlp_hidden: vec![16],
        };
        assert!(matches!(
            Din::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_n_history_exceeds_max() {
        let mut rng = make_rng();
        let cfg = DinConfig {
            embed_dim: 6,
            max_history: 3,
            attention_hidden: 8,
            mlp_hidden: vec![16],
        };
        let model = Din::new(cfg, &mut rng).unwrap();
        let history = random_vec(4 * 6, &mut rng);
        let target = random_vec(6, &mut rng);
        assert!(matches!(
            model.attention_over_history(&history, 4, &target),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_history_wrong_length() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = vec![0.0_f32; 3 * 6 - 1];
        let target = random_vec(6, &mut rng);
        assert!(matches!(
            model.attention_over_history(&history, 3, &target),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_target_wrong_length() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = random_vec(3 * 6, &mut rng);
        let target = vec![0.0_f32; 5];
        assert!(matches!(
            model.attention_over_history(&history, 3, &target),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn weights_not_softmax_normalized() {
        // DIN intentionally avoids softmax: the raw attention weights need
        // NOT sum to 1 (intensity preservation). With Kaiming-normal weights
        // and a non-trivial bias-free linear final layer, the chance of the
        // sum landing exactly on 1.0 by accident is negligible.
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let history = random_vec(5 * 6, &mut rng);
        let target = random_vec(6, &mut rng);
        let weights = model.attention_over_history(&history, 5, &target).unwrap();
        let s: f32 = weights.iter().sum();
        assert!(
            (s - 1.0).abs() > 1e-3,
            "attention weights should not be softmax-normalized (sum = {s})"
        );
    }

    #[test]
    fn constant_history_interest_proportional_to_h() {
        // If h_0 = h_1 = … = h_{n-1} = h, then each weight a_i is identical
        // (same input to the activation unit), and interest = (n·a)·h, i.e.
        // every component of `interest` is a uniform scalar times the
        // corresponding component of `h`.
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let d = 6_usize;
        let h = random_vec(d, &mut rng);
        let mut history = Vec::with_capacity(4 * d);
        for _ in 0..4 {
            history.extend_from_slice(&h);
        }
        let target = random_vec(d, &mut rng);
        let interest = model.interest_rep(&history, 4, &target).unwrap();
        // Look up the implied scalar from a non-zero component (use the one
        // with largest |h[k]| to maximise numerical stability).
        let (k_max, _) = h
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &0.0));
        let scalar = interest[k_max] / h[k_max];
        for k in 0..d {
            assert!(
                (interest[k] - scalar * h[k]).abs() < 1e-4,
                "interest[{k}] = {} not proportional to h[{k}] = {} (scalar {scalar})",
                interest[k],
                h[k]
            );
        }
    }

    #[test]
    fn identical_history_items_get_identical_attention() {
        let mut rng = make_rng();
        let model = Din::new(default_cfg(), &mut rng).unwrap();
        let d = 6_usize;
        let h = random_vec(d, &mut rng);
        let mut history = Vec::with_capacity(3 * d);
        for _ in 0..3 {
            history.extend_from_slice(&h);
        }
        let target = random_vec(d, &mut rng);
        let weights = model.attention_over_history(&history, 3, &target).unwrap();
        for i in 1..3 {
            assert!(
                (weights[0] - weights[i]).abs() < 1e-5,
                "identical histories must yield identical attention: \
                 weights[0]={} weights[{i}]={}",
                weights[0],
                weights[i]
            );
        }
    }
}
