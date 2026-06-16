//! Joint forecasting + classification from a shared encoder.
//!
//! A single MLP backbone embeds the input window into a shared representation
//! that feeds two task heads:
//!
//! * a **regression head** that forecasts the next `horizon` steps, and
//! * a **classification head** that predicts a discrete series label.
//!
//! The two tasks are trained with a convex combination of their losses,
//! `L = λ · MSE + (1 − λ) · CE`, so a single `λ ∈ [0, 1]` trades off forecasting
//! accuracy against classification accuracy. This mirrors the standard
//! multi-task-learning recipe of hard parameter sharing (Caruana, 1997) applied
//! to time series, where a shared trunk improves data efficiency and
//! regularises both heads.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for a [`MultiTaskForecaster`].
#[derive(Debug, Clone)]
pub struct MultiTaskConfig {
    /// Flattened input length (e.g. a univariate window, or `T*C` flattened).
    pub input_len: usize,
    /// Backbone hidden width.
    pub hidden_dim: usize,
    /// Shared embedding dimension fed to both heads.
    pub embed_dim: usize,
    /// Forecast horizon (regression-head output width).
    pub horizon: usize,
    /// Number of classes (classification-head output width).
    pub n_classes: usize,
}

impl MultiTaskConfig {
    /// Small configuration for tests and CPU smoke runs.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            input_len: 24,
            hidden_dim: 32,
            embed_dim: 16,
            horizon: 6,
            n_classes: 4,
        }
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// Shared-backbone parameters (a two-layer GELU MLP).
#[derive(Debug, Clone)]
pub struct BackboneWeights {
    /// First layer weight `[hidden_dim, input_len]`.
    pub w1: Vec<f32>,
    /// First layer bias `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Second layer weight `[embed_dim, hidden_dim]`.
    pub w2: Vec<f32>,
    /// Second layer bias `[embed_dim]`.
    pub b2: Vec<f32>,
}

/// A single linear head `[out_dim, embed_dim]`.
#[derive(Debug, Clone)]
pub struct HeadWeights {
    /// Weight `[out_dim, embed_dim]`.
    pub w: Vec<f32>,
    /// Bias `[out_dim]`.
    pub b: Vec<f32>,
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Multi-task forecaster: shared backbone + forecast head + classification head.
#[derive(Debug, Clone)]
pub struct MultiTaskForecaster {
    /// Shared backbone (affects both heads).
    pub backbone: BackboneWeights,
    /// Forecast (regression) head.
    pub forecast_head: HeadWeights,
    /// Classification head.
    pub class_head: HeadWeights,
    /// Model configuration.
    pub cfg: MultiTaskConfig,
}

impl MultiTaskForecaster {
    /// Build a multi-task forecaster, initialising all parameters from `rng`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `input_len == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `hidden_dim == 0` or `embed_dim == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::ShapeMismatch`] when `n_classes < 2`.
    pub fn new(cfg: MultiTaskConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.input_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.hidden_dim == 0 || cfg.embed_dim == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if cfg.n_classes < 2 {
            return Err(TsError::ShapeMismatch {
                msg: "n_classes must be >= 2".into(),
            });
        }

        let backbone = BackboneWeights {
            w1: xavier(cfg.hidden_dim, cfg.input_len, rng),
            b1: vec![0.0; cfg.hidden_dim],
            w2: xavier(cfg.embed_dim, cfg.hidden_dim, rng),
            b2: vec![0.0; cfg.embed_dim],
        };
        let forecast_head = HeadWeights {
            w: xavier(cfg.horizon, cfg.embed_dim, rng),
            b: vec![0.0; cfg.horizon],
        };
        let class_head = HeadWeights {
            w: xavier(cfg.n_classes, cfg.embed_dim, rng),
            b: vec![0.0; cfg.n_classes],
        };

        Ok(Self {
            backbone,
            forecast_head,
            class_head,
            cfg,
        })
    }

    /// Encode the input window into the shared `[embed_dim]` representation.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `series.len() != input_len`.
    pub fn encode(&self, series: &[f32]) -> TsResult<Vec<f32>> {
        if series.len() != self.cfg.input_len {
            return Err(TsError::DimensionMismatch {
                expected: self.cfg.input_len,
                got: series.len(),
            });
        }
        // Layer 1: input_len -> hidden_dim, GELU.
        let mut hidden = linear(
            series,
            &self.backbone.w1,
            &self.backbone.b1,
            self.cfg.input_len,
            self.cfg.hidden_dim,
        );
        for h in &mut hidden {
            *h = gelu(*h);
        }
        // Layer 2: hidden_dim -> embed_dim, GELU.
        let mut embed = linear(
            &hidden,
            &self.backbone.w2,
            &self.backbone.b2,
            self.cfg.hidden_dim,
            self.cfg.embed_dim,
        );
        for e in &mut embed {
            *e = gelu(*e);
        }
        Ok(embed)
    }

    /// Forward pass: returns `(forecast [horizon], class_logits [n_classes])`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `series.len() != input_len`.
    pub fn forward(&self, series: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        let embed = self.encode(series)?;
        let forecast = linear(
            &embed,
            &self.forecast_head.w,
            &self.forecast_head.b,
            self.cfg.embed_dim,
            self.cfg.horizon,
        );
        let class_logits = linear(
            &embed,
            &self.class_head.w,
            &self.class_head.b,
            self.cfg.embed_dim,
            self.cfg.n_classes,
        );
        Ok((forecast, class_logits))
    }

    /// Mean-squared-error forecasting loss.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `forecast.len() != target.len()`.
    /// - [`TsError::EmptyInput`] when both are empty.
    pub fn mse_loss(forecast: &[f32], target: &[f32]) -> TsResult<f32> {
        if forecast.len() != target.len() {
            return Err(TsError::DimensionMismatch {
                expected: forecast.len(),
                got: target.len(),
            });
        }
        if forecast.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "forecast must not be empty".into(),
            });
        }
        let sum: f32 = forecast
            .iter()
            .zip(target.iter())
            .map(|(p, y)| (p - y) * (p - y))
            .sum();
        Ok(sum / forecast.len() as f32)
    }

    /// Softmax cross-entropy classification loss against an integer `label`.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `logits` is empty.
    /// - [`TsError::ShapeMismatch`] when `label >= logits.len()`.
    pub fn cross_entropy_loss(logits: &[f32], label: usize) -> TsResult<f32> {
        if logits.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "logits must not be empty".into(),
            });
        }
        if label >= logits.len() {
            return Err(TsError::ShapeMismatch {
                msg: format!("label {label} out of range for {} classes", logits.len()),
            });
        }
        // log-softmax at the target index: log p_label = z_label - logsumexp(z).
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logsumexp = max + logits.iter().map(|&z| (z - max).exp()).sum::<f32>().ln();
        Ok(logsumexp - logits[label])
    }

    /// Numerically-stable softmax of class logits (sums to 1).
    #[must_use]
    pub fn class_probabilities(logits: &[f32]) -> Vec<f32> {
        let mut p = logits.to_vec();
        softmax_row(&mut p);
        p
    }

    /// Combined multi-task loss `L = λ · MSE + (1 − λ) · CE`.
    ///
    /// `λ = 1` ignores classification; `λ = 0` ignores forecasting.
    ///
    /// # Errors
    ///
    /// - [`TsError::ShapeMismatch`] when `lambda` is not in `[0, 1]`.
    /// - Any error from [`Self::mse_loss`] / [`Self::cross_entropy_loss`].
    pub fn loss(
        forecast: &[f32],
        target_forecast: &[f32],
        class_logits: &[f32],
        label: usize,
        lambda: f32,
    ) -> TsResult<f32> {
        if !(0.0..=1.0).contains(&lambda) {
            return Err(TsError::ShapeMismatch {
                msg: format!("lambda {lambda} must be in [0, 1]"),
            });
        }
        let mse = Self::mse_loss(forecast, target_forecast)?;
        let ce = Self::cross_entropy_loss(class_logits, label)?;
        Ok(lambda * mse + (1.0 - lambda) * ce)
    }
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Xavier-magnitude initialised `[rows, cols]` matrix.
fn xavier(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    let mut v = vec![0.0; rows * cols];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

/// Single linear layer: `y[o] = b[o] + Σ_k w[o*in+k] x[k]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_dim];
    for (o, ov) in out.iter_mut().enumerate() {
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = b[o];
        for k in 0..in_dim {
            acc += w_row[k] * x[k];
        }
        *ov = acc;
    }
    out
}

/// GELU (tanh approximation).
#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

/// In-place numerically-stable softmax over a row.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in row.iter_mut() {
        *v *= inv;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2025)
    }

    fn input(cfg: &MultiTaskConfig) -> Vec<f32> {
        (0..cfg.input_len)
            .map(|i| (i as f32 * 0.21).sin())
            .collect()
    }

    #[test]
    fn multitask_output_shapes() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (forecast, logits) = model.forward(&x).expect("forward");
        assert_eq!(forecast.len(), cfg.horizon);
        assert_eq!(logits.len(), cfg.n_classes);
    }

    #[test]
    fn multitask_output_finite() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0; cfg.input_len];
        rng.fill_normal(&mut x);
        let (forecast, logits) = model.forward(&x).expect("forward");
        assert!(forecast.iter().all(|v| v.is_finite()));
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn multitask_class_softmax_sums_to_one() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (_f, logits) = model.forward(&x).expect("forward");
        let probs = MultiTaskForecaster::class_probabilities(&logits);
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax sums to {s}");
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn multitask_combined_loss_decomposition() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (forecast, logits) = model.forward(&x).expect("forward");
        let target = vec![0.3_f32; cfg.horizon];
        let label = 2usize;
        let lambda = 0.7_f32;

        let mse = MultiTaskForecaster::mse_loss(&forecast, &target).expect("mse");
        let ce = MultiTaskForecaster::cross_entropy_loss(&logits, label).expect("ce");
        let combined =
            MultiTaskForecaster::loss(&forecast, &target, &logits, label, lambda).expect("loss");

        assert!(combined.is_finite());
        let expected = lambda * mse + (1.0 - lambda) * ce;
        assert!(
            (combined - expected).abs() < 1e-5,
            "combined {combined} != λ·MSE+(1-λ)·CE {expected}"
        );
    }

    #[test]
    fn multitask_lambda_one_is_pure_mse() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (forecast, logits) = model.forward(&x).expect("forward");
        let target = vec![0.1_f32; cfg.horizon];
        let mse = MultiTaskForecaster::mse_loss(&forecast, &target).expect("mse");
        let l = MultiTaskForecaster::loss(&forecast, &target, &logits, 1, 1.0).expect("loss");
        assert!((l - mse).abs() < 1e-6, "λ=1 should equal MSE: {l} vs {mse}");
    }

    #[test]
    fn multitask_lambda_zero_is_pure_ce() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (forecast, logits) = model.forward(&x).expect("forward");
        let target = vec![0.1_f32; cfg.horizon];
        let label = 1usize;
        let ce = MultiTaskForecaster::cross_entropy_loss(&logits, label).expect("ce");
        let l = MultiTaskForecaster::loss(&forecast, &target, &logits, label, 0.0).expect("loss");
        assert!((l - ce).abs() < 1e-6, "λ=0 should equal CE: {l} vs {ce}");
    }

    #[test]
    fn multitask_shared_backbone_affects_both_heads() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let mut model = MultiTaskForecaster::new(cfg.clone(), &mut rng).expect("build");
        let x = input(&cfg);
        let (f0, l0) = model.forward(&x).expect("forward");

        // Perturb a shared-backbone weight — both heads must respond.
        model.backbone.w1[0] += 5.0;
        model.backbone.w2[0] += 5.0;
        let (f1, l1) = model.forward(&x).expect("forward");

        let f_diff: f32 = f0.iter().zip(f1.iter()).map(|(a, b)| (a - b).abs()).sum();
        let l_diff: f32 = l0.iter().zip(l1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(f_diff > 1e-5, "forecast head did not respond to backbone");
        assert!(l_diff > 1e-5, "class head did not respond to backbone");
    }

    #[test]
    fn multitask_deterministic_under_seed() {
        let mut r1 = LcgRng::new(99);
        let mut r2 = LcgRng::new(99);
        let cfg = MultiTaskConfig::tiny();
        let m1 = MultiTaskForecaster::new(cfg.clone(), &mut r1).expect("build");
        let m2 = MultiTaskForecaster::new(cfg.clone(), &mut r2).expect("build");
        let x = input(&cfg);
        let (f1, l1) = m1.forward(&x).expect("f1");
        let (f2, l2) = m2.forward(&x).expect("f2");
        assert_eq!(f1, f2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn multitask_mse_zero_at_target() {
        let pred = vec![1.0_f32, 2.0, 3.0];
        let mse = MultiTaskForecaster::mse_loss(&pred, &pred).expect("mse");
        assert!(mse.abs() < 1e-9, "mse at target should be 0, got {mse}");
    }

    #[test]
    fn multitask_ce_non_negative() {
        let logits = vec![0.5_f32, -1.0, 2.0, 0.1];
        for label in 0..logits.len() {
            let ce = MultiTaskForecaster::cross_entropy_loss(&logits, label).expect("ce");
            assert!(ce >= 0.0, "CE must be >= 0, got {ce}");
        }
    }

    #[test]
    fn multitask_err_input_mismatch() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig::tiny();
        let model = MultiTaskForecaster::new(cfg, &mut rng).expect("build");
        let x = vec![0.0; 5]; // wrong length
        assert!(matches!(
            model.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn multitask_err_bad_lambda() {
        let logits = vec![0.1_f32, 0.2, 0.3];
        let forecast = vec![0.0_f32; 3];
        let target = vec![0.0_f32; 3];
        assert!(matches!(
            MultiTaskForecaster::loss(&forecast, &target, &logits, 0, 1.5).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn multitask_err_label_out_of_range() {
        let logits = vec![0.1_f32, 0.2, 0.3];
        assert!(matches!(
            MultiTaskForecaster::cross_entropy_loss(&logits, 5).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn multitask_err_n_classes_too_small() {
        let mut rng = make_rng();
        let cfg = MultiTaskConfig {
            n_classes: 1,
            ..MultiTaskConfig::tiny()
        };
        assert!(matches!(
            MultiTaskForecaster::new(cfg, &mut rng).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }
}
