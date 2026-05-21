//! Threshold-dependent Batch Normalisation (tdBN).
//!
//! Zheng et al. 2021 AAAI, "Going Deeper with Directly-Trained Larger Spiking
//! Neural Networks". tdBN normalises pre-activation currents jointly over the
//! batch *and* time dimensions per feature, then rescales by a factor tied to
//! the neuron firing threshold `Vth` so the normalised signal is centred around
//! the threshold and spike activity is balanced across depth:
//!
//! ```text
//! μ_c   = mean over (batch × time) of x[:, c]
//! σ²_c  = var  over (batch × time) of x[:, c]
//! x̂[n,c] = α · Vth · (x[n,c] − μ_c) / sqrt(σ²_c + ε)
//! y[n,c] = γ_c · x̂[n,c] + β_c
//! ```
//!
//! During [`crate::layer::td_bn::TdBn::forward_train`] the batch+time statistics are computed from the
//! current mini-batch and the running statistics are updated by `momentum`:
//!
//! ```text
//! running_mean ← (1 − momentum) · running_mean + momentum · μ
//! running_var  ← (1 − momentum) · running_var  + momentum · σ²
//! ```
//!
//! During [`crate::layer::td_bn::TdBn::forward_eval`] the (frozen) running statistics are used instead
//! and no state is mutated.
//!
//! Layout. The input is a flat row-major `[batch * time, n_features]` buffer; the
//! `batch * time` rows are pooled together when computing per-feature statistics,
//! exactly as tdBN prescribes.

use crate::error::{SnnError, SnnResult};

/// Configuration for [`TdBn`].
#[derive(Debug, Clone, Copy)]
pub struct TdBnConfig {
    /// Number of features (channels) — the size of the last dimension.
    pub n_features: usize,
    /// Neuron firing threshold `Vth` that scales the normalised activations.
    pub threshold: f32,
    /// Numerical stabiliser added to the variance; must be `> 0`.
    pub eps: f32,
    /// Exponential moving-average rate for the running statistics, in `[0, 1]`.
    pub momentum: f32,
    /// Extra gain `α` on the threshold scaling (paper default `1.0`).
    pub alpha: f32,
}

impl Default for TdBnConfig {
    fn default() -> Self {
        Self {
            n_features: 1,
            threshold: 1.0,
            eps: 1e-5,
            momentum: 0.1,
            alpha: 1.0,
        }
    }
}

/// Threshold-dependent batch-norm layer with learnable affine parameters and
/// running statistics for inference.
#[derive(Debug, Clone)]
pub struct TdBn {
    /// Per-feature scale `γ`, initialised to `1.0`.
    pub gamma: Vec<f32>,
    /// Per-feature shift `β`, initialised to `0.0`.
    pub beta: Vec<f32>,
    /// Running per-feature mean used at evaluation time.
    pub running_mean: Vec<f32>,
    /// Running per-feature variance used at evaluation time.
    pub running_var: Vec<f32>,
    /// Layer configuration.
    pub cfg: TdBnConfig,
}

/// Validate the numeric invariants of a [`TdBnConfig`].
fn validate_cfg(cfg: &TdBnConfig) -> SnnResult<()> {
    if cfg.n_features == 0 {
        return Err(SnnError::BadDim {
            got: cfg.n_features,
        });
    }
    if !cfg.threshold.is_finite() || cfg.threshold <= 0.0 {
        return Err(SnnError::BadThreshold {
            v_th: cfg.threshold,
        });
    }
    if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "eps".into(),
            val: cfg.eps,
        });
    }
    if !cfg.momentum.is_finite() || !(0.0..=1.0).contains(&cfg.momentum) {
        return Err(SnnError::OutOfRange {
            name: "momentum".into(),
            val: cfg.momentum,
        });
    }
    if !cfg.alpha.is_finite() || cfg.alpha <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "alpha".into(),
            val: cfg.alpha,
        });
    }
    Ok(())
}

impl TdBn {
    /// Allocate a tdBN layer with `γ = 1`, `β = 0`, `running_mean = 0`,
    /// `running_var = 1`.
    ///
    /// Returns an error when any configuration field is out of range.
    pub fn new(cfg: TdBnConfig) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        let f = cfg.n_features;
        Ok(Self {
            gamma: vec![1.0_f32; f],
            beta: vec![0.0_f32; f],
            running_mean: vec![0.0_f32; f],
            running_var: vec![1.0_f32; f],
            cfg,
        })
    }

    /// Number of rows (`batch * time`) implied by `input`, validating shape.
    fn n_rows(&self, input: &[f32]) -> SnnResult<usize> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let f = self.cfg.n_features;
        if !input.len().is_multiple_of(f) {
            return Err(SnnError::BadShape {
                expected: f,
                got: input.len(),
            });
        }
        Ok(input.len() / f)
    }

    /// Apply the affine threshold scaling for one feature value.
    #[inline]
    fn affine(&self, x: f32, mean: f32, var: f32, c: usize) -> SnnResult<f32> {
        let g = self.gamma.get(c).copied().ok_or(SnnError::Internal {
            msg: "gamma index out of range".into(),
        })?;
        let b = self.beta.get(c).copied().ok_or(SnnError::Internal {
            msg: "beta index out of range".into(),
        })?;
        let denom = (var + self.cfg.eps).sqrt();
        let x_hat = self.cfg.alpha * self.cfg.threshold * (x - mean) / denom;
        Ok(g * x_hat + b)
    }

    /// Compute per-feature mean and (population) variance over all `n_rows` rows.
    fn batch_stats(&self, input: &[f32], n_rows: usize) -> SnnResult<(Vec<f32>, Vec<f32>)> {
        let f = self.cfg.n_features;
        let mut mean = vec![0.0_f32; f];
        let mut var = vec![0.0_f32; f];
        for r in 0..n_rows {
            let off = r * f;
            let row = input.get(off..off + f).ok_or(SnnError::Internal {
                msg: "stats input row out of range".into(),
            })?;
            for (m, &x) in mean.iter_mut().zip(row.iter()) {
                *m += x;
            }
        }
        let inv_n = 1.0_f32 / n_rows as f32;
        for m in &mut mean {
            *m *= inv_n;
        }
        for r in 0..n_rows {
            let off = r * f;
            let row = input.get(off..off + f).ok_or(SnnError::Internal {
                msg: "stats input row out of range".into(),
            })?;
            for ((v, &x), &m) in var.iter_mut().zip(row.iter()).zip(mean.iter()) {
                let d = x - m;
                *v += d * d;
            }
        }
        for v in &mut var {
            *v *= inv_n;
        }
        Ok((mean, var))
    }

    /// Forward pass in **training** mode.
    ///
    /// Computes batch+time statistics, normalises, applies the affine transform,
    /// and updates the running statistics by `momentum`. Returns the output of
    /// the same shape as `input`.
    pub fn forward_train(&mut self, input: &[f32]) -> SnnResult<Vec<f32>> {
        let f = self.cfg.n_features;
        let n_rows = self.n_rows(input)?;
        let (mean, var) = self.batch_stats(input, n_rows)?;
        let mut out = vec![0.0_f32; input.len()];
        for r in 0..n_rows {
            let off = r * f;
            let row = input.get(off..off + f).ok_or(SnnError::Internal {
                msg: "train input row out of range".into(),
            })?;
            let out_row = out.get_mut(off..off + f).ok_or(SnnError::Internal {
                msg: "train output row out of range".into(),
            })?;
            for (c, (o, &x)) in out_row.iter_mut().zip(row.iter()).enumerate() {
                let m = mean.get(c).copied().ok_or(SnnError::Internal {
                    msg: "mean index out of range".into(),
                })?;
                let v = var.get(c).copied().ok_or(SnnError::Internal {
                    msg: "var index out of range".into(),
                })?;
                *o = self.affine(x, m, v, c)?;
            }
        }
        // Update running statistics by momentum.
        let mom = self.cfg.momentum;
        for ((rm, &m), (rv, &v)) in self
            .running_mean
            .iter_mut()
            .zip(mean.iter())
            .zip(self.running_var.iter_mut().zip(var.iter()))
        {
            *rm = (1.0 - mom) * *rm + mom * m;
            *rv = (1.0 - mom) * *rv + mom * v;
        }
        Ok(out)
    }

    /// Forward pass in **evaluation** mode using the frozen running statistics.
    ///
    /// Does not mutate any state. Returns the output of the same shape as
    /// `input`.
    pub fn forward_eval(&self, input: &[f32]) -> SnnResult<Vec<f32>> {
        let f = self.cfg.n_features;
        let n_rows = self.n_rows(input)?;
        let mut out = vec![0.0_f32; input.len()];
        for r in 0..n_rows {
            let off = r * f;
            let row = input.get(off..off + f).ok_or(SnnError::Internal {
                msg: "eval input row out of range".into(),
            })?;
            let out_row = out.get_mut(off..off + f).ok_or(SnnError::Internal {
                msg: "eval output row out of range".into(),
            })?;
            for (c, (o, &x)) in out_row.iter_mut().zip(row.iter()).enumerate() {
                let m = self
                    .running_mean
                    .get(c)
                    .copied()
                    .ok_or(SnnError::Internal {
                        msg: "running_mean index out of range".into(),
                    })?;
                let v = self.running_var.get(c).copied().ok_or(SnnError::Internal {
                    msg: "running_var index out of range".into(),
                })?;
                *o = self.affine(x, m, v, c)?;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-feature mean of a `[n_rows, n_features]` row-major buffer.
    fn feature_mean(data: &[f32], f: usize) -> Vec<f32> {
        let n = data.len() / f;
        let mut m = vec![0.0_f32; f];
        for r in 0..n {
            for c in 0..f {
                m[c] += data[r * f + c];
            }
        }
        for v in &mut m {
            *v /= n as f32;
        }
        m
    }

    /// Per-feature population std of a `[n_rows, n_features]` buffer.
    fn feature_std(data: &[f32], f: usize) -> Vec<f32> {
        let n = data.len() / f;
        let m = feature_mean(data, f);
        let mut v = vec![0.0_f32; f];
        for r in 0..n {
            for c in 0..f {
                let d = data[r * f + c] - m[c];
                v[c] += d * d;
            }
        }
        v.iter().map(|&s| (s / n as f32).sqrt()).collect()
    }

    fn sample_input(f: usize) -> Vec<f32> {
        // 4 rows (batch*time) x f features with distinct, non-degenerate values.
        let mut data = Vec::new();
        for r in 0..4 {
            for c in 0..f {
                data.push((r as f32) * 0.7 + (c as f32) * 1.3 - 0.4);
            }
        }
        data
    }

    #[test]
    fn train_output_mean_near_zero() {
        let f = 3;
        let cfg = TdBnConfig {
            n_features: f,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        let out = bn.forward_train(&input).expect("train");
        let m = feature_mean(&out, f);
        for &mc in &m {
            assert!(mc.abs() < 1e-4, "feature mean not ~0: {mc}");
        }
    }

    #[test]
    fn train_scaled_std_near_alpha_vth() {
        let f = 4;
        let cfg = TdBnConfig {
            n_features: f,
            threshold: 1.5,
            alpha: 1.0,
            eps: 1e-6,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        let out = bn.forward_train(&input).expect("train");
        let s = feature_std(&out, f);
        // With gamma=1, beta=0 the per-feature std equals alpha*Vth (eps small).
        for &sc in &s {
            assert!(
                (sc - cfg.alpha * cfg.threshold).abs() < 1e-3,
                "std={} expected={}",
                sc,
                cfg.alpha * cfg.threshold
            );
        }
    }

    #[test]
    fn running_stats_move_toward_batch_stats() {
        let f = 2;
        let cfg = TdBnConfig {
            n_features: f,
            momentum: 0.5,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        let (batch_mean, batch_var) = bn.batch_stats(&input, input.len() / f).expect("stats");
        let before_mean = bn.running_mean.clone();
        bn.forward_train(&input).expect("train");
        for (c, ((&rm, &bm0), &bm)) in bn
            .running_mean
            .iter()
            .zip(before_mean.iter())
            .zip(batch_mean.iter())
            .enumerate()
        {
            // running_mean started at 0; after one step with momentum 0.5 it is
            // halfway to the batch mean.
            let expected = 0.5 * bm0 + 0.5 * bm;
            assert!(
                (rm - expected).abs() < 1e-5,
                "running_mean[{c}]={rm} expected={expected}"
            );
        }
        // running_var moved away from its init of 1 toward the batch variance.
        for (&rv, &bv) in bn.running_var.iter().zip(batch_var.iter()) {
            let expected = 0.5 * 1.0 + 0.5 * bv;
            assert!((rv - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn eval_uses_running_and_differs_from_train() {
        let f = 3;
        let cfg = TdBnConfig {
            n_features: f,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        // Train output uses batch stats; eval uses default running (mean 0, var 1).
        let train_out = bn.forward_train(&input).expect("train");
        // Snapshot a fresh layer so eval uses the *initial* running stats (mean 0,
        // var 1) which differ from this batch's stats.
        let fresh = TdBn::new(cfg).expect("fresh");
        let eval_out = fresh.forward_eval(&input).expect("eval");
        assert_ne!(train_out, eval_out, "train and eval should differ");
    }

    #[test]
    fn eval_does_not_mutate_state() {
        let f = 2;
        let cfg = TdBnConfig {
            n_features: f,
            ..Default::default()
        };
        let bn = TdBn::new(cfg).expect("ctor");
        let rm_before = bn.running_mean.clone();
        let rv_before = bn.running_var.clone();
        let input = sample_input(f);
        let _ = bn.forward_eval(&input).expect("eval");
        assert_eq!(bn.running_mean, rm_before);
        assert_eq!(bn.running_var, rv_before);
    }

    #[test]
    fn default_gamma_one_beta_zero() {
        let cfg = TdBnConfig {
            n_features: 5,
            ..Default::default()
        };
        let bn = TdBn::new(cfg).expect("ctor");
        assert!(bn.gamma.iter().all(|&g| (g - 1.0).abs() < 1e-9));
        assert!(bn.beta.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn vth_two_doubles_spread_vs_vth_one() {
        let f = 3;
        let input = sample_input(f);
        let cfg1 = TdBnConfig {
            n_features: f,
            threshold: 1.0,
            ..Default::default()
        };
        let cfg2 = TdBnConfig {
            n_features: f,
            threshold: 2.0,
            ..Default::default()
        };
        let mut bn1 = TdBn::new(cfg1).expect("bn1");
        let mut bn2 = TdBn::new(cfg2).expect("bn2");
        let s1 = feature_std(&bn1.forward_train(&input).expect("t1"), f);
        let s2 = feature_std(&bn2.forward_train(&input).expect("t2"), f);
        for c in 0..f {
            assert!(
                (s2[c] - 2.0 * s1[c]).abs() < 1e-3,
                "Vth=2 std {} should be ~2x Vth=1 std {}",
                s2[c],
                s1[c]
            );
        }
    }

    #[test]
    fn alpha_scales_spread() {
        let f = 2;
        let input = sample_input(f);
        let cfg1 = TdBnConfig {
            n_features: f,
            alpha: 1.0,
            ..Default::default()
        };
        let cfg2 = TdBnConfig {
            n_features: f,
            alpha: 3.0,
            ..Default::default()
        };
        let mut bn1 = TdBn::new(cfg1).expect("bn1");
        let mut bn2 = TdBn::new(cfg2).expect("bn2");
        let s1 = feature_std(&bn1.forward_train(&input).expect("t1"), f);
        let s2 = feature_std(&bn2.forward_train(&input).expect("t2"), f);
        for c in 0..f {
            assert!((s2[c] - 3.0 * s1[c]).abs() < 1e-3);
        }
    }

    #[test]
    fn gamma_beta_affine_applied() {
        let f = 1;
        let cfg = TdBnConfig {
            n_features: f,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        // Set gamma=2, beta=5: output mean should be ~beta (since x_hat mean ~0).
        bn.gamma[0] = 2.0;
        bn.beta[0] = 5.0;
        let input = sample_input(f);
        let out = bn.forward_train(&input).expect("train");
        let m = feature_mean(&out, f);
        assert!((m[0] - 5.0).abs() < 1e-3, "mean with beta=5 was {}", m[0]);
    }

    #[test]
    fn momentum_one_replaces_running_with_batch() {
        let f = 2;
        let cfg = TdBnConfig {
            n_features: f,
            momentum: 1.0,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        let (batch_mean, batch_var) = bn.batch_stats(&input, input.len() / f).expect("stats");
        bn.forward_train(&input).expect("train");
        for c in 0..f {
            assert!((bn.running_mean[c] - batch_mean[c]).abs() < 1e-5);
            assert!((bn.running_var[c] - batch_var[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn momentum_zero_keeps_running_unchanged() {
        let f = 2;
        let cfg = TdBnConfig {
            n_features: f,
            momentum: 0.0,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let rm = bn.running_mean.clone();
        let rv = bn.running_var.clone();
        let input = sample_input(f);
        bn.forward_train(&input).expect("train");
        assert_eq!(bn.running_mean, rm);
        assert_eq!(bn.running_var, rv);
    }

    #[test]
    fn output_shape_matches_input() {
        let f = 4;
        let cfg = TdBnConfig {
            n_features: f,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let input = sample_input(f);
        let out = bn.forward_train(&input).expect("train");
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn n_features_mismatch_errors() {
        let cfg = TdBnConfig {
            n_features: 3,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        // 7 is not a multiple of 3.
        let input = vec![1.0_f32; 7];
        let err = bn.forward_train(&input);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn eps_non_positive_errors() {
        let cfg = TdBnConfig {
            n_features: 2,
            eps: 0.0,
            ..Default::default()
        };
        let err = TdBn::new(cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn empty_input_errors() {
        let cfg = TdBnConfig {
            n_features: 2,
            ..Default::default()
        };
        let mut bn = TdBn::new(cfg).expect("ctor");
        let err = bn.forward_train(&[]);
        assert!(matches!(err, Err(SnnError::EmptyInput)));
    }

    #[test]
    fn alpha_non_positive_errors() {
        let cfg = TdBnConfig {
            n_features: 2,
            alpha: 0.0,
            ..Default::default()
        };
        let err = TdBn::new(cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn zero_features_errors() {
        let cfg = TdBnConfig {
            n_features: 0,
            ..Default::default()
        };
        let err = TdBn::new(cfg);
        assert!(matches!(err, Err(SnnError::BadDim { .. })));
    }

    #[test]
    fn momentum_out_of_bounds_errors() {
        let cfg = TdBnConfig {
            n_features: 2,
            momentum: 1.5,
            ..Default::default()
        };
        let err = TdBn::new(cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn non_positive_threshold_errors() {
        let cfg = TdBnConfig {
            n_features: 2,
            threshold: -1.0,
            ..Default::default()
        };
        let err = TdBn::new(cfg);
        assert!(matches!(err, Err(SnnError::BadThreshold { .. })));
    }
}
