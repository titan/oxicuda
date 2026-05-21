//! AdapterFusion — non-destructive composition of multiple task adapters via attention.
//!
//! Reference: Pfeiffer J, Kamath A, Rücklé A, Cho K, Gurevych I (2021)
//! "AdapterFusion: Non-Destructive Task Composition for Transfer Learning",
//! EACL 2021. <https://arxiv.org/abs/2005.00247>
//!
//! AdapterFusion composes the outputs of `N` task-specific adapters at each
//! transformer layer using a scaled-dot-product attention scheme:
//!
//! ```text
//!   Q   = hidden · W_Q
//!   K_k = adapter_out_k · W_K
//!   V_k = adapter_out_k · W_V
//!   s_k = (Q · K_k) / (τ · √d_model)
//!   p   = softmax(s)
//!   out = Σ_k p_k · V_k
//! ```
//!
//! where `τ` is a temperature controlling the sharpness of the routing
//! distribution. The fusion module is parameter-efficient because only the
//! three `d_model × d_model` projection matrices `W_Q`, `W_K`, `W_V` are
//! trained while the underlying task adapters remain frozen.

use crate::error::{PeftError, PeftResult};
use crate::handle::PeftHandle;

/// Configuration for [`AdapterFusion::new`].
#[derive(Debug, Clone)]
pub struct AdapterFusionConfig {
    /// Hidden dimension of the host transformer.
    pub d_model: usize,
    /// Number of task adapters being fused.
    pub n_adapters: usize,
    /// Softmax temperature `τ`; `> 0`, with `1.0` recovering the unscaled variant.
    pub temperature: f32,
}

/// Attention-based fusion of multiple task adapter outputs.
#[derive(Debug, Clone)]
pub struct AdapterFusion {
    /// Query projection, row-major `d_model × d_model`.
    pub w_q: Vec<f32>,
    /// Key projection, row-major `d_model × d_model`.
    pub w_k: Vec<f32>,
    /// Value projection, row-major `d_model × d_model`.
    pub w_v: Vec<f32>,
    /// Configuration captured at construction time.
    pub cfg: AdapterFusionConfig,
}

impl AdapterFusion {
    /// Allocate the three projection matrices using Kaiming-uniform init drawn
    /// from `handle.rng`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if `d_model == 0`, `n_adapters == 0`,
    /// or `temperature ≤ 0`.
    pub fn new(cfg: AdapterFusionConfig, handle: &mut PeftHandle) -> PeftResult<Self> {
        validate_config(&cfg)?;
        let d = cfg.d_model;
        let elements = d * d;

        // Kaiming-uniform bounds for a fan-in == fan-out == d_model linear layer:
        //   limit = sqrt(6 / (d_model + d_model))
        let limit = (6.0_f32 / (2.0 * d as f32)).sqrt();
        let w_q = kaiming_uniform_matrix(elements, limit, handle);
        let w_k = kaiming_uniform_matrix(elements, limit, handle);
        let w_v = kaiming_uniform_matrix(elements, limit, handle);

        Ok(Self { w_q, w_k, w_v, cfg })
    }

    /// Compute the fused output and the per-adapter attention weights.
    ///
    /// Returns `(output, attention_weights)` where `output.len() == d_model`
    /// and `attention_weights.len() == n_adapters`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if any input length disagrees
    /// with the configuration.
    pub fn forward(
        &self,
        hidden: &[f32],
        adapter_outputs: &[Vec<f32>],
    ) -> PeftResult<(Vec<f32>, Vec<f32>)> {
        check_inputs(&self.cfg, hidden, adapter_outputs)?;
        let d = self.cfg.d_model;
        let n = self.cfg.n_adapters;

        // Q = hidden · W_Q (row-major matvec on the right).
        let q = right_matvec(hidden, &self.w_q, d);

        // K_k, V_k and raw scores.
        let scale_denom = (self.cfg.temperature * (d as f32).sqrt()).max(f32::EPSILON);
        let mut keys = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        let mut scores = vec![0.0_f32; n];
        for (k, adapter_out) in adapter_outputs.iter().enumerate() {
            let k_vec = right_matvec(adapter_out, &self.w_k, d);
            let v_vec = right_matvec(adapter_out, &self.w_v, d);
            let dot = dot_product(&q, &k_vec);
            scores[k] = dot / scale_denom;
            keys.push(k_vec);
            values.push(v_vec);
        }

        let probs = softmax_max_shift(&scores);

        let mut out = vec![0.0_f32; d];
        for (k, p_k) in probs.iter().enumerate() {
            let v_k = &values[k];
            for (slot, v) in out.iter_mut().zip(v_k.iter()) {
                *slot += p_k * v;
            }
        }
        // Touch `keys` so the symbol stays alive for the duration of the
        // computation even if a future change uses the cached buffer (e.g. for
        // a backward pass). This is a no-op at run time.
        debug_assert_eq!(keys.len(), n);

        Ok((out, probs))
    }

    /// Return only the attention weights without computing the fused output.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if any input length disagrees
    /// with the configuration.
    pub fn attention_weights(
        &self,
        hidden: &[f32],
        adapter_outputs: &[Vec<f32>],
    ) -> PeftResult<Vec<f32>> {
        check_inputs(&self.cfg, hidden, adapter_outputs)?;
        let d = self.cfg.d_model;
        let n = self.cfg.n_adapters;
        let q = right_matvec(hidden, &self.w_q, d);
        let scale_denom = (self.cfg.temperature * (d as f32).sqrt()).max(f32::EPSILON);
        let mut scores = vec![0.0_f32; n];
        for (k, adapter_out) in adapter_outputs.iter().enumerate() {
            let k_vec = right_matvec(adapter_out, &self.w_k, d);
            scores[k] = dot_product(&q, &k_vec) / scale_denom;
        }
        Ok(softmax_max_shift(&scores))
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Validate the configuration handed to [`AdapterFusion::new`].
fn validate_config(cfg: &AdapterFusionConfig) -> PeftResult<()> {
    if cfg.d_model == 0 {
        return Err(PeftError::Internal {
            msg: "AdapterFusion d_model must be > 0".to_string(),
        });
    }
    if cfg.n_adapters == 0 {
        return Err(PeftError::Internal {
            msg: "AdapterFusion n_adapters must be > 0".to_string(),
        });
    }
    if cfg.temperature.is_nan() || cfg.temperature <= 0.0 {
        return Err(PeftError::Internal {
            msg: format!(
                "AdapterFusion temperature must be > 0 and finite, got {}",
                cfg.temperature
            ),
        });
    }
    Ok(())
}

/// Validate the runtime shape of `hidden` and `adapter_outputs`.
fn check_inputs(
    cfg: &AdapterFusionConfig,
    hidden: &[f32],
    adapter_outputs: &[Vec<f32>],
) -> PeftResult<()> {
    if hidden.len() != cfg.d_model {
        return Err(PeftError::DimensionMismatch {
            expected: cfg.d_model,
            got: hidden.len(),
        });
    }
    if adapter_outputs.len() != cfg.n_adapters {
        return Err(PeftError::DimensionMismatch {
            expected: cfg.n_adapters,
            got: adapter_outputs.len(),
        });
    }
    for adapter_out in adapter_outputs.iter() {
        if adapter_out.len() != cfg.d_model {
            return Err(PeftError::DimensionMismatch {
                expected: cfg.d_model,
                got: adapter_out.len(),
            });
        }
    }
    Ok(())
}

/// Fill a buffer of `n` `f32` entries with i.i.d. samples from `U(-limit, limit)`,
/// using one `f32` draw from `handle.rng` per entry.
fn kaiming_uniform_matrix(n: usize, limit: f32, handle: &mut PeftHandle) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    let two_limit = 2.0 * limit;
    for slot in out.iter_mut() {
        let u = handle.rng.next_f32(); // ∈ [0, 1)
        *slot = u * two_limit - limit;
    }
    out
}

/// Compute `out[i] = Σ_j x[j] · w[j * d + i]` for a row-major `d × d` weight matrix.
///
/// This is the standard "matrix multiplied on the right" operation: `out = x · W`.
fn right_matvec(x: &[f32], w: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    for (j, &xj) in x.iter().enumerate().take(d) {
        if xj == 0.0 {
            continue;
        }
        let row_offset = j * d;
        for (i, slot) in out.iter_mut().enumerate().take(d) {
            *slot += xj * w[row_offset + i];
        }
    }
    out
}

/// Dot product of two equal-length `f32` slices. Caller guarantees the lengths match.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Numerically stable softmax with max shift.
fn softmax_max_shift(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0_f32; scores.len()];
    let mut sum = 0.0_f32;
    for (slot, &s) in out.iter_mut().zip(scores.iter()) {
        let e = (s - m).exp();
        *slot = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for slot in out.iter_mut() {
        *slot *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(d: usize, n: usize, temperature: f32) -> AdapterFusionConfig {
        AdapterFusionConfig {
            d_model: d,
            n_adapters: n,
            temperature,
        }
    }

    fn handle(seed: u64) -> PeftHandle {
        PeftHandle::new(80, seed)
    }

    #[test]
    fn single_adapter_weight_is_one() {
        let cfg = default_cfg(6, 1, 1.0);
        let mut h = handle(7);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![0.3_f32, -0.2, 0.4, 0.1, -0.5, 0.6];
        let outs = vec![vec![0.1_f32, 0.2, -0.3, 0.4, 0.5, -0.6]];
        let (_y, probs) = fusion.forward(&hidden, &outs).unwrap();
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6, "weight = {}", probs[0]);
    }

    #[test]
    fn two_identical_adapters_split_attention_evenly() {
        let cfg = default_cfg(5, 2, 1.0);
        let mut h = handle(11);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![0.1_f32, 0.2, 0.3, -0.1, 0.0];
        let same = vec![0.7_f32, -0.4, 0.2, 0.5, -0.3];
        let outs = vec![same.clone(), same];
        let probs = fusion.attention_weights(&hidden, &outs).unwrap();
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-5);
        assert!((probs[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn hidden_dim_mismatch_errors() {
        let cfg = default_cfg(4, 2, 1.0);
        let mut h = handle(1);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let bad_hidden = vec![0.1_f32; 5]; // wrong length
        let outs = vec![vec![0.0_f32; 4]; 2];
        let res = fusion.forward(&bad_hidden, &outs);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn n_adapter_count_mismatch_errors() {
        let cfg = default_cfg(4, 3, 1.0);
        let mut h = handle(2);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![0.1_f32; 4];
        let outs = vec![vec![0.0_f32; 4]; 2]; // only two adapters
        let res = fusion.forward(&hidden, &outs);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn adapter_inner_dim_mismatch_errors() {
        let cfg = default_cfg(4, 2, 1.0);
        let mut h = handle(3);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![0.1_f32; 4];
        let outs = vec![vec![0.0_f32; 4], vec![0.0_f32; 3]]; // second wrong
        let res = fusion.forward(&hidden, &outs);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn zero_n_adapters_in_config_errors() {
        let cfg = default_cfg(4, 0, 1.0);
        let mut h = handle(4);
        let res = AdapterFusion::new(cfg, &mut h);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn zero_d_model_in_config_errors() {
        let cfg = default_cfg(0, 2, 1.0);
        let mut h = handle(5);
        let res = AdapterFusion::new(cfg, &mut h);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn nonpositive_temperature_errors() {
        for &temp in &[0.0_f32, -1e-3, -100.0] {
            let cfg = default_cfg(4, 2, temp);
            let mut h = handle(6);
            let res = AdapterFusion::new(cfg, &mut h);
            assert!(
                matches!(res, Err(PeftError::Internal { .. })),
                "expected Internal for temperature {temp}"
            );
        }
        // NaN temperature also rejected.
        let cfg = default_cfg(4, 2, f32::NAN);
        let mut h = handle(7);
        let res = AdapterFusion::new(cfg, &mut h);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn attention_weights_sum_to_one() {
        let cfg = default_cfg(6, 5, 1.0);
        let mut h = handle(7);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden: Vec<f32> = (0..6).map(|i| 0.1 * i as f32 - 0.3).collect();
        let outs: Vec<Vec<f32>> = (0..5)
            .map(|k| (0..6).map(|i| 0.05 * i as f32 + 0.1 * k as f32).collect())
            .collect();
        let probs = fusion.attention_weights(&hidden, &outs).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn output_shape_matches_d_model() {
        let d = 8;
        let cfg = default_cfg(d, 3, 1.0);
        let mut h = handle(9);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![0.5_f32; d];
        let outs = vec![vec![0.25_f32; d]; 3];
        let (out, _probs) = fusion.forward(&hidden, &outs).unwrap();
        assert_eq!(out.len(), d);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let cfg = default_cfg(5, 4, 1.0);
        let mut h1 = handle(33);
        let mut h2 = handle(33);
        let a = AdapterFusion::new(cfg.clone(), &mut h1).unwrap();
        let b = AdapterFusion::new(cfg, &mut h2).unwrap();
        assert_eq!(a.w_q, b.w_q);
        assert_eq!(a.w_k, b.w_k);
        assert_eq!(a.w_v, b.w_v);
    }

    #[test]
    fn large_temperature_flattens_distribution() {
        let cfg = default_cfg(6, 5, 100.0);
        let mut h = handle(13);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden: Vec<f32> = (0..6).map(|i| 0.3 * i as f32 - 0.7).collect();
        let outs: Vec<Vec<f32>> = (0..5)
            .map(|k| (0..6).map(|i| 0.2 * (i as f32 + k as f32 * 1.1)).collect())
            .collect();
        let probs = fusion.attention_weights(&hidden, &outs).unwrap();
        let uniform = 1.0_f32 / 5.0;
        for &p in &probs {
            assert!(
                (p - uniform).abs() < 1e-2,
                "expected near-uniform weight, got {p}"
            );
        }
    }

    #[test]
    fn small_temperature_sharpens_distribution() {
        let cfg = default_cfg(6, 5, 1.0);
        let mut h = handle(15);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden: Vec<f32> = (0..6).map(|i| 0.3 * i as f32 - 0.7).collect();
        let outs: Vec<Vec<f32>> = (0..5)
            .map(|k| (0..6).map(|i| 0.2 * (i as f32 + k as f32 * 1.1)).collect())
            .collect();
        let probs_baseline = fusion.attention_weights(&hidden, &outs).unwrap();

        // Same data, sharper temperature.
        let cfg_sharp = default_cfg(6, 5, 0.01);
        let mut h_sharp = handle(15);
        let fusion_sharp = AdapterFusion::new(cfg_sharp, &mut h_sharp).unwrap();
        let probs_sharp = fusion_sharp.attention_weights(&hidden, &outs).unwrap();

        let max_baseline = probs_baseline.iter().copied().fold(0.0_f32, f32::max);
        let max_sharp = probs_sharp.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_sharp > max_baseline,
            "sharper temperature should concentrate mass: sharp_max={max_sharp} baseline_max={max_baseline}"
        );
        assert!(
            max_sharp > 0.9,
            "very small temperature should put almost all mass on one adapter, got max={max_sharp}"
        );
    }

    #[test]
    fn different_seeds_give_different_weights() {
        let cfg = default_cfg(5, 3, 1.0);
        let mut h1 = handle(1);
        let mut h2 = handle(2);
        let a = AdapterFusion::new(cfg.clone(), &mut h1).unwrap();
        let b = AdapterFusion::new(cfg, &mut h2).unwrap();
        let diff_q: f32 = a
            .w_q
            .iter()
            .zip(b.w_q.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        let diff_k: f32 = a
            .w_k
            .iter()
            .zip(b.w_k.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        let diff_v: f32 = a
            .w_v
            .iter()
            .zip(b.w_v.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff_q > 1e-3, "W_Q identical across seeds: diff={diff_q}");
        assert!(diff_k > 1e-3, "W_K identical across seeds: diff={diff_k}");
        assert!(diff_v > 1e-3, "W_V identical across seeds: diff={diff_v}");
    }

    #[test]
    fn weights_match_forward_attention() {
        let cfg = default_cfg(5, 4, 1.0);
        let mut h = handle(21);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden: Vec<f32> = (0..5).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let outs: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..5).map(|i| 0.05 * (i as f32 - k as f32)).collect())
            .collect();
        let (_y, probs_fwd) = fusion.forward(&hidden, &outs).unwrap();
        let probs_only = fusion.attention_weights(&hidden, &outs).unwrap();
        for (a, b) in probs_fwd.iter().zip(probs_only.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "forward vs weights mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn kaiming_uniform_bounds_respected() {
        let d = 8;
        let cfg = default_cfg(d, 2, 1.0);
        let mut h = handle(101);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let limit = (6.0_f32 / (2.0 * d as f32)).sqrt();
        for v in fusion
            .w_q
            .iter()
            .chain(fusion.w_k.iter())
            .chain(fusion.w_v.iter())
        {
            assert!(v.abs() <= limit + 1e-6, "weight {v} outside ±{limit}");
        }
    }

    #[test]
    fn extreme_score_does_not_produce_nan() {
        // Construct a fusion where one adapter's V_k is enormous; we want a
        // finite output and a valid probability distribution.
        let cfg = default_cfg(4, 2, 1.0);
        let mut h = handle(42);
        let fusion = AdapterFusion::new(cfg, &mut h).unwrap();
        let hidden = vec![1.0_f32; 4];
        let outs = vec![vec![1e6_f32; 4], vec![-1e6_f32; 4]];
        let (out, probs) = fusion.forward(&hidden, &outs).unwrap();
        for &v in &out {
            assert!(
                v.is_finite(),
                "extreme score produced non-finite output {v}"
            );
        }
        for &p in &probs {
            assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}
