//! Transductive Batch Normalisation (TBN) for few-shot meta-learning.
//!
//! Reference: Bronskill, Gordon, Requeima, Nowozin, Turner —
//! "TaskNorm: Rethinking Batch Normalization for Meta-Learning" (ICML 2020),
//! and the broader few-shot literature on transductive BN.
//!
//! At meta-**test** time, vanilla per-channel BN — which freezes the running
//! statistics learnt over the meta-training stream — discards a powerful source
//! of information: the *query* batch.  In the few-shot regime support sets are
//! tiny (often just one or five examples per class) so the support batch
//! statistics are extremely noisy and the training-time running statistics may
//! be a poor match for the new task.
//!
//! Transductive batch normalisation observes that the unlabeled query inputs
//! arrive *together* with the support set at meta-test time, and uses the
//! joint (support ∪ query) per-channel mean and variance to normalise *both*
//! the support and the query.  Because the query is treated as one big
//! transductive batch, the new task's data distribution is reflected directly
//! in the normalisation statistics, which empirically improves few-shot
//! classification accuracy.
//!
//! This module exposes:
//!
//! * [`batch_norm`](TransductiveBn::batch_norm) — standard per-channel BN
//!   over a single batch with the module's affine parameters γ, β.  This is
//!   the *non*-transductive path used at meta-training time.
//! * [`transductive_apply`](TransductiveBn::transductive_apply) — joint
//!   normalisation of `(support, query)`, returning both normalised batches.
//!
//! The module owns only the affine parameters γ (`scale`) and β (`shift`);
//! running statistics are intentionally omitted because in the transductive
//! setting the batch itself provides the moments.  This keeps the interface
//! aligned with the typical use of TBN in code such as CNAPs and TaskNorm
//! where the moments are recomputed from each task's data.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`TransductiveBn`].
#[derive(Debug, Clone)]
pub struct TbnConfig {
    /// Number of feature channels.  Every sample passed to the module must
    /// have exactly this many components.
    pub n_features: usize,
    /// Numerical stabiliser added to the per-channel variance before the
    /// square-root.  Must be strictly positive.
    pub eps: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// Transductive batch normalisation with learnable per-channel affine
/// parameters γ (`scale`) and β (`shift`).
///
/// Each sample is treated as a length-`n_features` row of a `n_samples ×
/// n_features` row-major matrix.  No running statistics are stored — at
/// inference time the moments are always computed from the current (possibly
/// joint) batch.
pub struct TransductiveBn {
    /// Per-channel multiplicative scale γ (length `n_features`).
    scale: Vec<f32>,
    /// Per-channel additive shift β (length `n_features`).
    shift: Vec<f32>,
    /// Frozen configuration (copied so it can be queried back).
    cfg: TbnConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-channel `(mean, variance)` over a row-major `n × d` matrix.
/// `n == 0` falls back to zero mean and zero variance — the caller is
/// expected to ensure that at least one sample exists in the joint batch.
fn per_channel_moments(x: &[f32], n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0_f32; n_features];
    let mut var = vec![0.0_f32; n_features];
    if n_samples == 0 {
        return (mean, var);
    }
    let inv_n = 1.0_f32 / n_samples as f32;
    for s in 0..n_samples {
        let row = &x[s * n_features..(s + 1) * n_features];
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m *= inv_n;
    }
    for s in 0..n_samples {
        let row = &x[s * n_features..(s + 1) * n_features];
        for (v_acc, (&v, &m)) in var.iter_mut().zip(row.iter().zip(mean.iter())) {
            let diff = v - m;
            *v_acc += diff * diff;
        }
    }
    for v in var.iter_mut() {
        *v *= inv_n;
    }
    (mean, var)
}

/// Affine-normalise a row-major `n × d` matrix in place with shared per-channel
/// statistics: `y = (x − μ) / sqrt(σ² + ε) · γ + β`.
fn apply_affine_normalise(
    out: &mut [f32],
    mean: &[f32],
    var: &[f32],
    scale: &[f32],
    shift: &[f32],
    n_samples: usize,
    n_features: usize,
    eps: f32,
) {
    for s in 0..n_samples {
        let row = &mut out[s * n_features..(s + 1) * n_features];
        for (idx, val) in row.iter_mut().enumerate() {
            let denom = (var[idx] + eps).sqrt();
            let normed = (*val - mean[idx]) / denom;
            *val = normed * scale[idx] + shift[idx];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// impl TransductiveBn
// ─────────────────────────────────────────────────────────────────────────────

impl TransductiveBn {
    /// Construct a TBN with `γ_i = 1`, `β_i = 0` (identity affine).
    ///
    /// `rng` is accepted to keep the constructor signature uniform with the
    /// rest of the network modules; the deterministic identity initialisation
    /// makes init-independent tests easy to write.
    pub fn new(cfg: TbnConfig, _rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.n_features == 0 {
            return Err(MetaError::InvalidFeatDim {
                dim: cfg.n_features,
            });
        }
        if cfg.eps.is_nan() || cfg.eps <= 0.0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("eps must be > 0, got {}", cfg.eps),
            });
        }
        Ok(Self {
            scale: vec![1.0_f32; cfg.n_features],
            shift: vec![0.0_f32; cfg.n_features],
            cfg,
        })
    }

    /// Number of feature channels — convenience accessor.
    pub fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    /// Numerical stabiliser ε used in the denominator — convenience accessor.
    pub fn eps(&self) -> f32 {
        self.cfg.eps
    }

    /// Manually override the affine parameters (used by tests of the
    /// γ = 2 / β = 1 affine path).
    pub fn set_affine(&mut self, scale: &[f32], shift: &[f32]) -> MetaResult<()> {
        if scale.len() != self.cfg.n_features {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.n_features,
                got: scale.len(),
            });
        }
        if shift.len() != self.cfg.n_features {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.n_features,
                got: shift.len(),
            });
        }
        self.scale.copy_from_slice(scale);
        self.shift.copy_from_slice(shift);
        Ok(())
    }

    /// Standard per-channel batch normalisation over `n_samples × n_features`,
    /// using the *current* batch's moments rather than any running estimates.
    ///
    /// `x`: row-major `n_samples × n_features`.
    /// Returns a fresh `Vec<f32>` of length `n_samples · n_features` with
    /// `y = (x − μ) / √(σ² + ε) · γ + β` applied per channel.
    pub fn batch_norm(&self, x: &[f32], n_samples: usize) -> MetaResult<Vec<f32>> {
        let d = self.cfg.n_features;
        let expected = n_samples.checked_mul(d).ok_or(MetaError::Internal {
            msg: "n_samples * n_features overflowed usize".into(),
        })?;
        if x.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if n_samples == 0 {
            // No samples — nothing to normalise.  Return an empty vector
            // (length 0) rather than NaN-poison anything.
            return Ok(Vec::new());
        }
        let (mean, var) = per_channel_moments(x, n_samples, d);
        let mut out = x.to_vec();
        apply_affine_normalise(
            &mut out,
            &mean,
            &var,
            &self.scale,
            &self.shift,
            n_samples,
            d,
            self.cfg.eps,
        );
        Ok(out)
    }

    /// Transductive normalisation over the union of `support` and `query`.
    ///
    /// 1. Concatenate `support` and `query` to obtain a joint `(n_support +
    ///    n_query) × n_features` batch.
    /// 2. Compute per-channel `(μ, σ²)` of that joint batch.
    /// 3. Normalise the support and the query *separately* using the **same**
    ///    joint statistics, then apply the affine `γ, β`.
    ///
    /// Returns the pair `(normalised_support, normalised_query)`.  Either
    /// half can be empty (`n_support == 0` or `n_query == 0`); if both are
    /// empty an empty pair is returned.
    pub fn transductive_apply(
        &self,
        support: &[f32],
        n_support: usize,
        query: &[f32],
        n_query: usize,
    ) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        let d = self.cfg.n_features;
        let expected_s = n_support.checked_mul(d).ok_or(MetaError::Internal {
            msg: "n_support * n_features overflowed usize".into(),
        })?;
        let expected_q = n_query.checked_mul(d).ok_or(MetaError::Internal {
            msg: "n_query * n_features overflowed usize".into(),
        })?;
        if support.len() != expected_s {
            return Err(MetaError::DimensionMismatch {
                expected: expected_s,
                got: support.len(),
            });
        }
        if query.len() != expected_q {
            return Err(MetaError::DimensionMismatch {
                expected: expected_q,
                got: query.len(),
            });
        }

        let total = n_support + n_query;
        if total == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Build the joint batch buffer to compute the shared statistics.
        let mut joint = Vec::with_capacity(total * d);
        joint.extend_from_slice(support);
        joint.extend_from_slice(query);
        let (mean, var) = per_channel_moments(&joint, total, d);

        let mut s_out = support.to_vec();
        if n_support > 0 {
            apply_affine_normalise(
                &mut s_out,
                &mean,
                &var,
                &self.scale,
                &self.shift,
                n_support,
                d,
                self.cfg.eps,
            );
        }

        let mut q_out = query.to_vec();
        if n_query > 0 {
            apply_affine_normalise(
                &mut q_out,
                &mean,
                &var,
                &self.scale,
                &self.shift,
                n_query,
                d,
                self.cfg.eps,
            );
        }
        Ok((s_out, q_out))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> TbnConfig {
        TbnConfig {
            n_features: 3,
            eps: 1e-5,
        }
    }

    fn make_tbn(cfg: TbnConfig) -> TransductiveBn {
        let mut rng = LcgRng::new(7);
        // unwrap is local to tests, where it is permitted
        TransductiveBn::new(cfg, &mut rng).expect("valid TBN config")
    }

    fn per_channel_mean(out: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut m = vec![0.0_f32; d];
        for s in 0..n {
            for (i, mi) in m.iter_mut().enumerate() {
                *mi += out[s * d + i];
            }
        }
        let inv = 1.0_f32 / n as f32;
        for mi in m.iter_mut() {
            *mi *= inv;
        }
        m
    }

    fn per_channel_var(out: &[f32], mean: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; d];
        for s in 0..n {
            for (i, vi) in v.iter_mut().enumerate() {
                let diff = out[s * d + i] - mean[i];
                *vi += diff * diff;
            }
        }
        let inv = 1.0_f32 / n as f32;
        for vi in v.iter_mut() {
            *vi *= inv;
        }
        v
    }

    // ── construction ────────────────────────────────────────────────────────

    #[test]
    fn new_valid_cfg_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(TransductiveBn::new(default_cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_zero_features_errs() {
        let mut rng = LcgRng::new(1);
        let cfg = TbnConfig {
            n_features: 0,
            eps: 1e-5,
        };
        assert!(matches!(
            TransductiveBn::new(cfg, &mut rng),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_non_positive_eps_errs() {
        let mut rng = LcgRng::new(1);
        let cfg = TbnConfig {
            n_features: 3,
            eps: 0.0,
        };
        assert!(matches!(
            TransductiveBn::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
        let cfg_neg = TbnConfig {
            n_features: 3,
            eps: -1e-5,
        };
        assert!(matches!(
            TransductiveBn::new(cfg_neg, &mut LcgRng::new(2)),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn n_features_accessor() {
        let tbn = make_tbn(default_cfg());
        assert_eq!(tbn.n_features(), 3);
    }

    // ── batch_norm ──────────────────────────────────────────────────────────

    #[test]
    fn batch_norm_output_length() {
        let tbn = make_tbn(default_cfg());
        let n = 4;
        let d = tbn.n_features();
        let x = vec![0.5_f32; n * d];
        let out = tbn.batch_norm(&x, n).expect("ok");
        assert_eq!(out.len(), n * d);
    }

    #[test]
    fn batch_norm_zero_mean_unit_var_per_channel() {
        let tbn = make_tbn(default_cfg());
        let n = 5;
        let d = tbn.n_features();
        // Distinct per-channel data so the variance is non-zero.
        let mut x = vec![0.0_f32; n * d];
        for s in 0..n {
            for i in 0..d {
                x[s * d + i] = (s as f32) * 0.7 + (i as f32) * 0.3 - 1.5;
            }
        }
        let out = tbn.batch_norm(&x, n).expect("ok");
        let mean = per_channel_mean(&out, n, d);
        let var = per_channel_var(&out, &mean, n, d);
        for &m in &mean {
            assert!(m.abs() < 1e-4, "per-channel mean ≈ 0, got {m}");
        }
        for &v in &var {
            // unit variance up to the ε perturbation
            assert!((v - 1.0).abs() < 1e-2, "per-channel var ≈ 1, got {v}");
        }
    }

    #[test]
    fn batch_norm_wrong_length_errs() {
        let tbn = make_tbn(default_cfg());
        let x = vec![0.0_f32; 5];
        assert!(matches!(
            tbn.batch_norm(&x, 2),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_norm_constant_input_is_finite() {
        // Variance is zero, so the ε floor is what keeps the denominator
        // strictly positive — the output must be finite (in fact zero +
        // shift, but we only assert finiteness here).
        let tbn = make_tbn(default_cfg());
        let n = 4;
        let d = tbn.n_features();
        let x = vec![3.5_f32; n * d];
        let out = tbn.batch_norm(&x, n).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batch_norm_deterministic() {
        let tbn = make_tbn(default_cfg());
        let n = 4;
        let d = tbn.n_features();
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32).sin()).collect();
        let a = tbn.batch_norm(&x, n).expect("ok");
        let b = tbn.batch_norm(&x, n).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn batch_norm_affine_gamma_two_beta_one() {
        let mut tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        tbn.set_affine(&vec![2.0_f32; d], &vec![1.0_f32; d])
            .expect("ok");
        let n = 5;
        let mut x = vec![0.0_f32; n * d];
        for s in 0..n {
            for i in 0..d {
                x[s * d + i] = (s as f32) - 2.0 + (i as f32) * 0.5;
            }
        }
        let out = tbn.batch_norm(&x, n).expect("ok");
        // With γ=2, β=1 the per-channel mean should be β=1 and the variance
        // should be γ² = 4 (modulo the ε perturbation).
        let mean = per_channel_mean(&out, n, d);
        let var = per_channel_var(&out, &mean, n, d);
        for &m in &mean {
            assert!((m - 1.0).abs() < 1e-3, "mean after β=1 ≈ 1, got {m}");
        }
        for &v in &var {
            assert!((v - 4.0).abs() < 1e-1, "var after γ=2 ≈ 4, got {v}");
        }
    }

    // ── transductive_apply ──────────────────────────────────────────────────

    #[test]
    fn transductive_apply_shapes_preserved() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let n_s = 4;
        let n_q = 3;
        let s = vec![0.25_f32; n_s * d];
        let q = vec![0.75_f32; n_q * d];
        let (ns, nq) = tbn.transductive_apply(&s, n_s, &q, n_q).expect("ok");
        assert_eq!(ns.len(), n_s * d);
        assert_eq!(nq.len(), n_q * d);
    }

    #[test]
    fn transductive_joint_mean_is_zero() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let n_s = 3;
        let n_q = 5;
        let mut s = vec![0.0_f32; n_s * d];
        for r in 0..n_s {
            for i in 0..d {
                s[r * d + i] = (r as f32) * 0.4 + (i as f32) * 0.2;
            }
        }
        let mut q = vec![0.0_f32; n_q * d];
        for r in 0..n_q {
            for i in 0..d {
                q[r * d + i] = -(r as f32) * 0.3 + (i as f32) * 0.5 + 1.0;
            }
        }
        let (ns, nq) = tbn.transductive_apply(&s, n_s, &q, n_q).expect("ok");
        // Concatenate and check per-channel mean of the joint normalised
        // output is approximately zero (γ=1, β=0).
        let mut joint = Vec::with_capacity((n_s + n_q) * d);
        joint.extend_from_slice(&ns);
        joint.extend_from_slice(&nq);
        let mean = per_channel_mean(&joint, n_s + n_q, d);
        for &m in &mean {
            assert!(m.abs() < 1e-4, "joint mean ≈ 0, got {m}");
        }
        let var = per_channel_var(&joint, &mean, n_s + n_q, d);
        for &v in &var {
            assert!((v - 1.0).abs() < 1e-2, "joint var ≈ 1, got {v}");
        }
    }

    #[test]
    fn transductive_single_support_works() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s = vec![0.5_f32, -0.25, 2.0];
        let q = vec![1.0_f32, 0.0, -1.0, 0.2, 0.7, 0.9];
        let (ns, nq) = tbn.transductive_apply(&s, 1, &q, 2).expect("ok");
        assert_eq!(ns.len(), d);
        assert_eq!(nq.len(), 2 * d);
        assert!(ns.iter().all(|v| v.is_finite()));
        assert!(nq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transductive_single_query_works() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s = vec![1.0_f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        let q = vec![0.5_f32, 0.25, -0.25];
        let (ns, nq) = tbn.transductive_apply(&s, 2, &q, 1).expect("ok");
        assert_eq!(ns.len(), 2 * d);
        assert_eq!(nq.len(), d);
        assert!(ns.iter().all(|v| v.is_finite()));
        assert!(nq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transductive_zero_query_reduces_to_support_only_bn() {
        // With n_query = 0 the joint batch is just the support — the result
        // for the support half should equal batch_norm(support).
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let n_s = 4;
        let mut s = vec![0.0_f32; n_s * d];
        for r in 0..n_s {
            for i in 0..d {
                s[r * d + i] = (r as f32) - 1.5 + (i as f32) * 0.3;
            }
        }
        let bn = tbn.batch_norm(&s, n_s).expect("ok");
        let (ns, nq) = tbn.transductive_apply(&s, n_s, &[], 0).expect("ok");
        assert!(nq.is_empty());
        assert_eq!(ns.len(), bn.len());
        for (a, b) in ns.iter().zip(bn.iter()) {
            assert!((a - b).abs() < 1e-6, "support path differs: {a} vs {b}");
        }
    }

    #[test]
    fn transductive_constant_joint_finite() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s = vec![1.5_f32; 3 * d];
        let q = vec![1.5_f32; 2 * d];
        let (ns, nq) = tbn.transductive_apply(&s, 3, &q, 2).expect("ok");
        assert!(ns.iter().all(|v| v.is_finite()));
        assert!(nq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transductive_changing_query_changes_support_norm() {
        // This is the defining property of TBN: because the support is
        // normalised by *joint* statistics, swapping the query changes the
        // support's normalised values.
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s: Vec<f32> = (0..3 * d).map(|i| (i as f32) * 0.15).collect();
        let q_a: Vec<f32> = (0..4 * d).map(|i| (i as f32) * 0.1).collect();
        let q_b: Vec<f32> = (0..4 * d).map(|i| (i as f32) * 0.1 + 5.0).collect();
        let (ns_a, _) = tbn.transductive_apply(&s, 3, &q_a, 4).expect("ok");
        let (ns_b, _) = tbn.transductive_apply(&s, 3, &q_b, 4).expect("ok");
        let mut differ = false;
        for (a, b) in ns_a.iter().zip(ns_b.iter()) {
            if (a - b).abs() > 1e-3 {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "support normalisation must change when query shifts (transductive property)"
        );
    }

    #[test]
    fn transductive_support_wrong_length_errs() {
        let tbn = make_tbn(default_cfg());
        let s = vec![0.0_f32; 5];
        let q = vec![0.0_f32; 6];
        assert!(matches!(
            tbn.transductive_apply(&s, 2, &q, 2),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn transductive_query_wrong_length_errs() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s = vec![0.0_f32; 2 * d];
        let q = vec![0.0_f32; 5]; // not a multiple of d
        assert!(matches!(
            tbn.transductive_apply(&s, 2, &q, 2),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn transductive_deterministic() {
        let tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        let s: Vec<f32> = (0..4 * d).map(|i| (i as f32).cos()).collect();
        let q: Vec<f32> = (0..3 * d).map(|i| (i as f32).sin()).collect();
        let a = tbn.transductive_apply(&s, 4, &q, 3).expect("ok");
        let b = tbn.transductive_apply(&s, 4, &q, 3).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn transductive_affine_gamma_two_beta_one() {
        let mut tbn = make_tbn(default_cfg());
        let d = tbn.n_features();
        tbn.set_affine(&vec![2.0_f32; d], &vec![1.0_f32; d])
            .expect("ok");
        let n_s = 3;
        let n_q = 4;
        let mut s = vec![0.0_f32; n_s * d];
        for r in 0..n_s {
            for i in 0..d {
                s[r * d + i] = (r as f32) - 1.0 + (i as f32) * 0.2;
            }
        }
        let mut q = vec![0.0_f32; n_q * d];
        for r in 0..n_q {
            for i in 0..d {
                q[r * d + i] = -(r as f32) + (i as f32) * 0.4 + 0.5;
            }
        }
        let (ns, nq) = tbn.transductive_apply(&s, n_s, &q, n_q).expect("ok");
        let mut joint = Vec::with_capacity((n_s + n_q) * d);
        joint.extend_from_slice(&ns);
        joint.extend_from_slice(&nq);
        let mean = per_channel_mean(&joint, n_s + n_q, d);
        let var = per_channel_var(&joint, &mean, n_s + n_q, d);
        for &m in &mean {
            assert!((m - 1.0).abs() < 1e-3, "joint mean (β=1) ≈ 1, got {m}");
        }
        for &v in &var {
            assert!((v - 4.0).abs() < 1e-1, "joint var (γ=2) ≈ 4, got {v}");
        }
    }

    #[test]
    fn transductive_double_empty_returns_empty() {
        let tbn = make_tbn(default_cfg());
        let (ns, nq) = tbn.transductive_apply(&[], 0, &[], 0).expect("ok");
        assert!(ns.is_empty());
        assert!(nq.is_empty());
    }

    #[test]
    fn set_affine_wrong_length_errs() {
        let mut tbn = make_tbn(default_cfg());
        assert!(matches!(
            tbn.set_affine(&[1.0_f32, 1.0], &vec![0.0_f32; tbn.n_features()]),
            Err(MetaError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            tbn.set_affine(&vec![1.0_f32; tbn.n_features()], &[0.0_f32, 0.0]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }
}
