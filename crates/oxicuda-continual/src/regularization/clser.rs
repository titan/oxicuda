//! CLS-ER — Complementary Learning Systems Experience Replay (Arani 2022).
//!
//! Reference: Arani, E., Sarfraz, F. & Zonooz, B. (2022). "Learning Fast,
//! Learning Slow: A General Continual Learning Method based on Complementary
//! Learning System." *International Conference on Learning Representations*
//! (ICLR 2022).
//!
//! # Overview
//!
//! CLS-ER is inspired by the mammalian **complementary learning systems**
//! theory: a fast-adapting hippocampus and a slow-consolidating neocortex.
//! Concretely, alongside the *working* (plastic) model `θ` it maintains two
//! exponential-moving-average (EMA) copies updated at different rates:
//!
//! - a **plastic / fast** semantic memory `θ_p`, updated frequently with a
//!   high decay `α_p` (closely tracks the working model);
//! - a **stable / slow** semantic memory `θ_s`, updated rarely with a high
//!   decay `α_s` (a long-horizon, drift-resistant knowledge base).
//!
//! Each EMA is a Polyak average:
//!
//! ```text
//!   θ_m ← α_m · θ_m + (1 − α_m) · θ            (m ∈ {plastic, stable})
//! ```
//!
//! At replay time the working model is regularised toward the **consistency
//! target** drawn from whichever semantic memory is more confident
//! (`max-logit`) on the replayed example — in this CPU primitive we expose the
//! per-coordinate consistency penalty against a chosen target logit vector:
//!
//! ```text
//!   L_consistency(z) = (reg / 2) · Σ_i ( z_i − z_target_i )²
//! ```
//!
//! and provide the closed-form gradient `reg · (z − z_target)`.
//!
//! Everything is FP32; updates are deterministic and side-effect-free except
//! for the explicit in-place EMA mutations.

use crate::error::{ContinualError, ContinualResult};

/// Configuration for CLS-ER.
#[derive(Debug, Clone)]
pub struct ClserConfig {
    /// EMA decay for the **plastic** (fast) semantic memory `α_p ∈ [0, 1]`.
    /// Higher → slower tracking. The plastic memory typically uses a *lower*
    /// decay than the stable one so it adapts faster.
    pub alpha_plastic: f32,
    /// EMA decay for the **stable** (slow) semantic memory `α_s ∈ [0, 1]`.
    /// Typically close to 1 (e.g. 0.999) for a drift-resistant knowledge base.
    pub alpha_stable: f32,
    /// Consistency-regularisation strength (`reg ≥ 0`, finite).
    pub reg: f32,
}

impl Default for ClserConfig {
    fn default() -> Self {
        Self {
            alpha_plastic: 0.9,
            alpha_stable: 0.999,
            reg: 1.0,
        }
    }
}

impl ClserConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`ContinualError::InvalidMomentum`] if either decay is outside `[0, 1]`
    ///   or non-finite.
    /// - [`ContinualError::InvalidLambda`] if `reg` is negative or non-finite.
    pub fn validate(&self) -> ContinualResult<()> {
        for a in [self.alpha_plastic, self.alpha_stable] {
            if !a.is_finite() || !(0.0..=1.0).contains(&a) {
                return Err(ContinualError::InvalidMomentum { momentum: a });
            }
        }
        if !self.reg.is_finite() || self.reg < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: self.reg });
        }
        Ok(())
    }
}

/// CLS-ER dual-memory state: a plastic and a stable EMA copy of the working
/// model parameters.
#[derive(Debug, Clone)]
pub struct ClserState {
    /// Plastic (fast) semantic-memory parameters `θ_p`.
    pub plastic: Vec<f32>,
    /// Stable (slow) semantic-memory parameters `θ_s`.
    pub stable: Vec<f32>,
    /// Number of plastic EMA updates performed.
    pub n_plastic_updates: usize,
    /// Number of stable EMA updates performed.
    pub n_stable_updates: usize,
    cfg: ClserConfig,
}

impl ClserState {
    /// Initialise both semantic memories from the current working parameters.
    ///
    /// # Errors
    /// - Propagates [`ClserConfig::validate`].
    /// - [`ContinualError::EmptyInput`] if `params` is empty.
    pub fn new(params: &[f32], cfg: ClserConfig) -> ContinualResult<Self> {
        cfg.validate()?;
        if params.is_empty() {
            return Err(ContinualError::EmptyInput);
        }
        Ok(Self {
            plastic: params.to_vec(),
            stable: params.to_vec(),
            n_plastic_updates: 0,
            n_stable_updates: 0,
            cfg,
        })
    }

    /// Number of parameters tracked.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.plastic.len()
    }

    /// Update the plastic EMA toward the working parameters:
    /// `θ_p ← α_p θ_p + (1 − α_p) θ`.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `params.len()` differs.
    pub fn update_plastic(&mut self, params: &[f32]) -> ContinualResult<()> {
        let n = self.plastic.len();
        if params.len() != n {
            return Err(ContinualError::DimensionMismatch {
                expected: n,
                got: params.len(),
            });
        }
        let a = self.cfg.alpha_plastic;
        for (slot, &p) in self.plastic.iter_mut().zip(params.iter()) {
            *slot = a * *slot + (1.0 - a) * p;
        }
        self.n_plastic_updates += 1;
        Ok(())
    }

    /// Update the stable EMA toward the working parameters:
    /// `θ_s ← α_s θ_s + (1 − α_s) θ`.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `params.len()` differs.
    pub fn update_stable(&mut self, params: &[f32]) -> ContinualResult<()> {
        let n = self.stable.len();
        if params.len() != n {
            return Err(ContinualError::DimensionMismatch {
                expected: n,
                got: params.len(),
            });
        }
        let a = self.cfg.alpha_stable;
        for (slot, &p) in self.stable.iter_mut().zip(params.iter()) {
            *slot = a * *slot + (1.0 - a) * p;
        }
        self.n_stable_updates += 1;
        Ok(())
    }

    /// Select the consistency target between the two semantic memories.
    ///
    /// Following CLS-ER, the target logits come from whichever memory is more
    /// *confident* — here measured by the maximum logit magnitude — on the
    /// replayed example. Returns a copy of the chosen memory's logits applied
    /// to a caller-supplied **logit pair** (`plastic_logits`, `stable_logits`)
    /// that the caller has already produced from each memory's forward pass.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if the two logit slices differ
    ///   in length.
    /// - [`ContinualError::EmptyInput`] if either logit slice is empty.
    pub fn consistency_target(
        &self,
        plastic_logits: &[f32],
        stable_logits: &[f32],
    ) -> ContinualResult<Vec<f32>> {
        if plastic_logits.is_empty() || stable_logits.is_empty() {
            return Err(ContinualError::EmptyInput);
        }
        if plastic_logits.len() != stable_logits.len() {
            return Err(ContinualError::DimensionMismatch {
                expected: plastic_logits.len(),
                got: stable_logits.len(),
            });
        }
        let conf = |z: &[f32]| z.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        if conf(stable_logits) >= conf(plastic_logits) {
            Ok(stable_logits.to_vec())
        } else {
            Ok(plastic_logits.to_vec())
        }
    }
}

/// Consistency-regularisation penalty between the working logits and a target:
/// `(reg / 2) · Σ_i (z_i − z_target_i)²`.
///
/// # Errors
/// - Propagates [`ClserConfig::validate`].
/// - [`ContinualError::DimensionMismatch`] if the slices differ in length.
/// - [`ContinualError::NanEncountered`] if the result is non-finite.
pub fn clser_consistency_loss(
    logits: &[f32],
    target: &[f32],
    cfg: &ClserConfig,
) -> ContinualResult<f32> {
    cfg.validate()?;
    if logits.len() != target.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: logits.len(),
            got: target.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (z, zt) in logits.iter().zip(target.iter()) {
        let d = z - zt;
        acc += d * d;
    }
    let out = 0.5 * cfg.reg * acc;
    if !out.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "clser_consistency_loss",
        });
    }
    Ok(out)
}

/// Closed-form gradient of [`clser_consistency_loss`] w.r.t. `logits`:
/// `reg · (z − z_target)`.
///
/// # Errors
/// - Propagates [`ClserConfig::validate`].
/// - [`ContinualError::DimensionMismatch`] if the slices differ in length.
pub fn clser_consistency_grad(
    logits: &[f32],
    target: &[f32],
    cfg: &ClserConfig,
) -> ContinualResult<Vec<f32>> {
    cfg.validate()?;
    if logits.len() != target.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: logits.len(),
            got: target.len(),
        });
    }
    Ok(logits
        .iter()
        .zip(target.iter())
        .map(|(z, zt)| cfg.reg * (z - zt))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ClserConfig {
        ClserConfig {
            alpha_plastic: 0.5,
            alpha_stable: 0.9,
            reg: 2.0,
        }
    }

    // -------------------- validation ---------------------------------------

    #[test]
    fn config_default_valid() {
        assert!(ClserConfig::default().validate().is_ok());
    }

    #[test]
    fn config_bad_alpha_error() {
        let c = ClserConfig {
            alpha_plastic: 1.5,
            ..cfg()
        };
        assert!(matches!(
            c.validate(),
            Err(ContinualError::InvalidMomentum { .. })
        ));
    }

    #[test]
    fn config_bad_reg_error() {
        let c = ClserConfig { reg: -1.0, ..cfg() };
        assert!(matches!(
            c.validate(),
            Err(ContinualError::InvalidLambda { .. })
        ));
    }

    #[test]
    fn new_empty_params_error() {
        let r = ClserState::new(&[], cfg());
        assert!(matches!(r, Err(ContinualError::EmptyInput)));
    }

    // -------------------- EMA behaviour ------------------------------------

    #[test]
    fn new_initialises_both_memories() {
        let params = vec![1.0_f32, -2.0, 3.0];
        let st = ClserState::new(&params, cfg())
            .expect("CLSER state should construct with valid params");
        assert_eq!(st.plastic, params);
        assert_eq!(st.stable, params);
        assert_eq!(st.n_params(), 3);
    }

    #[test]
    fn plastic_ema_known_value() {
        // α_p = 0.5: θ_p ← 0.5·θ_p + 0.5·θ. Start θ_p = 0, θ = 2 → 1.0.
        let mut st = ClserState::new(&[0.0_f32], cfg())
            .expect("CLSER state should construct with valid params");
        st.update_plastic(&[2.0])
            .expect("plastic network update should succeed");
        assert!((st.plastic[0] - 1.0).abs() < 1e-6);
        assert_eq!(st.n_plastic_updates, 1);
    }

    #[test]
    fn stable_ema_moves_less_than_plastic() {
        // Same target, stable (α=0.9) moves less than plastic (α=0.5).
        let mut st = ClserState::new(&[0.0_f32], cfg())
            .expect("CLSER state should construct with valid params");
        st.update_plastic(&[10.0])
            .expect("plastic network update should succeed");
        st.update_stable(&[10.0])
            .expect("stable network update should succeed");
        assert!(
            st.plastic[0] > st.stable[0],
            "plastic {} should move more than stable {}",
            st.plastic[0],
            st.stable[0]
        );
    }

    #[test]
    fn ema_converges_toward_target() {
        let mut st = ClserState::new(&[0.0_f32; 2], cfg())
            .expect("CLSER state should construct with valid params");
        for _ in 0..200 {
            st.update_plastic(&[5.0, -3.0])
                .expect("plastic network update should succeed");
            st.update_stable(&[5.0, -3.0])
                .expect("stable network update should succeed");
        }
        assert!((st.plastic[0] - 5.0).abs() < 1e-2);
        assert!((st.stable[1] + 3.0).abs() < 1e-2);
    }

    #[test]
    fn update_dim_mismatch_error() {
        let mut st = ClserState::new(&[0.0_f32; 3], cfg())
            .expect("CLSER state should construct with valid params");
        assert!(matches!(
            st.update_plastic(&[1.0, 2.0]),
            Err(ContinualError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            st.update_stable(&[1.0, 2.0]),
            Err(ContinualError::DimensionMismatch { .. })
        ));
    }

    // -------------------- consistency target & loss ------------------------

    #[test]
    fn consistency_target_picks_more_confident() {
        let st = ClserState::new(&[0.0_f32; 2], cfg())
            .expect("CLSER state should construct with valid params");
        // Stable more confident (max logit 5 > 2) → returns stable logits.
        let plastic_logits = vec![2.0_f32, 1.0];
        let stable_logits = vec![5.0_f32, 0.0];
        let tgt = st
            .consistency_target(&plastic_logits, &stable_logits)
            .expect("should succeed with valid test inputs");
        assert_eq!(tgt, stable_logits);

        // Plastic more confident → returns plastic logits.
        let plastic2 = vec![9.0_f32, 1.0];
        let stable2 = vec![3.0_f32, 0.0];
        let tgt2 = st
            .consistency_target(&plastic2, &stable2)
            .expect("consistency target should compute from valid inputs");
        assert_eq!(tgt2, plastic2);
    }

    #[test]
    fn consistency_target_dim_mismatch_error() {
        let st = ClserState::new(&[0.0_f32; 2], cfg())
            .expect("CLSER state should construct with valid params");
        let r = st.consistency_target(&[1.0, 2.0], &[1.0]);
        assert!(matches!(r, Err(ContinualError::DimensionMismatch { .. })));
    }

    #[test]
    fn consistency_loss_zero_at_target() {
        let z = vec![1.0_f32, -2.0, 3.0];
        let loss =
            clser_consistency_loss(&z, &z, &cfg()).expect("CLSER consistency loss should compute");
        assert!(loss.abs() < 1e-6, "loss should be 0 at target, got {loss}");
    }

    #[test]
    fn consistency_loss_nonneg_and_grows() {
        let z_target = vec![0.0_f32; 4];
        let z_near = vec![0.1_f32; 4];
        let z_far = vec![1.0_f32; 4];
        let near = clser_consistency_loss(&z_near, &z_target, &cfg())
            .expect("CLSER consistency loss should compute");
        let far = clser_consistency_loss(&z_far, &z_target, &cfg())
            .expect("CLSER consistency loss should compute");
        assert!(near >= 0.0 && far >= 0.0);
        assert!(far > near, "loss must grow with distance ({near} < {far})");
    }

    #[test]
    fn consistency_loss_known_value() {
        // reg=2, z=[1,0], target=[0,0]: 0.5·2·(1²) = 1.0.
        let loss = clser_consistency_loss(&[1.0, 0.0], &[0.0, 0.0], &cfg())
            .expect("CLSER consistency loss should compute");
        assert!((loss - 1.0).abs() < 1e-6, "got {loss}");
    }

    #[test]
    fn consistency_grad_shape_and_zero_at_target() {
        let z = vec![1.0_f32, 2.0, 3.0];
        let g0 = clser_consistency_grad(&z, &z, &cfg())
            .expect("CLSER consistency gradient should compute");
        assert_eq!(g0.len(), 3);
        assert!(g0.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn consistency_grad_known_value() {
        // reg=2, z=[3], target=[1] → grad = 2·(3−1) = 4.
        let g = clser_consistency_grad(&[3.0], &[1.0], &cfg())
            .expect("CLSER consistency gradient should compute");
        assert!((g[0] - 4.0).abs() < 1e-6, "got {}", g[0]);
    }

    #[test]
    fn consistency_loss_dim_mismatch_error() {
        let r = clser_consistency_loss(&[1.0, 2.0], &[1.0], &cfg());
        assert!(matches!(r, Err(ContinualError::DimensionMismatch { .. })));
    }
}
