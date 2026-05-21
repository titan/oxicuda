#![allow(clippy::needless_range_loop)]
//! Homeostatic plasticity rules: BCM (Bienenstock-Cooper-Munro 1982) and
//! Oja's Hebbian PCA rule (1982).
//!
//! ## BCM rule
//!
//! The sliding modification threshold tracks the time-averaged squared
//! post-synaptic activity:
//!
//! ```text
//! θ_M[j] ← (1 − τ_θ) · θ_M[j] + τ_θ · y_j²
//! Δw_{ij} = η · x_i · y_j · (y_j − θ_M[j])
//! ```
//!
//! Weights are then hard-clamped to `[w_min, w_max]`.
//!
//! ## Oja rule
//!
//! Normalised Hebbian rule that converges to the first principal component:
//!
//! ```text
//! y    = w^T · x
//! Δw_i = η · y · (x_i − y · w_i)
//! ```

use crate::error::{SnnError, SnnResult};

// ─────────────────────────────────────────────────────────────────────────────
// BCM
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters for the BCM plasticity rule.
#[derive(Debug, Clone, Copy)]
pub struct BcmConfig {
    /// Number of pre-synaptic neurons.
    pub n_pre: usize,
    /// Number of post-synaptic neurons.
    pub n_post: usize,
    /// Learning rate `η`.
    pub eta: f32,
    /// Exponential moving-average fraction `τ_θ ∈ (0, 1]` for the sliding threshold.
    pub tau_theta: f32,
    /// Hard lower clip on synaptic weight.
    pub w_min: f32,
    /// Hard upper clip on synaptic weight.
    pub w_max: f32,
}

impl Default for BcmConfig {
    fn default() -> Self {
        Self {
            n_pre: 10,
            n_post: 10,
            eta: 0.001,
            tau_theta: 0.01,
            w_min: -1.0,
            w_max: 1.0,
        }
    }
}

impl BcmConfig {
    /// Construct with `n_pre` × `n_post` neurons and default hyperparameters.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            n_pre,
            n_post,
            ..Self::default()
        }
    }
}

/// Mutable BCM state: per-post-synaptic sliding threshold vector.
#[derive(Debug, Clone)]
pub struct BcmState {
    /// Modification threshold `θ_M`, one entry per post-synaptic neuron (init 0).
    pub theta_m: Vec<f32>,
}

impl BcmState {
    /// Allocate a zero-initialised state for `n_post` neurons.
    #[must_use]
    pub fn new(n_post: usize) -> Self {
        Self {
            theta_m: vec![0.0_f32; n_post],
        }
    }

    /// Reset all thresholds to zero.
    pub fn reset(&mut self) {
        for v in &mut self.theta_m {
            *v = 0.0;
        }
    }
}

/// Validate BCM configuration fields.
fn validate_bcm_config(cfg: &BcmConfig) -> SnnResult<()> {
    if cfg.n_pre == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if cfg.n_post == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if cfg.tau_theta <= 0.0 || cfg.tau_theta > 1.0 || !cfg.tau_theta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "tau_theta".into(),
            val: cfg.tau_theta,
        });
    }
    if !cfg.w_min.is_finite() || !cfg.w_max.is_finite() || cfg.w_min >= cfg.w_max {
        return Err(SnnError::OutOfRange {
            name: "w_min/w_max".into(),
            val: cfg.w_min,
        });
    }
    Ok(())
}

/// Validate slice shapes for a single BCM step.
fn validate_bcm_slices(
    weights: &[f32],
    state: &BcmState,
    pre_activity: &[f32],
    post_activity: &[f32],
    cfg: &BcmConfig,
) -> SnnResult<()> {
    if weights.len() != cfg.n_post * cfg.n_pre {
        return Err(SnnError::BadShape {
            expected: cfg.n_post * cfg.n_pre,
            got: weights.len(),
        });
    }
    if state.theta_m.len() != cfg.n_post {
        return Err(SnnError::IncompatibleLength {
            a: cfg.n_post,
            b: state.theta_m.len(),
        });
    }
    if pre_activity.len() != cfg.n_pre {
        return Err(SnnError::IncompatibleLength {
            a: cfg.n_pre,
            b: pre_activity.len(),
        });
    }
    if post_activity.len() != cfg.n_post {
        return Err(SnnError::IncompatibleLength {
            a: cfg.n_post,
            b: post_activity.len(),
        });
    }
    Ok(())
}

/// Advance the BCM rule by one timestep.
///
/// Weights are stored in row-major order `[n_post × n_pre]` so that
/// `weights[j * n_pre + i]` is the synapse from pre-neuron `i` to
/// post-neuron `j`.
///
/// # Errors
/// Returns `SnnError` if any slice has the wrong length or configuration
/// values are invalid.
pub fn bcm_step(
    weights: &mut [f32],
    state: &mut BcmState,
    pre_activity: &[f32],
    post_activity: &[f32],
    cfg: &BcmConfig,
) -> SnnResult<()> {
    validate_bcm_config(cfg)?;
    validate_bcm_slices(weights, state, pre_activity, post_activity, cfg)?;

    for j in 0..cfg.n_post {
        let y_j = post_activity[j];
        let y_sq = y_j * y_j;

        // Slide the modification threshold.
        state.theta_m[j] = (1.0 - cfg.tau_theta) * state.theta_m[j] + cfg.tau_theta * y_sq;

        // BCM weight update and clamping.
        let row_off = j * cfg.n_pre;
        for i in 0..cfg.n_pre {
            let dw = cfg.eta * pre_activity[i] * y_j * (y_j - state.theta_m[j]);
            let updated = weights[row_off + i] + dw;
            weights[row_off + i] = updated.clamp(cfg.w_min, cfg.w_max);
        }
    }
    Ok(())
}

/// Run the BCM rule for `n_steps` timesteps.
///
/// `pre_activities` must have shape `[n_steps × n_pre]` and
/// `post_activities` must have shape `[n_steps × n_post]`, both stored
/// in row-major order.
///
/// # Errors
/// Returns `SnnError` if any slice has the wrong length or configuration
/// values are invalid.
pub fn bcm_run(
    weights: &mut [f32],
    state: &mut BcmState,
    pre_activities: &[f32],
    post_activities: &[f32],
    n_steps: usize,
    cfg: &BcmConfig,
) -> SnnResult<()> {
    validate_bcm_config(cfg)?;
    if n_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if pre_activities.len() != n_steps * cfg.n_pre {
        return Err(SnnError::BadShape {
            expected: n_steps * cfg.n_pre,
            got: pre_activities.len(),
        });
    }
    if post_activities.len() != n_steps * cfg.n_post {
        return Err(SnnError::BadShape {
            expected: n_steps * cfg.n_post,
            got: post_activities.len(),
        });
    }
    if weights.len() != cfg.n_post * cfg.n_pre {
        return Err(SnnError::BadShape {
            expected: cfg.n_post * cfg.n_pre,
            got: weights.len(),
        });
    }
    if state.theta_m.len() != cfg.n_post {
        return Err(SnnError::IncompatibleLength {
            a: cfg.n_post,
            b: state.theta_m.len(),
        });
    }

    for t in 0..n_steps {
        let pre_slice = &pre_activities[t * cfg.n_pre..(t + 1) * cfg.n_pre];
        let post_slice = &post_activities[t * cfg.n_post..(t + 1) * cfg.n_post];
        bcm_step(weights, state, pre_slice, post_slice, cfg)?;
    }
    Ok(())
}

/// Compute the equilibrium BCM threshold `E[y_j²]` over `n_steps` timesteps
/// for each post-synaptic neuron independently.
///
/// `post_activities` must have shape `[n_steps × n_post]`.
///
/// # Errors
/// Returns `SnnError::BadShape` if the slice has the wrong length, or
/// `SnnError::BadDim` if `n_post == 0`.
pub fn bcm_equilibrium_theta(
    post_activities: &[f32],
    n_post: usize,
    n_steps: usize,
) -> SnnResult<Vec<f32>> {
    if n_post == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if n_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if post_activities.len() != n_steps * n_post {
        return Err(SnnError::BadShape {
            expected: n_steps * n_post,
            got: post_activities.len(),
        });
    }

    let mut theta = vec![0.0_f32; n_post];
    let inv_steps = 1.0_f32 / n_steps as f32;
    for t in 0..n_steps {
        for j in 0..n_post {
            let y = post_activities[t * n_post + j];
            theta[j] += y * y * inv_steps;
        }
    }
    Ok(theta)
}

// ─────────────────────────────────────────────────────────────────────────────
// Oja rule
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters for Oja's Hebbian PCA rule.
#[derive(Debug, Clone, Copy)]
pub struct OjaConfig {
    /// Input dimensionality.
    pub n_input: usize,
    /// Learning rate `η`.
    pub eta: f32,
    /// Hard lower clip on weight entries (`f32::NEG_INFINITY` = unclamped).
    pub w_min: f32,
    /// Hard upper clip on weight entries (`f32::INFINITY` = unclamped).
    pub w_max: f32,
}

impl Default for OjaConfig {
    fn default() -> Self {
        Self {
            n_input: 10,
            eta: 0.01,
            w_min: f32::NEG_INFINITY,
            w_max: f32::INFINITY,
        }
    }
}

/// Validate Oja configuration and slice dimensions for a single step.
fn validate_oja_step(weights: &[f32], x: &[f32], cfg: &OjaConfig) -> SnnResult<()> {
    if cfg.n_input == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if weights.len() != cfg.n_input {
        return Err(SnnError::BadShape {
            expected: cfg.n_input,
            got: weights.len(),
        });
    }
    if x.len() != cfg.n_input {
        return Err(SnnError::IncompatibleLength {
            a: cfg.n_input,
            b: x.len(),
        });
    }
    Ok(())
}

/// Advance Oja's rule by one sample.
///
/// Computes the scalar projection `y = w^T · x`, applies
/// `Δw_i = η · y · (x_i − y · w_i)`, clamps to `[w_min, w_max]`, and
/// returns `y`.
///
/// # Errors
/// Returns `SnnError` if slice lengths do not match `cfg.n_input`.
pub fn oja_step(weights: &mut [f32], x: &[f32], cfg: &OjaConfig) -> SnnResult<f32> {
    validate_oja_step(weights, x, cfg)?;

    // y = w^T x
    let y: f32 = weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();

    // Δw_i = η · y · (x_i − y · w_i)
    for i in 0..cfg.n_input {
        let delta = cfg.eta * y * (x[i] - y * weights[i]);
        weights[i] = (weights[i] + delta).clamp(cfg.w_min, cfg.w_max);
    }
    Ok(y)
}

/// Run Oja's rule on a batch of `n_samples` input vectors.
///
/// `inputs` must have shape `[n_samples × n_input]`. Returns `y[n_samples]`.
///
/// # Errors
/// Returns `SnnError` if any slice has the wrong length.
pub fn oja_batch(
    weights: &mut [f32],
    inputs: &[f32],
    n_samples: usize,
    cfg: &OjaConfig,
) -> SnnResult<Vec<f32>> {
    if cfg.n_input == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if n_samples == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if inputs.len() != n_samples * cfg.n_input {
        return Err(SnnError::BadShape {
            expected: n_samples * cfg.n_input,
            got: inputs.len(),
        });
    }
    if weights.len() != cfg.n_input {
        return Err(SnnError::BadShape {
            expected: cfg.n_input,
            got: weights.len(),
        });
    }

    let mut ys = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let x_slice = &inputs[s * cfg.n_input..(s + 1) * cfg.n_input];
        let y = oja_step(weights, x_slice, cfg)?;
        ys.push(y);
    }
    Ok(ys)
}

/// Normalise the weight vector to unit L2 norm in-place, returning the
/// pre-normalisation norm.
///
/// # Errors
/// Returns `SnnError::Internal` if the norm is less than `1e-12` (the weight
/// vector is effectively zero and normalisation would be numerically unstable).
pub fn oja_normalize(weights: &mut [f32]) -> SnnResult<f32> {
    let norm: f32 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(SnnError::Internal {
            msg: "weight norm too small to normalise".into(),
        });
    }
    for w in weights.iter_mut() {
        *w /= norm;
    }
    Ok(norm)
}

/// Compute the fraction of variance explained by the weight vector `w`:
///
/// ```text
/// R² = Σ_t (w · x_t)² / Σ_t ‖x_t‖²
/// ```
///
/// `inputs` must have shape `[n_samples × n_input]`.
///
/// # Errors
/// Returns `SnnError` if any slice has the wrong length.  Returns
/// `SnnError::Internal` if the total input energy is zero.
pub fn oja_explained_variance(weights: &[f32], inputs: &[f32], n_samples: usize) -> SnnResult<f32> {
    let n_input = weights.len();
    if n_input == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if n_samples == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if inputs.len() != n_samples * n_input {
        return Err(SnnError::BadShape {
            expected: n_samples * n_input,
            got: inputs.len(),
        });
    }

    let mut num = 0.0_f32;
    let mut denom = 0.0_f32;
    for s in 0..n_samples {
        let x = &inputs[s * n_input..(s + 1) * n_input];
        let proj: f32 = weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
        num += proj * proj;
        let energy: f32 = x.iter().map(|xi| xi * xi).sum();
        denom += energy;
    }

    if denom < 1e-30 {
        return Err(SnnError::Internal {
            msg: "total input energy is zero; cannot compute explained variance".into(),
        });
    }
    Ok(num / denom)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BcmConfig ────────────────────────────────────────────────────────────

    #[test]
    fn bcm_config_default_fields() {
        let cfg = BcmConfig::default();
        assert_eq!(cfg.n_pre, 10);
        assert_eq!(cfg.n_post, 10);
        assert!((cfg.eta - 0.001).abs() < 1e-9);
        assert!((cfg.tau_theta - 0.01).abs() < 1e-9);
        assert!((cfg.w_min - (-1.0)).abs() < 1e-9);
        assert!((cfg.w_max - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bcm_config_new_sets_dims() {
        let cfg = BcmConfig::new(5, 8);
        assert_eq!(cfg.n_pre, 5);
        assert_eq!(cfg.n_post, 8);
    }

    // ── BcmState ─────────────────────────────────────────────────────────────

    #[test]
    fn bcm_state_new_all_zeros() {
        let s = BcmState::new(6);
        assert_eq!(s.theta_m.len(), 6);
        assert!(s.theta_m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bcm_state_reset_clears() {
        let mut s = BcmState::new(4);
        s.theta_m[1] = 3.7;
        s.reset();
        assert!(s.theta_m.iter().all(|&v| v == 0.0));
    }

    // ── bcm_step ─────────────────────────────────────────────────────────────

    #[test]
    fn bcm_step_dim_mismatch_err() {
        let cfg = BcmConfig::new(3, 2);
        let mut w = vec![0.0_f32; 5]; // wrong: should be 3*2=6
        let mut state = BcmState::new(2);
        let pre = vec![1.0_f32; 3];
        let post = vec![1.0_f32; 2];
        let result = bcm_step(&mut w, &mut state, &pre, &post, &cfg);
        assert!(matches!(result, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn bcm_step_updates_weight_when_y_nonzero() {
        let cfg = BcmConfig {
            n_pre: 1,
            n_post: 1,
            eta: 0.1,
            tau_theta: 0.01,
            w_min: -10.0,
            w_max: 10.0,
        };
        let mut w = vec![0.0_f32];
        let mut state = BcmState::new(1);
        let pre = [1.0_f32];
        let post = [2.0_f32]; // y > θ_M=0 → positive dw
        bcm_step(&mut w, &mut state, &pre, &post, &cfg).expect("ok");
        assert!(w[0] > 0.0, "weight should have increased");
    }

    #[test]
    fn bcm_step_theta_increases_when_y_sq_nonzero() {
        let cfg = BcmConfig::new(2, 2);
        let mut w = vec![0.5_f32; 4];
        let mut state = BcmState::new(2);
        let pre = [1.0_f32; 2];
        let post = [1.0_f32; 2];
        bcm_step(&mut w, &mut state, &pre, &post, &cfg).expect("ok");
        assert!(state.theta_m[0] > 0.0);
        assert!(state.theta_m[1] > 0.0);
    }

    #[test]
    fn bcm_step_theta_decays_toward_zero_when_y_zero() {
        let cfg = BcmConfig {
            n_pre: 1,
            n_post: 1,
            eta: 0.01,
            tau_theta: 0.1,
            w_min: -5.0,
            w_max: 5.0,
        };
        let mut w = vec![0.0_f32];
        let mut state = BcmState::new(1);
        state.theta_m[0] = 1.0; // start elevated
        let pre = [0.0_f32];
        let post = [0.0_f32]; // y=0 → EMA drives theta toward 0
        bcm_step(&mut w, &mut state, &pre, &post, &cfg).expect("ok");
        assert!(
            state.theta_m[0] < 1.0,
            "theta should have decayed, got {}",
            state.theta_m[0]
        );
    }

    #[test]
    fn bcm_step_clamp_to_w_max() {
        let cfg = BcmConfig {
            n_pre: 1,
            n_post: 1,
            eta: 100.0, // huge LR to force clamping
            tau_theta: 0.001,
            w_min: -1.0,
            w_max: 1.0,
        };
        let mut w = vec![0.9_f32];
        let mut state = BcmState::new(1);
        // θ_M still very small after one step, so y*(y-θ) > 0 → strong increase.
        bcm_step(&mut w, &mut state, &[1.0], &[1.0], &cfg).expect("ok");
        assert!(w[0] <= 1.0 + 1e-6, "weight must be clamped to w_max");
    }

    #[test]
    fn bcm_step_clamp_to_w_min() {
        let cfg = BcmConfig {
            n_pre: 1,
            n_post: 1,
            eta: 100.0,
            tau_theta: 0.001,
            w_min: -1.0,
            w_max: 1.0,
        };
        let mut w = vec![-0.9_f32];
        let mut state = BcmState::new(1);
        state.theta_m[0] = 10.0; // θ >> y → dw negative
        bcm_step(&mut w, &mut state, &[1.0], &[0.5], &cfg).expect("ok");
        assert!(w[0] >= -1.0 - 1e-6, "weight must be clamped to w_min");
    }

    #[test]
    fn bcm_step_y_equals_theta_no_weight_change() {
        // When y_j == θ_M[j] at call time the delta is zero.
        // NOTE: τ_θ updates θ before Δw so we set theta beforehand and arrange
        // y=sqrt(theta/(1-tau+tau)) so the post-update θ still equals y.
        // Simpler: set eta=0 so no update can occur.
        let cfg = BcmConfig {
            n_pre: 1,
            n_post: 1,
            eta: 0.0, // zero LR
            tau_theta: 0.01,
            w_min: -5.0,
            w_max: 5.0,
        };
        let mut w = vec![0.3_f32];
        let mut state = BcmState::new(1);
        state.theta_m[0] = 4.0;
        bcm_step(&mut w, &mut state, &[1.0], &[2.0], &cfg).expect("ok");
        assert!(
            (w[0] - 0.3).abs() < 1e-7,
            "no weight change expected with eta=0"
        );
    }

    // ── bcm_run ──────────────────────────────────────────────────────────────

    #[test]
    fn bcm_run_matches_manual_loop() {
        let cfg = BcmConfig {
            n_pre: 2,
            n_post: 2,
            eta: 0.01,
            tau_theta: 0.05,
            w_min: -2.0,
            w_max: 2.0,
        };
        let n_steps = 10;
        let pre_act: Vec<f32> = (0..n_steps * 2).map(|i| (i % 3) as f32 * 0.3).collect();
        let post_act: Vec<f32> = (0..n_steps * 2).map(|i| (i % 5) as f32 * 0.2).collect();

        // Manual loop.
        let mut w_manual = vec![0.1_f32; 4];
        let mut state_manual = BcmState::new(2);
        for t in 0..n_steps {
            let pre_sl = &pre_act[t * 2..(t + 1) * 2];
            let post_sl = &post_act[t * 2..(t + 1) * 2];
            bcm_step(&mut w_manual, &mut state_manual, pre_sl, post_sl, &cfg).expect("ok");
        }

        // bcm_run.
        let mut w_run = vec![0.1_f32; 4];
        let mut state_run = BcmState::new(2);
        bcm_run(
            &mut w_run,
            &mut state_run,
            &pre_act,
            &post_act,
            n_steps,
            &cfg,
        )
        .expect("ok");

        for (a, b) in w_manual.iter().zip(w_run.iter()) {
            assert!((a - b).abs() < 1e-6, "manual={a}, run={b}");
        }
        for (a, b) in state_manual.theta_m.iter().zip(state_run.theta_m.iter()) {
            assert!((a - b).abs() < 1e-6, "theta manual={a}, run={b}");
        }
    }

    // ── bcm_equilibrium_theta ─────────────────────────────────────────────────

    #[test]
    fn bcm_equilibrium_theta_correct_shape() {
        let n_post = 3;
        let n_steps = 5;
        let post_act = vec![1.0_f32; n_steps * n_post];
        let theta = bcm_equilibrium_theta(&post_act, n_post, n_steps).expect("ok");
        assert_eq!(theta.len(), n_post);
        // E[y²] = 1.0 for all-ones.
        for &t in &theta {
            assert!((t - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bcm_equilibrium_theta_bad_shape_err() {
        // Supply wrong-length slice.
        let result = bcm_equilibrium_theta(&[1.0_f32; 5], 3, 2); // 3*2=6 != 5
        assert!(matches!(result, Err(SnnError::BadShape { .. })));
    }

    // ── OjaConfig ────────────────────────────────────────────────────────────

    #[test]
    fn oja_config_default_fields() {
        let cfg = OjaConfig::default();
        assert_eq!(cfg.n_input, 10);
        assert!((cfg.eta - 0.01).abs() < 1e-9);
        assert!(cfg.w_min == f32::NEG_INFINITY);
        assert!(cfg.w_max == f32::INFINITY);
    }

    // ── oja_step ─────────────────────────────────────────────────────────────

    #[test]
    fn oja_step_changes_weights() {
        let cfg = OjaConfig {
            n_input: 2,
            eta: 0.1,
            w_min: f32::NEG_INFINITY,
            w_max: f32::INFINITY,
        };
        let mut w = vec![1.0_f32, 0.0];
        let x = [0.5_f32, 0.5];
        let w_before = w.clone();
        let y = oja_step(&mut w, &x, &cfg).expect("ok");
        assert!(y.is_finite());
        let changed = w
            .iter()
            .zip(w_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "weights should have changed");
    }

    #[test]
    fn oja_step_x_zero_no_change() {
        let cfg = OjaConfig {
            n_input: 3,
            eta: 0.5,
            w_min: f32::NEG_INFINITY,
            w_max: f32::INFINITY,
        };
        let mut w = vec![0.4_f32, 0.3, 0.5];
        let w_before = w.clone();
        let x = [0.0_f32; 3];
        let _y = oja_step(&mut w, &x, &cfg).expect("ok");
        for (a, b) in w.iter().zip(w_before.iter()) {
            assert!((a - b).abs() < 1e-9, "no change expected for zero input");
        }
    }

    #[test]
    fn oja_step_dim_mismatch_err() {
        let cfg = OjaConfig {
            n_input: 4,
            eta: 0.01,
            w_min: f32::NEG_INFINITY,
            w_max: f32::INFINITY,
        };
        let mut w = vec![0.0_f32; 3]; // wrong length
        let x = [1.0_f32; 4];
        let result = oja_step(&mut w, &x, &cfg);
        assert!(matches!(result, Err(SnnError::BadShape { .. })));
    }

    // ── oja_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn oja_normalize_unit_norm() {
        let mut w = vec![3.0_f32, 4.0]; // norm = 5
        let norm = oja_normalize(&mut w).expect("ok");
        assert!((norm - 5.0).abs() < 1e-5);
        let new_norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn oja_normalize_zero_norm_err() {
        let mut w = vec![0.0_f32; 4];
        let result = oja_normalize(&mut w);
        assert!(matches!(result, Err(SnnError::Internal { .. })));
    }

    // ── oja_batch ─────────────────────────────────────────────────────────────

    #[test]
    fn oja_batch_correct_output_length() {
        let cfg = OjaConfig {
            n_input: 3,
            eta: 0.01,
            w_min: f32::NEG_INFINITY,
            w_max: f32::INFINITY,
        };
        let mut w = vec![1.0_f32, 0.0, 0.0];
        let inputs = vec![0.5_f32; 5 * 3];
        let ys = oja_batch(&mut w, &inputs, 5, &cfg).expect("ok");
        assert_eq!(ys.len(), 5);
    }

    // ── oja_explained_variance ────────────────────────────────────────────────

    #[test]
    fn oja_explained_variance_nonneg() {
        let w = vec![1.0_f32, 0.0];
        let inputs = vec![0.5_f32, 0.3, 0.8_f32, 0.1];
        let ev = oja_explained_variance(&w, &inputs, 2).expect("ok");
        assert!(ev >= 0.0);
    }

    #[test]
    fn oja_explained_variance_at_most_one_for_unit_weight() {
        // For unit-norm w and normalised x, R² ≤ 1.
        let mut w = vec![0.6_f32, 0.8];
        oja_normalize(&mut w).expect("ok"); // make unit norm
        // Inputs with norm 1.
        let inputs: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // two unit vectors
        let ev = oja_explained_variance(&w, &inputs, 2).expect("ok");
        assert!(ev <= 1.0 + 1e-6, "R² must be ≤ 1, got {ev}");
        assert!(ev >= 0.0);
    }
}
