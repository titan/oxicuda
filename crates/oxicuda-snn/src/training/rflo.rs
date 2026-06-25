#![allow(clippy::needless_range_loop)]
//! Random Feedback Local Online learning (RFLO).
//!
//! Implements the RFLO rule of Murray (2019), "Local online learning in
//! recurrent networks with random feedback" (*eLife* 8:e43299). RFLO is an
//! online, fully-local approximation to backpropagation-through-time for a
//! continuous-time recurrent neural network. It avoids both the unrolled
//! storage of BPTT and the weight-transport problem of exact gradient descent.
//!
//! # Dynamics
//!
//! The recurrent unit `i` integrates its inputs through a leaky rate variable
//! `u_i` with firing rate `r_i = φ(u_i)` (here `φ = tanh`, `φ'(u) = 1 − tanh²(u)`).
//! Each plastic synapse `(i ← j)` carries a scalar **eligibility trace** `p_ij`
//! that low-pass-filters the product of the post-synaptic activation derivative
//! and the pre-synaptic activity:
//!
//! ```text
//! p_ij(t) = (1 − 1/τ) · p_ij(t−1)  +  (1/τ) · φ'(u_i(t)) · r_j(t)
//! ```
//!
//! The trace is the local quantity that, in exact BPTT, would be summed over the
//! past with the recurrent Jacobian; RFLO replaces that exact sum by the single
//! exponential filter above (`τ` is the network time constant, in units of `dt`).
//!
//! # Weight update with random feedback
//!
//! The output error `e(t) = y(t) − ŷ(t) ∈ ℝ^{n_out}` is projected back onto the
//! hidden layer through a **fixed random feedback matrix** `B ∈ ℝ^{n_out × n_hidden}`
//! (drawn once at construction and never trained — the central idea inherited
//! from feedback alignment, Lillicrap et al. 2016). The local learning signal
//! for hidden unit `i` is `[Bᵀ e(t)]_i = Σ_k B[k,i] · e_k(t)`, and the recurrent
//! weight update is the product of that signal with the eligibility trace:
//!
//! ```text
//! ΔW_ij = −lr · [Bᵀ e(t)]_i · p_ij(t)
//! ```
//!
//! The read-out weights `W_out ∈ ℝ^{n_out × n_hidden}` are trained by ordinary
//! local delta-rule descent on the same error, `ΔW_out[k,i] = −lr · e_k · r_i`.
//!
//! Because `B` is frozen and every update uses only locally-available quantities
//! (`e`, `p`, `r`), the rule is online and biologically plausible.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// RFLO hyper-parameters.
#[derive(Debug, Clone, Copy)]
pub struct RfloConfig {
    /// Number of recurrent (hidden) units.
    pub n_hidden: usize,
    /// Number of read-out (output) units.
    pub n_out: usize,
    /// Network time constant `τ` (in units of `dt`), controlling the eligibility
    /// trace low-pass filter `(1 − 1/τ)`. Must satisfy `τ ≥ 1`.
    pub tau: f32,
    /// Base learning rate `lr` (> 0).
    pub lr: f32,
}

impl Default for RfloConfig {
    fn default() -> Self {
        Self {
            n_hidden: 1,
            n_out: 1,
            tau: 10.0,
            lr: 0.01,
        }
    }
}

impl RfloConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::BadDim`] when `n_hidden` or `n_out` is zero,
    /// [`SnnError::BadTau`] when `τ < 1` or non-finite, and
    /// [`SnnError::OutOfRange`] when `lr ≤ 0` or non-finite.
    pub fn validate(&self) -> SnnResult<()> {
        if self.n_hidden == 0 {
            return Err(SnnError::BadDim { got: self.n_hidden });
        }
        if self.n_out == 0 {
            return Err(SnnError::BadDim { got: self.n_out });
        }
        if !self.tau.is_finite() || self.tau < 1.0 {
            return Err(SnnError::BadTau { tau: self.tau });
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: self.lr,
            });
        }
        Ok(())
    }
}

/// Mutable RFLO state: the per-synapse eligibility traces for the recurrent
/// weight matrix.
#[derive(Debug, Clone)]
pub struct RfloState {
    /// Eligibility traces `p_ij`, row-major `[n_hidden × n_in]`. Row `i` holds
    /// the traces of the synapses onto hidden unit `i`.
    pub eligibility: Vec<f32>,
    /// Number of hidden (post-synaptic) units — the rows of `eligibility`.
    pub n_hidden: usize,
    /// Number of pre-synaptic inputs — the columns of `eligibility`.
    pub n_in: usize,
}

impl RfloState {
    /// Allocate an all-zero trace matrix of shape `[n_hidden × n_in]`.
    #[must_use]
    pub fn new(n_hidden: usize, n_in: usize) -> Self {
        Self {
            eligibility: vec![0.0_f32; n_hidden * n_in],
            n_hidden,
            n_in,
        }
    }

    /// Reset all eligibility traces to zero.
    pub fn reset(&mut self) {
        for x in self.eligibility.iter_mut() {
            *x = 0.0;
        }
    }
}

/// `tanh` activation derivative `φ'(u) = 1 − tanh²(u)`, the default `φ` used by
/// RFLO. Provided as a free helper so callers can compute `phi_prime` to feed
/// [`RfloLearner::update_eligibility`].
#[must_use]
#[inline]
pub fn tanh_prime(u: f32) -> f32 {
    let t = u.tanh();
    1.0 - t * t
}

/// An RFLO recurrent learner.
///
/// Owns the trainable recurrent weights `W_rec` (`[n_hidden × n_in]`, row-major),
/// the trainable read-out weights `W_out` (`[n_out × n_hidden]`, row-major) and
/// the **frozen** random feedback matrix `B` (`[n_out × n_hidden]`, row-major).
/// `B` is drawn once in [`RfloLearner::new`] and never modified by any method.
#[derive(Debug, Clone)]
pub struct RfloLearner {
    /// Number of pre-synaptic inputs to the recurrent layer.
    n_in: usize,
    /// Number of recurrent (hidden) units.
    n_hidden: usize,
    /// Number of read-out (output) units.
    n_out: usize,
    /// Eligibility-trace low-pass coefficient `1/τ`.
    inv_tau: f32,
    /// Recurrent weights `W_rec`, shape `[n_hidden × n_in]`, row-major.
    w_rec: Vec<f32>,
    /// Read-out weights `W_out`, shape `[n_out × n_hidden]`, row-major.
    w_out: Vec<f32>,
    /// Frozen random feedback matrix `B`, shape `[n_out × n_hidden]`, row-major.
    b_feedback: Vec<f32>,
    /// Eligibility-trace state, shape `[n_hidden × n_in]`.
    state: RfloState,
    /// Cached firing rates `r_i = φ(u_i)` of the hidden units from the most
    /// recent [`RfloLearner::update_eligibility`] call, length `n_hidden`.
    /// Used by the read-out delta-rule in [`RfloLearner::step`].
    last_hidden_rates: Vec<f32>,
}

impl RfloLearner {
    /// Construct an RFLO learner with random initial weights and a frozen random
    /// feedback matrix, all drawn from `rng`.
    ///
    /// `W_rec` and `W_out` are scaled by `1/√fan_in` (LeCun-style), and the
    /// frozen feedback `B` by `1/√n_hidden` to keep the projected error at unit
    /// scale. The feedback matrix is statistically independent of `W_out` — it
    /// is **not** its transpose.
    ///
    /// # Errors
    ///
    /// Returns the error from [`RfloConfig::validate`], or [`SnnError::BadDim`]
    /// when `n_in` is zero.
    pub fn new(n_in: usize, cfg: RfloConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        cfg.validate()?;
        if n_in == 0 {
            return Err(SnnError::BadDim { got: n_in });
        }
        let n_hidden = cfg.n_hidden;
        let n_out = cfg.n_out;

        let rec_scale = 1.0 / (n_in as f32).sqrt();
        let out_scale = 1.0 / (n_hidden as f32).sqrt();
        let b_scale = 1.0 / (n_hidden as f32).sqrt();

        let w_rec: Vec<f32> = (0..n_hidden * n_in)
            .map(|_| rng.next_normal() * rec_scale)
            .collect();
        let w_out: Vec<f32> = (0..n_out * n_hidden)
            .map(|_| rng.next_normal() * out_scale)
            .collect();
        let b_feedback: Vec<f32> = (0..n_out * n_hidden)
            .map(|_| rng.next_normal() * b_scale)
            .collect();

        Ok(Self {
            n_in,
            n_hidden,
            n_out,
            inv_tau: 1.0 / cfg.tau,
            w_rec,
            w_out,
            b_feedback,
            state: RfloState::new(n_hidden, n_in),
            last_hidden_rates: vec![0.0_f32; n_hidden],
        })
    }

    /// Number of pre-synaptic inputs.
    #[must_use]
    #[inline]
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Number of recurrent (hidden) units.
    #[must_use]
    #[inline]
    pub fn n_hidden(&self) -> usize {
        self.n_hidden
    }

    /// Number of read-out (output) units.
    #[must_use]
    #[inline]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Immutable view of the recurrent weights `W_rec` (`[n_hidden × n_in]`).
    #[must_use]
    #[inline]
    pub fn w_rec(&self) -> &[f32] {
        &self.w_rec
    }

    /// Immutable view of the read-out weights `W_out` (`[n_out × n_hidden]`).
    #[must_use]
    #[inline]
    pub fn w_out(&self) -> &[f32] {
        &self.w_out
    }

    /// Immutable view of the frozen feedback matrix `B` (`[n_out × n_hidden]`).
    #[must_use]
    #[inline]
    pub fn b_feedback(&self) -> &[f32] {
        &self.b_feedback
    }

    /// Immutable view of the eligibility-trace state.
    #[must_use]
    #[inline]
    pub fn state(&self) -> &RfloState {
        &self.state
    }

    /// Hidden firing rates `r_i` recorded by the most recent [`RfloLearner::step`]
    /// (all zero before the first step or after [`RfloLearner::reset_state`]),
    /// length `n_hidden`. Useful for diagnostics and for chaining the same rates
    /// into a subsequent read-out evaluation.
    #[must_use]
    #[inline]
    pub fn last_hidden_rates(&self) -> &[f32] {
        &self.last_hidden_rates
    }

    /// Read-out forward pass `y = W_out · r`, where `r` are the hidden firing
    /// rates (length `n_hidden`). Returns the output vector of length `n_out`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::IncompatibleLength`] when `hidden_rates.len() != n_hidden`.
    pub fn readout(&self, hidden_rates: &[f32]) -> SnnResult<Vec<f32>> {
        if hidden_rates.len() != self.n_hidden {
            return Err(SnnError::IncompatibleLength {
                a: self.n_hidden,
                b: hidden_rates.len(),
            });
        }
        let mut y = vec![0.0_f32; self.n_out];
        for k in 0..self.n_out {
            let row_off = k * self.n_hidden;
            let mut acc = 0.0_f32;
            for i in 0..self.n_hidden {
                acc += self.w_out[row_off + i] * hidden_rates[i];
            }
            y[k] = acc;
        }
        Ok(y)
    }

    /// Low-pass update of every recurrent-synapse eligibility trace for one
    /// timestep:
    ///
    /// ```text
    /// p_ij(t) = (1 − 1/τ) · p_ij(t−1) + (1/τ) · φ'(u_i) · r_j
    /// ```
    ///
    /// * `pre_activity` — pre-synaptic activity `r_j`, length `n_in`.
    /// * `phi_prime`    — post-synaptic activation derivative `φ'(u_i)`,
    ///   length `n_hidden`.
    ///
    /// The post-synaptic firing rates are *not* required for the trace, but the
    /// caller also supplies them implicitly: this method records `φ'`-paired
    /// rates only via `pre_activity`. The hidden firing rates used by the
    /// read-out delta-rule are taken from a subsequent [`RfloLearner::step`]
    /// call's `hidden_rates`; to keep the API minimal they are cached here when
    /// `phi_prime` is supplied as the activation derivative. See [`RfloLearner::step`].
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::IncompatibleLength`] when either input length is wrong.
    pub fn update_eligibility(&mut self, pre_activity: &[f32], phi_prime: &[f32]) -> SnnResult<()> {
        if pre_activity.len() != self.n_in {
            return Err(SnnError::IncompatibleLength {
                a: self.n_in,
                b: pre_activity.len(),
            });
        }
        if phi_prime.len() != self.n_hidden {
            return Err(SnnError::IncompatibleLength {
                a: self.n_hidden,
                b: phi_prime.len(),
            });
        }
        let decay = 1.0 - self.inv_tau;
        let gain = self.inv_tau;
        for i in 0..self.n_hidden {
            let row_off = i * self.n_in;
            let h_i = phi_prime[i];
            for j in 0..self.n_in {
                let p = &mut self.state.eligibility[row_off + j];
                *p = decay * *p + gain * h_i * pre_activity[j];
            }
        }
        Ok(())
    }

    /// Project the output error through the frozen feedback matrix `B`,
    /// `[Bᵀ e]_i = Σ_k B[k,i] · e_k`, returning a length-`n_hidden` vector.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::IncompatibleLength`] when `error_out.len() != n_out`.
    pub fn project_error(&self, error_out: &[f32]) -> SnnResult<Vec<f32>> {
        if error_out.len() != self.n_out {
            return Err(SnnError::IncompatibleLength {
                a: self.n_out,
                b: error_out.len(),
            });
        }
        let mut projected = vec![0.0_f32; self.n_hidden];
        for k in 0..self.n_out {
            let row_off = k * self.n_hidden;
            let e_k = error_out[k];
            for i in 0..self.n_hidden {
                projected[i] += self.b_feedback[row_off + i] * e_k;
            }
        }
        Ok(projected)
    }

    /// Apply one RFLO weight update from the output error.
    ///
    /// The recurrent weights are updated with the random-feedback-projected local
    /// learning signal times the eligibility trace:
    ///
    /// ```text
    /// ΔW_rec[i,j] = −lr · [Bᵀ e]_i · p_ij
    /// ```
    ///
    /// and the read-out weights by the local delta rule on the supplied
    /// `hidden_rates` `r_i`:
    ///
    /// ```text
    /// ΔW_out[k,i] = −lr · e_k · r_i
    /// ```
    ///
    /// `lr` overrides the configured base rate for this step (pass the config's
    /// `lr` to keep it unchanged). The frozen feedback matrix `B` is never
    /// modified.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::IncompatibleLength`] for wrong-length inputs and
    /// [`SnnError::OutOfRange`] when `lr ≤ 0` or non-finite.
    pub fn step(&mut self, error_out: &[f32], hidden_rates: &[f32], lr: f32) -> SnnResult<()> {
        if error_out.len() != self.n_out {
            return Err(SnnError::IncompatibleLength {
                a: self.n_out,
                b: error_out.len(),
            });
        }
        if hidden_rates.len() != self.n_hidden {
            return Err(SnnError::IncompatibleLength {
                a: self.n_hidden,
                b: hidden_rates.len(),
            });
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: lr,
            });
        }

        // Cache the rates for inspection / reuse.
        self.last_hidden_rates.copy_from_slice(hidden_rates);

        // Random-feedback-projected learning signal: l_i = [Bᵀ e]_i.
        let projected = self.project_error(error_out)?;

        // Recurrent update: ΔW_rec[i,j] = −lr · l_i · p_ij.
        for i in 0..self.n_hidden {
            let row_off = i * self.n_in;
            let l_i = projected[i];
            let step_i = -lr * l_i;
            for j in 0..self.n_in {
                self.w_rec[row_off + j] += step_i * self.state.eligibility[row_off + j];
            }
        }

        // Read-out update: ΔW_out[k,i] = −lr · e_k · r_i (exact local delta rule).
        for k in 0..self.n_out {
            let row_off = k * self.n_hidden;
            let step_k = -lr * error_out[k];
            for i in 0..self.n_hidden {
                self.w_out[row_off + i] += step_k * hidden_rates[i];
            }
        }
        Ok(())
    }

    /// Reset the eligibility-trace state (does not touch weights or `B`).
    pub fn reset_state(&mut self) {
        self.state.reset();
        for r in self.last_hidden_rates.iter_mut() {
            *r = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_hidden: usize, n_out: usize) -> RfloConfig {
        RfloConfig {
            n_hidden,
            n_out,
            tau: 10.0,
            lr: 0.05,
        }
    }

    // 1. Config validation: zero dims / bad tau / bad lr.
    #[test]
    fn config_validation() {
        assert!(matches!(
            RfloConfig {
                n_hidden: 0,
                ..cfg(1, 1)
            }
            .validate(),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            RfloConfig {
                n_out: 0,
                ..cfg(1, 1)
            }
            .validate(),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            RfloConfig {
                tau: 0.5,
                ..cfg(2, 2)
            }
            .validate(),
            Err(SnnError::BadTau { .. })
        ));
        assert!(matches!(
            RfloConfig {
                lr: -1.0,
                ..cfg(2, 2)
            }
            .validate(),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    // 2. Constructor shapes are correct.
    #[test]
    fn constructor_shapes() {
        let mut rng = LcgRng::new(1);
        let n_in = 4_usize;
        let learner = RfloLearner::new(n_in, cfg(3, 2), &mut rng).expect("new");
        assert_eq!(learner.w_rec().len(), 3 * n_in);
        assert_eq!(learner.w_out().len(), 2 * 3);
        assert_eq!(learner.b_feedback().len(), 2 * 3);
        assert_eq!(learner.state().eligibility.len(), 3 * n_in);
        assert_eq!(learner.n_in(), n_in);
        assert_eq!(learner.n_hidden(), 3);
        assert_eq!(learner.n_out(), 2);
    }

    // 3. Constructor rejects n_in == 0.
    #[test]
    fn constructor_rejects_zero_in() {
        let mut rng = LcgRng::new(2);
        assert!(matches!(
            RfloLearner::new(0, cfg(2, 2), &mut rng),
            Err(SnnError::BadDim { .. })
        ));
    }

    // 4. B is frozen across updates.
    #[test]
    fn feedback_matrix_is_frozen() {
        let mut rng = LcgRng::new(3);
        let mut learner = RfloLearner::new(3, cfg(4, 2), &mut rng).expect("new");
        let b_before = learner.b_feedback().to_vec();
        for _ in 0..20 {
            learner
                .update_eligibility(&[0.5, -0.3, 0.7], &[0.2, 0.4, 0.1, 0.9])
                .expect("elig");
            learner
                .step(&[0.3, -0.6], &[0.1, 0.2, 0.3, 0.4], 0.05)
                .expect("step");
        }
        let b_after = learner.b_feedback();
        for (a, b) in b_before.iter().zip(b_after.iter()) {
            assert_eq!(a, b, "feedback matrix B must remain frozen");
        }
    }

    // 5. Eligibility low-passes: decays toward zero with no input.
    #[test]
    fn eligibility_decays_without_input() {
        let mut rng = LcgRng::new(4);
        let mut learner = RfloLearner::new(2, cfg(2, 1), &mut rng).expect("new");
        // Seed traces with non-zero values.
        for x in learner.state.eligibility.iter_mut() {
            *x = 1.0;
        }
        // No pre-synaptic activity, no activation derivative → pure decay.
        for _ in 0..50 {
            learner
                .update_eligibility(&[0.0, 0.0], &[0.0, 0.0])
                .expect("elig");
        }
        for &p in &learner.state().eligibility {
            assert!(p < 0.01, "trace should decay toward zero, got {p}");
        }
    }

    // 6. Eligibility low-pass: one positive step raises a trace toward φ'·r.
    #[test]
    fn eligibility_low_pass_step_response() {
        let mut rng = LcgRng::new(5);
        // n_in = 1, n_hidden = 1 for an exact scalar check.
        let mut learner = RfloLearner::new(1, cfg(1, 1), &mut rng).expect("new");
        let pre = [2.0_f32];
        let phi = [0.5_f32];
        // Target steady state of the filter p_ss = φ'·r = 1.0.
        let target = phi[0] * pre[0];
        for _ in 0..500 {
            learner.update_eligibility(&pre, &phi).expect("elig");
        }
        let p = learner.state().eligibility[0];
        assert!(
            (p - target).abs() < 1e-3,
            "trace should converge to φ'·r={target}, got {p}"
        );
        assert!(p > 0.0, "trace should be positive");
    }

    // 7. project_error computes Bᵀ e correctly on a hand-set matrix.
    #[test]
    fn project_error_matches_manual() {
        let mut rng = LcgRng::new(6);
        let mut learner = RfloLearner::new(2, cfg(3, 2), &mut rng).expect("new");
        // Overwrite B with a known matrix: [n_out × n_hidden] = 2×3.
        // B = [[1, 2, 3], [4, 5, 6]]
        learner
            .b_feedback
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let e = [0.5_f32, -0.5];
        let proj = learner.project_error(&e).expect("project");
        // [Bᵀ e]_i = B[0,i]*e0 + B[1,i]*e1
        // i=0: 1*0.5 + 4*(-0.5) = 0.5 - 2.0 = -1.5
        // i=1: 2*0.5 + 5*(-0.5) = 1.0 - 2.5 = -1.5
        // i=2: 3*0.5 + 6*(-0.5) = 1.5 - 3.0 = -1.5
        assert!((proj[0] + 1.5).abs() < 1e-6);
        assert!((proj[1] + 1.5).abs() < 1e-6);
        assert!((proj[2] + 1.5).abs() < 1e-6);
    }

    // 8. step shape validation.
    #[test]
    fn step_shape_validation() {
        let mut rng = LcgRng::new(7);
        let mut learner = RfloLearner::new(2, cfg(3, 2), &mut rng).expect("new");
        // wrong error length
        assert!(matches!(
            learner.step(&[0.1], &[0.0, 0.0, 0.0], 0.01),
            Err(SnnError::IncompatibleLength { .. })
        ));
        // wrong hidden_rates length
        assert!(matches!(
            learner.step(&[0.1, 0.2], &[0.0, 0.0], 0.01),
            Err(SnnError::IncompatibleLength { .. })
        ));
        // bad lr
        assert!(matches!(
            learner.step(&[0.1, 0.2], &[0.0, 0.0, 0.0], 0.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    // 9. update_eligibility shape validation.
    #[test]
    fn update_eligibility_shape_validation() {
        let mut rng = LcgRng::new(8);
        let mut learner = RfloLearner::new(2, cfg(3, 1), &mut rng).expect("new");
        assert!(matches!(
            learner.update_eligibility(&[0.1], &[0.0, 0.0, 0.0]),
            Err(SnnError::IncompatibleLength { .. })
        ));
        assert!(matches!(
            learner.update_eligibility(&[0.1, 0.2], &[0.0, 0.0]),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    // 10. tanh_prime properties: 1 at 0, decays to 0, always in (0,1].
    #[test]
    fn tanh_prime_properties() {
        assert!((tanh_prime(0.0) - 1.0).abs() < 1e-6);
        assert!(tanh_prime(10.0) < 1e-3);
        for &u in &[-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let d = tanh_prime(u);
            assert!(d > 0.0 && d <= 1.0 + 1e-6, "φ'({u})={d}");
        }
    }

    // 11. A tiny linear-readout task reduces output MSE over iterations.
    //
    //   Hidden rates are a fixed feature vector r; the target is a fixed
    //   y* in R^{n_out}. With a frozen feature representation, the read-out
    //   delta rule alone (ΔW_out = −lr·e·rᵀ) is exact gradient descent on
    //   ½‖W_out r − y*‖² and must monotonically reduce the error. The
    //   recurrent eligibility branch runs alongside (and does not corrupt
    //   the read-out learning since it touches only W_rec).
    #[test]
    fn tiny_task_reduces_mse() {
        let mut rng = LcgRng::new(123);
        let n_in = 3_usize;
        let mut learner = RfloLearner::new(
            n_in,
            RfloConfig {
                n_hidden: 4,
                n_out: 2,
                tau: 8.0,
                lr: 0.05,
            },
            &mut rng,
        )
        .expect("new");

        // Fixed hidden feature vector and input.
        let hidden_rates = [0.7_f32, -0.3, 0.5, 0.2];
        let pre = [0.4_f32, 0.6, -0.2];
        let phi = [0.9_f32, 0.8, 0.7, 0.95];
        let target = [0.5_f32, -0.4];

        let mse = |learner: &RfloLearner| -> f32 {
            let y = learner.readout(&hidden_rates).expect("readout");
            let mut s = 0.0_f32;
            for k in 0..2 {
                let d = y[k] - target[k];
                s += d * d;
            }
            s / 2.0
        };

        let mse_before = mse(&learner);
        for _ in 0..200 {
            learner.update_eligibility(&pre, &phi).expect("elig");
            let y = learner.readout(&hidden_rates).expect("readout");
            // error e = y − y*
            let err = [y[0] - target[0], y[1] - target[1]];
            learner.step(&err, &hidden_rates, 0.05).expect("step");
        }
        let mse_after = mse(&learner);
        assert!(
            mse_after < mse_before * 0.1,
            "MSE should drop substantially: before={mse_before}, after={mse_after}"
        );
    }

    // 12. step with non-zero traces and projected error changes W_rec.
    #[test]
    fn step_changes_recurrent_weights() {
        let mut rng = LcgRng::new(9);
        let mut learner = RfloLearner::new(3, cfg(2, 2), &mut rng).expect("new");
        // Build non-zero eligibility traces.
        learner
            .update_eligibility(&[1.0, 1.0, 1.0], &[0.5, 0.5])
            .expect("elig");
        let w_before = learner.w_rec().to_vec();
        learner.step(&[1.0, -1.0], &[0.3, 0.4], 0.1).expect("step");
        let changed = learner
            .w_rec()
            .iter()
            .zip(w_before.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-9);
        assert!(changed, "W_rec should change after a non-trivial step");
    }

    // 13. reset_state zeroes traces but preserves weights and B.
    #[test]
    fn reset_state_clears_traces_only() {
        let mut rng = LcgRng::new(10);
        let mut learner = RfloLearner::new(2, cfg(2, 2), &mut rng).expect("new");
        learner
            .update_eligibility(&[1.0, 1.0], &[1.0, 1.0])
            .expect("elig");
        let w_rec = learner.w_rec().to_vec();
        let b = learner.b_feedback().to_vec();
        learner.reset_state();
        assert!(learner.state().eligibility.iter().all(|&x| x == 0.0));
        assert_eq!(learner.w_rec(), w_rec.as_slice());
        assert_eq!(learner.b_feedback(), b.as_slice());
    }

    // 14. readout shape validation.
    #[test]
    fn readout_shape_validation() {
        let mut rng = LcgRng::new(11);
        let learner = RfloLearner::new(2, cfg(3, 2), &mut rng).expect("new");
        assert!(matches!(
            learner.readout(&[0.0, 0.0]),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    // 15. last_hidden_rates records the rates from the most recent step and is
    //     zeroed by reset_state.
    #[test]
    fn last_hidden_rates_tracking() {
        let mut rng = LcgRng::new(12);
        let mut learner = RfloLearner::new(2, cfg(3, 2), &mut rng).expect("new");
        assert!(
            learner.last_hidden_rates().iter().all(|&x| x == 0.0),
            "rates should start at zero"
        );
        learner
            .update_eligibility(&[0.5, 0.5], &[0.1, 0.2, 0.3])
            .expect("elig");
        let rates = [0.3_f32, -0.4, 0.9];
        learner.step(&[0.2, -0.1], &rates, 0.05).expect("step");
        assert_eq!(learner.last_hidden_rates(), &rates);
        learner.reset_state();
        assert!(learner.last_hidden_rates().iter().all(|&x| x == 0.0));
    }
}
