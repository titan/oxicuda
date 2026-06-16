//! Liquid-S4 — Liquid Structural State-Space Model (Hasani et al. 2022).
//!
//! # Background
//!
//! Liquid-S4 ("Liquid Structural State-Space Models", Hasani, Lechner, Wang,
//! Chahine, Amini & Rus, 2022) augments the diagonal S4 / S4D state-space
//! layer with **input-dependent ("liquid") dynamics** and a learnable
//! per-neuron time-constant `τ`.  Where a plain diagonal SSM discretizes a
//! continuous-time system once with a fixed step `Δ`, Liquid-S4 lets the
//! effective step depend on the input, so the contraction rate of each state
//! mode *adapts* to the signal — exactly the property of a Liquid-Time-Constant
//! (LTC) network carried into the structured SSM setting.
//!
//! ## Faithful CPU form (input-modulated `Δ`)
//!
//! The paper presents two complementary mechanisms: an explicit second-order
//! liquid kernel `K_liquid` capturing `u·u` input correlations, and an
//! input-dependent time-constant.  This module implements the latter — the
//! **input-modulated-`Δ`** recurrence, which is the simplest faithful form:
//!
//! ```text
//! Δ_eff(t, d) = Δ · softplus( τ_d + w_d · u_t )          (positive, adaptive)
//! Ā(t, d, n)  = exp( Δ_eff(t, d) · A[d, n] )             (per-step ZOH)
//! B̄(t, d, n)  = (Ā − 1) / A[d, n] · B[d, n]              (ZOH, L'Hôpital A→0)
//! h(t, d, n)  = Ā(t, d, n) · h(t-1, d, n) + B̄(t, d, n) · u_t[d]
//! y_t[d]      = Σ_n C[d, n] · h(t, d, n)
//! ```
//!
//! Each of the `D` channels is an independent SISO diagonal SSM with `N`
//! stable real modes `A[d, n] < 0`.  The continuous parameters `A, B, C` are
//! input-*independent* (as in S4); the **liquidity** lives entirely in the
//! input-dependent step `Δ_eff`, driven by the per-channel time-constant `τ_d`
//! and a liquid projection `w_d` of the current input vector `u_t`.
//!
//! ## Reduction to plain S4
//!
//! When the liquid coupling is disabled (`w ≡ 0`, or [`LiquidS4Config::liquid`]
//! `= false`) the step collapses to `Δ_eff(d) = Δ · softplus(τ_d)`, a constant
//! per-channel time-step, and the recurrence is exactly a diagonal S4 / S4D
//! layer with a learnable per-neuron `Δ`.  This reduction is asserted by the
//! unit tests.
//!
//! The per-step ZOH discretization reuses [`crate::ssm::discretize::discretize`]
//! and the numerically-stable [`crate::mamba::selective_scan::softplus`].

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::mamba::selective_scan::softplus;
use crate::ssm::discretize::{Discretization, discretize};

/// Floor applied to the effective step `Δ_eff` so the per-step discretization
/// always receives a strictly-positive `Δ` (guards the `softplus → 0` tail).
const MIN_DELTA: f32 = 1e-6;

// ─── LiquidS4Config ──────────────────────────────────────────────────────────

/// Configuration for a [`LiquidS4Layer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidS4Config {
    /// Number of channels `D` (independent SISO diagonal SSMs).
    pub d_model: usize,
    /// Diagonal state order `N` (real modes per channel).
    pub d_state: usize,
    /// Base discretization step `Δ > 0` (the liquid factor multiplies this).
    pub delta: f32,
    /// Initial value of every per-channel time-constant `τ_d`.
    pub tau_init: f32,
    /// Whether the input-dependent liquid coupling `w · u_t` is active.
    ///
    /// When `false` the layer reduces to a plain diagonal S4 layer with a
    /// constant per-channel step `Δ · softplus(τ_d)`.
    pub liquid: bool,
}

impl LiquidS4Config {
    /// Create a new configuration (`Δ = 0.1`, `τ_init = 1.0`, liquid enabled).
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`] — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`] — if `d_state == 0`.
    pub fn new(d_model: usize, d_state: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        Ok(Self {
            d_model,
            d_state,
            delta: 0.1_f32,
            tau_init: 1.0_f32,
            liquid: true,
        })
    }

    /// Override the base discretization step `Δ`.
    ///
    /// # Errors
    ///
    /// [`MambaError::NonPositiveDelta`] if `delta ≤ 0`.
    pub fn with_delta(mut self, delta: f32) -> MambaResult<Self> {
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }
        self.delta = delta;
        Ok(self)
    }

    /// Override the initial per-channel time-constant `τ_init`.
    #[must_use]
    pub fn with_tau_init(mut self, tau_init: f32) -> Self {
        self.tau_init = tau_init;
        self
    }

    /// Enable or disable the input-dependent liquid coupling.
    #[must_use]
    pub fn with_liquid(mut self, liquid: bool) -> Self {
        self.liquid = liquid;
        self
    }
}

// ─── LiquidS4Layer ───────────────────────────────────────────────────────────

/// A Liquid-S4 sequence-to-sequence layer (CPU reference).
///
/// Input `u` is laid out row-major `[L × D]` (`u[t·D + d]`); the output `y` has
/// the same shape.  State is initialised to zero at `t = 0`.
#[derive(Debug, Clone)]
pub struct LiquidS4Layer {
    config: LiquidS4Config,
    /// Continuous diagonal `A[d, n] < 0`, row-major `[D × N]`.
    a_diag: Vec<f32>,
    /// Continuous input vector `B[d, n]`, row-major `[D × N]`.
    b: Vec<f32>,
    /// Output mixing `C[d, n]`, row-major `[D × N]`.
    c: Vec<f32>,
    /// Per-channel time-constant `τ_d`, length `D` (raw; `softplus` applied).
    tau: Vec<f32>,
    /// Liquid input projection `w[d, j]`, row-major `[D × D]`.
    ///
    /// Row `d` maps the full input vector `u_t` to the scalar modulation
    /// `w_d · u_t = Σ_j w[d, j] · u_t[j]`.
    w_liquid: Vec<f32>,
}

impl LiquidS4Layer {
    /// Construct a Liquid-S4 layer with paper-faithful initialization:
    ///
    /// * `A[d, n] = −(n + 1)` — stable HiPPO-LegS-style real poles.
    /// * `B[d, n] = 1` — the S4D convention (the `C·B̄` product carries scale).
    /// * `C[d, n] ~ N(0, 1)` — random output mixing.
    /// * `τ_d = τ_init` — uniform initial time-constant.
    /// * `w[d, j]` — small Xavier-uniform liquid projection (scale `0.1`).
    ///
    /// # Errors
    ///
    /// Propagates [`LiquidS4Config`] validation (`d_model` / `d_state`).
    pub fn new(config: LiquidS4Config, rng: &mut LcgRng) -> MambaResult<Self> {
        let d = config.d_model;
        let n = config.d_state;

        let mut a_diag = Vec::with_capacity(d * n);
        let mut b = Vec::with_capacity(d * n);
        let mut c = Vec::with_capacity(d * n);
        for _ in 0..d {
            for k in 0..n {
                a_diag.push(-((k + 1) as f32));
                b.push(1.0_f32);
                let (cr, _) = rng.next_normal_pair();
                c.push(cr);
            }
        }

        let tau = vec![config.tau_init; d];

        // Small symmetric liquid projection in [-scale, scale].
        let scale = 0.1_f32;
        let w_liquid: Vec<f32> = (0..d * d)
            .map(|_| rng.next_f32() * 2.0 * scale - scale)
            .collect();

        Ok(Self {
            config,
            a_diag,
            b,
            c,
            tau,
            w_liquid,
        })
    }

    /// Return a reference to the configuration.
    #[inline]
    pub fn config(&self) -> &LiquidS4Config {
        &self.config
    }

    /// Read-only view of the continuous diagonal `A` (length `D·N`).
    #[inline]
    pub fn a_diag(&self) -> &[f32] {
        &self.a_diag
    }

    /// Read-only view of the continuous input vector `B` (length `D·N`).
    #[inline]
    pub fn b_weights(&self) -> &[f32] {
        &self.b
    }

    /// Read-only view of the output mixing `C` (length `D·N`).
    #[inline]
    pub fn c_weights(&self) -> &[f32] {
        &self.c
    }

    /// Read-only view of the per-channel time-constants `τ` (length `D`).
    #[inline]
    pub fn tau(&self) -> &[f32] {
        &self.tau
    }

    /// Mutable view of the per-channel time-constants `τ` (length `D`).
    ///
    /// Useful for inspecting how `τ` modulates the adaptive step.
    #[inline]
    pub fn tau_mut(&mut self) -> &mut [f32] {
        &mut self.tau
    }

    /// Enable or disable the input-dependent liquid coupling in place.
    #[inline]
    pub fn set_liquid(&mut self, liquid: bool) {
        self.config.liquid = liquid;
    }

    /// The input-modulated effective step `Δ_eff(t, d)` for channel `ch`.
    ///
    /// `Δ_eff = Δ · softplus(τ_d + w_d · u_t)`, floored at [`MIN_DELTA`].
    /// When the liquid coupling is disabled the `w_d · u_t` term is dropped.
    #[inline]
    fn effective_delta(&self, ch: usize, u_t: &[f32]) -> f32 {
        let mut pre = self.tau[ch];
        if self.config.liquid {
            let d = self.config.d_model;
            let w_row = &self.w_liquid[ch * d..(ch + 1) * d];
            let dot: f32 = w_row.iter().zip(u_t.iter()).map(|(&w, &u)| w * u).sum();
            pre += dot;
        }
        (self.config.delta * softplus(pre)).max(MIN_DELTA)
    }

    /// Forward pass over a full sequence.
    ///
    /// # Arguments
    ///
    /// * `u`       — input, row-major `[seq_len × D]`, length `seq_len · D`.
    /// * `seq_len` — number of time steps.
    ///
    /// # Returns
    ///
    /// Output `y`, row-major `[seq_len × D]`, length `seq_len · D`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`]     — if `seq_len == 0`.
    /// * [`MambaError::DimensionMismatch`] — if `u.len() != seq_len · D`.
    /// * [`MambaError::NonPositiveDelta`]  — propagated from discretization
    ///   (cannot occur in practice because `Δ_eff` is floored positive).
    pub fn forward(&self, u: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        let d = self.config.d_model;
        let n = self.config.d_state;
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        let expected = seq_len * d;
        if u.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: u.len(),
            });
        }

        let mut y = vec![0.0_f32; expected];
        // Hidden state h: [D × N], initialised to zero.
        let mut h = vec![0.0_f32; d * n];

        for t in 0..seq_len {
            let u_t = &u[t * d..(t + 1) * d];
            for ch in 0..d {
                // Input-dependent step → per-step ZOH discretization (reused).
                let delta_eff = self.effective_delta(ch, u_t);
                let a_row = &self.a_diag[ch * n..(ch + 1) * n];
                let b_row = &self.b[ch * n..(ch + 1) * n];
                let (a_bar, b_bar) = discretize(a_row, b_row, delta_eff, Discretization::Zoh)?;

                let u_val = u_t[ch];
                let c_row = &self.c[ch * n..(ch + 1) * n];
                let h_ch = &mut h[ch * n..(ch + 1) * n];
                let mut y_val = 0.0_f32;
                for (j, ((&ab, &bb), &cv)) in
                    a_bar.iter().zip(b_bar.iter()).zip(c_row.iter()).enumerate()
                {
                    let h_new = ab * h_ch[j] + bb * u_val;
                    h_ch[j] = h_new;
                    y_val += cv * h_new;
                }
                y[t * d + ch] = y_val;
            }
        }

        Ok(y)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::discretize::{Discretization, discretize};

    const D_MODEL: usize = 4;
    const D_STATE: usize = 8;
    const SEQ_LEN: usize = 16;

    fn rng() -> LcgRng {
        LcgRng::new(2024)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn tiny_layer(liquid: bool) -> LiquidS4Layer {
        let cfg = LiquidS4Config::new(D_MODEL, D_STATE)
            .expect("config")
            .with_liquid(liquid);
        LiquidS4Layer::new(cfg, &mut rng()).expect("layer")
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn config_rejects_zero_dims() {
        assert!(matches!(
            LiquidS4Config::new(0, 4),
            Err(MambaError::InvalidModelDim(0))
        ));
        assert!(matches!(
            LiquidS4Config::new(4, 0),
            Err(MambaError::InvalidSsmOrder(0))
        ));
    }

    #[test]
    fn config_with_delta_rejects_nonpositive() {
        let cfg = LiquidS4Config::new(2, 4).expect("cfg");
        assert!(matches!(
            cfg.with_delta(0.0),
            Err(MambaError::NonPositiveDelta(_))
        ));
        assert!(matches!(
            cfg.with_delta(-1.0),
            Err(MambaError::NonPositiveDelta(_))
        ));
    }

    #[test]
    fn config_builders() {
        let cfg = LiquidS4Config::new(2, 4)
            .expect("cfg")
            .with_delta(0.05)
            .expect("delta")
            .with_tau_init(2.0)
            .with_liquid(false);
        assert!((cfg.delta - 0.05).abs() < 1e-7);
        assert!((cfg.tau_init - 2.0).abs() < 1e-7);
        assert!(!cfg.liquid);
    }

    // ── Initialization ────────────────────────────────────────────────────────

    #[test]
    fn new_layer_shapes_and_stable_poles() {
        let layer = tiny_layer(true);
        assert_eq!(layer.a_diag().len(), D_MODEL * D_STATE);
        assert_eq!(layer.b_weights().len(), D_MODEL * D_STATE);
        assert_eq!(layer.c_weights().len(), D_MODEL * D_STATE);
        assert_eq!(layer.tau().len(), D_MODEL);
        // All continuous poles strictly negative (stable / contractive).
        for &a in layer.a_diag() {
            assert!(a < 0.0, "A pole {a} must be < 0 for stability");
        }
    }

    // ── Shape ─────────────────────────────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let layer = tiny_layer(true);
        let mut r = rng();
        let u = randn(&mut r, SEQ_LEN * D_MODEL);
        let y = layer.forward(&u, SEQ_LEN).expect("forward");
        assert_eq!(y.len(), SEQ_LEN * D_MODEL, "output shape [L·D]");
    }

    #[test]
    fn forward_rejects_bad_length() {
        let layer = tiny_layer(true);
        let u = vec![0.0_f32; SEQ_LEN * D_MODEL + 1];
        assert!(matches!(
            layer.forward(&u, SEQ_LEN),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_rejects_zero_seq_len() {
        let layer = tiny_layer(true);
        assert!(matches!(
            layer.forward(&[], 0),
            Err(MambaError::InvalidSeqLen(0))
        ));
    }

    // ── Stability ─────────────────────────────────────────────────────────────

    #[test]
    fn forward_stable_bounded_finite() {
        // Stable A + bounded input → bounded, finite outputs (no NaN/Inf).
        let layer = tiny_layer(true);
        let mut r = rng();
        let u = randn(&mut r, SEQ_LEN * D_MODEL); // ~N(0,1), bounded
        let y = layer.forward(&u, SEQ_LEN).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} must be finite");
            assert!(v.abs() < 1e3, "y[{i}]={v} unexpectedly large");
        }
    }

    #[test]
    fn forward_long_sequence_finite() {
        // Longer roll-out must remain bounded for stable poles.
        let layer = tiny_layer(true);
        let l = 256_usize;
        let u = vec![0.5_f32; l * D_MODEL]; // bounded constant drive
        let y = layer.forward(&u, l).expect("forward");
        assert_eq!(y.len(), l * D_MODEL);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} must stay finite at L={l}");
        }
    }

    // ── Zero input → zero output ──────────────────────────────────────────────

    #[test]
    fn forward_zero_input_zero_output() {
        let layer = tiny_layer(true);
        let u = vec![0.0_f32; SEQ_LEN * D_MODEL];
        let y = layer.forward(&u, SEQ_LEN).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < 1e-9, "y[{i}]={v} should be zero for zero input");
        }
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn forward_deterministic_under_fixed_seed() {
        let cfg = LiquidS4Config::new(D_MODEL, D_STATE).expect("cfg");
        let a = LiquidS4Layer::new(cfg, &mut LcgRng::new(7)).expect("a");
        let b = LiquidS4Layer::new(cfg, &mut LcgRng::new(7)).expect("b");
        // Identical seeds ⇒ identical weights ⇒ identical outputs.
        assert_eq!(a.a_diag(), b.a_diag());
        assert_eq!(a.c_weights(), b.c_weights());
        let mut r = LcgRng::new(99);
        let u = randn(&mut r, SEQ_LEN * D_MODEL);
        let ya = a.forward(&u, SEQ_LEN).expect("fa");
        let yb = b.forward(&u, SEQ_LEN).expect("fb");
        for (i, (&va, &vb)) in ya.iter().zip(yb.iter()).enumerate() {
            assert!(
                (va - vb).abs() < 1e-7,
                "non-deterministic at {i}: {va} vs {vb}"
            );
        }
    }

    // ── τ modulates dynamics ──────────────────────────────────────────────────

    #[test]
    fn tau_modulates_outputs() {
        // Same weights, different τ ⇒ different Δ_eff ⇒ different outputs.
        let base = tiny_layer(true);
        let mut small_tau = base.clone();
        let mut large_tau = base.clone();
        for t in small_tau.tau_mut() {
            *t = -1.0; // softplus(-1) ≈ 0.31 → small step
        }
        for t in large_tau.tau_mut() {
            *t = 3.0; // softplus(3) ≈ 3.05 → large step
        }
        let mut r = rng();
        let u = randn(&mut r, SEQ_LEN * D_MODEL);
        let y_small = small_tau.forward(&u, SEQ_LEN).expect("small");
        let y_large = large_tau.forward(&u, SEQ_LEN).expect("large");
        let max_diff = y_small
            .iter()
            .zip(y_large.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "changing τ must change the dynamics, max_diff={max_diff}"
        );
    }

    // ── Reduction to plain S4 when liquid disabled ────────────────────────────

    #[test]
    fn disabled_liquid_reduces_to_plain_s4() {
        // With liquid disabled, Δ_eff(d) = Δ·softplus(τ_d) is constant in time,
        // so the layer is a plain diagonal S4 recurrence — recomputed here
        // independently from the public weights and compared element-wise.
        let layer = tiny_layer(false);
        let mut r = rng();
        let u = randn(&mut r, SEQ_LEN * D_MODEL);
        let y = layer.forward(&u, SEQ_LEN).expect("forward");

        let d = D_MODEL;
        let n = D_STATE;
        let cfg = *layer.config();
        let a_diag = layer.a_diag();
        let b = layer.b_weights();
        let c = layer.c_weights();
        let tau = layer.tau();

        let mut y_ref = vec![0.0_f32; SEQ_LEN * d];
        let mut h = vec![0.0_f32; d * n];
        for t in 0..SEQ_LEN {
            for ch in 0..d {
                let delta_c = (cfg.delta * softplus(tau[ch])).max(MIN_DELTA);
                let a_row = &a_diag[ch * n..(ch + 1) * n];
                let b_row = &b[ch * n..(ch + 1) * n];
                let (a_bar, b_bar) =
                    discretize(a_row, b_row, delta_c, Discretization::Zoh).expect("disc");
                let u_val = u[t * d + ch];
                let mut acc = 0.0_f32;
                for j in 0..n {
                    let idx = ch * n + j;
                    let h_new = a_bar[j] * h[idx] + b_bar[j] * u_val;
                    h[idx] = h_new;
                    acc += c[idx] * h_new;
                }
                y_ref[t * d + ch] = acc;
            }
        }

        for (i, (&got, &want)) in y.iter().zip(y_ref.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "plain-S4 mismatch at {i}: got {got}, want {want}"
            );
        }
    }

    // ── Liquid coupling actually changes the output vs. its plain limit ───────

    #[test]
    fn liquid_on_differs_from_liquid_off() {
        // Same weights/seed; toggling the liquid coupling must change outputs
        // (the input-dependent step is genuinely active).
        let mut on = tiny_layer(true);
        on.set_liquid(true);
        let mut off = on.clone();
        off.set_liquid(false);
        // Drive with a non-trivial input so w·u ≠ 0.
        let mut r = LcgRng::new(123);
        let u = randn(&mut r, SEQ_LEN * D_MODEL);
        let y_on = on.forward(&u, SEQ_LEN).expect("on");
        let y_off = off.forward(&u, SEQ_LEN).expect("off");
        let max_diff = y_on
            .iter()
            .zip(y_off.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "liquid coupling should alter dynamics, max_diff={max_diff}"
        );
    }
}
