//! RWKV-5 (Eagle) time-mixing layer: multi-head WKV with group normalization.
//!
//! # Theory (Peng et al. 2023/2024 — RWKV-5 Eagle)
//!
//! RWKV-5 extends RWKV-4 by partitioning the model dimension `d_model` into
//! `n_heads` independent heads of size `d_head = d_model / n_heads`.  Each head
//! runs its own numerically-stable WKV recurrence (identical to RWKV-4) and its
//! output is independently normalised with a per-head group-norm before being
//! concatenated and projected.
//!
//! ## New features over RWKV-4
//!
//! 1. **Multi-head WKV**: the `d_model` channels are split into `n_heads` heads,
//!    each of size `d_head = d_model / n_heads`.  WKV is computed per-head.
//! 2. **Per-head GroupNorm**: after WKV each head's `d_head`-vector is layer-
//!    normalised independently using shared `gn_gamma` / `gn_beta` parameters.
//! 3. **Gate vector G**: an additional sigmoid-activated gate (like receptance but
//!    applied after WKV and after GroupNorm).
//! 4. **Separate token-shift per projection**: `r`, `k`, `v`, and `g` each have
//!    their own element-wise mix coefficients (`mix_r/k/v/g`).
//!
//! ## Per-step algorithm
//!
//! ```text
//! x_r = mix_r ⊙ x_t + (1 − mix_r) ⊙ x_prev      (token-shift per projection)
//! x_k = mix_k ⊙ x_t + (1 − mix_k) ⊙ x_prev
//! x_v = mix_v ⊙ x_t + (1 − mix_v) ⊙ x_prev
//! x_g = mix_g ⊙ x_t + (1 − mix_g) ⊙ x_prev
//!
//! r = sigmoid(W_r @ x_r)          receptance
//! k = W_k @ x_k                   key
//! v = W_v @ x_v                   value
//! g = sigmoid(W_g @ x_g)          gate
//!
//! For each head h in 0..n_heads:
//!   Slice [h*d_head..(h+1)*d_head] of r, k, v, w, u.
//!   Run numerically-stable WKV per channel c in the head:
//!     q  = max(state.p[c], u[c] + k[c])
//!     wkv_c = (exp(state.p[c] − q)*state.a[c] + exp(u[c]+k[c]−q)*v[c])
//!           / (exp(state.p[c] − q)*state.b[c] + exp(u[c]+k[c]−q))
//!     q2 = max(state.p[c] − w[c], k[c])
//!     new_a[c] = exp(state.p[c]−w[c]−q2)*state.a[c] + exp(k[c]−q2)*v[c]
//!     new_b[c] = exp(state.p[c]−w[c]−q2)*state.b[c] + exp(k[c]−q2)
//!     new_p[c] = q2
//!     wkv_head[c] = r[c] * wkv_c      (receptance gate)
//!   Apply per-head GroupNorm to wkv_head:
//!     μ = mean(wkv_head)
//!     σ² = variance(wkv_head)
//!     wkv_head_norm = (wkv_head − μ) / sqrt(σ² + 1e-5) * gn_gamma_h + gn_beta_h
//!
//! output = W_o @ (wkv_all ⊙ g)        (gate then project)
//! ```

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::rwkv::time_mixing::sigmoid;

// ─── Rwkv5WkvState ───────────────────────────────────────────────────────────

/// Per-head numerically stable WKV recurrent state for RWKV-5.
///
/// Holds the running numerator (`a`), denominator (`b`), and log-space maximum
/// (`p`) vectors for one attention head of size `d_head`.
#[derive(Debug, Clone)]
pub struct Rwkv5WkvState {
    /// Running numerator accumulator per head-channel.  Length `d_head`.
    pub a: Vec<f32>,
    /// Running denominator accumulator per head-channel.  Length `d_head`.
    pub b: Vec<f32>,
    /// Running log-space maximum per head-channel.  Length `d_head`.
    /// Initialised to `f32::NEG_INFINITY` (empty-history identity).
    pub p: Vec<f32>,
}

impl Rwkv5WkvState {
    /// Create a fresh (zero-numerator / zero-denominator / -∞ log-max) WKV state
    /// for one head of size `d_head`.
    #[must_use]
    pub fn new(d_head: usize) -> Self {
        Self {
            a: vec![0.0_f32; d_head],
            b: vec![0.0_f32; d_head],
            p: vec![f32::NEG_INFINITY; d_head],
        }
    }
}

// ─── Rwkv5TimeMixWeights ─────────────────────────────────────────────────────

/// Learned parameters for an RWKV-5 (Eagle) time-mixing layer.
#[derive(Debug, Clone)]
pub struct Rwkv5TimeMixWeights {
    /// Per-channel time decay `w_c > 0` applied as `exp(−w_c)`.  Length `d_model`.
    pub w: Vec<f32>,
    /// Per-channel current-token bonus `u_c`.  Length `d_model`.
    pub u: Vec<f32>,
    /// Token-shift blend for receptance projection.  Length `d_model`.
    pub mix_r: Vec<f32>,
    /// Token-shift blend for key projection.  Length `d_model`.
    pub mix_k: Vec<f32>,
    /// Token-shift blend for value projection.  Length `d_model`.
    pub mix_v: Vec<f32>,
    /// Token-shift blend for gate projection.  Length `d_model`.
    pub mix_g: Vec<f32>,
    /// Receptance projection `W_r`.  Row-major `[d_model × d_model]`.
    pub w_r: Vec<f32>,
    /// Key projection `W_k`.  Row-major `[d_model × d_model]`.
    pub w_k: Vec<f32>,
    /// Value projection `W_v`.  Row-major `[d_model × d_model]`.
    pub w_v: Vec<f32>,
    /// Gate projection `W_g`.  Row-major `[d_model × d_model]`.
    pub w_g: Vec<f32>,
    /// Output projection `W_o`.  Row-major `[d_model × d_model]`.
    pub w_o: Vec<f32>,
    /// GroupNorm scale `γ`.  Length `d_model` (d_head values per head, repeated n_heads times).
    pub gn_gamma: Vec<f32>,
    /// GroupNorm bias `β`.  Length `d_model`.
    pub gn_beta: Vec<f32>,
}

impl Rwkv5TimeMixWeights {
    /// Initialize weights.
    ///
    /// * Linear projection matrices (`w_r/k/v/g/o`): Xavier uniform.
    /// * `w`: ones (moderate decay placeholder; all positive for stable decay).
    /// * `u`: ones.
    /// * `mix_r/k/v/g`: ones (full current-token weight by default).
    /// * `gn_gamma`: ones; `gn_beta`: zeros.
    ///
    /// # Errors
    ///
    /// [`MambaError::InvalidModelDim`] if `d_model == 0`.
    /// [`MambaError::HeadDimMismatch`] if `n_heads == 0` or `d_model % n_heads != 0`.
    pub fn new(d_model: usize, n_heads: usize, rng: &mut LcgRng) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(MambaError::HeadDimMismatch { n_heads, d_model });
        }
        let dd = d_model * d_model;
        let xavier = |fan_in: usize, fan_out: usize, rng: &mut LcgRng, n: usize| -> Vec<f32> {
            let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            (0..n)
                .map(|_| rng.next_f32() * 2.0 * scale - scale)
                .collect()
        };
        Ok(Self {
            w: vec![1.0_f32; d_model],
            u: vec![1.0_f32; d_model],
            mix_r: vec![1.0_f32; d_model],
            mix_k: vec![1.0_f32; d_model],
            mix_v: vec![1.0_f32; d_model],
            mix_g: vec![1.0_f32; d_model],
            w_r: xavier(d_model, d_model, rng, dd),
            w_k: xavier(d_model, d_model, rng, dd),
            w_v: xavier(d_model, d_model, rng, dd),
            w_g: xavier(d_model, d_model, rng, dd),
            w_o: xavier(d_model, d_model, rng, dd),
            gn_gamma: vec![1.0_f32; d_model],
            gn_beta: vec![0.0_f32; d_model],
        })
    }
}

// ─── Rwkv5TimeMixLayer ────────────────────────────────────────────────────────

/// RWKV-5 (Eagle) time-mixing layer.
///
/// Partitions `d_model` into `n_heads` heads of size `d_head = d_model / n_heads`,
/// runs numerically-stable WKV on each head independently, applies per-head
/// group normalisation, and gates the combined output before the output projection.
#[derive(Debug, Clone)]
pub struct Rwkv5TimeMixLayer {
    /// Total model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per head (`d_model / n_heads`).
    pub d_head: usize,
}

impl Rwkv5TimeMixLayer {
    /// Construct a new layer, validating that `d_model` is divisible by `n_heads`.
    ///
    /// # Errors
    ///
    /// [`MambaError::InvalidModelDim`] if `d_model == 0`.
    /// [`MambaError::HeadDimMismatch`] if `n_heads == 0` or `d_model % n_heads != 0`.
    pub fn new(d_model: usize, n_heads: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(MambaError::HeadDimMismatch { n_heads, d_model });
        }
        Ok(Self {
            d_model,
            n_heads,
            d_head: d_model / n_heads,
        })
    }

    /// Process one time step.
    ///
    /// # Arguments
    ///
    /// * `x_t`    — current token embedding, length `d_model`.
    /// * `x_prev` — previous token embedding (for token-shift), length `d_model`.
    ///   Pass zeros for the very first step.
    /// * `states` — per-head WKV states, length `n_heads` (each of size `d_head`).
    /// * `weights` — model parameters.
    ///
    /// # Returns
    ///
    /// `(output, new_states)` where `output` has length `d_model`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if any slice length is wrong.
    pub fn step(
        &self,
        x_t: &[f32],
        x_prev: &[f32],
        states: &[Rwkv5WkvState],
        weights: &Rwkv5TimeMixWeights,
    ) -> MambaResult<(Vec<f32>, Vec<Rwkv5WkvState>)> {
        let d = self.d_model;
        let h = self.n_heads;
        let dh = self.d_head;

        // Validate inputs.
        if x_t.len() != d {
            return Err(MambaError::DimensionMismatch {
                expected: d,
                got: x_t.len(),
            });
        }
        if x_prev.len() != d {
            return Err(MambaError::DimensionMismatch {
                expected: d,
                got: x_prev.len(),
            });
        }
        if states.len() != h {
            return Err(MambaError::DimensionMismatch {
                expected: h,
                got: states.len(),
            });
        }

        // ── Token-shift blending ──────────────────────────────────────────────
        let token_shift = |mix: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; d];
            for c in 0..d {
                out[c] = mix[c] * x_t[c] + (1.0 - mix[c]) * x_prev[c];
            }
            out
        };

        let x_r = token_shift(&weights.mix_r);
        let x_k = token_shift(&weights.mix_k);
        let x_v = token_shift(&weights.mix_v);
        let x_g = token_shift(&weights.mix_g);

        // ── Linear projections ────────────────────────────────────────────────
        let matvec = |w: &[f32], x: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; d];
            for o in 0..d {
                let mut acc = 0.0_f32;
                for i in 0..d {
                    acc += w[o * d + i] * x[i];
                }
                out[o] = acc;
            }
            out
        };

        let r_raw = matvec(&weights.w_r, &x_r);
        let k_vec = matvec(&weights.w_k, &x_k);
        let v_vec = matvec(&weights.w_v, &x_v);
        let g_raw = matvec(&weights.w_g, &x_g);

        // Apply sigmoid to r and g.
        let r_vec: Vec<f32> = r_raw.iter().map(|&v| sigmoid(v)).collect();
        let g_vec: Vec<f32> = g_raw.iter().map(|&v| sigmoid(v)).collect();

        // ── Multi-head WKV + GroupNorm ─────────────────────────────────────────
        let mut wkv_all = vec![0.0_f32; d];
        let mut new_states: Vec<Rwkv5WkvState> = Vec::with_capacity(h);

        for (head, state) in states.iter().enumerate().take(h) {
            let start = head * dh;

            let mut new_a = vec![0.0_f32; dh];
            let mut new_b = vec![0.0_f32; dh];
            let mut new_p = vec![0.0_f32; dh];
            let mut wkv_head = vec![0.0_f32; dh];

            for c in 0..dh {
                let gc = start + c; // global channel index

                let kk = k_vec[gc];
                let vv = v_vec[gc];
                let ww = weights.w[gc]; // positive decay value
                let uu = weights.u[gc];

                let p_c = state.p[c];
                let a_c = state.a[c];
                let b_c = state.b[c];

                // Normaliser for current WKV output.
                let q = p_c.max(uu + kk);
                let num = (p_c - q).exp() * a_c + (uu + kk - q).exp() * vv;
                let den = (p_c - q).exp() * b_c + (uu + kk - q).exp();
                let wkv_c = if den.abs() > 1e-30 { num / den } else { 0.0 };

                // State update: decay with w > 0 means log-space: p - w.
                let q2 = (p_c - ww).max(kk);
                new_a[c] = (p_c - ww - q2).exp() * a_c + (kk - q2).exp() * vv;
                new_b[c] = (p_c - ww - q2).exp() * b_c + (kk - q2).exp();
                new_p[c] = q2;

                // Apply receptance gate.
                wkv_head[c] = r_vec[gc] * wkv_c;
            }

            new_states.push(Rwkv5WkvState {
                a: new_a,
                b: new_b,
                p: new_p,
            });

            // Per-head GroupNorm.
            let mean = wkv_head.iter().sum::<f32>() / dh as f32;
            let var = wkv_head
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / dh as f32;
            let inv_std = (var + 1e-5_f32).sqrt().recip();

            for (c, &wkv_c_val) in wkv_head.iter().enumerate().take(dh) {
                let gc = start + c;
                let x_hat = (wkv_c_val - mean) * inv_std;
                wkv_all[gc] = weights.gn_gamma[gc] * x_hat + weights.gn_beta[gc];
            }
        }

        // ── Output gate + projection ──────────────────────────────────────────
        // gated = wkv_all ⊙ g
        let mut gated = vec![0.0_f32; d];
        for c in 0..d {
            gated[c] = wkv_all[c] * g_vec[c];
        }

        let output = matvec(&weights.w_o, &gated);

        Ok((output, new_states))
    }

    /// Forward pass over a full sequence `x: [L × d_model]`.
    ///
    /// Initialises WKV states to zero and `x_prev` to zero at `t=0`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `x.len() != seq_len * d_model`.
    /// [`MambaError::InvalidSeqLen`] if `seq_len == 0`.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        weights: &Rwkv5TimeMixWeights,
    ) -> MambaResult<Vec<f32>> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        let d = self.d_model;
        let expected = seq_len * d;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut states: Vec<Rwkv5WkvState> = (0..self.n_heads)
            .map(|_| Rwkv5WkvState::new(self.d_head))
            .collect();
        let mut x_prev = vec![0.0_f32; d];
        let mut output = vec![0.0_f32; expected];

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            let (out_t, new_states) = self.step(x_t, &x_prev, &states, weights)?;
            output[t * d..(t + 1) * d].copy_from_slice(&out_t);
            states = new_states;
            x_prev.copy_from_slice(x_t);
        }

        Ok(output)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    // ── Rwkv5WkvState ─────────────────────────────────────────────────────────

    #[test]
    fn rwkv5_state_new() {
        let d_head = 4;
        let s = Rwkv5WkvState::new(d_head);
        assert_eq!(s.a.len(), d_head);
        assert_eq!(s.b.len(), d_head);
        assert_eq!(s.p.len(), d_head);
        assert!(s.a.iter().all(|&v| v == 0.0), "a should be zeros");
        assert!(s.b.iter().all(|&v| v == 0.0), "b should be zeros");
        assert!(
            s.p.iter().all(|&v| v == f32::NEG_INFINITY),
            "p should be NEG_INFINITY"
        );
    }

    // ── Rwkv5TimeMixWeights ───────────────────────────────────────────────────

    #[test]
    fn rwkv5_weights_new_valid() {
        let mut rng = make_rng();
        let w = Rwkv5TimeMixWeights::new(8, 2, &mut rng).expect("d_model=8, n_heads=2 valid");
        assert_eq!(w.w.len(), 8);
        assert_eq!(w.u.len(), 8);
        assert_eq!(w.w_r.len(), 64);
        assert_eq!(w.gn_gamma.len(), 8);
        assert_eq!(w.gn_beta.len(), 8);
    }

    #[test]
    fn rwkv5_head_dim_mismatch() {
        let mut rng = make_rng();
        // 7 is not divisible by 3
        let err = Rwkv5TimeMixWeights::new(7, 3, &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::HeadDimMismatch { .. }));
    }

    // ── Rwkv5TimeMixLayer ─────────────────────────────────────────────────────

    #[test]
    fn rwkv5_d_model_zero_error() {
        let err = Rwkv5TimeMixLayer::new(0, 1).expect_err("d_model=0 must fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    #[test]
    fn rwkv5_n_heads_zero_error() {
        let err = Rwkv5TimeMixLayer::new(8, 0).expect_err("n_heads=0 must fail");
        assert!(matches!(err, MambaError::HeadDimMismatch { .. }));
    }

    #[test]
    fn rwkv5_step_output_shape() {
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);
        let x_prev = randn(&mut rng, d);
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();
        let (out, _) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step ok");
        assert_eq!(out.len(), d, "output length must be d_model");
    }

    #[test]
    fn rwkv5_step_states_updated() {
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);
        let x_prev = vec![0.0_f32; d];
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();
        let (_, new_states) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step ok");
        // After one step with non-zero input, at least one state value should change.
        let any_p_changed = new_states
            .iter()
            .any(|s| s.p.iter().any(|&v| v != f32::NEG_INFINITY));
        assert!(any_p_changed, "states.p must be updated after step");
    }

    #[test]
    fn rwkv5_step_finite_output() {
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);
        let x_prev = randn(&mut rng, d);
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();
        let (out, _) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}]={v} is not finite");
        }
    }

    #[test]
    fn rwkv5_forward_output_shape() {
        let d = 8;
        let nh = 2;
        let l = 6;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x = randn(&mut rng, l * d);
        let out = layer.forward(&x, l, &weights).expect("forward ok");
        assert_eq!(out.len(), l * d, "output must be [L × d_model]");
    }

    #[test]
    fn rwkv5_forward_seq1_matches_step() {
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);

        // forward with L=1
        let out_fwd = layer.forward(&x_t, 1, &weights).expect("forward ok");

        // step with zero x_prev and fresh states
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();
        let x_prev = vec![0.0_f32; d];
        let (out_step, _) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step ok");

        for (i, (&a, &b)) in out_fwd.iter().zip(out_step.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "forward/step mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rwkv5_forward_causal_direction() {
        let d = 8;
        let nh = 2;
        let l = 4;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x = randn(&mut rng, l * d);

        // reverse the sequence
        let mut x_rev = vec![0.0_f32; l * d];
        for t in 0..l {
            x_rev[t * d..(t + 1) * d].copy_from_slice(&x[(l - 1 - t) * d..(l - t) * d]);
        }

        let out_fwd = layer.forward(&x, l, &weights).expect("forward ok");
        let out_rev = layer.forward(&x_rev, l, &weights).expect("forward rev ok");

        // At least one element should differ (causality makes reversal non-trivial).
        let any_diff = out_fwd
            .iter()
            .zip(out_rev.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "reversed input should produce different output");
    }

    #[test]
    fn rwkv5_forward_zero_input() {
        let d = 8;
        let nh = 2;
        let l = 4;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x = vec![0.0_f32; l * d];
        let out = layer.forward(&x, l, &weights).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "zero input must produce finite output"
        );
    }

    #[test]
    fn rwkv5_forward_constant_input() {
        let d = 8;
        let nh = 2;
        let l = 8;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        // All tokens identical
        let one_token: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();
        let x: Vec<f32> = one_token.iter().copied().cycle().take(l * d).collect();
        let out = layer.forward(&x, l, &weights).expect("forward ok");
        // Output must remain finite for constant input.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "constant input must produce finite output"
        );
    }

    #[test]
    fn rwkv5_token_shift_effect() {
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        // Use 0.5 mix so x_prev contributes equally to x_t.
        let mut weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        weights.mix_r = vec![0.5_f32; d];
        weights.mix_k = vec![0.5_f32; d];
        weights.mix_v = vec![0.5_f32; d];
        weights.mix_g = vec![0.5_f32; d];
        let x_t = randn(&mut rng, d);
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();

        let x_prev_a = vec![0.0_f32; d];
        let x_prev_b = randn(&mut rng, d);

        let (out_a, _) = layer
            .step(&x_t, &x_prev_a, &states, &weights)
            .expect("step a");
        let (out_b, _) = layer
            .step(&x_t, &x_prev_b, &states, &weights)
            .expect("step b");

        // Different x_prev should produce different output when mix != 1.0.
        let any_diff = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "different x_prev must produce different output");
    }

    #[test]
    fn rwkv5_gating_gate_active() {
        // With a non-trivial gate g, the output should be scaled differently
        // than with a zero gate (g = sigmoid(-inf) ≈ 0).
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);
        let x_prev = vec![0.0_f32; d];
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();

        let (out, _) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step ok");

        // Create weights with nearly-zero w_g (gate ≈ 0.5 by sigmoid(0)).
        let mut weights_zero_g = weights.clone();
        weights_zero_g.w_g = vec![0.0_f32; d * d]; // sigmoid(0) = 0.5 for all
        let (out_zero_g, _) = layer
            .step(&x_t, &x_prev, &states, &weights_zero_g)
            .expect("step zero_g ok");

        // Outputs should differ because w_g changed.
        let any_diff = out
            .iter()
            .zip(out_zero_g.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "gate changes must affect output magnitude");
    }

    #[test]
    fn rwkv5_group_norm_applied() {
        // Test that group norm parameters (gn_gamma/gn_beta) affect the output.
        let d = 8;
        let nh = 2;
        let dh = d / nh;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        let x_t = randn(&mut rng, d);
        let x_prev = vec![0.0_f32; d];
        let states: Vec<Rwkv5WkvState> = (0..nh).map(|_| Rwkv5WkvState::new(dh)).collect();

        let (out_a, _) = layer
            .step(&x_t, &x_prev, &states, &weights)
            .expect("step a");

        let mut weights_b = weights.clone();
        weights_b.gn_gamma = vec![2.0_f32; d]; // scale ×2
        let (out_b, _) = layer
            .step(&x_t, &x_prev, &states, &weights_b)
            .expect("step b");

        let any_diff = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "gn_gamma changes must affect output");
    }

    #[test]
    fn rwkv5_forward_wrong_shape_error() {
        let d = 8;
        let nh = 2;
        let l = 4;
        let mut rng = make_rng();
        let layer = Rwkv5TimeMixLayer::new(d, nh).expect("valid");
        let weights = Rwkv5TimeMixWeights::new(d, nh, &mut rng).expect("valid");
        // Wrong length: l*d + 1
        let x = vec![0.0_f32; l * d + 1];
        let err = layer.forward(&x, l, &weights).expect_err("must fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    #[test]
    fn rwkv5_multi_head_output_consistent() {
        // Same d_model, different n_heads: both should produce finite output of
        // the same shape.
        let d = 8;
        let l = 4;
        let mut rng_a = LcgRng::new(1);
        let mut rng_b = LcgRng::new(1);

        let layer_1h = Rwkv5TimeMixLayer::new(d, 1).expect("valid 1 head");
        let weights_1h = Rwkv5TimeMixWeights::new(d, 1, &mut rng_a).expect("valid");

        let layer_2h = Rwkv5TimeMixLayer::new(d, 2).expect("valid 2 heads");
        let weights_2h = Rwkv5TimeMixWeights::new(d, 2, &mut rng_b).expect("valid");

        let mut rng = LcgRng::new(42);
        let x = randn(&mut rng, l * d);

        let out_1h = layer_1h.forward(&x, l, &weights_1h).expect("1h forward");
        let out_2h = layer_2h.forward(&x, l, &weights_2h).expect("2h forward");

        assert_eq!(out_1h.len(), l * d);
        assert_eq!(out_2h.len(), l * d);
        assert!(out_1h.iter().all(|v| v.is_finite()));
        assert!(out_2h.iter().all(|v| v.is_finite()));
    }
}
