//! RWKV time-mixing layer: the WKV (Weighted Key-Value) recurrent operation.
//!
//! # Theory (Peng et al., 2023 — RWKV-4)
//!
//! Time-mixing replaces self-attention with a linear recurrence over the sequence.
//! For each channel `c` and time step `t`, the WKV output is:
//!
//! ```text
//! wkv_{t,c} = (Σ_{i<t} exp((i-t+1)*w_c + k_{i,c}) * v_{i,c}
//!              + exp(u_c + k_{t,c}) * v_{t,c})
//!           / (Σ_{i<t} exp((i-t+1)*w_c + k_{i,c})
//!              + exp(u_c + k_{t,c}))
//! ```
//!
//! where `w_c > 0` is a per-channel learnable decay and `u_c` is a per-channel bonus
//! that amplifies the current token's contribution relative to the history.
//!
//! ## Numerically stable recurrence (running-max trick)
//!
//! ```text
//! Initialize: a=0, b=0, p=-inf
//! For t = 0..L:
//!   kk = k[t,c]
//!   q  = max(p, u_c + kk)       // normaliser for current output
//!   q2 = max(p + w_c, kk)       // normaliser for state update
//!   wkv = (exp(p - q)*a + exp(u_c + kk - q)*v[t,c])
//!       / (exp(p - q)*b + exp(u_c + kk - q))
//!   a  = exp(p + w_c - q2)*a + exp(kk - q2)*v[t,c]
//!   b  = exp(p + w_c - q2)*b + exp(kk - q2)
//!   p  = q2
//! ```
//!
//! Final gating: `output_{t,c} = sigmoid(r_{t,c}) * wkv_{t,c}`
//!
//! Token-shift blending mixes the current token embedding `x_t` with the
//! previous token `x_{t-1}` using per-channel learnable mix coefficients
//! `time_mix_r/k/v ∈ [0, 1]`:
//!
//! ```text
//! r_raw_t = W_r * (time_mix_r * x_t + (1 - time_mix_r) * x_{t-1})
//! k_t     = W_k * (time_mix_k * x_t + (1 - time_mix_k) * x_{t-1})
//! v_t     = W_v * (time_mix_v * x_t + (1 - time_mix_v) * x_{t-1})
//! r_t     = sigmoid(r_raw_t)
//! ```

use crate::error::{MambaError, MambaResult};

// ─── WkvState ────────────────────────────────────────────────────────────────

/// Per-channel numerically stable WKV recurrent state for incremental decoding.
///
/// Holds the running numerator (`a`), denominator (`b`), and log-space maximum
/// (`p`) for one channel. A full state for dimension `D` is `Vec<WkvState>`.
#[derive(Debug, Clone)]
pub struct WkvState {
    /// Running numerator accumulator (log-space normalised).
    pub a: f32,
    /// Running denominator accumulator (log-space normalised).
    pub b: f32,
    /// Running maximum in log-space (for numerical stability).
    pub p: f32,
}

impl WkvState {
    /// Create a fresh (zeroed) WKV state.
    ///
    /// `a = 0`, `b = 0`, `p = f32::NEG_INFINITY` — the empty-history identity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            p: f32::NEG_INFINITY,
        }
    }
}

impl Default for WkvState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── TimeMixingConfig ────────────────────────────────────────────────────────

/// Configuration for an RWKV time-mixing layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeMixingConfig {
    /// Model dimension `D` (number of channels).
    pub d_model: usize,
    /// Sequence length `L`.
    pub seq_len: usize,
}

impl TimeMixingConfig {
    /// Create a new configuration, validating dimensions.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] if `d_model == 0`
    /// - [`MambaError::InvalidSeqLen`] if `seq_len == 0`
    pub fn new(d_model: usize, seq_len: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        Ok(Self { d_model, seq_len })
    }
}

// ─── TimeMixingWeights ───────────────────────────────────────────────────────

/// Learned parameters for an RWKV time-mixing layer.
#[derive(Debug, Clone)]
pub struct TimeMixingWeights {
    /// Per-channel time decay `w`: `[D]`. Decay factor per step is `exp(-w_c)`;
    /// typically `w_c > 0` so the decay is sub-unity.
    pub w: Vec<f32>,
    /// Per-channel "bonus" for current token `u`: `[D]`.
    pub u: Vec<f32>,
    /// Receptance projection weight `W_r`: `[D, D]` (row-major: row=out, col=in).
    pub w_r: Vec<f32>,
    /// Key projection weight `W_k`: `[D, D]` (row-major).
    pub w_k: Vec<f32>,
    /// Value projection weight `W_v`: `[D, D]` (row-major).
    pub w_v: Vec<f32>,
    /// Output projection weight `W_o`: `[D, D]` (row-major).
    pub w_o: Vec<f32>,
    /// Pre-layer-norm weight `γ`: `[D]`.
    pub ln_weight: Vec<f32>,
    /// Pre-layer-norm bias `β`: `[D]`.
    pub ln_bias: Vec<f32>,
    /// Token-shift blend coefficient for receptance `μ_r`: `[D]`.
    pub time_mix_r: Vec<f32>,
    /// Token-shift blend coefficient for key `μ_k`: `[D]`.
    pub time_mix_k: Vec<f32>,
    /// Token-shift blend coefficient for value `μ_v`: `[D]`.
    pub time_mix_v: Vec<f32>,
}

impl TimeMixingWeights {
    /// Allocate all weight tensors as zeros (for the given config).
    #[must_use]
    pub fn zeros(config: &TimeMixingConfig) -> Self {
        let d = config.d_model;
        Self {
            w: vec![0.0; d],
            u: vec![0.0; d],
            w_r: vec![0.0; d * d],
            w_k: vec![0.0; d * d],
            w_v: vec![0.0; d * d],
            w_o: vec![0.0; d * d],
            ln_weight: vec![0.0; d],
            ln_bias: vec![0.0; d],
            time_mix_r: vec![0.0; d],
            time_mix_k: vec![0.0; d],
            time_mix_v: vec![0.0; d],
        }
    }

    /// Return default-initialised weights suitable for unit-testing:
    ///
    /// - `w = [2.0; D]` — moderate time decay
    /// - `u = [0.0; D]` — no current-token bonus
    /// - `ln_weight = [1.0; D]`, `ln_bias = [0.0; D]` — identity layer norm
    /// - `time_mix_* = [0.5; D]` — equal blend of current and previous token
    /// - Projection matrices = scaled identity (`1/D` on diagonal, 0 elsewhere)
    #[must_use]
    pub fn default_init(config: &TimeMixingConfig) -> Self {
        let d = config.d_model;
        let scale = if d > 0 { 1.0 / d as f32 } else { 1.0 };

        // Build a scaled identity matrix of shape [d, d].
        let scaled_identity = {
            let mut m = vec![0.0_f32; d * d];
            for i in 0..d {
                m[i * d + i] = scale;
            }
            m
        };

        Self {
            w: vec![2.0; d],
            u: vec![0.0; d],
            w_r: scaled_identity.clone(),
            w_k: scaled_identity.clone(),
            w_v: scaled_identity.clone(),
            w_o: scaled_identity,
            ln_weight: vec![1.0; d],
            ln_bias: vec![0.0; d],
            time_mix_r: vec![0.5; d],
            time_mix_k: vec![0.5; d],
            time_mix_v: vec![0.5; d],
        }
    }

    /// Randomly initialise weights with N(0, 1) samples scaled by `1/sqrt(D)`.
    ///
    /// `w` is always initialised to positive values `2.0 + |normal|` to ensure
    /// stable decay. `ln_weight` is initialised to 1.0, `ln_bias` to 0.0.
    pub fn random(config: &TimeMixingConfig, rng: &mut crate::handle::LcgRng) -> Self {
        let d = config.d_model;
        let scale = if d > 0 {
            (d as f32).sqrt().recip()
        } else {
            1.0
        };

        let sample_scaled = |rng: &mut crate::handle::LcgRng, n: usize| -> Vec<f32> {
            let mut buf = vec![0.0_f32; n];
            rng.fill_normal(&mut buf);
            buf.iter_mut().for_each(|v| *v *= scale);
            buf
        };

        // Decay w must be positive — use abs to ensure non-negativity then add offset.
        let mut w = vec![0.0_f32; d];
        rng.fill_normal(&mut w);
        w.iter_mut().for_each(|v| *v = (*v).abs() + 2.0);

        // Bonus u centred around zero.
        let mut u = vec![0.0_f32; d];
        rng.fill_normal(&mut u);

        Self {
            w,
            u,
            w_r: sample_scaled(rng, d * d),
            w_k: sample_scaled(rng, d * d),
            w_v: sample_scaled(rng, d * d),
            w_o: sample_scaled(rng, d * d),
            ln_weight: vec![1.0; d],
            ln_bias: vec![0.0; d],
            time_mix_r: vec![0.5; d],
            time_mix_k: vec![0.5; d],
            time_mix_v: vec![0.5; d],
        }
    }
}

// ─── TimeMixingLayer ─────────────────────────────────────────────────────────

/// RWKV time-mixing layer.
///
/// Applies pre-layer-norm, token-shift, WKV recurrence, and receptance gating.
pub struct TimeMixingLayer {
    config: TimeMixingConfig,
    weights: TimeMixingWeights,
}

impl TimeMixingLayer {
    /// Create a new time-mixing layer from config and weights.
    ///
    /// # Errors
    ///
    /// - [`MambaError::WeightShapeMismatch`] if any weight tensor has the wrong length.
    pub fn new(config: TimeMixingConfig, weights: TimeMixingWeights) -> MambaResult<Self> {
        let d = config.d_model;
        let dd = d * d;

        let check = |name: &'static str, got: usize, expected: usize| -> MambaResult<()> {
            if got != expected {
                return Err(MambaError::WeightShapeMismatch {
                    name,
                    expected: vec![expected],
                    got: vec![got],
                });
            }
            Ok(())
        };

        check("w", weights.w.len(), d)?;
        check("u", weights.u.len(), d)?;
        check("w_r", weights.w_r.len(), dd)?;
        check("w_k", weights.w_k.len(), dd)?;
        check("w_v", weights.w_v.len(), dd)?;
        check("w_o", weights.w_o.len(), dd)?;
        check("ln_weight", weights.ln_weight.len(), d)?;
        check("ln_bias", weights.ln_bias.len(), d)?;
        check("time_mix_r", weights.time_mix_r.len(), d)?;
        check("time_mix_k", weights.time_mix_k.len(), d)?;
        check("time_mix_v", weights.time_mix_v.len(), d)?;

        Ok(Self { config, weights })
    }

    /// Forward pass: `x: [L * D]` → `output: [L * D]`.
    ///
    /// 1. Pre-layer-norm on the full sequence.
    /// 2. Token-shift blending for r/k/v projections.
    /// 3. Linear projection to get raw r, k, v.
    /// 4. WKV recurrence per channel.
    /// 5. Receptance gate: `out_t = sigmoid(r_t) ⊙ wkv_t`.
    /// 6. Output projection.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != L * D`
    /// - [`MambaError::NonFinite`] if any intermediate value is not finite
    pub fn forward(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        let d = self.config.d_model;
        let l = self.config.seq_len;
        let expected = l * d;

        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let w = &self.weights;

        // ── 1. Pre-layer norm ─────────────────────────────────────────────────
        let x_norm = layer_norm(x, &w.ln_weight, &w.ln_bias, l, d, 1e-5)?;

        // ── 2. Token-shift blending ───────────────────────────────────────────
        // x_shifted[t, c] = time_mix[c] * x_norm[t, c] + (1 - time_mix[c]) * x_norm[t-1, c]
        // At t=0 the "previous" token is treated as zeros.
        let token_shift = |time_mix: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; l * d];
            for t in 0..l {
                let prev_row = if t > 0 {
                    &x_norm[(t - 1) * d..t * d]
                } else {
                    &[] as &[f32]
                };
                let curr_row = &x_norm[t * d..(t + 1) * d];
                for c in 0..d {
                    let prev_val = if t > 0 { prev_row[c] } else { 0.0 };
                    out[t * d + c] = time_mix[c] * curr_row[c] + (1.0 - time_mix[c]) * prev_val;
                }
            }
            out
        };

        let shifted_r = token_shift(&w.time_mix_r);
        let shifted_k = token_shift(&w.time_mix_k);
        let shifted_v = token_shift(&w.time_mix_v);

        // ── 3. Linear projections: r_raw, k, v ───────────────────────────────
        // matmul: [L, D] × [D, D]^T → [L, D]
        // (w_r is stored as [D_out, D_in], row-major)
        let linear_proj = |input: &[f32], weight: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; l * d];
            for t in 0..l {
                for o in 0..d {
                    let mut acc = 0.0_f32;
                    for i in 0..d {
                        acc += weight[o * d + i] * input[t * d + i];
                    }
                    out[t * d + o] = acc;
                }
            }
            out
        };

        let r_raw = linear_proj(&shifted_r, &w.w_r);
        let k = linear_proj(&shifted_k, &w.w_k);
        let v = linear_proj(&shifted_v, &w.w_v);

        // ── 4. WKV recurrence per channel ─────────────────────────────────────
        let mut wkv = vec![0.0_f32; l * d];

        for c in 0..d {
            let w_c = w.w[c];
            let u_c = w.u[c];

            let mut a = 0.0_f32; // numerator accumulator
            let mut b = 0.0_f32; // denominator accumulator
            let mut p = f32::NEG_INFINITY; // running log-space max

            for t in 0..l {
                let kk = k[t * d + c];
                let vv = v[t * d + c];

                // Normaliser for current WKV output computation.
                let q = p.max(u_c + kk);
                // Output for this time step.
                let num = (p - q).exp() * a + (u_c + kk - q).exp() * vv;
                let den = (p - q).exp() * b + (u_c + kk - q).exp();

                let wkv_val = if den.abs() > 1e-30 { num / den } else { 0.0 };
                wkv[t * d + c] = wkv_val;

                // State update with numerically stable running-max.
                let q2 = (p + w_c).max(kk);
                let decay_factor = (p + w_c - q2).exp();
                let input_factor = (kk - q2).exp();

                a = decay_factor * a + input_factor * vv;
                b = decay_factor * b + input_factor;
                p = q2;
            }
        }

        // ── 5. Receptance gate: out = sigmoid(r) ⊙ wkv ───────────────────────
        let mut gated = vec![0.0_f32; l * d];
        for i in 0..l * d {
            gated[i] = sigmoid(r_raw[i]) * wkv[i];
        }

        // ── 6. Output projection ───────────────────────────────────────────────
        let output = linear_proj(&gated, &w.w_o);

        // ── Finiteness check ───────────────────────────────────────────────────
        for (i, &v) in output.iter().enumerate() {
            if !v.is_finite() {
                return Err(MambaError::NonFinite("time-mixing output"));
            }
            let _ = i; // suppress unused-variable warning in release builds
        }

        Ok(output)
    }

    /// Return a reference to the layer configuration.
    #[must_use]
    pub fn config(&self) -> &TimeMixingConfig {
        &self.config
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Layer normalisation with learnable scale and shift.
///
/// Normalises each token vector (row) of `x` to zero mean and unit variance,
/// then applies the affine transform `y = γ * x̂ + β`.
///
/// # Inputs
///
/// * `x`       — `[seq_len * d_model]` (row-major, rows are token vectors)
/// * `weight`  — `[d_model]` scale (γ)
/// * `bias`    — `[d_model]` shift (β)
/// * `eps`     — small constant to avoid division by zero (typically `1e-5`)
///
/// # Errors
///
/// - [`MambaError::DimensionMismatch`] if weight or bias length ≠ `d_model`
/// - [`MambaError::NonFinite`] if any normalised value is not finite
pub fn layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq_len: usize,
    d_model: usize,
    eps: f32,
) -> MambaResult<Vec<f32>> {
    if weight.len() != d_model {
        return Err(MambaError::DimensionMismatch {
            expected: d_model,
            got: weight.len(),
        });
    }
    if bias.len() != d_model {
        return Err(MambaError::DimensionMismatch {
            expected: d_model,
            got: bias.len(),
        });
    }

    let mut out = vec![0.0_f32; seq_len * d_model];

    for t in 0..seq_len {
        let row = &x[t * d_model..(t + 1) * d_model];

        // Compute mean.
        let mean = row.iter().sum::<f32>() / d_model as f32;

        // Compute variance.
        let var = row.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f32>() / d_model as f32;

        // Normalise and scale.
        let inv_std = (var + eps).sqrt().recip();
        for c in 0..d_model {
            let x_hat = (row[c] - mean) * inv_std;
            let val = weight[c] * x_hat + bias[c];
            if !val.is_finite() {
                return Err(MambaError::NonFinite("layer norm output"));
            }
            out[t * d_model + c] = val;
        }
    }

    Ok(out)
}

/// Sigmoid activation: `σ(x) = 1 / (1 + exp(-x))`.
///
/// Numerically stable: uses `exp(-|x|)` formulation to avoid overflow.
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const EPS: f32 = 1e-5;

    // ── WkvState ──────────────────────────────────────────────────────────────

    #[test]
    fn wkv_state_default_initial() {
        let s = WkvState::default();
        assert_eq!(s.a, 0.0, "a should be 0");
        assert_eq!(s.b, 0.0, "b should be 0");
        assert!(s.p.is_infinite() && s.p < 0.0, "p should be NEG_INFINITY");
    }

    #[test]
    fn wkv_state_new_equals_default() {
        let a = WkvState::new();
        let b = WkvState::default();
        assert_eq!(a.a, b.a);
        assert_eq!(a.b, b.b);
        assert_eq!(a.p, b.p);
    }

    // ── sigmoid ───────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_zero() {
        let v = sigmoid(0.0);
        assert!((v - 0.5).abs() < EPS, "sigmoid(0)={v}, expected 0.5");
    }

    #[test]
    fn sigmoid_large_pos() {
        let v = sigmoid(100.0);
        assert!((v - 1.0).abs() < 1e-4, "sigmoid(100)={v}, expected ≈1.0");
    }

    #[test]
    fn sigmoid_large_neg() {
        let v = sigmoid(-100.0);
        assert!(v.abs() < 1e-4, "sigmoid(-100)={v}, expected ≈0.0");
    }

    #[test]
    fn sigmoid_monotonic() {
        let xs = [-5.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let ys: Vec<f32> = xs.iter().map(|&x| sigmoid(x)).collect();
        for i in 1..ys.len() {
            assert!(ys[i] > ys[i - 1], "sigmoid not monotonic at index {i}");
        }
    }

    #[test]
    fn sigmoid_symmetry() {
        for x in [1.0_f32, 2.0, 3.0, 4.5, 7.0] {
            let pos = sigmoid(x);
            let neg = sigmoid(-x);
            assert!(
                (pos + neg - 1.0).abs() < EPS,
                "sigmoid({x})+sigmoid(-{x}) ≠ 1"
            );
        }
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_non_constant_input() {
        // x = [1.0, -1.0] per token, L=1, D=2
        let x = vec![1.0_f32, -1.0];
        let w = vec![1.0_f32, 1.0];
        let b = vec![0.0_f32, 0.0];
        let out = layer_norm(&x, &w, &b, 1, 2, 1e-5).expect("layer_norm ok");
        assert_eq!(out.len(), 2);
        // mean = 0.0, std = 1.0 → normalised = [1.0, -1.0]
        assert!((out[0] - 1.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - (-1.0)).abs() < 1e-4, "out[1]={}", out[1]);
    }

    #[test]
    fn layer_norm_shape_correct() {
        let l = 5_usize;
        let d = 8_usize;
        let x = vec![0.5_f32; l * d];
        let w = vec![1.0_f32; d];
        let b = vec![0.0_f32; d];
        let out = layer_norm(&x, &w, &b, l, d, 1e-5).expect("layer_norm shape ok");
        assert_eq!(out.len(), l * d, "output length should be L*D");
    }

    #[test]
    fn layer_norm_finite() {
        let mut rng = LcgRng::new(42);
        let l = 4_usize;
        let d = 16_usize;
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let w = vec![1.0_f32; d];
        let b = vec![0.0_f32; d];
        let out = layer_norm(&x, &w, &b, l, d, 1e-5).expect("layer_norm finite");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}]={v} not finite");
        }
    }

    #[test]
    fn layer_norm_zero_bias_preserves_zero_mean() {
        // After normalisation with zero bias, each token row should have mean ≈ 0.
        let l = 3_usize;
        let d = 8_usize;
        let x: Vec<f32> = (0..l * d).map(|i| i as f32).collect();
        let w = vec![1.0_f32; d];
        let b = vec![0.0_f32; d];
        let out = layer_norm(&x, &w, &b, l, d, 1e-5).expect("layer_norm mean");
        for t in 0..l {
            let row = &out[t * d..(t + 1) * d];
            let mean = row.iter().sum::<f32>() / d as f32;
            assert!(mean.abs() < 1e-4, "token {t} mean={mean}, expected ≈0");
        }
    }

    #[test]
    fn layer_norm_affine_scale_shift() {
        // Scale by 2, shift by 1 → output = 2*x_hat + 1.
        let l = 2_usize;
        let d = 4_usize;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 2.0_f32, 4.0, 6.0, 8.0];
        let w = vec![2.0_f32; d];
        let b = vec![1.0_f32; d];
        let out = layer_norm(&x, &w, &b, l, d, 1e-5).expect("layer_norm affine");
        // Each output element should equal 2 * x_hat + 1.
        // Verify mean of affine output per token = 1.0 (since mean of normalised = 0).
        for t in 0..l {
            let row = &out[t * d..(t + 1) * d];
            let mean = row.iter().sum::<f32>() / d as f32;
            assert!(
                (mean - 1.0).abs() < 1e-4,
                "token {t} mean={mean}, expected 1.0"
            );
        }
    }

    // ── TimeMixingConfig ──────────────────────────────────────────────────────

    #[test]
    fn time_mixing_config_valid() {
        let cfg = TimeMixingConfig::new(8, 16).expect("valid config");
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.seq_len, 16);
    }

    #[test]
    fn time_mixing_config_zero_d_model() {
        let err = TimeMixingConfig::new(0, 8).expect_err("should fail on d_model=0");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    #[test]
    fn time_mixing_config_zero_seq_len() {
        let err = TimeMixingConfig::new(8, 0).expect_err("should fail on seq_len=0");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    // ── TimeMixingWeights ─────────────────────────────────────────────────────

    #[test]
    fn time_mixing_weights_default_w_value() {
        let cfg = TimeMixingConfig::new(4, 8).expect("valid config");
        let wts = TimeMixingWeights::default_init(&cfg);
        assert_eq!(wts.w.len(), 4);
        for (i, &wv) in wts.w.iter().enumerate() {
            assert!((wv - 2.0).abs() < EPS, "w[{i}]={wv}, expected 2.0");
        }
    }

    #[test]
    fn time_mixing_weights_zero_shapes() {
        let cfg = TimeMixingConfig::new(4, 8).expect("valid config");
        let wts = TimeMixingWeights::zeros(&cfg);
        assert_eq!(wts.w.len(), 4);
        assert_eq!(wts.w_r.len(), 16);
        assert_eq!(wts.w_k.len(), 16);
        assert_eq!(wts.w_v.len(), 16);
        assert_eq!(wts.w_o.len(), 16);
        assert_eq!(wts.ln_weight.len(), 4);
        assert_eq!(wts.ln_bias.len(), 4);
    }

    #[test]
    fn time_mixing_weights_random_finite() {
        let cfg = TimeMixingConfig::new(6, 4).expect("valid config");
        let mut rng = LcgRng::new(13);
        let wts = TimeMixingWeights::random(&cfg, &mut rng);
        assert!(
            wts.w.iter().all(|v| v.is_finite() && *v > 0.0),
            "w should be positive finite"
        );
        assert!(
            wts.w_r.iter().all(|v| v.is_finite()),
            "w_r should be finite"
        );
        assert!(wts.u.iter().all(|v| v.is_finite()), "u should be finite");
    }

    // ── TimeMixingLayer ───────────────────────────────────────────────────────

    #[test]
    fn time_mixing_layer_forward_shape() {
        let d = 8_usize;
        let l = 4_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let wts = TimeMixingWeights::default_init(&cfg);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![0.1_f32; l * d];
        let out = layer.forward(&x).expect("forward ok");
        assert_eq!(out.len(), l * d, "output should have L*D elements");
    }

    #[test]
    fn time_mixing_layer_forward_finite() {
        let d = 8_usize;
        let l = 6_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let mut rng = LcgRng::new(77);
        let wts = TimeMixingWeights::random(&cfg, &mut rng);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let out = layer.forward(&x).expect("forward finite");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}]={v} not finite");
        }
    }

    #[test]
    fn time_mixing_layer_single_token() {
        let d = 4_usize;
        let l = 1_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let wts = TimeMixingWeights::default_init(&cfg);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![1.0_f32, -1.0, 0.5, -0.5];
        let out = layer.forward(&x).expect("single token forward");
        assert_eq!(out.len(), d, "single-token output should have D elements");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    #[test]
    fn time_mixing_layer_deterministic() {
        let d = 6_usize;
        let l = 4_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let wts = TimeMixingWeights::default_init(&cfg);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let x: Vec<f32> = (0..l * d).map(|i| i as f32 * 0.1).collect();
        let out_a = layer.forward(&x).expect("forward a");
        let out_b = layer.forward(&x).expect("forward b");
        for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert_eq!(a, b, "non-determinism at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn time_mixing_layer_wrong_input_size() {
        let d = 4_usize;
        let l = 4_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let wts = TimeMixingWeights::default_init(&cfg);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![0.0_f32; d * l + 1]; // wrong size
        let err = layer.forward(&x).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    #[test]
    fn time_mixing_layer_config_accessor() {
        let d = 8_usize;
        let l = 4_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let wts = TimeMixingWeights::zeros(&cfg);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        assert_eq!(layer.config().d_model, d);
        assert_eq!(layer.config().seq_len, l);
    }

    #[test]
    fn time_mixing_layer_random_long_sequence() {
        let d = 8_usize;
        let l = 32_usize;
        let cfg = TimeMixingConfig::new(d, l).expect("valid config");
        let mut rng = LcgRng::new(999);
        let wts = TimeMixingWeights::random(&cfg, &mut rng);
        let layer = TimeMixingLayer::new(cfg, wts).expect("layer ok");
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let out = layer.forward(&x).expect("long sequence forward");
        assert_eq!(out.len(), l * d);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "long sequence output must be finite"
        );
    }
}
