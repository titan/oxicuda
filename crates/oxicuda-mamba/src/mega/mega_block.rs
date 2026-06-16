//! MEGA — Moving Average Equipped Gated Attention (Ma et al. 2022).
//!
//! MEGA combines an Exponential Moving Average (EMA) sub-layer with a
//! single-headed gated attention mechanism, achieving linear-time sequence
//! modelling competitive with full quadratic attention on many tasks.
//!
//! # Architecture
//!
//! ```text
//! input x  [seq_len × d_model]
//!   │
//!   ├── EMA layer  (per-dim α / δ coefficients, state h ∈ ℝ^d_ema)
//!   │     h_t = α ⊙ h_{t-1} + (1-α) ⊙ δ ⊙ x_t_proj
//!   │     ema_out [seq_len × d_ema]
//!   │
//!   ├── Gated attention
//!   │     Q = q_w @ ema_out    [seq_len × d_head]
//!   │     K = k_w @ ema_out    [seq_len × d_head]
//!   │     V = v_w @ x          [seq_len × d_expand]
//!   │     attn = softmax(QK^T / sqrt(d_head)) @ V
//!   │
//!   ├── Gate: sigmoid(ema_ctx) * attn
//!   │
//!   └── out_proj  [d_expand → d_model] + residual
//! ```
//!
//! # Reference
//!
//! Ma, Chunxi et al. (2022) "Mega: Moving Average Equipped Gated Attention".
//! <https://arxiv.org/abs/2209.10655>

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── MegaConfig ──────────────────────────────────────────────────────────────

/// Configuration for a single MEGA block.
#[derive(Debug, Clone)]
pub struct MegaConfig {
    /// Input / output model dimension `D`.
    pub d_model: usize,
    /// Attention head dimension (Q / K dimension per head).
    pub d_head: usize,
    /// EMA state dimension (number of EMA channels).
    pub d_ema: usize,
    /// Value expansion factor: `d_expand = expand * d_model`.
    pub expand: usize,
}

impl MegaConfig {
    /// Validate that all dimensions are positive.
    fn validate(&self) -> MambaResult<()> {
        if self.d_model == 0 {
            return Err(MambaError::InvalidModelDim(self.d_model));
        }
        if self.d_head == 0 {
            return Err(MambaError::Internal("d_head must be > 0".into()));
        }
        if self.d_ema == 0 {
            return Err(MambaError::InvalidSsmOrder(self.d_ema));
        }
        if self.expand == 0 {
            return Err(MambaError::Internal("expand must be > 0".into()));
        }
        Ok(())
    }

    /// Output dimension of the value branch.
    #[inline]
    fn d_expand(&self) -> usize {
        self.expand * self.d_model
    }
}

// ─── MegaBlock ────────────────────────────────────────────────────────────────

/// MEGA block: Exponential Moving Average + Gated Attention.
///
/// Stores all learned parameters; no mutable state is held between forward
/// calls (stateless / recurrence recreated each call).
pub struct MegaBlock {
    /// EMA forgetting coefficients (before sigmoid): `[d_ema]`.
    ema_alpha: Vec<f32>,
    /// EMA input-gate (before sigmoid): `[d_ema]`.
    ema_delta: Vec<f32>,
    /// EMA input projection: `[d_ema × d_model]` row-major.
    ema_in_w: Vec<f32>,
    /// Query projection: `[d_head × d_ema]` row-major.
    q_w: Vec<f32>,
    /// Key projection: `[d_head × d_ema]` row-major.
    k_w: Vec<f32>,
    /// Value projection: `[d_expand × d_model]` row-major.
    v_w: Vec<f32>,
    /// Context gate projection: `[d_expand × d_ema]` row-major.
    gate_w: Vec<f32>,
    /// Output projection: `[d_model × d_expand]` row-major.
    out_w: Vec<f32>,
    /// Block configuration.
    config: MegaConfig,
}

impl MegaBlock {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new MEGA block with Xavier-uniform initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`MambaError::InvalidModelDim`] / [`MambaError::InvalidSsmOrder`]
    /// / [`MambaError::Internal`] if any configuration dimension is zero.
    pub fn new(config: MegaConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        config.validate()?;

        let d_model = config.d_model;
        let d_head = config.d_head;
        let d_ema = config.d_ema;
        let d_expand = config.d_expand();

        /// Xavier-uniform for a `[rows × cols]` weight matrix.
        fn xavier(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
            let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
            (0..rows * cols)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
                .collect()
        }

        // EMA parameters initialised in log-space so sigmoid(·) ∈ (0,1).
        let ema_alpha: Vec<f32> = (0..d_ema).map(|_| rng.next_f32() - 0.5).collect();
        let ema_delta: Vec<f32> = (0..d_ema).map(|_| rng.next_f32() - 0.5).collect();
        let ema_in_w = xavier(d_ema, d_model, rng);
        let q_w = xavier(d_head, d_ema, rng);
        let k_w = xavier(d_head, d_ema, rng);
        let v_w = xavier(d_expand, d_model, rng);
        let gate_w = xavier(d_expand, d_ema, rng);
        let out_w = xavier(d_model, d_expand, rng);

        Ok(Self {
            ema_alpha,
            ema_delta,
            ema_in_w,
            q_w,
            k_w,
            v_w,
            gate_w,
            out_w,
            config,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return the model dimension `D`.
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Run the MEGA block on an input sequence.
    ///
    /// # Arguments
    ///
    /// * `x`       — flat `[seq_len × d_model]` row-major input.
    /// * `seq_len` — number of tokens `L`.
    ///
    /// # Returns
    ///
    /// Flat `[seq_len × d_model]` row-major output (same shape as input).
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`] — if `seq_len == 0`.
    /// * [`MambaError::DimensionMismatch`] — if `x.len() != seq_len * d_model`.
    /// * [`MambaError::NonFinite`] — if any intermediate value is NaN/Inf.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        let d_model = self.config.d_model;
        let d_head = self.config.d_head;
        let d_ema = self.config.d_ema;
        let d_expand = self.config.d_expand();

        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        if x.len() != seq_len * d_model {
            return Err(MambaError::DimensionMismatch {
                expected: seq_len * d_model,
                got: x.len(),
            });
        }

        // ── 1. EMA sub-layer ─────────────────────────────────────────────────
        // Project input to EMA dimension: x_ema [seq_len × d_ema]
        // h_t = α ⊙ h_{t-1} + (1-α) ⊙ δ ⊙ x_ema_t
        let alpha: Vec<f32> = self.ema_alpha.iter().map(|&v| sigmoid(v)).collect();
        let delta: Vec<f32> = self.ema_delta.iter().map(|&v| sigmoid(v)).collect();

        // Project x → EMA space
        let x_ema = mat_mul(x, &self.ema_in_w, seq_len, d_model, d_ema)?;
        // x_ema shape: [seq_len × d_ema]

        let mut h = vec![0.0_f32; d_ema]; // hidden EMA state
        let mut ema_out = vec![0.0_f32; seq_len * d_ema];

        for t in 0..seq_len {
            let xt = &x_ema[t * d_ema..(t + 1) * d_ema];
            for d in 0..d_ema {
                // h_t = α * h_{t-1} + (1-α) * δ * x_t
                h[d] = alpha[d] * h[d] + (1.0 - alpha[d]) * delta[d] * xt[d];
                ema_out[t * d_ema + d] = h[d];
            }
        }

        // ── 2. Gated attention ───────────────────────────────────────────────
        // Q = ema_out @ q_w^T  [seq_len × d_head]
        // K = ema_out @ k_w^T  [seq_len × d_head]
        // V = x       @ v_w^T  [seq_len × d_expand]

        let q = mat_mul(&ema_out, &self.q_w, seq_len, d_ema, d_head)?;
        let k = mat_mul(&ema_out, &self.k_w, seq_len, d_ema, d_head)?;
        let v = mat_mul(x, &self.v_w, seq_len, d_model, d_expand)?;

        // Attention scores: A = Q K^T / sqrt(d_head)  [seq_len × seq_len]
        let scale = 1.0_f32 / (d_head as f32).sqrt();
        let attn_scores = qk_dot(&q, &k, seq_len, d_head, scale)?;

        // Causal softmax per query row
        let attn_weights = causal_softmax(&attn_scores, seq_len)?;

        // Context: attn_weights @ V  [seq_len × d_expand]
        let attn_out = mat_mul(&attn_weights, &v, seq_len, seq_len, d_expand)?;

        // ── 3. Gating ─────────────────────────────────────────────────────────
        // gate_ctx = ema_out @ gate_w^T  [seq_len × d_expand]
        // gated = sigmoid(gate_ctx) ⊙ attn_out
        let gate_ctx = mat_mul(&ema_out, &self.gate_w, seq_len, d_ema, d_expand)?;
        let mut gated = vec![0.0_f32; seq_len * d_expand];
        for i in 0..seq_len * d_expand {
            gated[i] = sigmoid(gate_ctx[i]) * attn_out[i];
        }

        // ── 4. Output projection + residual ───────────────────────────────────
        // out = gated @ out_w^T  [seq_len × d_model]
        // output = x + out
        let projected = mat_mul(&gated, &self.out_w, seq_len, d_expand, d_model)?;
        let mut output = vec![0.0_f32; seq_len * d_model];
        for i in 0..seq_len * d_model {
            output[i] = x[i] + projected[i];
        }

        // Finite-value guard
        if output.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("MEGA forward output"));
        }

        Ok(output)
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Element-wise sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dense matrix–matrix multiply: `C = A * W^T`.
///
/// * `a`     — `[m × k]` row-major.
/// * `w`     — `[n × k]` row-major (rows = output features, cols = input features).
///
/// Returns `[m × n]` row-major.
fn mat_mul(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> MambaResult<Vec<f32>> {
    if a.len() != m * k {
        return Err(MambaError::DimensionMismatch {
            expected: m * k,
            got: a.len(),
        });
    }
    if w.len() != n * k {
        return Err(MambaError::DimensionMismatch {
            expected: n * k,
            got: w.len(),
        });
    }
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for l in 0..k {
                acc += a[i * k + l] * w[j * k + l];
            }
            c[i * n + j] = acc;
        }
    }
    Ok(c)
}

/// Compute `A = Q * K^T * scale` where Q,K are `[seq × d_head]`.
///
/// Returns `[seq × seq]` row-major.
fn qk_dot(q: &[f32], k: &[f32], seq: usize, d: usize, scale: f32) -> MambaResult<Vec<f32>> {
    let mut a = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut acc = 0.0_f32;
            for l in 0..d {
                acc += q[i * d + l] * k[j * d + l];
            }
            a[i * seq + j] = acc * scale;
        }
    }
    Ok(a)
}

/// Row-wise causal softmax over `[seq × seq]` attention logits.
///
/// Token `i` attends only to tokens `j <= i` (lower triangular).
fn causal_softmax(scores: &[f32], seq: usize) -> MambaResult<Vec<f32>> {
    let mut out = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        // Find max over causal positions for numerical stability.
        let row_max = (0..=i)
            .map(|j| scores[i * seq + j])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for j in 0..=i {
            let e = (scores[i * seq + j] - row_max).exp();
            out[i * seq + j] = e;
            sum += e;
        }
        // Positions j > i remain 0 (masked out).
        if sum > 0.0 {
            for j in 0..=i {
                out[i * seq + j] /= sum;
            }
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> MegaConfig {
        MegaConfig {
            d_model: 4,
            d_head: 2,
            d_ema: 3,
            expand: 2,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── 1. Output shape ───────────────────────────────────────────────────────
    #[test]
    fn output_shape() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let seq = 5;
        let x: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.1).collect();
        let out = block.forward(&x, seq).expect("mega: forward");
        assert_eq!(out.len(), seq * d);
    }

    // ── 2. Output finite ──────────────────────────────────────────────────────
    #[test]
    fn output_finite() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let seq = 8;
        let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 - 16.0) * 0.05).collect();
        let out = block.forward(&x, seq).expect("mega: forward");
        assert!(out.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
    }

    // ── 3. seq_len == 1 ───────────────────────────────────────────────────────
    #[test]
    fn seq_len_1() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let x = vec![1.0_f32; d];
        let out = block.forward(&x, 1).expect("mega: forward seq_len=1");
        assert_eq!(out.len(), d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 4. EMA smooths sequence (output changes with longer history) ──────────
    #[test]
    fn ema_smooths_sequence() {
        let cfg = MegaConfig {
            d_model: 4,
            d_head: 2,
            d_ema: 3,
            expand: 2,
        };
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");

        // Short context
        let x_short: Vec<f32> = (0..3 * d).map(|i| i as f32 * 0.1).collect();
        let out_short = block.forward(&x_short, 3).expect("forward short");

        // Long context: same prefix + extra tokens
        let x_long: Vec<f32> = (0..7 * d).map(|i| i as f32 * 0.1).collect();
        let out_long = block.forward(&x_long, 7).expect("forward long");

        // The first token in short and long should differ because MEGA builds context
        // — actually for position 0 in both, both have same x but different block
        // states because the long version has a larger attn context too.
        // The key property: last position differs.
        let last_short = &out_short[(3 - 1) * d..3 * d];
        let last_long = &out_long[(7 - 1) * d..7 * d];
        let diff: f32 = last_short
            .iter()
            .zip(last_long.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "EMA should produce context-dependent outputs, diff={diff}"
        );
    }

    // ── 5. Gating values are bounded (sigmoid output ∈ (0,1)) ─────────────────
    #[test]
    fn gating_bounded() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        // Verify sigmoid stays in (0,1) for a range of inputs.
        for &v in &[-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(v);
            assert!(s > 0.0 && s < 1.0, "sigmoid({v}) = {s} out of (0,1)");
        }
        // Verify forward output is finite (implicitly tests gate is bounded).
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let x = vec![0.5_f32; 4 * d];
        let out = block.forward(&x, 4).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 6. Different inputs → different outputs ────────────────────────────────
    #[test]
    fn different_inputs_different_outputs() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let seq = 4;
        let x1: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.1).collect();
        let x2: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.2 + 1.0).collect();
        let out1 = block.forward(&x1, seq).expect("forward x1");
        let out2 = block.forward(&x2, seq).expect("forward x2");
        let diff: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different inputs should yield different outputs"
        );
    }

    // ── 7. d_model == 0 → error ───────────────────────────────────────────────
    #[test]
    fn d_model_0_error() {
        let cfg = MegaConfig {
            d_model: 0,
            d_head: 2,
            d_ema: 3,
            expand: 2,
        };
        let mut rng = make_rng();
        let result = MegaBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for d_model=0");
    }

    // ── 8. expand == 0 → error ────────────────────────────────────────────────
    #[test]
    fn expand_0_error() {
        let cfg = MegaConfig {
            d_model: 4,
            d_head: 2,
            d_ema: 3,
            expand: 0,
        };
        let mut rng = make_rng();
        let result = MegaBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for expand=0");
    }

    // ── 9. d_ema == 0 → error ─────────────────────────────────────────────────
    #[test]
    fn d_ema_0_error() {
        let cfg = MegaConfig {
            d_model: 4,
            d_head: 2,
            d_ema: 0,
            expand: 2,
        };
        let mut rng = make_rng();
        let result = MegaBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for d_ema=0");
    }

    // ── 10. d_head == 0 → error ────────────────────────────────────────────────
    #[test]
    fn d_head_0_error() {
        let cfg = MegaConfig {
            d_model: 4,
            d_head: 0,
            d_ema: 3,
            expand: 2,
        };
        let mut rng = make_rng();
        let result = MegaBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for d_head=0");
    }

    // ── 11. seq_len == 0 → error ──────────────────────────────────────────────
    #[test]
    fn seq_len_0_error() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        let result = block.forward(&[], 0);
        assert!(result.is_err(), "expected error for seq_len=0");
    }

    // ── 12. dimension mismatch → error ────────────────────────────────────────
    #[test]
    fn input_length_mismatch_error() {
        let cfg = tiny_config();
        let d = cfg.d_model;
        let mut rng = make_rng();
        let block = MegaBlock::new(cfg, &mut rng).expect("mega: new");
        // Provide wrong length (3*d instead of 4*d)
        let x = vec![0.0_f32; 3 * d];
        let result = block.forward(&x, 4);
        assert!(result.is_err(), "expected DimensionMismatch");
    }
}
