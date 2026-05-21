//! mLSTM — Matrix LSTM with Associative Memory (Beck et al. 2024).
//!
//! The mLSTM cell replaces the scalar cell state `c ∈ ℝ^d` with a matrix
//! `C ∈ ℝ^{d×d}`.  Keys and values are projected from input, and the memory
//! is updated via an outer product `v ⊗ k`, enabling content-addressable
//! retrieval with query `q` via `C @ q`.  Scalar (per-head) gates are
//! stabilized by a max-state tracker.
//!
//! ## Reference
//!
//! Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory",
//! ICML 2024. <https://arxiv.org/abs/2405.04517>

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the matrix LSTM (mLSTM) cell.
#[derive(Debug, Clone)]
pub struct MLstmConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Key/value head dimension (d_k = d_v = head_dim).
    pub head_dim: usize,
    /// Number of independent heads.
    pub n_heads: usize,
}

impl MLstmConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] if `input_dim` or `head_dim` is zero.
    /// - [`MambaError::HeadDimMismatch`] if `n_heads` is zero.
    pub fn validate(&self) -> MambaResult<()> {
        if self.input_dim == 0 {
            return Err(MambaError::InvalidModelDim(self.input_dim));
        }
        if self.head_dim == 0 {
            return Err(MambaError::InvalidModelDim(self.head_dim));
        }
        if self.n_heads == 0 {
            return Err(MambaError::HeadDimMismatch {
                n_heads: 0,
                d_model: self.head_dim,
            });
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weight tensors for the mLSTM cell.
///
/// Gate projections `w_i`/`w_f` produce a **scalar** per head (d_gate=1).
/// All other projections produce `head_dim` values per head.
#[derive(Debug, Clone)]
pub struct MLstmWeights {
    /// Query projection: `[n_heads, head_dim, input_dim]`.
    pub w_q: Vec<f32>,
    /// Key projection: `[n_heads, head_dim, input_dim]`.
    pub w_k: Vec<f32>,
    /// Value projection: `[n_heads, head_dim, input_dim]`.
    pub w_v: Vec<f32>,
    /// Input gate projection (scalar per head): `[n_heads, 1, input_dim]`.
    pub w_i: Vec<f32>,
    /// Forget gate projection (scalar per head): `[n_heads, 1, input_dim]`.
    pub w_f: Vec<f32>,
    /// Output gate projection: `[n_heads, head_dim, input_dim]`.
    pub w_o: Vec<f32>,
    /// Query bias: `[n_heads, head_dim]`.
    pub b_q: Vec<f32>,
    /// Key bias: `[n_heads, head_dim]`.
    pub b_k: Vec<f32>,
    /// Value bias: `[n_heads, head_dim]`.
    pub b_v: Vec<f32>,
    /// Input gate bias (scalar per head): `[n_heads]`.
    pub b_i: Vec<f32>,
    /// Forget gate bias (scalar per head): `[n_heads]` (initialized to 3.0).
    pub b_f: Vec<f32>,
    /// Output gate bias: `[n_heads, head_dim]`.
    pub b_o: Vec<f32>,
}

impl MLstmWeights {
    /// Initialize weights with Kaiming uniform for W matrices and zero biases
    /// (except `b_f` set to 3.0 for long-range dependency learning).
    pub fn random(cfg: &MLstmConfig, rng: &mut LcgRng) -> Self {
        let n_heads = cfg.n_heads;
        let hd = cfg.head_dim;
        let id = cfg.input_dim;

        let qkvo_size = n_heads * hd * id;
        let gate_size = n_heads * id; // scalar gate per head
        let hd_bias_size = n_heads * hd;
        let scalar_bias_size = n_heads;

        let w_scale = (6.0_f32 / id as f32).sqrt();

        let mut fill_w = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * w_scale)
                .collect()
        };

        let w_q = fill_w(qkvo_size);
        let w_k = fill_w(qkvo_size);
        let w_v = fill_w(qkvo_size);
        let w_i = fill_w(gate_size);
        let w_f = fill_w(gate_size);
        let w_o = fill_w(qkvo_size);

        let b_q = vec![0.0_f32; hd_bias_size];
        let b_k = vec![0.0_f32; hd_bias_size];
        let b_v = vec![0.0_f32; hd_bias_size];
        let b_i = vec![0.0_f32; scalar_bias_size];
        let b_f = vec![3.0_f32; scalar_bias_size]; // long-range initialization
        let b_o = vec![0.0_f32; hd_bias_size];

        Self {
            w_q,
            w_k,
            w_v,
            w_i,
            w_f,
            w_o,
            b_q,
            b_k,
            b_v,
            b_i,
            b_f,
            b_o,
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Recurrent state for the mLSTM cell.
#[derive(Debug, Clone)]
pub struct MLstmState {
    /// Covariance/memory matrix: `[n_heads * head_dim * head_dim]` (row-major per head).
    pub c: Vec<f32>,
    /// Normalizer vector: `[n_heads * head_dim]`.
    pub n: Vec<f32>,
    /// Max-stabilizer (scalar per head): `[n_heads]`.
    pub m: Vec<f32>,
}

// ─── mLSTM cell ───────────────────────────────────────────────────────────────

/// Matrix LSTM (mLSTM) with outer-product memory and multi-head support.
pub struct MLstm {
    /// Cell configuration.
    pub cfg: MLstmConfig,
    /// Learnable parameters.
    pub weights: MLstmWeights,
}

impl MLstm {
    /// Create a new mLSTM with randomly initialized weights.
    ///
    /// # Errors
    ///
    /// [`MambaError::InvalidModelDim`] / [`MambaError::HeadDimMismatch`] on invalid config.
    pub fn new(cfg: MLstmConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        cfg.validate()?;
        let weights = MLstmWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Create a zero-initialized recurrent state.
    #[must_use]
    pub fn init_state(&self) -> MLstmState {
        let n_heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim;
        MLstmState {
            c: vec![0.0_f32; n_heads * hd * hd],
            n: vec![0.0_f32; n_heads * hd],
            m: vec![f32::NEG_INFINITY; n_heads],
        }
    }

    /// Process a single time step.
    ///
    /// Returns `(h_t, new_state)` where `h_t` has shape `n_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != input_dim`.
    /// - [`MambaError::NonFinite`] if output contains NaN or infinity.
    pub fn step(&self, x: &[f32], state: &MLstmState) -> MambaResult<(Vec<f32>, MLstmState)> {
        let n_heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim;
        let id = self.cfg.input_dim;

        if x.len() != id {
            return Err(MambaError::DimensionMismatch {
                expected: id,
                got: x.len(),
            });
        }

        let mut new_c = vec![0.0_f32; n_heads * hd * hd];
        let mut new_n = vec![0.0_f32; n_heads * hd];
        let mut new_m = vec![0.0_f32; n_heads];
        let mut h_out = vec![0.0_f32; n_heads * hd];

        for head in 0..n_heads {
            // ── Project inputs ─────────────────────────────────────────────
            let q_t = proj_head(&self.weights.w_q, &self.weights.b_q, head, hd, id, x);
            // Normalize key to unit L2 norm (standard xLSTM practice to keep C bounded)
            let k_t_raw = proj_head(&self.weights.w_k, &self.weights.b_k, head, hd, id, x);
            let k_norm = k_t_raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            let k_t: Vec<f32> = k_t_raw.iter().map(|v| v / k_norm).collect();
            let v_t = proj_head(&self.weights.w_v, &self.weights.b_v, head, hd, id, x);
            let o_t = proj_head(&self.weights.w_o, &self.weights.b_o, head, hd, id, x);

            // Scalar gate projections (d_gate = 1 per head)
            let i_tilde = dot_slice(&self.weights.w_i[head * id..(head + 1) * id], x)
                + self.weights.b_i[head];

            let f_tilde = dot_slice(&self.weights.w_f[head * id..(head + 1) * id], x)
                + self.weights.b_f[head];

            // ── Stabilized gates ───────────────────────────────────────────
            // The max-stabilizer ensures i_gate, f_gate ∈ (0, 1] without overflow.
            let prev_m = state.m[head];
            let f_carry = if prev_m.is_finite() {
                f_tilde + prev_m
            } else {
                f32::NEG_INFINITY
            };
            // m_t = max(f_tilde + m_{t-1}, i_tilde) — no external clamping needed here
            // because by construction i_tilde - m_t <= 0 and f_carry - m_t <= 0.
            let m_t = f_carry.max(i_tilde);
            // Clamp m_t only to prevent f32 saturation on the stored state
            let m_t_stored = m_t.clamp(-100.0, 100.0);

            // Both exponents are <= 0, so gates are in (0, 1]
            let i_gate = (i_tilde - m_t).exp();
            let f_gate = if prev_m.is_finite() {
                (f_tilde - m_t + prev_m).exp()
            } else {
                0.0_f32
            };

            // ── Matrix memory update: C_t = f * C_{t-1} + i * (v ⊗ k) ────
            let c_off = head * hd * hd;
            let n_off = head * hd;

            // Outer product v ⊗ k: shape [hd, hd], row i = k[j] * v[i]
            // C[i,j] += i_gate * v[i] * k[j]
            for row in 0..hd {
                for col in 0..hd {
                    let prev = state.c[c_off + row * hd + col];
                    new_c[c_off + row * hd + col] = f_gate * prev + i_gate * v_t[row] * k_t[col];
                }
            }

            // ── Normalizer update: n_t = f * n_{t-1} + i * k ──────────────
            for j in 0..hd {
                new_n[n_off + j] = f_gate * state.n[n_off + j] + i_gate * k_t[j];
            }

            new_m[head] = m_t_stored;

            // ── Output: h_t = sigmoid(o) ⊙ (C_t @ q) / max(|n_t^T q|, 1) ─
            // C_t @ q_t: shape [hd]
            let mut cq = vec![0.0_f32; hd];
            for row in 0..hd {
                let mut acc = 0.0_f32;
                for col in 0..hd {
                    acc += new_c[c_off + row * hd + col] * q_t[col];
                }
                cq[row] = acc;
            }

            // n_t^T @ q_t (scalar)
            let nq: f32 = new_n[n_off..n_off + hd]
                .iter()
                .zip(q_t.iter())
                .map(|(ni, qi)| ni * qi)
                .sum();
            let denom = nq.abs().max(1.0_f32);

            for j in 0..hd {
                h_out[head * hd + j] = sigmoid(o_t[j]) * cq[j] / denom;
            }
        }

        if h_out.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("mLSTM hidden state"));
        }

        let new_state = MLstmState {
            c: new_c,
            n: new_n,
            m: new_m,
        };
        Ok((h_out, new_state))
    }

    /// Process a full sequence.
    ///
    /// `x` is laid out as `[seq_len, input_dim]` (row-major).
    /// Returns `[seq_len, n_heads, head_dim]` (row-major).
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != seq_len * input_dim`.
    /// - [`MambaError::InvalidSeqLen`] if `seq_len == 0`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        let id = self.cfg.input_dim;
        let hd = self.cfg.head_dim;
        let n_heads = self.cfg.n_heads;
        let expected = seq_len * id;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let out_stride = n_heads * hd;
        let mut output = vec![0.0_f32; seq_len * out_stride];
        let mut state = self.init_state();

        for t in 0..seq_len {
            let x_t = &x[t * id..(t + 1) * id];
            let (h_t, new_state) = self.step(x_t, &state)?;
            let out_start = t * out_stride;
            output[out_start..out_start + out_stride].copy_from_slice(&h_t);
            state = new_state;
        }
        Ok(output)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Project a single head: `W[head, :, :] @ x + b[head, :]`.
///
/// `w` is stored as `[n_heads, hd, id]` row-major.
/// `b` is stored as `[n_heads, hd]`.
#[inline]
fn proj_head(w: &[f32], b: &[f32], head: usize, hd: usize, id: usize, x: &[f32]) -> Vec<f32> {
    let base = head * hd * id;
    let b_base = head * hd;
    let mut out = vec![0.0_f32; hd];
    for row in 0..hd {
        let row_start = base + row * id;
        let dot: f32 = w[row_start..row_start + id]
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi * xi)
            .sum();
        out[row] = dot + b[b_base + row];
    }
    out
}

/// Dot product of two equal-length slices.
#[inline]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> MLstmConfig {
        MLstmConfig {
            input_dim: 8,
            head_dim: 4,
            n_heads: 2,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    // ── Shape tests ───────────────────────────────────────────────────────────

    #[test]
    fn mlstm_step_output_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.head_dim;
        let id = cfg.input_dim;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let state = model.init_state();
        let x: Vec<f32> = (0..id).map(|i| i as f32 * 0.1).collect();
        let (h, _) = model.step(&x, &state).expect("step");
        assert_eq!(h.len(), n_heads * hd);
    }

    #[test]
    fn mlstm_matrix_state_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.head_dim;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let s = model.init_state();
        assert_eq!(s.c.len(), n_heads * hd * hd);
        assert_eq!(s.n.len(), n_heads * hd);
        assert_eq!(s.m.len(), n_heads);
    }

    #[test]
    fn mlstm_forward_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.head_dim;
        let id = cfg.input_dim;
        let seq_len = 5;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let x: Vec<f32> = (0..seq_len * id).map(|i| i as f32 * 0.01).collect();
        let y = model.forward(&x, seq_len).expect("forward");
        assert_eq!(y.len(), seq_len * n_heads * hd);
    }

    #[test]
    fn mlstm_output_finite() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let seq_len = 12;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let x: Vec<f32> = (0..seq_len * id)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let y = model.forward(&x, seq_len).expect("forward");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    #[test]
    fn mlstm_associative_memory_content() {
        // After encoding a single (v, k) pair with i=1, f=0:
        // C should contain v ⊗ k.
        // We test this by constructing a minimal 1-head model and
        // verifying C accumulates correctly.
        let cfg = MLstmConfig {
            input_dim: 4,
            head_dim: 2,
            n_heads: 1,
        };
        let mut rng = LcgRng::new(0);
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let state = model.init_state();
        // Construct an input that yields known v and k through identity-like weights
        // We verify that after a step, C is updated (i.e., not all zeros)
        let x = vec![0.5_f32, -0.3, 0.8, -0.1];
        let (h, new_state) = model.step(&x, &state).expect("step");
        // C should no longer be all-zero after one step
        let c_nonzero = new_state.c.iter().any(|&v| v.abs() > 1e-8);
        assert!(c_nonzero, "C should accumulate outer product content");
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mlstm_normalized_output_bounded() {
        // Output h_t is bounded by |sigmoid(o_t)| * reasonable factor
        // Sigmoid max is 1.0, so h_t should not blow up
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let seq_len = 20;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let x: Vec<f32> = (0..seq_len * id)
            .map(|_| rng.next_f32() * 4.0 - 2.0)
            .collect();
        let y = model.forward(&x, seq_len).expect("forward");
        // All values must be finite and within a generous bound
        let max_val = y.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(max_val.is_finite(), "max output must be finite");
        // With normalization, output should be bounded
        assert!(max_val < 1e6, "output too large: {max_val}");
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn err_input_dim_zero() {
        let cfg = MLstmConfig {
            input_dim: 0,
            head_dim: 4,
            n_heads: 1,
        };
        let mut rng = make_rng();
        assert!(MLstm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_head_dim_zero() {
        let cfg = MLstmConfig {
            input_dim: 8,
            head_dim: 0,
            n_heads: 1,
        };
        let mut rng = make_rng();
        assert!(MLstm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_n_heads_zero() {
        let cfg = MLstmConfig {
            input_dim: 8,
            head_dim: 4,
            n_heads: 0,
        };
        let mut rng = make_rng();
        assert!(MLstm::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_input_length_mismatch() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        let state = model.init_state();
        let bad_x = vec![0.0_f32; id + 1];
        assert!(model.step(&bad_x, &state).is_err());
    }

    #[test]
    fn err_forward_length_mismatch() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let model = MLstm::new(cfg, &mut rng).expect("MLstm::new");
        // seq_len=3 but x.len() = 2 * id (wrong)
        let x = vec![0.0_f32; 2 * id];
        assert!(model.forward(&x, 3).is_err());
    }
}
