//! sLSTM — Scalar LSTM with Exponential Gating (Beck et al. 2024).
//!
//! The sLSTM cell replaces the standard sigmoid input/forget gates with
//! exponential gates and a per-element max-normalizer for numerical stability.
//! Multiple heads are supported; the recurrent connection uses a flattened
//! head-specific hidden state projected back through `R_*` weight matrices.
//!
//! ## Reference
//!
//! Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory",
//! ICML 2024. <https://arxiv.org/abs/2405.04517>

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the scalar LSTM (sLSTM) cell.
#[derive(Debug, Clone)]
pub struct SLstmConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden dimension per head (= head_dim).
    pub hidden_dim: usize,
    /// Number of independent heads.
    pub n_heads: usize,
    /// Recurrent expansion factor. Recurrent input width = head_dim * r.
    pub r: usize,
}

impl SLstmConfig {
    /// Validate the configuration and return it if valid.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] if `input_dim`, `hidden_dim`, or `r` is zero.
    /// - [`MambaError::HeadDimMismatch`] if `n_heads` is zero.
    pub fn validate(&self) -> MambaResult<()> {
        if self.input_dim == 0 {
            return Err(MambaError::InvalidModelDim(self.input_dim));
        }
        if self.hidden_dim == 0 {
            return Err(MambaError::InvalidModelDim(self.hidden_dim));
        }
        if self.n_heads == 0 {
            return Err(MambaError::HeadDimMismatch {
                n_heads: 0,
                d_model: self.hidden_dim,
            });
        }
        if self.r == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        Ok(())
    }

    /// Recurrent hidden state width = head_dim * r.
    #[must_use]
    pub fn recurrent_dim(&self) -> usize {
        self.hidden_dim * self.r
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weight tensors for the sLSTM cell.
///
/// All matrices are stored row-major.
/// Shapes: `W_*`: `n_heads × hidden_dim × input_dim`,
///         `R_*`: `n_heads × hidden_dim × (hidden_dim * r)`,
///         biases `b_*`: `n_heads × hidden_dim`.
#[derive(Debug, Clone)]
pub struct SLstmWeights {
    /// Input gate input projection: `[n_heads, hidden_dim, input_dim]`.
    pub w_i: Vec<f32>,
    /// Forget gate input projection: `[n_heads, hidden_dim, input_dim]`.
    pub w_f: Vec<f32>,
    /// Cell input projection: `[n_heads, hidden_dim, input_dim]`.
    pub w_z: Vec<f32>,
    /// Output gate input projection: `[n_heads, hidden_dim, input_dim]`.
    pub w_o: Vec<f32>,
    /// Input gate recurrent projection: `[n_heads, hidden_dim, recurrent_dim]`.
    pub r_i: Vec<f32>,
    /// Forget gate recurrent projection: `[n_heads, hidden_dim, recurrent_dim]`.
    pub r_f: Vec<f32>,
    /// Cell recurrent projection: `[n_heads, hidden_dim, recurrent_dim]`.
    pub r_z: Vec<f32>,
    /// Output gate recurrent projection: `[n_heads, hidden_dim, recurrent_dim]`.
    pub r_o: Vec<f32>,
    /// Input gate bias: `[n_heads, hidden_dim]`.
    pub b_i: Vec<f32>,
    /// Forget gate bias: `[n_heads, hidden_dim]` (initialized to 3.0).
    pub b_f: Vec<f32>,
    /// Cell bias: `[n_heads, hidden_dim]`.
    pub b_z: Vec<f32>,
    /// Output gate bias: `[n_heads, hidden_dim]`.
    pub b_o: Vec<f32>,
}

impl SLstmWeights {
    /// Initialize weights with Kaiming uniform for W/R matrices and zero biases
    /// (except b_f initialized to 3.0 for long-range dependency learning).
    pub fn random(cfg: &SLstmConfig, rng: &mut LcgRng) -> Self {
        let n_heads = cfg.n_heads;
        let hd = cfg.hidden_dim;
        let id = cfg.input_dim;
        let rd = cfg.recurrent_dim();

        let w_size = n_heads * hd * id;
        let r_size = n_heads * hd * rd;
        let b_size = n_heads * hd;

        // Kaiming uniform scale = sqrt(6 / fan_in)
        let w_scale = (6.0_f32 / id as f32).sqrt();
        let r_scale = (6.0_f32 / rd as f32).sqrt();

        let mut fill_w = |scale: f32, size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect()
        };

        let w_i = fill_w(w_scale, w_size);
        let w_f = fill_w(w_scale, w_size);
        let w_z = fill_w(w_scale, w_size);
        let w_o = fill_w(w_scale, w_size);
        let r_i = fill_w(r_scale, r_size);
        let r_f = fill_w(r_scale, r_size);
        let r_z = fill_w(r_scale, r_size);
        let r_o = fill_w(r_scale, r_size);

        // Biases: zero except b_f = 3.0
        let b_i = vec![0.0_f32; b_size];
        let b_f = vec![3.0_f32; b_size]; // long-range initialization
        let b_z = vec![0.0_f32; b_size];
        let b_o = vec![0.0_f32; b_size];

        Self {
            w_i,
            w_f,
            w_z,
            w_o,
            r_i,
            r_f,
            r_z,
            r_o,
            b_i,
            b_f,
            b_z,
            b_o,
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Recurrent state for the sLSTM cell.
#[derive(Debug, Clone)]
pub struct SLstmState {
    /// Cell state: `[n_heads * hidden_dim]`.
    pub c: Vec<f32>,
    /// Normalizer state: `[n_heads * hidden_dim]`.
    pub n: Vec<f32>,
    /// Hidden output: `[n_heads * hidden_dim]`.
    pub h: Vec<f32>,
    /// Max stabilizer: `[n_heads * hidden_dim]`.
    pub m: Vec<f32>,
}

// ─── sLSTM cell ───────────────────────────────────────────────────────────────

/// Scalar LSTM (sLSTM) with exponential gating and multi-head support.
pub struct SLstm {
    /// Cell configuration.
    pub cfg: SLstmConfig,
    /// Learnable parameters.
    pub weights: SLstmWeights,
}

impl SLstm {
    /// Create a new sLSTM with randomly initialized weights.
    ///
    /// # Errors
    ///
    /// [`MambaError::InvalidModelDim`] / [`MambaError::HeadDimMismatch`] on invalid config.
    pub fn new(cfg: SLstmConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        cfg.validate()?;
        let weights = SLstmWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Create a zero-initialized recurrent state.
    #[must_use]
    pub fn init_state(&self) -> SLstmState {
        let sz = self.cfg.n_heads * self.cfg.hidden_dim;
        SLstmState {
            c: vec![0.0_f32; sz],
            n: vec![0.0_f32; sz],
            h: vec![0.0_f32; sz],
            m: vec![f32::NEG_INFINITY; sz],
        }
    }

    /// Process a single time step.
    ///
    /// Returns `(y_t, new_state)` where `y_t` is `n_heads * hidden_dim`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != input_dim`.
    /// - [`MambaError::NonFinite`] if output contains NaN or infinity.
    pub fn step(&self, x: &[f32], state: &SLstmState) -> MambaResult<(Vec<f32>, SLstmState)> {
        let n_heads = self.cfg.n_heads;
        let hd = self.cfg.hidden_dim;
        let id = self.cfg.input_dim;
        let rd = self.cfg.recurrent_dim();

        if x.len() != id {
            return Err(MambaError::DimensionMismatch {
                expected: id,
                got: x.len(),
            });
        }

        let state_sz = n_heads * hd;
        let mut new_c = vec![0.0_f32; state_sz];
        let mut new_n = vec![0.0_f32; state_sz];
        let mut new_h = vec![0.0_f32; state_sz];
        let mut new_m = vec![0.0_f32; state_sz];

        for head in 0..n_heads {
            // Slice of current hidden state for this head (flattened, used for recurrence)
            let h_prev = &state.h[head * hd..(head + 1) * hd];

            // For each output unit j in [0, hd):
            for j in 0..hd {
                let w_row = head * hd + j; // row into the per-head weight block

                // Input projections via W matrices: W @ x
                let i_in = mat_vec_row(&self.weights.w_i, w_row * id, id, x)
                    + self.weights.b_i[head * hd + j];

                let f_in = mat_vec_row(&self.weights.w_f, w_row * id, id, x)
                    + self.weights.b_f[head * hd + j];

                let z_in = mat_vec_row(&self.weights.w_z, w_row * id, id, x)
                    + self.weights.b_z[head * hd + j];

                let o_in = mat_vec_row(&self.weights.w_o, w_row * id, id, x)
                    + self.weights.b_o[head * hd + j];

                // Recurrent projections via R matrices: R @ h_prev
                let r_row_start = (head * hd + j) * rd;

                let i_rec = dot(&self.weights.r_i[r_row_start..r_row_start + rd], h_prev);
                let f_rec = dot(&self.weights.r_f[r_row_start..r_row_start + rd], h_prev);
                let z_rec = dot(&self.weights.r_z[r_row_start..r_row_start + rd], h_prev);
                let o_rec = dot(&self.weights.r_o[r_row_start..r_row_start + rd], h_prev);

                let i_tilde = i_in + i_rec;
                let f_tilde = f_in + f_rec;
                let z_t = z_in + z_rec;
                let o_t = o_in + o_rec;

                // Stabilized exponential gates
                let prev_m = state.m[head * hd + j];
                let m_candidate = f_tilde + prev_m.max(f32::NEG_INFINITY);
                let m_t = if m_candidate > i_tilde {
                    m_candidate
                } else {
                    i_tilde
                };

                // Clamp for numerical safety
                let safe_m_t = m_t.min(30.0_f32);
                let i_gate = (i_tilde - safe_m_t).exp().min(1e9_f32);
                let f_gate = if prev_m.is_finite() {
                    (f_tilde - safe_m_t + prev_m).exp().min(1e9_f32)
                } else {
                    0.0_f32
                };

                let prev_c = state.c[head * hd + j];
                let prev_n = state.n[head * hd + j];

                let c_t = f_gate * prev_c + i_gate * z_t.tanh();
                let n_t = f_gate * prev_n + i_gate;

                // Stabilized output: divide by max(|n_t|, 1)
                let n_norm = n_t.abs().max(1.0_f32);
                let h_t = sigmoid(o_t) * (c_t / n_norm);

                let out_idx = head * hd + j;
                new_c[out_idx] = c_t;
                new_n[out_idx] = n_t;
                new_m[out_idx] = m_t.min(30.0_f32);
                new_h[out_idx] = h_t;
            }
        }

        // Verify output finiteness
        if new_h.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("sLSTM hidden state"));
        }

        let y = new_h.clone();
        let new_state = SLstmState {
            c: new_c,
            n: new_n,
            h: new_h,
            m: new_m,
        };
        Ok((y, new_state))
    }

    /// Process a full sequence.
    ///
    /// `x` is laid out as `[seq_len, input_dim]` (row-major).
    /// Returns `[seq_len, n_heads, hidden_dim]` (row-major) = `seq_len * n_heads * hidden_dim`.
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
        let hd = self.cfg.hidden_dim;
        let n_heads = self.cfg.n_heads;
        let expected_len = seq_len * id;
        if x.len() != expected_len {
            return Err(MambaError::DimensionMismatch {
                expected: expected_len,
                got: x.len(),
            });
        }

        let mut state = self.init_state();
        let out_stride = n_heads * hd;
        let mut output = vec![0.0_f32; seq_len * out_stride];

        for t in 0..seq_len {
            let x_t = &x[t * id..(t + 1) * id];
            let (y_t, new_state) = self.step(x_t, &state)?;
            let out_start = t * out_stride;
            output[out_start..out_start + out_stride].copy_from_slice(&y_t);
            state = new_state;
        }
        Ok(output)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Single row of a matrix-vector product: `row_slice @ v`.
/// `row_start` is the byte offset into `mat` for the start of this row.
/// `row_len` is the number of columns.
#[inline]
fn mat_vec_row(mat: &[f32], row_start: usize, row_len: usize, v: &[f32]) -> f32 {
    let row = &mat[row_start..row_start + row_len];
    dot(row, v)
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> SLstmConfig {
        SLstmConfig {
            input_dim: 8,
            hidden_dim: 4,
            n_heads: 2,
            r: 2,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Shape tests ───────────────────────────────────────────────────────────

    #[test]
    fn slstm_step_output_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.hidden_dim;
        let id = cfg.input_dim;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new should succeed");
        let state = model.init_state();
        let x: Vec<f32> = (0..id).map(|i| i as f32 * 0.1).collect();
        let (y, _) = model.step(&x, &state).expect("step should succeed");
        assert_eq!(y.len(), n_heads * hd, "step output shape");
    }

    #[test]
    fn slstm_state_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.hidden_dim;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let s = model.init_state();
        assert_eq!(s.c.len(), n_heads * hd);
        assert_eq!(s.n.len(), n_heads * hd);
        assert_eq!(s.h.len(), n_heads * hd);
        assert_eq!(s.m.len(), n_heads * hd);
    }

    #[test]
    fn slstm_forward_shape() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let n_heads = cfg.n_heads;
        let hd = cfg.hidden_dim;
        let id = cfg.input_dim;
        let seq_len = 6;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let x: Vec<f32> = (0..seq_len * id).map(|i| i as f32 * 0.01).collect();
        let y = model.forward(&x, seq_len).expect("forward");
        assert_eq!(y.len(), seq_len * n_heads * hd);
    }

    #[test]
    fn slstm_output_finite() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let seq_len = 10;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let x: Vec<f32> = (0..seq_len * id)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let y = model.forward(&x, seq_len).expect("forward");
        assert!(y.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn slstm_stabilization() {
        // Max stabilizer m should remain bounded and not cause NaN
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let hd = cfg.hidden_dim;
        let n_heads = cfg.n_heads;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let mut state = model.init_state();
        // Feed large-magnitude inputs to stress-test stabilization
        let x: Vec<f32> = (0..id).map(|i| i as f32 * 2.0).collect();
        let mut prev_m_max = f32::NEG_INFINITY;
        for _ in 0..8 {
            let (y, new_state) = model.step(&x, &state).expect("step");
            assert!(y.iter().all(|v| v.is_finite()), "outputs must stay finite");
            // m state values should be bounded by our 30.0 clamp
            let cur_m_max = new_state
                .m
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(cur_m_max <= 31.0, "m_t exceeded clamp: {cur_m_max}");
            assert!(
                cur_m_max >= prev_m_max - 1.0,
                "m_max regressed unexpectedly: {cur_m_max} < {prev_m_max}"
            );
            prev_m_max = cur_m_max;
            state = new_state;
        }
        let _ = (n_heads, hd); // suppress unused
    }

    #[test]
    fn slstm_single_step_matches_forward() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let hd = cfg.hidden_dim;
        let n_heads = cfg.n_heads;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let x0: Vec<f32> = (0..id).map(|i| i as f32 * 0.05).collect();
        // step result
        let state = model.init_state();
        let (y_step, _) = model.step(&x0, &state).expect("step");
        // forward result (seq_len=1)
        let y_fwd = model.forward(&x0, 1).expect("forward");
        assert_eq!(y_step.len(), n_heads * hd);
        assert_eq!(y_fwd.len(), n_heads * hd);
        for (s, f) in y_step.iter().zip(y_fwd.iter()) {
            assert!((s - f).abs() < 1e-5, "step vs forward mismatch: {s} vs {f}");
        }
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn err_input_dim_zero() {
        let cfg = SLstmConfig {
            input_dim: 0,
            hidden_dim: 4,
            n_heads: 1,
            r: 1,
        };
        let mut rng = make_rng();
        let result = SLstm::new(cfg, &mut rng);
        assert!(result.is_err(), "zero input_dim should fail");
    }

    #[test]
    fn err_head_dim_zero() {
        let cfg = SLstmConfig {
            input_dim: 8,
            hidden_dim: 0,
            n_heads: 1,
            r: 1,
        };
        let mut rng = make_rng();
        let result = SLstm::new(cfg, &mut rng);
        assert!(result.is_err(), "zero hidden_dim should fail");
    }

    #[test]
    fn err_n_heads_zero() {
        let cfg = SLstmConfig {
            input_dim: 8,
            hidden_dim: 4,
            n_heads: 0,
            r: 1,
        };
        let mut rng = make_rng();
        let result = SLstm::new(cfg, &mut rng);
        assert!(result.is_err(), "zero n_heads should fail");
    }

    #[test]
    fn err_input_length_mismatch() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let id = cfg.input_dim;
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let short_x = vec![0.0_f32; id - 1];
        let state = model.init_state();
        let result = model.step(&short_x, &state);
        assert!(result.is_err(), "input length mismatch should fail");
    }

    #[test]
    fn err_seq_len_zero() {
        let cfg = make_cfg();
        let mut rng = make_rng();
        let model = SLstm::new(cfg, &mut rng).expect("SLstm::new");
        let result = model.forward(&[], 0);
        assert!(result.is_err(), "zero seq_len should fail");
    }
}
