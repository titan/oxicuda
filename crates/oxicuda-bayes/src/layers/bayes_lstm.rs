//! Bayesian LSTM cell via Bayes-by-Backprop (BBB).
//!
//! Each weight in the four LSTM gates (input `i`, forget `f`, cell candidate
//! `g`, output `o`) carries a learned diagonal-Gaussian posterior
//! `q(w) = N(w_mu, σ²)` with `σ = exp(w_log_sigma)`.  At sampling time weights
//! are drawn from `q` via the reparameterisation trick `w = μ + σ·ε`,
//! `ε ~ N(0, 1)`.
//!
//! The prior is the standard isotropic Gaussian `p(w) = N(0, 1)`, so the
//! per-weight KL has the closed form
//!
//! ```text
//! KL(N(μ, σ²) ‖ N(0, 1)) = ½ (μ² + σ² − log σ² − 1).
//! ```
//!
//! Unlike the GRU cell (which re-samples weights at every step), the BBB
//! recurrent scheme of Fortunato et al. (2017) samples one weight set per
//! sequence and shares it across all time steps; [`BayesLstm::forward_seq`]
//! follows that convention.
//!
//! The LSTM update equations follow Hochreiter & Schmidhuber (1997):
//! ```text
//! i  = σ(W_ii·x + W_hi·h + b_i)         (input gate)
//! f  = σ(W_if·x + W_hf·h + b_f)         (forget gate)
//! g  = tanh(W_ig·x + W_hg·h + b_g)      (cell candidate)
//! o  = σ(W_io·x + W_ho·h + b_o)         (output gate)
//! c' = f ⊙ c + i ⊙ g                    (new cell state)
//! h' = o ⊙ tanh(c')                     (new hidden state)
//! ```
//!
//! **References:**
//! - Fortunato, M., Blundell, C., & Vinyals, O. (2017). Bayesian Recurrent
//!   Neural Networks. *arXiv:1704.02798*.
//! - Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
//!   Weight Uncertainty in Neural Networks. *ICML 2015*.
//! - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
//!   *Neural Computation, 9(8)*.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Constructor configuration for [`BayesLstm`].
#[derive(Debug, Clone)]
pub struct BayesLstmConfig {
    /// Dimensionality of the input vector `x`.
    pub input_dim: usize,
    /// Dimensionality of the hidden / cell state `h`, `c`.
    pub hidden_dim: usize,
}

// ─── Variational parameters ──────────────────────────────────────────────────

/// All variational parameters (`μ`, `log σ`) for the four LSTM gates.
///
/// Every gate owns an input-to-hidden matrix `[hidden × input]`, a
/// hidden-to-hidden matrix `[hidden × hidden]` and a bias `[hidden]`.  Each
/// tensor is stored as a `_mu` (mean) and a `_log_sigma`
/// (`σ = exp(log_sigma)`) component.  Matrices are row-major:
/// `index(row, col) = row * cols + col`.
#[derive(Debug, Clone)]
pub struct BayesLstmWeights {
    // ── Input gate: i = σ(W_ii·x + W_hi·h + b_i) ────────────────────────────
    /// Input-to-hidden mean for the input gate, shape `[hidden × input]`.
    pub w_ii_mu: Vec<f32>,
    /// Input-to-hidden log-σ for the input gate, shape `[hidden × input]`.
    pub w_ii_log_sigma: Vec<f32>,
    /// Hidden-to-hidden mean for the input gate, shape `[hidden × hidden]`.
    pub w_hi_mu: Vec<f32>,
    /// Hidden-to-hidden log-σ for the input gate, shape `[hidden × hidden]`.
    pub w_hi_log_sigma: Vec<f32>,
    /// Bias mean for the input gate, shape `[hidden]`.
    pub b_i_mu: Vec<f32>,
    /// Bias log-σ for the input gate, shape `[hidden]`.
    pub b_i_log_sigma: Vec<f32>,

    // ── Forget gate: f = σ(W_if·x + W_hf·h + b_f) ───────────────────────────
    /// Input-to-hidden mean for the forget gate, shape `[hidden × input]`.
    pub w_if_mu: Vec<f32>,
    /// Input-to-hidden log-σ for the forget gate, shape `[hidden × input]`.
    pub w_if_log_sigma: Vec<f32>,
    /// Hidden-to-hidden mean for the forget gate, shape `[hidden × hidden]`.
    pub w_hf_mu: Vec<f32>,
    /// Hidden-to-hidden log-σ for the forget gate, shape `[hidden × hidden]`.
    pub w_hf_log_sigma: Vec<f32>,
    /// Bias mean for the forget gate, shape `[hidden]`.
    pub b_f_mu: Vec<f32>,
    /// Bias log-σ for the forget gate, shape `[hidden]`.
    pub b_f_log_sigma: Vec<f32>,

    // ── Cell candidate: g = tanh(W_ig·x + W_hg·h + b_g) ─────────────────────
    /// Input-to-hidden mean for the cell candidate, shape `[hidden × input]`.
    pub w_ig_mu: Vec<f32>,
    /// Input-to-hidden log-σ for the cell candidate, shape `[hidden × input]`.
    pub w_ig_log_sigma: Vec<f32>,
    /// Hidden-to-hidden mean for the cell candidate, shape `[hidden × hidden]`.
    pub w_hg_mu: Vec<f32>,
    /// Hidden-to-hidden log-σ for the cell candidate, shape `[hidden × hidden]`.
    pub w_hg_log_sigma: Vec<f32>,
    /// Bias mean for the cell candidate, shape `[hidden]`.
    pub b_g_mu: Vec<f32>,
    /// Bias log-σ for the cell candidate, shape `[hidden]`.
    pub b_g_log_sigma: Vec<f32>,

    // ── Output gate: o = σ(W_io·x + W_ho·h + b_o) ───────────────────────────
    /// Input-to-hidden mean for the output gate, shape `[hidden × input]`.
    pub w_io_mu: Vec<f32>,
    /// Input-to-hidden log-σ for the output gate, shape `[hidden × input]`.
    pub w_io_log_sigma: Vec<f32>,
    /// Hidden-to-hidden mean for the output gate, shape `[hidden × hidden]`.
    pub w_ho_mu: Vec<f32>,
    /// Hidden-to-hidden log-σ for the output gate, shape `[hidden × hidden]`.
    pub w_ho_log_sigma: Vec<f32>,
    /// Bias mean for the output gate, shape `[hidden]`.
    pub b_o_mu: Vec<f32>,
    /// Bias log-σ for the output gate, shape `[hidden]`.
    pub b_o_log_sigma: Vec<f32>,
}

// ─── Sampled weights ─────────────────────────────────────────────────────────

/// A concrete realisation `w ~ q(w)` of every LSTM weight, produced by
/// [`BayesLstm::sample_weights`] and consumed by
/// [`BayesLstm::forward_step`].
#[derive(Debug, Clone)]
pub struct BayesLstmSampledWeights {
    /// Input gate, input-to-hidden `[hidden × input]`.
    pub w_ii: Vec<f32>,
    /// Input gate, hidden-to-hidden `[hidden × hidden]`.
    pub w_hi: Vec<f32>,
    /// Input gate bias `[hidden]`.
    pub b_i: Vec<f32>,
    /// Forget gate, input-to-hidden `[hidden × input]`.
    pub w_if: Vec<f32>,
    /// Forget gate, hidden-to-hidden `[hidden × hidden]`.
    pub w_hf: Vec<f32>,
    /// Forget gate bias `[hidden]`.
    pub b_f: Vec<f32>,
    /// Cell candidate, input-to-hidden `[hidden × input]`.
    pub w_ig: Vec<f32>,
    /// Cell candidate, hidden-to-hidden `[hidden × hidden]`.
    pub w_hg: Vec<f32>,
    /// Cell candidate bias `[hidden]`.
    pub b_g: Vec<f32>,
    /// Output gate, input-to-hidden `[hidden × input]`.
    pub w_io: Vec<f32>,
    /// Output gate, hidden-to-hidden `[hidden × hidden]`.
    pub w_ho: Vec<f32>,
    /// Output gate bias `[hidden]`.
    pub b_o: Vec<f32>,
}

// ─── Main struct ─────────────────────────────────────────────────────────────

/// Bayesian Long Short-Term Memory cell using Bayes-by-Backprop (BBB).
///
/// # Usage
/// ```text
/// let mut rng = LcgRng::new(42);
/// let lstm = BayesLstm::new(BayesLstmConfig { input_dim: 4, hidden_dim: 8 }, &mut rng)?;
/// let ws = lstm.sample_weights(&mut rng);
/// let (h0, c0) = (vec![0.0; 8], vec![0.0; 8]);
/// let (h1, c1) = lstm.forward_step(&x, &h0, &c0, &ws)?;
/// let kl = lstm.kl_divergence();
/// ```
#[derive(Debug, Clone)]
pub struct BayesLstm {
    /// Configuration (sizes).
    pub cfg: BayesLstmConfig,
    /// Variational parameters for all four gates.
    pub weights: BayesLstmWeights,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Sample a full tensor `w = μ + exp(log_σ)·ε`, `ε ~ N(0,1)`.
fn sample_tensor(mu: &[f32], log_sigma: &[f32], rng: &mut LcgRng) -> Vec<f32> {
    mu.iter()
        .zip(log_sigma.iter())
        .map(|(&m, &ls)| {
            let sigma = ls.exp();
            let (eps, _) = rng.next_normal_pair();
            m + sigma * eps
        })
        .collect()
}

/// Matrix-vector product `out[r] = Σ_c M[r, c]·v[c]` for a row-major
/// `[rows × cols]` matrix.
fn mv(mat: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let off = r * cols;
            mat[off..off + cols]
                .iter()
                .zip(v.iter().take(cols))
                .map(|(&m, &x)| m * x)
                .sum()
        })
        .collect()
}

/// Per-weight KL `½ (μ² + σ² − log σ² − 1)` with `σ = exp(log_sigma)`,
/// summed over a `(μ, log σ)` tensor pair against the `N(0, 1)` prior.
fn kl_tensor(mu: &[f32], log_sigma: &[f32]) -> f32 {
    mu.iter()
        .zip(log_sigma.iter())
        .map(|(&m, &ls)| {
            let sigma_sq = (2.0 * ls).exp();
            0.5 * (m * m + sigma_sq - 2.0 * ls - 1.0)
        })
        .sum()
}

impl BayesLstm {
    /// Construct a new `BayesLstm` with small-variance weight initialisation.
    ///
    /// - `w_mu  ~ N(0, scale)` with `scale = 0.1 / sqrt(input + hidden)`
    /// - `w_log_sigma = -3.0` (so `σ = exp(-3) ≈ 0.0498`)
    /// - `b_mu = 0.0`, `b_log_sigma = -3.0`
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] — `input_dim == 0`.
    /// - [`BayesError::InsufficientSamples`] — `hidden_dim == 0`.
    pub fn new(cfg: BayesLstmConfig, rng: &mut LcgRng) -> BayesResult<Self> {
        if cfg.input_dim == 0 {
            return Err(BayesError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.hidden_dim == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }

        let h = cfg.hidden_dim;
        let i = cfg.input_dim;
        let scale = 0.1_f32 / ((i + h) as f32).sqrt();

        let make_mat_mu = |rows: usize, cols: usize, rng: &mut LcgRng| -> Vec<f32> {
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in v.iter_mut() {
                *x *= scale;
            }
            v
        };
        let make_log_sigma = |len: usize| vec![-3.0_f32; len];
        let make_bias_mu = |len: usize| vec![0.0_f32; len];

        let weights = BayesLstmWeights {
            // Input gate
            w_ii_mu: make_mat_mu(h, i, rng),
            w_ii_log_sigma: make_log_sigma(h * i),
            w_hi_mu: make_mat_mu(h, h, rng),
            w_hi_log_sigma: make_log_sigma(h * h),
            b_i_mu: make_bias_mu(h),
            b_i_log_sigma: make_log_sigma(h),
            // Forget gate
            w_if_mu: make_mat_mu(h, i, rng),
            w_if_log_sigma: make_log_sigma(h * i),
            w_hf_mu: make_mat_mu(h, h, rng),
            w_hf_log_sigma: make_log_sigma(h * h),
            b_f_mu: make_bias_mu(h),
            b_f_log_sigma: make_log_sigma(h),
            // Cell candidate
            w_ig_mu: make_mat_mu(h, i, rng),
            w_ig_log_sigma: make_log_sigma(h * i),
            w_hg_mu: make_mat_mu(h, h, rng),
            w_hg_log_sigma: make_log_sigma(h * h),
            b_g_mu: make_bias_mu(h),
            b_g_log_sigma: make_log_sigma(h),
            // Output gate
            w_io_mu: make_mat_mu(h, i, rng),
            w_io_log_sigma: make_log_sigma(h * i),
            w_ho_mu: make_mat_mu(h, h, rng),
            w_ho_log_sigma: make_log_sigma(h * h),
            b_o_mu: make_bias_mu(h),
            b_o_log_sigma: make_log_sigma(h),
        };

        Ok(Self { cfg, weights })
    }

    /// Draw a single realisation `w ~ q(w)` of every weight via the
    /// reparameterisation trick.
    ///
    /// The returned set is shared across all time steps of one sequence
    /// (Fortunato et al. 2017).
    #[must_use]
    pub fn sample_weights(&self, rng: &mut LcgRng) -> BayesLstmSampledWeights {
        let w = &self.weights;
        BayesLstmSampledWeights {
            w_ii: sample_tensor(&w.w_ii_mu, &w.w_ii_log_sigma, rng),
            w_hi: sample_tensor(&w.w_hi_mu, &w.w_hi_log_sigma, rng),
            b_i: sample_tensor(&w.b_i_mu, &w.b_i_log_sigma, rng),
            w_if: sample_tensor(&w.w_if_mu, &w.w_if_log_sigma, rng),
            w_hf: sample_tensor(&w.w_hf_mu, &w.w_hf_log_sigma, rng),
            b_f: sample_tensor(&w.b_f_mu, &w.b_f_log_sigma, rng),
            w_ig: sample_tensor(&w.w_ig_mu, &w.w_ig_log_sigma, rng),
            w_hg: sample_tensor(&w.w_hg_mu, &w.w_hg_log_sigma, rng),
            b_g: sample_tensor(&w.b_g_mu, &w.b_g_log_sigma, rng),
            w_io: sample_tensor(&w.w_io_mu, &w.w_io_log_sigma, rng),
            w_ho: sample_tensor(&w.w_ho_mu, &w.w_ho_log_sigma, rng),
            b_o: sample_tensor(&w.b_o_mu, &w.b_o_log_sigma, rng),
        }
    }

    /// Single LSTM cell step using a pre-sampled weight set.
    ///
    /// Returns `(h', c')`, the new hidden and cell states.  Because
    /// `h' = o ⊙ tanh(c')` with `o ∈ (0, 1)` and `tanh(c') ∈ (−1, 1)`, every
    /// component of `h'` lies strictly in `(−1, 1)`.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `x.len() != input_dim`,
    /// `h.len() != hidden_dim`, or `c.len() != hidden_dim`.
    pub fn forward_step(
        &self,
        x: &[f32],
        h: &[f32],
        c: &[f32],
        ws: &BayesLstmSampledWeights,
    ) -> BayesResult<(Vec<f32>, Vec<f32>)> {
        let (h_sz, i_sz) = (self.cfg.hidden_dim, self.cfg.input_dim);
        if x.len() != i_sz {
            return Err(BayesError::DimensionMismatch {
                expected: i_sz,
                got: x.len(),
            });
        }
        if h.len() != h_sz {
            return Err(BayesError::DimensionMismatch {
                expected: h_sz,
                got: h.len(),
            });
        }
        if c.len() != h_sz {
            return Err(BayesError::DimensionMismatch {
                expected: h_sz,
                got: c.len(),
            });
        }

        // Gate pre-activations: W_i·x + W_h·h + b.
        let gate = |w_i: &[f32], w_h: &[f32], b: &[f32]| -> Vec<f32> {
            let ix = mv(w_i, x, h_sz, i_sz);
            let hh = mv(w_h, h, h_sz, h_sz);
            ix.iter()
                .zip(hh.iter())
                .zip(b.iter())
                .map(|((&a, &d), &bb)| a + d + bb)
                .collect()
        };

        let i_gate: Vec<f32> = gate(&ws.w_ii, &ws.w_hi, &ws.b_i)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();
        let f_gate: Vec<f32> = gate(&ws.w_if, &ws.w_hf, &ws.b_f)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();
        let g_gate: Vec<f32> = gate(&ws.w_ig, &ws.w_hg, &ws.b_g)
            .iter()
            .map(|&v| v.tanh())
            .collect();
        let o_gate: Vec<f32> = gate(&ws.w_io, &ws.w_ho, &ws.b_o)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();

        // c' = f ⊙ c + i ⊙ g
        let c_new: Vec<f32> = f_gate
            .iter()
            .zip(c.iter())
            .zip(i_gate.iter().zip(g_gate.iter()))
            .map(|((&fg, &cv), (&ig, &gg))| fg * cv + ig * gg)
            .collect();

        // h' = o ⊙ tanh(c')
        let h_new: Vec<f32> = o_gate
            .iter()
            .zip(c_new.iter())
            .map(|(&og, &cv)| og * cv.tanh())
            .collect();

        Ok((h_new, c_new))
    }

    /// Unroll the cell over a flat input sequence with one shared weight sample.
    ///
    /// `xs` is the row-major `[seq_len × input_dim]` sequence.  The hidden and
    /// cell states start at zero.  Returns the concatenated hidden states, a
    /// flat `[seq_len × hidden_dim]` buffer (`output[t·hidden + j]`).
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `xs.len() != seq_len * input_dim`.
    pub fn forward_seq(
        &self,
        xs: &[f32],
        seq_len: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Vec<f32>> {
        let (h_sz, i_sz) = (self.cfg.hidden_dim, self.cfg.input_dim);
        if xs.len() != seq_len * i_sz {
            return Err(BayesError::DimensionMismatch {
                expected: seq_len * i_sz,
                got: xs.len(),
            });
        }

        // One weight realisation shared across the whole sequence.
        let ws = self.sample_weights(rng);

        let mut h = vec![0.0_f32; h_sz];
        let mut c = vec![0.0_f32; h_sz];
        let mut out = Vec::with_capacity(seq_len * h_sz);

        for t in 0..seq_len {
            let x_t = &xs[t * i_sz..(t + 1) * i_sz];
            let (h_new, c_new) = self.forward_step(x_t, &h, &c, &ws)?;
            out.extend_from_slice(&h_new);
            h = h_new;
            c = c_new;
        }

        Ok(out)
    }

    /// Total KL divergence `Σ ½(μ² + σ² − log σ² − 1)` over all weights,
    /// against the isotropic `N(0, 1)` prior.
    #[must_use]
    pub fn kl_divergence(&self) -> f32 {
        let w = &self.weights;
        let mut kl = 0.0_f32;
        // 8 weight matrices (input-to-hidden + hidden-to-hidden) × 4 gates.
        kl += kl_tensor(&w.w_ii_mu, &w.w_ii_log_sigma);
        kl += kl_tensor(&w.w_hi_mu, &w.w_hi_log_sigma);
        kl += kl_tensor(&w.w_if_mu, &w.w_if_log_sigma);
        kl += kl_tensor(&w.w_hf_mu, &w.w_hf_log_sigma);
        kl += kl_tensor(&w.w_ig_mu, &w.w_ig_log_sigma);
        kl += kl_tensor(&w.w_hg_mu, &w.w_hg_log_sigma);
        kl += kl_tensor(&w.w_io_mu, &w.w_io_log_sigma);
        kl += kl_tensor(&w.w_ho_mu, &w.w_ho_log_sigma);
        // 4 bias vectors.
        kl += kl_tensor(&w.b_i_mu, &w.b_i_log_sigma);
        kl += kl_tensor(&w.b_f_mu, &w.b_f_log_sigma);
        kl += kl_tensor(&w.b_g_mu, &w.b_g_log_sigma);
        kl += kl_tensor(&w.b_o_mu, &w.b_o_log_sigma);
        kl
    }

    /// Total number of variational parameters (`μ` + `log σ` per weight/bias).
    ///
    /// Formula: `2 × (8·input·hidden + 8·hidden² + 8·hidden)`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let i = self.cfg.input_dim;
        let h = self.cfg.hidden_dim;
        2 * (8 * i * h + 8 * h * h + 8 * h)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> BayesLstmConfig {
        BayesLstmConfig {
            input_dim: 3,
            hidden_dim: 4,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_succeeds_with_valid_config() {
        let mut rng = make_rng();
        assert!(BayesLstm::new(small_cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_fails_with_zero_input_dim() {
        let mut rng = make_rng();
        let cfg = BayesLstmConfig {
            input_dim: 0,
            hidden_dim: 4,
        };
        let r = BayesLstm::new(cfg, &mut rng);
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn new_fails_with_zero_hidden_dim() {
        let mut rng = make_rng();
        let cfg = BayesLstmConfig {
            input_dim: 3,
            hidden_dim: 0,
        };
        let r = BayesLstm::new(cfg, &mut rng);
        assert!(
            matches!(r, Err(BayesError::InsufficientSamples { .. })),
            "got {r:?}"
        );
    }

    // ── forward_step shapes ──────────────────────────────────────────────────

    #[test]
    fn forward_step_output_shapes() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let ws = lstm.sample_weights(&mut rng);
        let h = vec![0.0_f32; small_cfg().hidden_dim];
        let c = vec![0.0_f32; small_cfg().hidden_dim];
        let x = vec![0.5_f32; small_cfg().input_dim];
        let (h_t, c_t) = lstm.forward_step(&x, &h, &c, &ws).expect("step");
        assert_eq!(h_t.len(), small_cfg().hidden_dim);
        assert_eq!(c_t.len(), small_cfg().hidden_dim);
    }

    // ── h_t in (-1, 1) and finite ────────────────────────────────────────────

    #[test]
    fn forward_step_h_in_open_minus_one_one_and_finite() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let ws = lstm.sample_weights(&mut rng);
        // Start from a non-trivial cell state to exercise the tanh squashing.
        let h = vec![0.2_f32; small_cfg().hidden_dim];
        let c = vec![3.0_f32; small_cfg().hidden_dim];
        let x = vec![1.0_f32; small_cfg().input_dim];
        let (h_t, c_t) = lstm.forward_step(&x, &h, &c, &ws).expect("step");
        for &v in &h_t {
            assert!(v.is_finite(), "h component must be finite, got {v}");
            assert!((-1.0..1.0).contains(&v), "h component {v} not in (-1, 1)");
        }
        for &v in &c_t {
            assert!(v.is_finite(), "c component must be finite, got {v}");
        }
    }

    // ── determinism & stochasticity ──────────────────────────────────────────

    #[test]
    fn same_seed_same_weights_deterministic() {
        let mut rng_build = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng_build).expect("new");
        let x = vec![0.4_f32; small_cfg().input_dim];
        let h = vec![0.0_f32; small_cfg().hidden_dim];
        let c = vec![0.0_f32; small_cfg().hidden_dim];

        let mut rng_a = LcgRng::new(7);
        let mut rng_b = LcgRng::new(7);
        let ws_a = lstm.sample_weights(&mut rng_a);
        let ws_b = lstm.sample_weights(&mut rng_b);
        let (h_a, _) = lstm.forward_step(&x, &h, &c, &ws_a).expect("step");
        let (h_b, _) = lstm.forward_step(&x, &h, &c, &ws_b).expect("step");
        for (a, b) in h_a.iter().zip(h_b.iter()) {
            assert!((a - b).abs() < 1e-9, "same seed must be deterministic");
        }
    }

    #[test]
    fn different_seed_different_output() {
        let mut rng_build = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng_build).expect("new");
        let x = vec![0.4_f32; small_cfg().input_dim];
        let h = vec![0.0_f32; small_cfg().hidden_dim];
        let c = vec![0.0_f32; small_cfg().hidden_dim];

        let mut rng_a = LcgRng::new(1);
        let mut rng_b = LcgRng::new(987_654);
        let ws_a = lstm.sample_weights(&mut rng_a);
        let ws_b = lstm.sample_weights(&mut rng_b);
        let (h_a, _) = lstm.forward_step(&x, &h, &c, &ws_a).expect("step");
        let (h_b, _) = lstm.forward_step(&x, &h, &c, &ws_b).expect("step");
        assert!(
            h_a.iter()
                .zip(h_b.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "different seeds must produce different outputs"
        );
    }

    // ── KL divergence ────────────────────────────────────────────────────────

    #[test]
    fn kl_divergence_non_negative_default_init() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let kl = lstm.kl_divergence();
        assert!(kl >= 0.0 && kl.is_finite(), "KL = {kl}");
    }

    #[test]
    fn kl_divergence_positive_for_nonzero_mu() {
        // Build a layer then set σ = 1 (log σ = 0) everywhere and one μ = 2.
        // Then KL = ½ μ² = 2 exactly (all other terms vanish at σ = 1, μ = 0).
        let mut rng = make_rng();
        let mut lstm = BayesLstm::new(
            BayesLstmConfig {
                input_dim: 1,
                hidden_dim: 1,
            },
            &mut rng,
        )
        .expect("new");
        let w = &mut lstm.weights;
        for t in [
            &mut w.w_ii_mu,
            &mut w.w_hi_mu,
            &mut w.b_i_mu,
            &mut w.w_if_mu,
            &mut w.w_hf_mu,
            &mut w.b_f_mu,
            &mut w.w_ig_mu,
            &mut w.w_hg_mu,
            &mut w.b_g_mu,
            &mut w.w_io_mu,
            &mut w.w_ho_mu,
            &mut w.b_o_mu,
        ] {
            for v in t.iter_mut() {
                *v = 0.0;
            }
        }
        for t in [
            &mut w.w_ii_log_sigma,
            &mut w.w_hi_log_sigma,
            &mut w.b_i_log_sigma,
            &mut w.w_if_log_sigma,
            &mut w.w_hf_log_sigma,
            &mut w.b_f_log_sigma,
            &mut w.w_ig_log_sigma,
            &mut w.w_hg_log_sigma,
            &mut w.b_g_log_sigma,
            &mut w.w_io_log_sigma,
            &mut w.w_ho_log_sigma,
            &mut w.b_o_log_sigma,
        ] {
            for v in t.iter_mut() {
                *v = 0.0;
            }
        }
        // Now KL == 0 with all μ = 0, σ = 1.
        assert!(lstm.kl_divergence().abs() < 1e-5);
        // Set one μ = 2 ⇒ KL = ½·4 = 2.
        lstm.weights.w_ii_mu[0] = 2.0;
        let kl = lstm.kl_divergence();
        assert!(kl > 0.0, "KL must be > 0 for non-zero μ, got {kl}");
        assert!((kl - 2.0).abs() < 1e-4, "expected KL = 2, got {kl}");
    }

    // ── forward_seq ──────────────────────────────────────────────────────────

    #[test]
    fn forward_seq_output_shape() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let seq_len = 5;
        let xs = vec![0.1_f32; seq_len * small_cfg().input_dim];
        let out = lstm.forward_seq(&xs, seq_len, &mut rng).expect("seq");
        assert_eq!(out.len(), seq_len * small_cfg().hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_seq_deterministic_same_seed() {
        let mut rng_build = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng_build).expect("new");
        let seq_len = 4;
        let xs = vec![0.3_f32; seq_len * small_cfg().input_dim];
        let out_a = lstm
            .forward_seq(&xs, seq_len, &mut LcgRng::new(11))
            .expect("seq");
        let out_b = lstm
            .forward_seq(&xs, seq_len, &mut LcgRng::new(11))
            .expect("seq");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    // ── error paths ──────────────────────────────────────────────────────────

    #[test]
    fn forward_step_dim_mismatch_on_x() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let ws = lstm.sample_weights(&mut rng);
        let h = vec![0.0_f32; small_cfg().hidden_dim];
        let c = vec![0.0_f32; small_cfg().hidden_dim];
        let x = vec![0.5_f32; small_cfg().input_dim + 1];
        let r = lstm.forward_step(&x, &h, &c, &ws);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_step_dim_mismatch_on_c() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let ws = lstm.sample_weights(&mut rng);
        let h = vec![0.0_f32; small_cfg().hidden_dim];
        let c = vec![0.0_f32; small_cfg().hidden_dim + 2];
        let x = vec![0.5_f32; small_cfg().input_dim];
        let r = lstm.forward_step(&x, &h, &c, &ws);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_seq_dim_mismatch() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let seq_len = 4;
        // Wrong length: not seq_len * input_dim.
        let xs = vec![0.1_f32; seq_len * small_cfg().input_dim + 1];
        let r = lstm.forward_seq(&xs, seq_len, &mut rng);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn n_params_formula_is_correct() {
        let mut rng = make_rng();
        let lstm = BayesLstm::new(small_cfg(), &mut rng).expect("new");
        let expected = 2 * (8 * 3 * 4 + 8 * 4 * 4 + 8 * 4);
        assert_eq!(lstm.n_params(), expected);
    }
}
