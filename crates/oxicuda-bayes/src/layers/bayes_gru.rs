//! Bayesian GRU cell via Bayes-by-Backprop (BBB).
//!
//! Each weight matrix in the three GRU gates (reset `r`, update `z`, new `n`)
//! carries a learned Gaussian posterior
//! `q(W) = N(W_mu, softplus(W_rho)²)` (Blundell et al. 2015).
//!
//! At sampling time weights are drawn from `q`; the KL divergence to the
//! isotropic Gaussian prior `p(W) = N(0, prior_sigma²)` is accumulated and
//! returned so that callers can minimise the ELBO.
//!
//! The GRU update equations follow Cho et al. (2014):
//! ```text
//! r  = σ(W_ir·x + W_hr·h + b_r)         (reset gate)
//! z  = σ(W_iz·x + W_hz·h + b_z)         (update gate)
//! n  = tanh(W_in·x + W_hn·(r ⊙ h) + b_n) (new gate)
//! h' = (1 - z) ⊙ n + z ⊙ h             (output hidden state)
//! ```
//!
//! **References:**
//! - Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
//!   Weight Uncertainty in Neural Networks. *ICML 2015*.
//! - Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
//!   Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using
//!   RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP 2014*.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;
use crate::layers::bayes_linear::softplus;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Constructor configuration for [`BayesGru`].
#[derive(Debug, Clone)]
pub struct BayesGruConfig {
    /// Dimensionality of the input vector `x`.
    pub input_size: usize,
    /// Dimensionality of the hidden state `h`.
    pub hidden_size: usize,
    /// Standard deviation of the isotropic Gaussian prior on all weights.
    pub prior_sigma: f32,
}

// ─── Weight container ────────────────────────────────────────────────────────

/// All variational parameters for the three GRU gates.
///
/// Each gate has two weight matrices (input-to-hidden and hidden-to-hidden)
/// plus a bias vector.  Every matrix / vector is stored with a `_mu` (mean)
/// and `_rho` (rho, where `σ = softplus(ρ)`) component.
///
/// Memory layout of the `[hidden × in]` matrices: row-major, i.e.
/// `index(row, col) = row * width + col`.
#[derive(Debug, Clone)]
pub struct BayesGruWeights {
    // ── Reset gate: r = σ(W_ir·x + W_hr·h + b_r) ────────────────────────────
    /// Input-to-hidden weight mean for reset gate, shape `[hidden × input]`.
    pub w_ir_mu: Vec<f32>,
    /// Input-to-hidden weight rho for reset gate, shape `[hidden × input]`.
    pub w_ir_rho: Vec<f32>,
    /// Hidden-to-hidden weight mean for reset gate, shape `[hidden × hidden]`.
    pub w_hr_mu: Vec<f32>,
    /// Hidden-to-hidden weight rho for reset gate, shape `[hidden × hidden]`.
    pub w_hr_rho: Vec<f32>,
    /// Bias mean for reset gate, shape `[hidden]`.
    pub b_r_mu: Vec<f32>,
    /// Bias rho for reset gate, shape `[hidden]`.
    pub b_r_rho: Vec<f32>,

    // ── Update gate: z = σ(W_iz·x + W_hz·h + b_z) ───────────────────────────
    /// Input-to-hidden weight mean for update gate, shape `[hidden × input]`.
    pub w_iz_mu: Vec<f32>,
    /// Input-to-hidden weight rho for update gate, shape `[hidden × input]`.
    pub w_iz_rho: Vec<f32>,
    /// Hidden-to-hidden weight mean for update gate, shape `[hidden × hidden]`.
    pub w_hz_mu: Vec<f32>,
    /// Hidden-to-hidden weight rho for update gate, shape `[hidden × hidden]`.
    pub w_hz_rho: Vec<f32>,
    /// Bias mean for update gate, shape `[hidden]`.
    pub b_z_mu: Vec<f32>,
    /// Bias rho for update gate, shape `[hidden]`.
    pub b_z_rho: Vec<f32>,

    // ── New gate: n = tanh(W_in·x + W_hn·(r⊙h) + b_n) ──────────────────────
    /// Input-to-hidden weight mean for new gate, shape `[hidden × input]`.
    pub w_in_mu: Vec<f32>,
    /// Input-to-hidden weight rho for new gate, shape `[hidden × input]`.
    pub w_in_rho: Vec<f32>,
    /// Hidden-to-hidden weight mean for new gate, shape `[hidden × hidden]`.
    pub w_hn_mu: Vec<f32>,
    /// Hidden-to-hidden weight rho for new gate, shape `[hidden × hidden]`.
    pub w_hn_rho: Vec<f32>,
    /// Bias mean for new gate, shape `[hidden]`.
    pub b_n_mu: Vec<f32>,
    /// Bias rho for new gate, shape `[hidden]`.
    pub b_n_rho: Vec<f32>,
}

// ─── Hidden state ────────────────────────────────────────────────────────────

/// GRU hidden state.
#[derive(Debug, Clone)]
pub struct BayesGruState {
    /// Hidden-state vector, length = `hidden_size`.
    pub h: Vec<f32>,
}

// ─── Main struct ─────────────────────────────────────────────────────────────

/// Bayesian Gated Recurrent Unit cell using Bayes-by-Backprop (BBB).
///
/// # Usage
/// ```text
/// let mut rng = LcgRng::new(42);
/// let gru = BayesGru::new(BayesGruConfig { input_size: 4, hidden_size: 8, prior_sigma: 1.0 }, &mut rng)?;
/// let state = gru.zero_state();
/// let (new_state, kl) = gru.forward_sample(&x, &state, &mut rng)?;
/// ```
#[derive(Debug, Clone)]
pub struct BayesGru {
    /// Configuration (sizes and prior).
    pub cfg: BayesGruConfig,
    /// Variational parameters for all gates.
    pub weights: BayesGruWeights,
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

/// Sample ε ~ N(0,1) from the `LcgRng` and return `mu + softplus(rho) * ε`.
#[inline]
fn sample_weight(mu: f32, rho: f32, rng: &mut LcgRng) -> f32 {
    let (eps, _) = rng.next_normal_pair();
    mu + softplus(rho) * eps
}

/// KL(N(mu, sigma²) ‖ N(0, prior_sigma²)) — analytic formula.
/// `= log(prior_sigma/sigma) + (sigma² + mu²)/(2·prior_sigma²) - 0.5`
#[inline]
fn kl_single(mu: f32, sigma: f32, prior_sigma: f32) -> f32 {
    let log_ratio = prior_sigma.ln() - sigma.ln();
    let sigma_sq = sigma * sigma;
    let prior_sq = prior_sigma * prior_sigma;
    log_ratio + (sigma_sq + mu * mu) / (2.0 * prior_sq) - 0.5
}

/// Matrix-vector multiply: `out[i] += M[i, :] · v[:]` where `M` is `[rows × cols]` row-major.
#[inline]
fn mv_add(out: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for (r, o) in out.iter_mut().take(rows).enumerate() {
        let row_off = r * cols;
        let acc: f32 = mat[row_off..row_off + cols]
            .iter()
            .zip(vec.iter().take(cols))
            .map(|(&m, &v)| m * v)
            .sum();
        *o += acc;
    }
}

/// Sample a matrix and compute its contribution to `out` and KL.
/// Returns the KL accumulated over all sampled weights.
fn sample_mat_mv(
    out: &mut [f32],
    mu: &[f32],
    rho: &[f32],
    vec: &[f32],
    rows: usize,
    cols: usize,
    prior_sigma: f32,
    rng: &mut LcgRng,
) -> f32 {
    let mut kl = 0.0_f32;
    // Row-major iteration: process one row at a time to stay cache-friendly.
    for (r, o) in out.iter_mut().take(rows).enumerate() {
        let row_off = r * cols;
        let mut acc = 0.0_f32;
        for (c, &vc) in vec.iter().take(cols).enumerate() {
            let idx = row_off + c;
            let w = sample_weight(mu[idx], rho[idx], rng);
            acc += w * vc;
            let sigma = softplus(rho[idx]);
            kl += kl_single(mu[idx], sigma, prior_sigma);
        }
        *o += acc;
    }
    kl
}

/// Sample a bias vector and accumulate into `out`; returns KL contribution.
fn sample_bias_add(
    out: &mut [f32],
    mu: &[f32],
    rho: &[f32],
    prior_sigma: f32,
    rng: &mut LcgRng,
) -> f32 {
    let mut kl = 0.0_f32;
    for (i, o) in out.iter_mut().enumerate() {
        let b = sample_weight(mu[i], rho[i], rng);
        *o += b;
        let sigma = softplus(rho[i]);
        kl += kl_single(mu[i], sigma, prior_sigma);
    }
    kl
}

/// Analytic KL for a whole (mu, rho) vector pair.
fn kl_vec(mu: &[f32], rho: &[f32], prior_sigma: f32) -> f32 {
    mu.iter()
        .zip(rho.iter())
        .map(|(&m, &r)| kl_single(m, softplus(r), prior_sigma))
        .sum()
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl BayesGru {
    /// Construct a new `BayesGru` with Kaiming-like weight initialisation.
    ///
    /// - `w_mu  ~ N(0, 0.1 / sqrt(input_size + hidden_size))`
    /// - `w_rho  = -3.0` (small initial posterior variance: `softplus(-3) ≈ 0.049`)
    /// - `b_mu  = 0.0`, `b_rho = -3.0`
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] — `input_size == 0` (expected ≥ 1, got 0).
    /// - [`BayesError::InsufficientSamples`] — `hidden_size == 0`.
    /// - [`BayesError::InvalidPriorVariance`] — `prior_sigma ≤ 0` or non-finite.
    pub fn new(cfg: BayesGruConfig, rng: &mut LcgRng) -> BayesResult<Self> {
        if cfg.input_size == 0 {
            return Err(BayesError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.hidden_size == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        if cfg.prior_sigma <= 0.0 || !cfg.prior_sigma.is_finite() {
            return Err(BayesError::InvalidPriorVariance);
        }

        let h = cfg.hidden_size;
        let i = cfg.input_size;
        let scale = 0.1_f32 / ((i + h) as f32).sqrt();

        // Helper: allocate [rows × cols] matrix with mu ~ N(0, scale).
        let make_mat_mu = |rows: usize, cols: usize, rng: &mut LcgRng| -> Vec<f32> {
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in v.iter_mut() {
                *x *= scale;
            }
            v
        };
        let make_rho = |len: usize| vec![-3.0_f32; len];
        let make_bias_mu = |len: usize| vec![0.0_f32; len];

        let weights = BayesGruWeights {
            // Reset gate
            w_ir_mu: make_mat_mu(h, i, rng),
            w_ir_rho: make_rho(h * i),
            w_hr_mu: make_mat_mu(h, h, rng),
            w_hr_rho: make_rho(h * h),
            b_r_mu: make_bias_mu(h),
            b_r_rho: make_rho(h),
            // Update gate
            w_iz_mu: make_mat_mu(h, i, rng),
            w_iz_rho: make_rho(h * i),
            w_hz_mu: make_mat_mu(h, h, rng),
            w_hz_rho: make_rho(h * h),
            b_z_mu: make_bias_mu(h),
            b_z_rho: make_rho(h),
            // New gate
            w_in_mu: make_mat_mu(h, i, rng),
            w_in_rho: make_rho(h * i),
            w_hn_mu: make_mat_mu(h, h, rng),
            w_hn_rho: make_rho(h * h),
            b_n_mu: make_bias_mu(h),
            b_n_rho: make_rho(h),
        };

        Ok(Self { cfg, weights })
    }

    // ─── Zero state ──────────────────────────────────────────────────────────

    /// Return an all-zero hidden state of length `hidden_size`.
    #[must_use]
    pub fn zero_state(&self) -> BayesGruState {
        BayesGruState {
            h: vec![0.0_f32; self.cfg.hidden_size],
        }
    }

    // ─── Stochastic forward ──────────────────────────────────────────────────

    /// Stochastic GRU cell forward pass: sample weights, compute `h'`, accumulate KL.
    ///
    /// Returns `(new_state, kl_contribution)` where `kl_contribution` is the
    /// KL divergence KL(q ‖ p) summed over all sampled weight matrices for this step.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `x.len() != input_size` or `state.h.len() != hidden_size`.
    pub fn forward_sample(
        &self,
        x: &[f32],
        state: &BayesGruState,
        rng: &mut LcgRng,
    ) -> BayesResult<(BayesGruState, f32)> {
        let (h_sz, i_sz) = (self.cfg.hidden_size, self.cfg.input_size);
        if x.len() != i_sz {
            return Err(BayesError::DimensionMismatch {
                expected: i_sz,
                got: x.len(),
            });
        }
        if state.h.len() != h_sz {
            return Err(BayesError::DimensionMismatch {
                expected: h_sz,
                got: state.h.len(),
            });
        }

        let prior = self.cfg.prior_sigma;
        let w = &self.weights;
        let h = &state.h;
        let mut total_kl = 0.0_f32;

        // ── Reset gate ───────────────────────────────────────────────────────
        let mut r_logit = vec![0.0_f32; h_sz];
        total_kl += sample_mat_mv(
            &mut r_logit,
            &w.w_ir_mu,
            &w.w_ir_rho,
            x,
            h_sz,
            i_sz,
            prior,
            rng,
        );
        total_kl += sample_mat_mv(
            &mut r_logit,
            &w.w_hr_mu,
            &w.w_hr_rho,
            h,
            h_sz,
            h_sz,
            prior,
            rng,
        );
        total_kl += sample_bias_add(&mut r_logit, &w.b_r_mu, &w.b_r_rho, prior, rng);
        let r_gate: Vec<f32> = r_logit.iter().map(|&v| sigmoid(v)).collect();

        // ── Update gate ──────────────────────────────────────────────────────
        let mut z_logit = vec![0.0_f32; h_sz];
        total_kl += sample_mat_mv(
            &mut z_logit,
            &w.w_iz_mu,
            &w.w_iz_rho,
            x,
            h_sz,
            i_sz,
            prior,
            rng,
        );
        total_kl += sample_mat_mv(
            &mut z_logit,
            &w.w_hz_mu,
            &w.w_hz_rho,
            h,
            h_sz,
            h_sz,
            prior,
            rng,
        );
        total_kl += sample_bias_add(&mut z_logit, &w.b_z_mu, &w.b_z_rho, prior, rng);
        let z_gate: Vec<f32> = z_logit.iter().map(|&v| sigmoid(v)).collect();

        // ── New gate ─────────────────────────────────────────────────────────
        // Hadamard product r ⊙ h
        let rh: Vec<f32> = r_gate
            .iter()
            .zip(h.iter())
            .map(|(&r, &hv)| r * hv)
            .collect();

        let mut n_logit = vec![0.0_f32; h_sz];
        total_kl += sample_mat_mv(
            &mut n_logit,
            &w.w_in_mu,
            &w.w_in_rho,
            x,
            h_sz,
            i_sz,
            prior,
            rng,
        );
        total_kl += sample_mat_mv(
            &mut n_logit,
            &w.w_hn_mu,
            &w.w_hn_rho,
            &rh,
            h_sz,
            h_sz,
            prior,
            rng,
        );
        total_kl += sample_bias_add(&mut n_logit, &w.b_n_mu, &w.b_n_rho, prior, rng);
        let n_gate: Vec<f32> = n_logit.iter().map(|&v| v.tanh()).collect();

        // ── Output hidden state ───────────────────────────────────────────────
        let h_new: Vec<f32> = z_gate
            .iter()
            .zip(n_gate.iter().zip(h.iter()))
            .map(|(&z, (&n, &hv))| (1.0 - z) * n + z * hv)
            .collect();

        Ok((BayesGruState { h: h_new }, total_kl))
    }

    // ─── Deterministic forward (inference) ───────────────────────────────────

    /// Mean-parameter GRU step: use `w_mu` / `b_mu` directly, no sampling.
    ///
    /// Suitable for test-time inference where the single best-guess output is required.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — size mismatch on `x` or `state`.
    pub fn forward_mean(&self, x: &[f32], state: &BayesGruState) -> BayesResult<BayesGruState> {
        let (h_sz, i_sz) = (self.cfg.hidden_size, self.cfg.input_size);
        if x.len() != i_sz {
            return Err(BayesError::DimensionMismatch {
                expected: i_sz,
                got: x.len(),
            });
        }
        if state.h.len() != h_sz {
            return Err(BayesError::DimensionMismatch {
                expected: h_sz,
                got: state.h.len(),
            });
        }

        let w = &self.weights;
        let h = &state.h;

        // ── Reset gate ───────────────────────────────────────────────────────
        let mut r_logit = vec![0.0_f32; h_sz];
        mv_add(&mut r_logit, &w.w_ir_mu, x, h_sz, i_sz);
        mv_add(&mut r_logit, &w.w_hr_mu, h, h_sz, h_sz);
        for (v, &b) in r_logit.iter_mut().zip(w.b_r_mu.iter()) {
            *v += b;
        }
        let r_gate: Vec<f32> = r_logit.iter().map(|&v| sigmoid(v)).collect();

        // ── Update gate ──────────────────────────────────────────────────────
        let mut z_logit = vec![0.0_f32; h_sz];
        mv_add(&mut z_logit, &w.w_iz_mu, x, h_sz, i_sz);
        mv_add(&mut z_logit, &w.w_hz_mu, h, h_sz, h_sz);
        for (v, &b) in z_logit.iter_mut().zip(w.b_z_mu.iter()) {
            *v += b;
        }
        let z_gate: Vec<f32> = z_logit.iter().map(|&v| sigmoid(v)).collect();

        // ── New gate ─────────────────────────────────────────────────────────
        let rh: Vec<f32> = r_gate
            .iter()
            .zip(h.iter())
            .map(|(&r, &hv)| r * hv)
            .collect();

        let mut n_logit = vec![0.0_f32; h_sz];
        mv_add(&mut n_logit, &w.w_in_mu, x, h_sz, i_sz);
        mv_add(&mut n_logit, &w.w_hn_mu, &rh, h_sz, h_sz);
        for (v, &b) in n_logit.iter_mut().zip(w.b_n_mu.iter()) {
            *v += b;
        }
        let n_gate: Vec<f32> = n_logit.iter().map(|&v| v.tanh()).collect();

        // ── Output ───────────────────────────────────────────────────────────
        let h_new: Vec<f32> = z_gate
            .iter()
            .zip(n_gate.iter().zip(h.iter()))
            .map(|(&z, (&n, &hv))| (1.0 - z) * n + z * hv)
            .collect();

        Ok(BayesGruState { h: h_new })
    }

    // ─── Analytic KL ─────────────────────────────────────────────────────────

    /// Total KL divergence KL(q ‖ p) summed analytically over all 18 parameter tensors.
    ///
    /// Uses the closed-form: `KL(N(μ,σ²) ‖ N(0,σ_p²)) = log(σ_p/σ) + (σ²+μ²)/(2σ_p²) - 0.5`.
    #[must_use]
    pub fn kl_divergence(&self) -> f32 {
        let p = self.cfg.prior_sigma;
        let w = &self.weights;
        let mut kl = 0.0_f32;
        // 6 weight matrix pairs (input-to-hidden + hidden-to-hidden) × 3 gates
        kl += kl_vec(&w.w_ir_mu, &w.w_ir_rho, p);
        kl += kl_vec(&w.w_hr_mu, &w.w_hr_rho, p);
        kl += kl_vec(&w.w_iz_mu, &w.w_iz_rho, p);
        kl += kl_vec(&w.w_hz_mu, &w.w_hz_rho, p);
        kl += kl_vec(&w.w_in_mu, &w.w_in_rho, p);
        kl += kl_vec(&w.w_hn_mu, &w.w_hn_rho, p);
        // 6 bias pairs
        kl += kl_vec(&w.b_r_mu, &w.b_r_rho, p);
        kl += kl_vec(&w.b_z_mu, &w.b_z_rho, p);
        kl += kl_vec(&w.b_n_mu, &w.b_n_rho, p);
        kl
    }

    // ─── Sequence unrolling ───────────────────────────────────────────────────

    /// Unroll the GRU over a sequence of `T` inputs, returning the hidden state
    /// at each time step and the cumulative KL divergence.
    ///
    /// `xs[t]` must have length `input_size`.
    ///
    /// Returns `(hidden_states, total_kl)` where `hidden_states[t]` has length
    /// `hidden_size`.  An empty sequence produces an empty output with `kl = 0`.
    ///
    /// # Errors
    /// Propagates [`BayesError::DimensionMismatch`] from [`Self::forward_sample`].
    pub fn forward_sequence_sample(
        &self,
        xs: &[Vec<f32>],
        rng: &mut LcgRng,
    ) -> BayesResult<(Vec<Vec<f32>>, f32)> {
        let mut state = self.zero_state();
        let mut hiddens = Vec::with_capacity(xs.len());
        let mut total_kl = 0.0_f32;

        for x in xs {
            let (new_state, kl) = self.forward_sample(x, &state, rng)?;
            hiddens.push(new_state.h.clone());
            total_kl += kl;
            state = new_state;
        }

        Ok((hiddens, total_kl))
    }

    // ─── Utilities ───────────────────────────────────────────────────────────

    /// Total number of variational parameters (mu + rho for every weight and bias).
    ///
    /// Formula: `2 × (6·input_size·hidden_size + 6·hidden_size² + 6·hidden_size)`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let i = self.cfg.input_size;
        let h = self.cfg.hidden_size;
        2 * (6 * i * h + 6 * h * h + 6 * h)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> BayesGruConfig {
        BayesGruConfig {
            input_size: 3,
            hidden_size: 4,
            prior_sigma: 1.0,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_succeeds_with_valid_config() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng);
        assert!(gru.is_ok(), "new must succeed with valid config");
    }

    #[test]
    fn new_fails_with_zero_input_size() {
        let mut rng = make_rng();
        let cfg = BayesGruConfig {
            input_size: 0,
            hidden_size: 4,
            prior_sigma: 1.0,
        };
        let r = BayesGru::new(cfg, &mut rng);
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn new_fails_with_zero_hidden_size() {
        let mut rng = make_rng();
        let cfg = BayesGruConfig {
            input_size: 3,
            hidden_size: 0,
            prior_sigma: 1.0,
        };
        let r = BayesGru::new(cfg, &mut rng);
        assert!(
            matches!(r, Err(BayesError::InsufficientSamples { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn new_fails_with_zero_prior_sigma() {
        let mut rng = make_rng();
        let cfg = BayesGruConfig {
            input_size: 3,
            hidden_size: 4,
            prior_sigma: 0.0,
        };
        let r = BayesGru::new(cfg, &mut rng);
        assert!(
            matches!(r, Err(BayesError::InvalidPriorVariance)),
            "got {r:?}"
        );
    }

    // ── zero_state ────────────────────────────────────────────────────────────

    #[test]
    fn zero_state_has_correct_length() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let s = gru.zero_state();
        assert_eq!(s.h.len(), small_cfg().hidden_size);
        assert!(s.h.iter().all(|&v| v == 0.0));
    }

    // ── forward_sample ────────────────────────────────────────────────────────

    #[test]
    fn forward_sample_output_length_equals_hidden_size() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.5_f32; small_cfg().input_size];
        let (new_state, _kl) = gru
            .forward_sample(&x, &state, &mut rng)
            .expect("forward_sample must succeed");
        assert_eq!(new_state.h.len(), small_cfg().hidden_size);
    }

    #[test]
    fn forward_sample_kl_is_positive() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![1.0_f32; small_cfg().input_size];
        let (_state, kl) = gru
            .forward_sample(&x, &state, &mut rng)
            .expect("forward_sample must succeed");
        assert!(kl > 0.0, "KL must be positive, got {kl}");
    }

    #[test]
    fn forward_sample_stochastic_across_rng_states() {
        // Two forward passes with different rng states must differ.
        let mut rng1 = LcgRng::new(1);
        let mut rng2 = LcgRng::new(99999);
        let gru = BayesGru::new(small_cfg(), &mut rng1.clone()).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.5_f32; small_cfg().input_size];
        let (s1, _) = gru
            .forward_sample(&x, &state, &mut rng1)
            .expect("forward_sample must succeed");
        let (s2, _) = gru
            .forward_sample(&x, &state, &mut rng2)
            .expect("forward_sample must succeed");
        // They will almost certainly differ.
        assert!(
            s1.h.iter()
                .zip(s2.h.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "stochastic forward must vary with different rng"
        );
    }

    #[test]
    fn forward_sample_dim_mismatch_on_x() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        // Wrong input size
        let x = vec![0.5_f32; small_cfg().input_size + 1];
        let r = gru.forward_sample(&x, &state, &mut rng);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    // ── forward_mean ─────────────────────────────────────────────────────────

    #[test]
    fn forward_mean_output_length_equals_hidden_size() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.3_f32; small_cfg().input_size];
        let new_state = gru
            .forward_mean(&x, &state)
            .expect("forward_mean must succeed");
        assert_eq!(new_state.h.len(), small_cfg().hidden_size);
    }

    #[test]
    fn forward_mean_is_deterministic() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.7_f32; small_cfg().input_size];
        let s1 = gru
            .forward_mean(&x, &state)
            .expect("forward_mean must succeed");
        let s2 = gru
            .forward_mean(&x, &state)
            .expect("forward_mean must succeed");
        for (a, b) in s1.h.iter().zip(s2.h.iter()) {
            assert!((a - b).abs() < 1e-9, "forward_mean must be deterministic");
        }
    }

    #[test]
    fn forward_mean_h_in_minus_one_to_one() {
        // GRU combines sigmoid gates with tanh; h components must lie in [-1, 1].
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.5_f32; small_cfg().input_size];
        let new_state = gru
            .forward_mean(&x, &state)
            .expect("forward_mean must succeed");
        for &v in &new_state.h {
            assert!(
                (-1.0_f32..=1.0_f32).contains(&v),
                "h component {v} outside [-1,1]"
            );
        }
    }

    #[test]
    fn forward_mean_dim_mismatch_on_x() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let state = gru.zero_state();
        let x = vec![0.0_f32; small_cfg().input_size + 2];
        let r = gru.forward_mean(&x, &state);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    // ── kl_divergence ────────────────────────────────────────────────────────

    #[test]
    fn kl_divergence_is_non_negative() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let kl = gru.kl_divergence();
        assert!(kl >= 0.0, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn kl_divergence_with_default_rho_is_small_but_positive() {
        // With w_rho = -3.0, sigma = softplus(-3) ≈ 0.049 < 1 = prior_sigma.
        // KL per weight = log(1/0.049) + (0.049² + ~0)/(2) - 0.5 > 0.
        let mut rng = make_rng();
        let cfg = BayesGruConfig {
            input_size: 2,
            hidden_size: 2,
            prior_sigma: 1.0,
        };
        let gru = BayesGru::new(cfg, &mut rng).expect("new must succeed");
        let kl = gru.kl_divergence();
        // Should be > 0 but finite.
        assert!(kl > 0.0 && kl.is_finite(), "KL = {kl}");
    }

    // ── n_params ──────────────────────────────────────────────────────────────

    #[test]
    fn n_params_formula_is_correct() {
        let mut rng = make_rng();
        let cfg = BayesGruConfig {
            input_size: 3,
            hidden_size: 4,
            prior_sigma: 1.0,
        };
        let gru = BayesGru::new(cfg, &mut rng).expect("new must succeed");
        let expected = 2 * (6 * 3 * 4 + 6 * 4 * 4 + 6 * 4);
        assert_eq!(gru.n_params(), expected, "n_params formula mismatch");
    }

    // ── forward_sequence_sample ───────────────────────────────────────────────

    #[test]
    fn forward_sequence_returns_hidden_per_timestep() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let t = 5;
        let xs: Vec<Vec<f32>> = (0..t)
            .map(|_| vec![0.1_f32; small_cfg().input_size])
            .collect();
        let (hiddens, _kl) = gru
            .forward_sequence_sample(&xs, &mut rng)
            .expect("sequence must succeed");
        assert_eq!(hiddens.len(), t);
        for h in &hiddens {
            assert_eq!(h.len(), small_cfg().hidden_size);
        }
    }

    #[test]
    fn forward_sequence_length_equals_input_length() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let xs: Vec<Vec<f32>> = vec![vec![0.0_f32; small_cfg().input_size]; 7];
        let (hs, _) = gru
            .forward_sequence_sample(&xs, &mut rng)
            .expect("sequence must succeed");
        assert_eq!(hs.len(), xs.len());
    }

    #[test]
    fn forward_sequence_empty_returns_empty() {
        let mut rng = make_rng();
        let gru = BayesGru::new(small_cfg(), &mut rng).expect("new must succeed");
        let (hs, kl) = gru
            .forward_sequence_sample(&[], &mut rng)
            .expect("empty sequence must succeed");
        assert!(hs.is_empty());
        assert_eq!(kl, 0.0);
    }
}
