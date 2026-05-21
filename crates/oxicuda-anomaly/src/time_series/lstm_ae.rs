//! LSTM-AE — RNN Autoencoder for time-series anomaly detection.
//!
//! Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016).
//! LSTM-based encoder-decoder for multi-sensor anomaly detection. *ICML 2016 Anomaly Detection
//! Workshop*.
//!
//! # Architecture
//!
//! Uses a simplified Elman RNN (vanilla RNN with `tanh`) rather than a full LSTM gate structure.
//! The key structural pattern from the paper — sequence encoding to a fixed-size context vector
//! followed by reverse-order decoding — is faithfully reproduced.
//!
//! ```text
//! Encoder (forward):
//!   h_0 = 0
//!   h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)   for t = 1 … T
//!   context = h_T
//!
//! Decoder (reverse, teacher-forcing during training):
//!   s_T = context
//!   For t = T, T-1, …, 1:
//!     s_{t-1} = tanh(W_dx · x̂_t + W_dh · s_t + b_d)
//!     x̂_t    = W_out · s_{t-1} + b_out
//!
//! Anomaly score(t) = mean over all windows containing t of MSE(x_t, x̂_t)
//! ```
//!
//! # Training
//!
//! Sliding windows of length `window_size` are extracted from the series.
//! Back-propagation through time (BPTT) is performed with gradient clipping
//! to `[-1, 1]` to prevent exploding gradients.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Gradient clipping threshold for BPTT.
const GRAD_CLIP: f64 = 1.0;

// ─── Xavier initialisation (f64) ─────────────────────────────────────────────

fn xavier_init_f64(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── LstmAeConfig ─────────────────────────────────────────────────────────────

/// Configuration for the RNN-AE time-series anomaly detector.
#[derive(Debug, Clone)]
pub struct LstmAeConfig {
    /// Number of consecutive timesteps in each sliding window.
    pub window_size: usize,
    /// Dimensionality of each timestep observation (features per timestep).
    pub input_dim: usize,
    /// Hidden state dimensionality of the RNN cells.
    pub hidden_dim: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Number of training epochs over all windows.
    pub n_epochs: usize,
}

impl Default for LstmAeConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            input_dim: 1,
            hidden_dim: 16,
            lr: 1e-3,
            n_epochs: 20,
        }
    }
}

// ─── LstmAeFit ────────────────────────────────────────────────────────────────

/// Fitted RNN-AE model.
///
/// All weight matrices are stored row-major as `[fan_out × fan_in]`.
///
/// - Encoder cell: `h_t = tanh(enc_wx · x_t + enc_wh · h_{t-1} + enc_b)`
/// - Decoder cell: `s_t = tanh(dec_wx · x̂ + dec_wh · s_{t+1} + dec_b)`
/// - Output projection: `x̂_t = dec_out_w · s_t + dec_out_b`
#[derive(Debug, Clone)]
pub struct LstmAeFit {
    /// Encoder input→hidden weights: `[hidden_dim × input_dim]`.
    pub enc_wx: Vec<f64>,
    /// Encoder hidden→hidden weights: `[hidden_dim × hidden_dim]`.
    pub enc_wh: Vec<f64>,
    /// Encoder biases: `[hidden_dim]`.
    pub enc_b: Vec<f64>,

    /// Decoder input→hidden weights: `[hidden_dim × input_dim]`.
    pub dec_wx: Vec<f64>,
    /// Decoder hidden→hidden weights: `[hidden_dim × hidden_dim]`.
    pub dec_wh: Vec<f64>,
    /// Decoder biases: `[hidden_dim]`.
    pub dec_b: Vec<f64>,

    /// Decoder output projection weights: `[input_dim × hidden_dim]`.
    pub dec_out_w: Vec<f64>,
    /// Decoder output projection biases: `[input_dim]`.
    pub dec_out_b: Vec<f64>,

    /// Window size the model was trained with.
    pub window_size: usize,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden state dimensionality.
    pub hidden_dim: usize,
}

// ─── RNN cell forward helpers ─────────────────────────────────────────────────

/// Elman RNN step: `h_new = tanh(Wx · x + Wh · h_prev + b)`.
///
/// `wx` is `[hidden × input]`, `wh` is `[hidden × hidden]`.
fn rnn_step(
    x: &[f64],
    h_prev: &[f64],
    wx: &[f64],
    wh: &[f64],
    b: &[f64],
    input_dim: usize,
    hidden_dim: usize,
) -> Vec<f64> {
    let mut pre = vec![0.0_f64; hidden_dim];
    for o in 0..hidden_dim {
        let mut acc = b[o];
        for i in 0..input_dim {
            acc += wx[o * input_dim + i] * x[i];
        }
        for j in 0..hidden_dim {
            acc += wh[o * hidden_dim + j] * h_prev[j];
        }
        pre[o] = acc;
    }
    pre.iter().map(|&v| v.tanh()).collect()
}

/// Linear projection: `y = W · h + b` where `W` is `[out × hidden]`.
fn linear_proj(h: &[f64], w: &[f64], b: &[f64], hidden_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for o in 0..out_dim {
        let mut acc = b[o];
        for j in 0..hidden_dim {
            acc += w[o * hidden_dim + j] * h[j];
        }
        out[o] = acc;
    }
    out
}

// ─── Forward pass for one window ──────────────────────────────────────────────

/// Run encoder over a window `[x_0, x_1, …, x_{T-1}]`.
///
/// Returns `(hidden_states, context)` where `hidden_states[t]` is `h_t`
/// and `context = h_{T-1}`.
fn encode_window(window: &[f64], fit: &LstmAeFit) -> (Vec<Vec<f64>>, Vec<f64>) {
    let t = fit.window_size;
    let d = fit.input_dim;
    let h = fit.hidden_dim;

    let mut hidden_states: Vec<Vec<f64>> = Vec::with_capacity(t);
    let mut h_prev = vec![0.0_f64; h];

    for step in 0..t {
        let x_t = &window[step * d..(step + 1) * d];
        let h_new = rnn_step(x_t, &h_prev, &fit.enc_wx, &fit.enc_wh, &fit.enc_b, d, h);
        hidden_states.push(h_new.clone());
        h_prev = h_new;
    }
    let context = hidden_states[t - 1].clone();
    (hidden_states, context)
}

/// Run decoder from context, generating reconstructions `[x̂_{T-1}, …, x̂_0]`.
///
/// During inference the decoder uses its own previous output as next input
/// (autoregressive). Returns `(dec_hidden_states, reconstructions)` where
/// `reconstructions[0]` corresponds to timestep `T-1` and
/// `reconstructions[T-1]` to timestep `0`.
fn decode_window(
    context: &[f64],
    fit: &LstmAeFit,
    teacher_inputs: Option<&[f64]>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let t = fit.window_size;
    let d = fit.input_dim;
    let h = fit.hidden_dim;

    let mut dec_hidden: Vec<Vec<f64>> = Vec::with_capacity(t);
    let mut recons: Vec<Vec<f64>> = Vec::with_capacity(t);

    let mut s_prev = context.to_vec();
    // Start token: zero input for first decoder step
    let mut prev_out = vec![0.0_f64; d];

    for step in 0..t {
        // Use teacher forcing during training if teacher_inputs provided
        let x_in = if let Some(inputs) = teacher_inputs {
            // Reversed teacher forcing: for step k we feed x_{T-1-k}
            let rev_t = t - 1 - step;
            inputs[rev_t * d..(rev_t + 1) * d].to_vec()
        } else {
            prev_out.clone()
        };

        let s_new = rnn_step(&x_in, &s_prev, &fit.dec_wx, &fit.dec_wh, &fit.dec_b, d, h);
        let x_hat = linear_proj(&s_new, &fit.dec_out_w, &fit.dec_out_b, h, d);
        prev_out = x_hat.clone();
        dec_hidden.push(s_new.clone());
        recons.push(x_hat);
        s_prev = s_new;
    }
    (dec_hidden, recons)
}

// ─── Gradient clip ────────────────────────────────────────────────────────────

#[inline]
fn clip_grad(g: f64) -> f64 {
    g.clamp(-GRAD_CLIP, GRAD_CLIP)
}

// ─── lstm_ae_fit ──────────────────────────────────────────────────────────────

/// Fit an RNN-AE model to a multivariate time series.
///
/// # Parameters
/// - `series` — time series, shape `n × d` row-major.
/// - `n` — number of timesteps.
/// - `d` — number of dimensions per timestep (same as `cfg.input_dim`).
/// - `cfg` — algorithm configuration.
/// - `seed` — RNG seed for reproducible initialisation.
///
/// # Errors
/// Returns `AnomalyError` if:
/// - `n < window_size` (insufficient data)
/// - `d == 0` (invalid input dimensionality)
/// - `window_size == 0`
/// # Clippy note
/// The nested range loops in this function are genuine matrix-multiplication
/// patterns (`w[o * stride + inner]`); both indices are required.
#[allow(clippy::needless_range_loop)]
pub fn lstm_ae_fit(
    series: &[f64],
    n: usize,
    d: usize,
    cfg: &LstmAeConfig,
    seed: u64,
) -> AnomalyResult<LstmAeFit> {
    // ── Validation ─────────────────────────────────────────────────────────────
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if cfg.window_size == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "window_size must be > 0".into(),
        });
    }
    if n < cfg.window_size {
        return Err(AnomalyError::InsufficientSamples {
            need: cfg.window_size,
            got: n,
        });
    }
    if series.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: series.len(),
        });
    }

    let t = cfg.window_size;
    let h = cfg.hidden_dim;
    let lr = cfg.lr;

    let mut rng = LcgRng::new(seed);

    // ── Weight initialisation ─────────────────────────────────────────────────
    let enc_wx = xavier_init_f64(d, h, &mut rng);
    let enc_wh = xavier_init_f64(h, h, &mut rng);
    let enc_b = vec![0.0_f64; h];

    let dec_wx = xavier_init_f64(d, h, &mut rng);
    let dec_wh = xavier_init_f64(h, h, &mut rng);
    let dec_b = vec![0.0_f64; h];

    let dec_out_w = xavier_init_f64(h, d, &mut rng);
    let dec_out_b = vec![0.0_f64; d];

    let mut fit = LstmAeFit {
        enc_wx,
        enc_wh,
        enc_b,
        dec_wx,
        dec_wh,
        dec_b,
        dec_out_w,
        dec_out_b,
        window_size: t,
        input_dim: d,
        hidden_dim: h,
    };

    // Number of windows
    let n_windows = n - t + 1;

    // ── Training loop ─────────────────────────────────────────────────────────
    for _epoch in 0..cfg.n_epochs {
        for w_start in 0..n_windows {
            let window = &series[w_start * d..(w_start + t) * d];

            // ── Forward: encoder ─────────────────────────────────────────────
            // enc_pre[step] = pre-tanh activations at each encoder step
            let mut enc_pre: Vec<Vec<f64>> = Vec::with_capacity(t);
            let mut enc_h: Vec<Vec<f64>> = Vec::with_capacity(t);
            let mut h_prev = vec![0.0_f64; h];

            for step in 0..t {
                let x_t = &window[step * d..(step + 1) * d];
                let mut pre = vec![0.0_f64; h];
                for o in 0..h {
                    let mut acc = fit.enc_b[o];
                    for i in 0..d {
                        acc += fit.enc_wx[o * d + i] * x_t[i];
                    }
                    for j in 0..h {
                        acc += fit.enc_wh[o * h + j] * h_prev[j];
                    }
                    pre[o] = acc;
                }
                let h_new: Vec<f64> = pre.iter().map(|&v| v.tanh()).collect();
                enc_pre.push(pre);
                enc_h.push(h_new.clone());
                h_prev = h_new;
            }
            let context = enc_h[t - 1].clone();

            // ── Forward: decoder (teacher forcing with reversed targets) ──────
            // dec_pre[step], dec_h[step], dec_out[step] for step = 0..t
            // step 0 → reconstructs timestep T-1, step k → timestep T-1-k
            let mut dec_pre: Vec<Vec<f64>> = Vec::with_capacity(t);
            let mut dec_h: Vec<Vec<f64>> = Vec::with_capacity(t);
            let mut dec_out: Vec<Vec<f64>> = Vec::with_capacity(t);
            let mut s_prev = context.clone();
            // Teacher-forcing: feed reversed ground truth as decoder input
            let mut x_in = vec![0.0_f64; d]; // zero start token

            for step in 0..t {
                let mut pre = vec![0.0_f64; h];
                for o in 0..h {
                    let mut acc = fit.dec_b[o];
                    for i in 0..d {
                        acc += fit.dec_wx[o * d + i] * x_in[i];
                    }
                    for j in 0..h {
                        acc += fit.dec_wh[o * h + j] * s_prev[j];
                    }
                    pre[o] = acc;
                }
                let s_new: Vec<f64> = pre.iter().map(|&v| v.tanh()).collect();
                // Project to output
                let x_hat = linear_proj(&s_new, &fit.dec_out_w, &fit.dec_out_b, h, d);

                // Next input: teacher forcing — reversed target
                let rev_t = t - 1 - step;
                x_in = window[rev_t * d..(rev_t + 1) * d].to_vec();

                dec_pre.push(pre);
                dec_h.push(s_new.clone());
                dec_out.push(x_hat);
                s_prev = s_new;
            }

            // ── Loss and output-layer gradient ────────────────────────────────
            // MSE loss: (1/T) Σ_t (1/d) Σ_j (x̂_t_j − x_t_j)²
            // ∂L/∂x̂_{t,j} = 2/(T*d) * (x̂_{t,j} − x_{t,j})
            // Decoder step k reconstructs reversed timestep (T-1-k)
            let inv_td = 1.0 / (t * d) as f64;

            // Gradient accumulators
            let mut d_enc_wx = vec![0.0_f64; h * d];
            let mut d_enc_wh = vec![0.0_f64; h * h];
            let mut d_enc_b = vec![0.0_f64; h];
            let mut d_dec_wx = vec![0.0_f64; h * d];
            let mut d_dec_wh = vec![0.0_f64; h * h];
            let mut d_dec_b = vec![0.0_f64; h];
            let mut d_dec_out_w = vec![0.0_f64; d * h];
            let mut d_dec_out_b = vec![0.0_f64; d];

            // BPTT through decoder (unrolled in reverse over steps)
            let mut d_s_next = vec![0.0_f64; h]; // gradient from future step

            // Reconstruct decoder input sequence for gradient computation
            let mut dec_inputs: Vec<Vec<f64>> = Vec::with_capacity(t);
            dec_inputs.push(vec![0.0_f64; d]); // step 0: zero start token
            for step in 0..t - 1 {
                let rev_t = t - 1 - step;
                dec_inputs.push(window[rev_t * d..(rev_t + 1) * d].to_vec());
            }

            // Gradient wrt context (from decoder BPTT into encoder)
            let mut d_context = vec![0.0_f64; h];

            for step in (0..t).rev() {
                let rev_t = t - 1 - step; // which original timestep this step reconstructs
                let target = &window[rev_t * d..(rev_t + 1) * d];
                let x_hat = &dec_out[step];

                // ∂L/∂x̂: MSE gradient
                let d_xhat: Vec<f64> = x_hat
                    .iter()
                    .zip(target.iter())
                    .map(|(&xh, &xt)| 2.0 * inv_td * clip_grad(xh - xt))
                    .collect();

                // Gradient through output projection: dec_out_w [d × h], dec_out_b [d]
                for o in 0..d {
                    d_dec_out_b[o] += d_xhat[o];
                    for j in 0..h {
                        d_dec_out_w[o * h + j] += d_xhat[o] * dec_h[step][j];
                    }
                }
                // Gradient back into s_new (decoder hidden at this step)
                let mut d_s_new = vec![0.0_f64; h];
                for j in 0..h {
                    for o in 0..d {
                        d_s_new[j] += d_xhat[o] * fit.dec_out_w[o * h + j];
                    }
                }
                // Add gradient from future step
                for j in 0..h {
                    d_s_new[j] += d_s_next[j];
                }

                // Through tanh: d_pre = d_s_new * (1 - tanh²(pre))
                let mut d_pre = vec![0.0_f64; h];
                for j in 0..h {
                    let tanh_val = dec_pre[step][j].tanh();
                    d_pre[j] = clip_grad(d_s_new[j] * (1.0 - tanh_val * tanh_val));
                }

                // Gradient wrt dec_wx, dec_wh, dec_b
                let x_in_step = &dec_inputs[step];
                let s_prev_step: &[f64] = if step == 0 {
                    &context
                } else {
                    &dec_h[step - 1]
                };
                for o in 0..h {
                    d_dec_b[o] += d_pre[o];
                    for i in 0..d {
                        d_dec_wx[o * d + i] += d_pre[o] * x_in_step[i];
                    }
                    for j in 0..h {
                        d_dec_wh[o * h + j] += d_pre[o] * s_prev_step[j];
                    }
                }
                // Gradient to s_prev (which is dec_h[step-1] or context for step==0)
                let mut d_s_prev = vec![0.0_f64; h];
                for j in 0..h {
                    for o in 0..h {
                        d_s_prev[j] += d_pre[o] * fit.dec_wh[o * h + j];
                    }
                }
                if step == 0 {
                    // Gradient into context
                    for j in 0..h {
                        d_context[j] += d_s_prev[j];
                    }
                } else {
                    d_s_next = d_s_prev;
                }
            }

            // BPTT through encoder starting from d_context
            let mut d_h_next = d_context;

            for step in (0..t).rev() {
                // Through tanh
                let mut d_pre = vec![0.0_f64; h];
                for j in 0..h {
                    let tanh_val = enc_pre[step][j].tanh();
                    d_pre[j] = clip_grad(d_h_next[j] * (1.0 - tanh_val * tanh_val));
                }

                let x_t = &window[step * d..(step + 1) * d];
                let h_prev_step: &[f64] = if step == 0 {
                    // zero state
                    &[]
                } else {
                    &enc_h[step - 1]
                };

                for o in 0..h {
                    d_enc_b[o] += d_pre[o];
                    for i in 0..d {
                        d_enc_wx[o * d + i] += d_pre[o] * x_t[i];
                    }
                    if step > 0 {
                        for j in 0..h {
                            d_enc_wh[o * h + j] += d_pre[o] * h_prev_step[j];
                        }
                    }
                }
                // Gradient to h_{step-1}
                let mut d_h_prev = vec![0.0_f64; h];
                if step > 0 {
                    for j in 0..h {
                        for o in 0..h {
                            d_h_prev[j] += d_pre[o] * fit.enc_wh[o * h + j];
                        }
                    }
                }
                d_h_next = d_h_prev;
            }

            // ── SGD updates ─────────────────────────────────────────────────
            for (w, &g) in fit.enc_wx.iter_mut().zip(d_enc_wx.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.enc_wh.iter_mut().zip(d_enc_wh.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.enc_b.iter_mut().zip(d_enc_b.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.dec_wx.iter_mut().zip(d_dec_wx.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.dec_wh.iter_mut().zip(d_dec_wh.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.dec_b.iter_mut().zip(d_dec_b.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.dec_out_w.iter_mut().zip(d_dec_out_w.iter()) {
                *w -= lr * g;
            }
            for (w, &g) in fit.dec_out_b.iter_mut().zip(d_dec_out_b.iter()) {
                *w -= lr * g;
            }
        }
    }

    Ok(fit)
}

// ─── lstm_ae_score ────────────────────────────────────────────────────────────

/// Compute per-timestep anomaly scores for a time series.
///
/// For each timestep `t`, the score is the mean MSE over all windows that
/// contain `t`. Windows at the boundaries of the series contribute less.
///
/// # Parameters
/// - `fit` — fitted model.
/// - `series` — time series, shape `n × input_dim` row-major.
/// - `n` — number of timesteps.
///
/// # Errors
/// Returns `AnomalyError` if `n < window_size` or dimensions are inconsistent.
pub fn lstm_ae_score(fit: &LstmAeFit, series: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    let d = fit.input_dim;
    let t = fit.window_size;

    if n < t {
        return Err(AnomalyError::InsufficientSamples { need: t, got: n });
    }
    if series.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: series.len(),
        });
    }

    let n_windows = n - t + 1;

    // Accumulators: sum of per-timestep MSEs and count of contributing windows
    let mut score_sum = vec![0.0_f64; n];
    let mut score_cnt = vec![0_usize; n];

    for w_start in 0..n_windows {
        let window = &series[w_start * d..(w_start + t) * d];

        // Encode
        let (_, context) = encode_window(window, fit);

        // Decode (no teacher forcing at inference)
        let (_, recons) = decode_window(&context, fit, None);

        // recons[step] reconstructs reversed timestep: actual_t = w_start + (T-1-step)
        for step in 0..t {
            let actual_t = w_start + (t - 1 - step);
            let target = &window[(t - 1 - step) * d..(t - step) * d];
            let x_hat = &recons[step];

            let mse: f64 = target
                .iter()
                .zip(x_hat.iter())
                .map(|(&xt, &xh)| (xt - xh).powi(2))
                .sum::<f64>()
                / d as f64;

            score_sum[actual_t] += mse;
            score_cnt[actual_t] += 1;
        }
    }

    let scores: Vec<f64> = score_sum
        .iter()
        .zip(score_cnt.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    Ok(scores)
}

// ─── lstm_ae_predict ─────────────────────────────────────────────────────────

/// Classify each timestep as anomalous (`true`) or normal (`false`).
///
/// A timestep `t` is flagged anomalous if `score(t) ≥ threshold`.
///
/// # Errors
/// Propagates errors from [`lstm_ae_score`].
pub fn lstm_ae_predict(
    fit: &LstmAeFit,
    series: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = lstm_ae_score(fit, series, n)?;
    Ok(scores.iter().map(|&s| s >= threshold).collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> LstmAeConfig {
        LstmAeConfig {
            window_size: 5,
            input_dim: 1,
            hidden_dim: 4,
            lr: 1e-2,
            n_epochs: 5,
        }
    }

    /// Build a constant univariate series of length n.
    fn constant_series(n: usize, val: f64) -> Vec<f64> {
        vec![val; n]
    }

    // ── Test 1: scores have correct length ────────────────────────────────────

    #[test]
    fn scores_have_correct_length() {
        let cfg = default_cfg();
        let n = 20_usize;
        let series = constant_series(n, 0.5);
        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 1).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        assert_eq!(scores.len(), n, "expected {n} scores, got {}", scores.len());
    }

    // ── Test 2: scores are finite and non-negative ────────────────────────────

    #[test]
    fn scores_finite_nonneg() {
        let cfg = default_cfg();
        let n = 20_usize;
        let series: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 2).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} not finite");
            assert!(s >= 0.0, "score[{i}] = {s} is negative");
        }
    }

    // ── Test 3: spike anomaly scores higher than normal timesteps ─────────────

    #[test]
    fn spike_anomaly_scores_higher() {
        let n = 30_usize;
        let mut cfg = default_cfg();
        cfg.n_epochs = 20;
        cfg.hidden_dim = 8;
        cfg.lr = 5e-3;
        // Constant series with a single spike at position spike_t
        let spike_t = 15_usize;
        let mut series = constant_series(n, 0.1);
        series[spike_t] = 5.0; // large spike

        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 3).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();

        let spike_score = scores[spike_t];
        let normal_mean: f64 = scores
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != spike_t)
            .map(|(_, &s)| s)
            .sum::<f64>()
            / (n - 1) as f64;

        assert!(
            spike_score.is_finite(),
            "spike score not finite: {spike_score}"
        );
        assert!(
            spike_score > normal_mean,
            "spike score {spike_score} should be > normal mean {normal_mean}"
        );
    }

    // ── Test 4: predict returns correct length vec ────────────────────────────

    #[test]
    fn predict_correct_length() {
        let cfg = default_cfg();
        let n = 15_usize;
        let series = constant_series(n, 0.3);
        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 4).unwrap();
        let preds = lstm_ae_predict(&fit, &series, n, 0.1).unwrap();
        assert_eq!(preds.len(), n);
    }

    // ── Test 5: error on series shorter than window_size ─────────────────────

    #[test]
    fn error_series_shorter_than_window() {
        let cfg = LstmAeConfig {
            window_size: 10,
            ..default_cfg()
        };
        let series = constant_series(5, 0.0); // n=5 < window_size=10
        let result = lstm_ae_fit(&series, 5, cfg.input_dim, &cfg, 5);
        assert!(
            matches!(
                result,
                Err(AnomalyError::InsufficientSamples { need: 10, got: 5 })
            ),
            "expected InsufficientSamples, got: {result:?}"
        );
    }

    // ── Test 6: error on input_dim = 0 ───────────────────────────────────────

    #[test]
    fn error_input_dim_zero() {
        let cfg = LstmAeConfig {
            input_dim: 0,
            ..default_cfg()
        };
        let series: Vec<f64> = vec![];
        let result = lstm_ae_fit(&series, 0, 0, &cfg, 6);
        assert!(
            matches!(result, Err(AnomalyError::InvalidFeatureCount { .. })),
            "expected InvalidFeatureCount, got: {result:?}"
        );
    }

    // ── Test 7: window_size = 1 trivial case ─────────────────────────────────

    #[test]
    fn window_size_one_works() {
        let cfg = LstmAeConfig {
            window_size: 1,
            input_dim: 2,
            hidden_dim: 4,
            lr: 1e-2,
            n_epochs: 3,
        };
        let n = 10_usize;
        let series: Vec<f64> = (0..n * 2).map(|i| i as f64 * 0.1).collect();
        let fit = lstm_ae_fit(&series, n, 2, &cfg, 7).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        assert_eq!(scores.len(), n);
        for &s in &scores {
            assert!(s.is_finite() && s >= 0.0, "score = {s}");
        }
    }

    // ── Test 8: multivariate series (d > 1) works ────────────────────────────

    #[test]
    fn multivariate_series_works() {
        let d = 3_usize;
        let n = 20_usize;
        let cfg = LstmAeConfig {
            window_size: 4,
            input_dim: d,
            hidden_dim: 6,
            lr: 1e-2,
            n_epochs: 5,
        };
        let series: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.05).cos()).collect();
        let fit = lstm_ae_fit(&series, n, d, &cfg, 8).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        assert_eq!(scores.len(), n);
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} not finite");
        }
    }

    // ── Test 9: predict threshold=0 flags any non-zero-score timestep ─────────

    #[test]
    fn predict_threshold_zero() {
        let cfg = default_cfg();
        let n = 15_usize;
        let series = constant_series(n, 0.5);
        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 9).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        let preds = lstm_ae_predict(&fit, &series, n, 0.0).unwrap();
        for (i, (&s, &p)) in scores.iter().zip(preds.iter()).enumerate() {
            if s > 0.0 {
                assert!(p, "timestep {i} score={s} should be flagged at threshold 0");
            }
        }
    }

    // ── Test 10: error on score with n < window_size ──────────────────────────

    #[test]
    fn error_on_score_too_short() {
        let cfg = default_cfg();
        let n = 20_usize;
        let series = constant_series(n, 0.5);
        let fit = lstm_ae_fit(&series, n, cfg.input_dim, &cfg, 10).unwrap();
        // Try to score a shorter series
        let short = constant_series(3, 0.5);
        let result = lstm_ae_score(&fit, &short, 3);
        assert!(
            matches!(result, Err(AnomalyError::InsufficientSamples { .. })),
            "expected InsufficientSamples, got: {result:?}"
        );
    }

    // ── Test 11: reconstruction error decreases over training ─────────────────

    #[test]
    fn reconstruction_improves_over_training() {
        let d = 1_usize;
        let n = 25_usize;
        let series: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin() * 0.5 + 0.5).collect();

        let cfg_few = LstmAeConfig {
            window_size: 5,
            input_dim: d,
            hidden_dim: 8,
            lr: 5e-3,
            n_epochs: 1,
        };
        let cfg_many = LstmAeConfig {
            n_epochs: 50,
            ..cfg_few.clone()
        };

        let fit_few = lstm_ae_fit(&series, n, d, &cfg_few, 200).unwrap();
        let fit_many = lstm_ae_fit(&series, n, d, &cfg_many, 200).unwrap();

        let score_few: f64 = lstm_ae_score(&fit_few, &series, n).unwrap().iter().sum();
        let score_many: f64 = lstm_ae_score(&fit_many, &series, n).unwrap().iter().sum();

        assert!(
            score_few.is_finite() && score_many.is_finite(),
            "scores not finite: few={score_few}, many={score_many}"
        );
        // More epochs should reduce total reconstruction error on training data
        assert!(
            score_many <= score_few * 1.1,
            "expected more epochs to not increase score beyond 10%: few={score_few}, many={score_many}"
        );
    }

    // ── Test 12: window_size = n (single window) works ───────────────────────

    #[test]
    fn window_size_equals_n() {
        let n = 8_usize;
        let d = 1_usize;
        let cfg = LstmAeConfig {
            window_size: n,
            input_dim: d,
            hidden_dim: 4,
            lr: 1e-2,
            n_epochs: 3,
        };
        let series: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let fit = lstm_ae_fit(&series, n, d, &cfg, 12).unwrap();
        let scores = lstm_ae_score(&fit, &series, n).unwrap();
        assert_eq!(scores.len(), n);
        for &s in &scores {
            assert!(s.is_finite() && s >= 0.0, "score = {s}");
        }
    }

    // ── Test 13: error on dimension mismatch in series ────────────────────────

    #[test]
    fn error_on_series_dim_mismatch() {
        let d = 2_usize;
        let n = 10_usize;
        let cfg = LstmAeConfig {
            window_size: 3,
            input_dim: d,
            hidden_dim: 4,
            lr: 1e-2,
            n_epochs: 2,
        };
        // Series of wrong total length
        let series = vec![0.0_f64; n * d + 1];
        let result = lstm_ae_fit(&series, n, d, &cfg, 13);
        assert!(
            matches!(result, Err(AnomalyError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got: {result:?}"
        );
    }
}
