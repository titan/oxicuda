//! ENAS reinforcement-learning controller (Pham et al., 2018).
//!
//! "Efficient Neural Architecture Search via Parameter Sharing" trains a small
//! LSTM *controller* that autoregressively emits an architecture as a sequence
//! of categorical decisions. At each step the controller runs one LSTM cell,
//! decodes a softmax over the choices for that step, samples one, and feeds the
//! embedding of the chosen action back in as the next step's input. The
//! controller is trained with the REINFORCE policy-gradient rule using an
//! exponential-moving-average (EMA) reward baseline for variance reduction.
//!
//! This module is fully self-contained — the LSTM cell, the embedding table,
//! the softmax decoder, sampling, and back-propagation-through-time (BPTT) are
//! implemented directly over flat `Vec<f32>` parameter buffers with no external
//! linear-algebra or autodiff dependency.
//!
//! # Math (one step `t`)
//!
//! ```text
//! i_t = σ(W_ii x_t + W_hi h_{t-1} + b_i)
//! f_t = σ(W_if x_t + W_hf h_{t-1} + b_f)
//! g_t = tanh(W_ig x_t + W_hg h_{t-1} + b_g)
//! o_t = σ(W_io x_t + W_ho h_{t-1} + b_o)
//! c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//! h_t = o_t ⊙ tanh(c_t)
//! z_t = W_dec h_t + b_dec                       (logits, length n_choices)
//! p_t = softmax(z_t / temperature)
//! a_t ~ Categorical(p_t)                         (sampled action)
//! x_{t+1} = embedding[a_t]                       (fed back as next input)
//! ```
//!
//! The REINFORCE objective minimised by a gradient step is
//!
//! ```text
//! L = −(reward − baseline) · Σ_t log p_t[a_t]  −  entropy_weight · Σ_t H(p_t)
//! ```
//!
//! Gradient *descent* on `L` with advantage `A = reward − baseline > 0`
//! increases `Σ_t log p_t[a_t]`, i.e. genuine policy improvement toward the
//! sampled (high-reward) trajectory.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

/// Divisor that maps [`LcgRng::next_u32`] (range `[0, 2³¹)`) onto `[0, 1)`.
///
/// `LcgRng::next_u32` returns the top bits via a `state >> 33` shift, so its
/// range is `[0, 2³¹)`. Dividing by `2³¹` therefore yields a proper unit-uniform
/// `f32` (the crate's `next_f32` divides by `2³²` and so only spans `[0, 0.5)` —
/// we deliberately avoid it).
const U31: f32 = 2_147_483_648.0;

/// Draw a unit-uniform `f32` in `[0, 1)` from the LCG, using the documented
/// `next_u32 / 2³¹` convention (see [`U31`]).
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / U31
}

/// Hyper-parameters for the ENAS controller.
#[derive(Debug, Clone)]
pub struct EnasConfig {
    /// LSTM hidden dimension (also the embedding / input dimension).
    pub hidden_dim: usize,
    /// Number of autoregressive decisions per sampled architecture.
    pub n_steps: usize,
    /// Number of categorical choices available at each step.
    pub n_choices: usize,
    /// Gradient-ascent step size for the controller parameters.
    pub learning_rate: f32,
    /// EMA decay for the reward baseline: `b ← decay·b + (1−decay)·reward`.
    pub ema_baseline_decay: f32,
    /// Softmax temperature applied to the decoder logits (`> 0`).
    pub temperature: f32,
    /// Coefficient of the entropy bonus in the REINFORCE loss.
    pub entropy_weight: f32,
}

impl Default for EnasConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 32,
            n_steps: 4,
            n_choices: 8,
            learning_rate: 1e-2,
            ema_baseline_decay: 0.95,
            temperature: 1.0,
            entropy_weight: 1e-4,
        }
    }
}

impl EnasConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`NasError::InvalidNumOps`] if `n_choices < 2` or `hidden_dim == 0`.
    /// - [`NasError::InvalidNumNodes`] if `n_steps == 0`.
    /// - [`NasError::NanInArchParams`] if `temperature`, `learning_rate`,
    ///   `ema_baseline_decay`, or `entropy_weight` are non-finite, or
    ///   `temperature <= 0`, or `ema_baseline_decay` is outside `[0, 1)`.
    pub fn validate(&self) -> NasResult<()> {
        if self.hidden_dim == 0 || self.n_choices < 2 {
            return Err(NasError::InvalidNumOps);
        }
        if self.n_steps == 0 {
            return Err(NasError::InvalidNumNodes { min: 1, got: 0 });
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(NasError::NanInArchParams);
        }
        if !self.learning_rate.is_finite() || !self.entropy_weight.is_finite() {
            return Err(NasError::NanInArchParams);
        }
        if !self.ema_baseline_decay.is_finite() || !(0.0..1.0).contains(&self.ema_baseline_decay) {
            return Err(NasError::NanInArchParams);
        }
        Ok(())
    }
}

/// The four LSTM gates, in canonical order `i, f, g, o`.
const N_GATES: usize = 4;
const GATE_I: usize = 0;
const GATE_F: usize = 1;
const GATE_G: usize = 2;
const GATE_O: usize = 3;

/// Per-step values cached during the forward pass and consumed by BPTT.
#[derive(Debug, Clone)]
struct StepCache {
    /// Input `x_t` (embedding of the previous action; zeros at `t = 0`).
    x: Vec<f32>,
    /// Previous hidden state `h_{t-1}` (zeros at `t = 0`).
    h_prev: Vec<f32>,
    /// Previous cell state `c_{t-1}` (zeros at `t = 0`).
    c_prev: Vec<f32>,
    /// Post-activation gate values, one `Vec<f32>` of length `hidden_dim` per gate.
    gates: [Vec<f32>; N_GATES],
    /// Cell state `c_t`.
    c: Vec<f32>,
    /// `tanh(c_t)` (reused in backward).
    tanh_c: Vec<f32>,
    /// Hidden state `h_t`.
    h: Vec<f32>,
    /// Softmax probabilities `p_t` over the `n_choices` actions.
    probs: Vec<f32>,
    /// The action selected at this step (the previous-action index for `t+1`).
    action: usize,
}

/// Accumulated gradients of the REINFORCE loss w.r.t. every controller parameter.
#[derive(Debug, Clone)]
struct Grads {
    w_ih: [Vec<f32>; N_GATES],
    w_hh: [Vec<f32>; N_GATES],
    bias: [Vec<f32>; N_GATES],
    embedding: Vec<f32>,
    w_dec: Vec<f32>,
    b_dec: Vec<f32>,
}

impl Grads {
    fn zeros(hidden_dim: usize, n_choices: usize) -> Self {
        let g = || vec![0.0_f32; hidden_dim * hidden_dim];
        let b = || vec![0.0_f32; hidden_dim];
        Self {
            w_ih: [g(), g(), g(), g()],
            w_hh: [g(), g(), g(), g()],
            bias: [b(), b(), b(), b()],
            embedding: vec![0.0_f32; n_choices * hidden_dim],
            w_dec: vec![0.0_f32; n_choices * hidden_dim],
            b_dec: vec![0.0_f32; n_choices],
        }
    }
}

/// LSTM-based autoregressive ENAS controller trained by REINFORCE.
#[derive(Debug, Clone)]
pub struct EnasController {
    cfg: EnasConfig,
    /// Input→gate weights `[gate][hidden_dim × hidden_dim]` (row-major, row = output unit).
    w_ih: [Vec<f32>; N_GATES],
    /// Hidden→gate weights `[gate][hidden_dim × hidden_dim]`.
    w_hh: [Vec<f32>; N_GATES],
    /// Gate biases `[gate][hidden_dim]`.
    bias: [Vec<f32>; N_GATES],
    /// Action embedding table `[n_choices × hidden_dim]` (row = action).
    embedding: Vec<f32>,
    /// Decoder weights `[n_choices × hidden_dim]` (row = action logit).
    w_dec: Vec<f32>,
    /// Decoder bias `[n_choices]`.
    b_dec: Vec<f32>,
    /// EMA reward baseline.
    baseline: f32,
    /// Cached most-recent sampled trajectory (actions), used for the update.
    last_actions: Vec<usize>,
}

impl EnasController {
    /// Create a controller with weights initialised from a small uniform
    /// distribution `U(−scale, scale)` using `rng` and the `next_u32 / 2³¹`
    /// convention (see `U31`).
    ///
    /// # Errors
    /// Propagates [`EnasConfig::validate`].
    pub fn new(cfg: EnasConfig, rng: &mut LcgRng) -> NasResult<Self> {
        cfg.validate()?;
        let h = cfg.hidden_dim;
        let n_choices = cfg.n_choices;
        // Small symmetric init; scale ~ 1/sqrt(fan_in) keeps gate pre-activations modest.
        let scale = (1.0_f32 / h as f32).sqrt() * 0.5;
        let mut uniform = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (unit_uniform(rng) * 2.0 - 1.0) * scale)
                .collect()
        };
        let mat = |u: &mut dyn FnMut(usize) -> Vec<f32>| u(h * h);
        let vecn = |u: &mut dyn FnMut(usize) -> Vec<f32>| u(h);
        let w_ih = [
            mat(&mut uniform),
            mat(&mut uniform),
            mat(&mut uniform),
            mat(&mut uniform),
        ];
        let w_hh = [
            mat(&mut uniform),
            mat(&mut uniform),
            mat(&mut uniform),
            mat(&mut uniform),
        ];
        let bias = [
            vecn(&mut uniform),
            vecn(&mut uniform),
            vecn(&mut uniform),
            vecn(&mut uniform),
        ];
        let embedding = uniform(n_choices * h);
        let w_dec = uniform(n_choices * h);
        let b_dec = uniform(n_choices);
        Ok(Self {
            cfg,
            w_ih,
            w_hh,
            bias,
            embedding,
            w_dec,
            b_dec,
            baseline: 0.0,
            last_actions: Vec::new(),
        })
    }

    /// Current EMA reward baseline.
    #[must_use]
    pub fn baseline(&self) -> f32 {
        self.baseline
    }

    /// The controller configuration.
    #[must_use]
    pub fn config(&self) -> &EnasConfig {
        &self.cfg
    }

    // ── Forward primitives ────────────────────────────────────────────────────

    /// Run one LSTM cell + decoder, returning the per-step cache (sans action /
    /// probs filled by the caller is unnecessary — probs are computed here).
    ///
    /// `action` is recorded into the cache by the caller after sampling /
    /// teacher-forcing; this routine fills every field except `action`.
    fn cell_forward(&self, x: &[f32], h_prev: &[f32], c_prev: &[f32]) -> StepCache {
        let h = self.cfg.hidden_dim;
        // Pre-activations per gate: b + W_ih·x + W_hh·h_prev.
        let mut pre = [
            self.bias[GATE_I].clone(),
            self.bias[GATE_F].clone(),
            self.bias[GATE_G].clone(),
            self.bias[GATE_O].clone(),
        ];
        for (gate, acc) in pre.iter_mut().enumerate() {
            let w_ih = &self.w_ih[gate];
            let w_hh = &self.w_hh[gate];
            for (row, acc_r) in acc.iter_mut().enumerate() {
                let ih_row = &w_ih[row * h..row * h + h];
                let hh_row = &w_hh[row * h..row * h + h];
                let mut s = *acc_r;
                for k in 0..h {
                    s += ih_row[k] * x[k] + hh_row[k] * h_prev[k];
                }
                *acc_r = s;
            }
        }
        // Activations.
        let i_gate: Vec<f32> = pre[GATE_I].iter().map(|&v| sigmoid(v)).collect();
        let f_gate: Vec<f32> = pre[GATE_F].iter().map(|&v| sigmoid(v)).collect();
        let g_gate: Vec<f32> = pre[GATE_G].iter().map(|&v| v.tanh()).collect();
        let o_gate: Vec<f32> = pre[GATE_O].iter().map(|&v| sigmoid(v)).collect();
        // Cell + hidden.
        let mut c = vec![0.0_f32; h];
        let mut tanh_c = vec![0.0_f32; h];
        let mut h_new = vec![0.0_f32; h];
        for j in 0..h {
            c[j] = f_gate[j] * c_prev[j] + i_gate[j] * g_gate[j];
            let tc = c[j].tanh();
            tanh_c[j] = tc;
            h_new[j] = o_gate[j] * tc;
        }
        // Decoder logits + temperature softmax.
        let probs = self.decode(&h_new);
        StepCache {
            x: x.to_vec(),
            h_prev: h_prev.to_vec(),
            c_prev: c_prev.to_vec(),
            gates: [i_gate, f_gate, g_gate, o_gate],
            c,
            tanh_c,
            h: h_new,
            probs,
            action: 0,
        }
    }

    /// Decoder + temperature softmax → probability vector of length `n_choices`.
    fn decode(&self, h: &[f32]) -> Vec<f32> {
        let hd = self.cfg.hidden_dim;
        let n = self.cfg.n_choices;
        let inv_t = 1.0 / self.cfg.temperature;
        let mut logits = vec![0.0_f32; n];
        for (k, lk) in logits.iter_mut().enumerate() {
            let row = &self.w_dec[k * hd..k * hd + hd];
            let mut s = self.b_dec[k];
            for (w, &hv) in row.iter().zip(h.iter()) {
                s += w * hv;
            }
            *lk = s * inv_t;
        }
        softmax(&logits)
    }

    /// Row `a` of the embedding table.
    fn embed(&self, a: usize) -> Vec<f32> {
        let hd = self.cfg.hidden_dim;
        self.embedding[a * hd..a * hd + hd].to_vec()
    }

    /// Run the autoregressive forward over a *fixed* action sequence
    /// (teacher forcing). Returns the full per-step cache. Used by both the
    /// loss evaluation and BPTT so the gradient matches the loss exactly.
    fn forward_actions(&self, actions: &[usize]) -> Vec<StepCache> {
        let h = self.cfg.hidden_dim;
        let mut caches = Vec::with_capacity(actions.len());
        let mut h_prev = vec![0.0_f32; h];
        let mut c_prev = vec![0.0_f32; h];
        let mut x = vec![0.0_f32; h]; // step-0 input is zero (no previous action)
        for (t, &a) in actions.iter().enumerate() {
            let mut cache = self.cell_forward(&x, &h_prev, &c_prev);
            cache.action = a;
            h_prev = cache.h.clone();
            c_prev = cache.c.clone();
            if t + 1 < actions.len() {
                x = self.embed(a);
            }
            caches.push(cache);
        }
        caches
    }

    // ── Sampling ────────────────────────────────────────────────────────────────

    /// Autoregressively sample an architecture.
    ///
    /// Returns `(actions, log_probs, total_entropy)` where `actions[t]` is the
    /// sampled category at step `t`, `log_probs[t] = ln p_t[actions[t]]`, and
    /// `total_entropy = Σ_t H(p_t) ≥ 0`. The sampled trajectory is cached for a
    /// subsequent [`Self::reinforce_update`].
    ///
    /// Sampling uses the inverse-CDF method with a unit-uniform drawn via
    /// `rng.next_u32() / 2³¹` (see `U31`).
    ///
    /// # Errors
    /// Propagates [`EnasConfig::validate`].
    pub fn sample_architecture(
        &mut self,
        rng: &mut LcgRng,
    ) -> NasResult<(Vec<usize>, Vec<f32>, f32)> {
        self.cfg.validate()?;
        let h = self.cfg.hidden_dim;
        let n_choices = self.cfg.n_choices;
        let mut h_prev = vec![0.0_f32; h];
        let mut c_prev = vec![0.0_f32; h];
        let mut x = vec![0.0_f32; h];
        let mut actions = Vec::with_capacity(self.cfg.n_steps);
        let mut log_probs = Vec::with_capacity(self.cfg.n_steps);
        let mut total_entropy = 0.0_f32;
        for t in 0..self.cfg.n_steps {
            let cache = self.cell_forward(&x, &h_prev, &c_prev);
            let probs = &cache.probs;
            // Inverse-CDF categorical sample.
            let u = unit_uniform(rng);
            let mut cdf = 0.0_f32;
            let mut action = n_choices - 1;
            for (k, &p) in probs.iter().enumerate() {
                cdf += p;
                if u < cdf {
                    action = k;
                    break;
                }
            }
            let p_a = probs[action].max(f32::MIN_POSITIVE);
            log_probs.push(p_a.ln());
            total_entropy += entropy(probs);
            h_prev = cache.h.clone();
            c_prev = cache.c.clone();
            if t + 1 < self.cfg.n_steps {
                x = self.embed(action);
            }
            actions.push(action);
        }
        self.last_actions = actions.clone();
        Ok((actions, log_probs, total_entropy.max(0.0)))
    }

    /// Sum of per-step log-probabilities `Σ_t ln p_t[actions[t]]` of a *given*
    /// action sequence under the current parameters (teacher-forced).
    ///
    /// Useful to verify genuine policy improvement after an update.
    ///
    /// # Errors
    /// - [`NasError::DimensionMismatch`] if `actions.len() != n_steps`.
    /// - [`NasError::InvalidArchEncoding`] if any action is `>= n_choices`.
    pub fn log_prob_of_actions(&self, actions: &[usize]) -> NasResult<f32> {
        if actions.len() != self.cfg.n_steps {
            return Err(NasError::DimensionMismatch {
                expected: self.cfg.n_steps,
                got: actions.len(),
            });
        }
        for &a in actions {
            if a >= self.cfg.n_choices {
                return Err(NasError::InvalidArchEncoding);
            }
        }
        let caches = self.forward_actions(actions);
        let mut total = 0.0_f32;
        for cache in &caches {
            let p_a = cache.probs[cache.action].max(f32::MIN_POSITIVE);
            total += p_a.ln();
        }
        Ok(total)
    }

    // ── REINFORCE update ──────────────────────────────────────────────────────

    /// Apply one REINFORCE gradient-ascent step for the most recently sampled
    /// trajectory and update the EMA baseline.
    ///
    /// `log_probs` must be the per-step log-probabilities returned by the
    /// matching [`Self::sample_architecture`] call (its length is validated
    /// against `n_steps`). The advantage is `A = reward − baseline` using the
    /// baseline *before* this update. Returns the scalar loss
    /// `L = −A·Σ log_probs − entropy_weight·entropy` (computed by re-running the
    /// forward pass over the cached actions, so the returned loss is exactly the
    /// quantity whose gradient is taken).
    ///
    /// After the parameter step the baseline is updated as
    /// `baseline ← decay·baseline + (1−decay)·reward`.
    ///
    /// # Errors
    /// - [`NasError::DimensionMismatch`] if `log_probs.len() != n_steps`.
    /// - [`NasError::NoFeasibleArchitecture`] if no trajectory has been sampled.
    /// - [`NasError::NanInArchParams`] if `reward` is non-finite or the loss
    ///   becomes non-finite.
    pub fn reinforce_update(&mut self, log_probs: &[f32], reward: f32) -> NasResult<f32> {
        if log_probs.len() != self.cfg.n_steps {
            return Err(NasError::DimensionMismatch {
                expected: self.cfg.n_steps,
                got: log_probs.len(),
            });
        }
        if self.last_actions.len() != self.cfg.n_steps {
            return Err(NasError::NoFeasibleArchitecture);
        }
        if !reward.is_finite() {
            return Err(NasError::NanInArchParams);
        }
        let advantage = reward - self.baseline;
        let actions = self.last_actions.clone();

        // Forward over the fixed actions, then BPTT → analytic gradients.
        let caches = self.forward_actions(&actions);
        let (loss, grads) = self.loss_and_grads(&caches, advantage);
        if !loss.is_finite() {
            return Err(NasError::NanInArchParams);
        }

        // Gradient *descent* on L (== ascent on the REINFORCE objective).
        self.apply_grads(&grads);

        // EMA baseline update (after the parameter step).
        let d = self.cfg.ema_baseline_decay;
        self.baseline = d * self.baseline + (1.0 - d) * reward;
        Ok(loss)
    }

    /// Compute the loss and its analytic gradient (BPTT) for a forward cache.
    ///
    /// `advantage` is treated as a constant (it does not depend on the
    /// controller parameters), matching standard REINFORCE.
    fn loss_and_grads(&self, caches: &[StepCache], advantage: f32) -> (f32, Grads) {
        let h = self.cfg.hidden_dim;
        let n_choices = self.cfg.n_choices;
        let inv_t = 1.0 / self.cfg.temperature;
        let ew = self.cfg.entropy_weight;
        let mut grads = Grads::zeros(h, n_choices);

        // Loss value.
        let mut sum_lp = 0.0_f32;
        let mut sum_entropy = 0.0_f32;
        for cache in caches {
            let p_a = cache.probs[cache.action].max(f32::MIN_POSITIVE);
            sum_lp += p_a.ln();
            sum_entropy += entropy(&cache.probs);
        }
        let loss = -advantage * sum_lp - ew * sum_entropy;

        // Backward through time. dh_next / dc_next carry gradients from step t+1.
        let mut dh_next = vec![0.0_f32; h];
        let mut dc_next = vec![0.0_f32; h];
        for t in (0..caches.len()).rev() {
            let cache = &caches[t];
            let probs = &cache.probs;
            let entropy_t = entropy(probs);

            // dL/dz_t : combine log-prob and entropy gradients (both / temperature).
            //   d(-A·log p[a])/dz_k = -A·(δ_{k,a} - p_k)/τ
            //   d(-ew·H)/dz_k       = -ew·(-p_k(ln p_k + H))/τ = ew·p_k(ln p_k + H)/τ
            let mut dz = vec![0.0_f32; n_choices];
            for (k, dzk) in dz.iter_mut().enumerate() {
                let pk = probs[k];
                let indicator = if k == cache.action { 1.0 } else { 0.0 };
                let dlogp = -advantage * (indicator - pk);
                let ln_pk = pk.max(f32::MIN_POSITIVE).ln();
                let dent = ew * pk * (ln_pk + entropy_t);
                *dzk = (dlogp + dent) * inv_t;
            }

            // Decoder grads + dL/dh from the decoder at this step.
            let mut dh = vec![0.0_f32; h];
            for (k, &dzk) in dz.iter().enumerate() {
                grads.b_dec[k] += dzk;
                let wrow = &self.w_dec[k * h..k * h + h];
                let grow = &mut grads.w_dec[k * h..k * h + h];
                for j in 0..h {
                    grow[j] += dzk * cache.h[j];
                    dh[j] += dzk * wrow[j];
                }
            }
            // Add the gradient flowing back from step t+1's recurrence.
            for j in 0..h {
                dh[j] += dh_next[j];
            }

            // LSTM cell backward.
            let i_gate = &cache.gates[GATE_I];
            let f_gate = &cache.gates[GATE_F];
            let g_gate = &cache.gates[GATE_G];
            let o_gate = &cache.gates[GATE_O];

            // dc_t includes the path h_t = o ⊙ tanh(c_t).
            let mut dc = vec![0.0_f32; h];
            for j in 0..h {
                let dtanh = dh[j] * o_gate[j] * (1.0 - cache.tanh_c[j] * cache.tanh_c[j]);
                dc[j] = dc_next[j] + dtanh;
            }

            // Pre-activation gate gradients.
            let mut di_pre = vec![0.0_f32; h];
            let mut df_pre = vec![0.0_f32; h];
            let mut dg_pre = vec![0.0_f32; h];
            let mut do_pre = vec![0.0_f32; h];
            for j in 0..h {
                let do_act = dh[j] * cache.tanh_c[j];
                do_pre[j] = do_act * o_gate[j] * (1.0 - o_gate[j]);
                let di_act = dc[j] * g_gate[j];
                di_pre[j] = di_act * i_gate[j] * (1.0 - i_gate[j]);
                let df_act = dc[j] * cache.c_prev[j];
                df_pre[j] = df_act * f_gate[j] * (1.0 - f_gate[j]);
                let dg_act = dc[j] * i_gate[j];
                dg_pre[j] = dg_act * (1.0 - g_gate[j] * g_gate[j]);
            }
            let pre_gates = [&di_pre, &df_pre, &dg_pre, &do_pre];

            // Accumulate weight / bias grads and propagate to x_t and h_{t-1}.
            let mut dx = vec![0.0_f32; h];
            let mut dh_prev = vec![0.0_f32; h];
            for (gate, &gp) in pre_gates.iter().enumerate() {
                let w_ih = &self.w_ih[gate];
                let w_hh = &self.w_hh[gate];
                let g_ih = &mut grads.w_ih[gate];
                let b_gate = &mut grads.bias[gate];
                for (row, &gpr) in gp.iter().enumerate() {
                    b_gate[row] += gpr;
                    let ih_row = &mut g_ih[row * h..row * h + h];
                    for k in 0..h {
                        ih_row[k] += gpr * cache.x[k];
                        dx[k] += gpr * w_ih[row * h + k];
                    }
                }
                let g_hh = &mut grads.w_hh[gate];
                for (row, &gpr) in gp.iter().enumerate() {
                    let hh_row = &mut g_hh[row * h..row * h + h];
                    for k in 0..h {
                        hh_row[k] += gpr * cache.h_prev[k];
                        dh_prev[k] += gpr * w_hh[row * h + k];
                    }
                }
            }

            // dc_prev = dc ⊙ f → becomes dc_next for step t-1.
            for j in 0..h {
                dc_next[j] = dc[j] * f_gate[j];
            }
            // dh_prev → becomes dh_next for step t-1.
            dh_next = dh_prev;

            // x_t is the embedding of action_{t-1} for t > 0 (zeros at t = 0).
            if t > 0 {
                let prev_a = caches[t - 1].action;
                let erow = &mut grads.embedding[prev_a * h..prev_a * h + h];
                for k in 0..h {
                    erow[k] += dx[k];
                }
            }
        }
        (loss, grads)
    }

    /// Gradient-descent parameter update `θ ← θ − lr · ∂L/∂θ`.
    fn apply_grads(&mut self, grads: &Grads) {
        let lr = self.cfg.learning_rate;
        for gate in 0..N_GATES {
            axpy(&mut self.w_ih[gate], &grads.w_ih[gate], -lr);
            axpy(&mut self.w_hh[gate], &grads.w_hh[gate], -lr);
            axpy(&mut self.bias[gate], &grads.bias[gate], -lr);
        }
        axpy(&mut self.embedding, &grads.embedding, -lr);
        axpy(&mut self.w_dec, &grads.w_dec, -lr);
        axpy(&mut self.b_dec, &grads.b_dec, -lr);
    }
}

// ─── Math helpers ────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for e in &mut exps {
        *e *= inv;
    }
    exps
}

/// Shannon entropy `−Σ p_k ln p_k` of a probability vector (`≥ 0`).
fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h.max(0.0)
}

/// In-place `y ← y + α·x`.
#[inline]
fn axpy(y: &mut [f32], x: &[f32], alpha: f32) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> EnasConfig {
        EnasConfig {
            hidden_dim: 4,
            n_steps: 3,
            n_choices: 5,
            learning_rate: 0.05,
            ema_baseline_decay: 0.9,
            temperature: 1.0,
            entropy_weight: 0.0,
        }
    }

    #[test]
    fn config_rejects_few_choices() {
        let mut cfg = tiny_cfg();
        cfg.n_choices = 1;
        assert!(cfg.validate().is_err());
        let mut rng = LcgRng::new(1);
        assert!(EnasController::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn config_rejects_zero_steps() {
        let mut cfg = tiny_cfg();
        cfg.n_steps = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_bad_temperature() {
        let mut cfg = tiny_cfg();
        cfg.temperature = 0.0;
        assert!(cfg.validate().is_err());
        cfg.temperature = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sampled_actions_in_range() {
        let mut rng = LcgRng::new(7);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        for _ in 0..50 {
            let (actions, _, _) = ctrl.sample_architecture(&mut rng).expect("sample");
            assert_eq!(actions.len(), 3);
            assert!(actions.iter().all(|&a| a < 5));
        }
    }

    #[test]
    fn log_probs_match_sampled_categorical() {
        let mut rng = LcgRng::new(11);
        let ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let mut srng = LcgRng::new(123);
        let mut ctrl2 = ctrl.clone();
        let (actions, log_probs, _) = ctrl2.sample_architecture(&mut srng).expect("sample");
        // Recompute log-probs over the fixed actions via the public helper.
        let caches = ctrl.forward_actions(&actions);
        for (t, cache) in caches.iter().enumerate() {
            let expected = cache.probs[actions[t]].ln();
            assert!(
                (log_probs[t] - expected).abs() < 1e-5,
                "step {t}: {} vs {}",
                log_probs[t],
                expected
            );
        }
    }

    #[test]
    fn log_probs_sum_to_log_joint() {
        let mut rng = LcgRng::new(31);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (actions, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        let sum: f32 = log_probs.iter().sum();
        let joint = ctrl.log_prob_of_actions(&actions).expect("joint");
        assert!((sum - joint).abs() < 1e-5, "{sum} vs {joint}");
    }

    #[test]
    fn entropy_non_negative() {
        let mut rng = LcgRng::new(5);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        for _ in 0..20 {
            let (_, _, ent) = ctrl.sample_architecture(&mut rng).expect("sample");
            assert!(ent >= 0.0, "entropy = {ent}");
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let mut r1 = LcgRng::new(99);
        let mut c1 = EnasController::new(tiny_cfg(), &mut r1).expect("c1");
        let mut s1 = LcgRng::new(2024);
        let (a1, lp1, e1) = c1.sample_architecture(&mut s1).expect("s1");

        let mut r2 = LcgRng::new(99);
        let mut c2 = EnasController::new(tiny_cfg(), &mut r2).expect("c2");
        let mut s2 = LcgRng::new(2024);
        let (a2, lp2, e2) = c2.sample_architecture(&mut s2).expect("s2");

        assert_eq!(a1, a2);
        assert_eq!(lp1, lp2);
        assert_eq!(e1, e2);
    }

    #[test]
    fn high_temperature_near_uniform() {
        let mut cfg = tiny_cfg();
        cfg.temperature = 1000.0;
        let mut rng = LcgRng::new(3);
        let ctrl = EnasController::new(cfg, &mut rng).expect("ctrl");
        let h = vec![0.5_f32; 4];
        let probs = ctrl.decode(&h);
        let uniform = 1.0 / 5.0;
        for &p in &probs {
            assert!((p - uniform).abs() < 1e-2, "p = {p}");
        }
    }

    #[test]
    fn policy_improves_after_positive_advantage() {
        // entropy_weight = 0 so the policy-gradient term acts alone.
        let mut rng = LcgRng::new(17);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (actions, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        let before = ctrl.log_prob_of_actions(&actions).expect("before");
        // reward 1.0 ≫ baseline 0.0 → positive advantage.
        let _loss = ctrl.reinforce_update(&log_probs, 1.0).expect("update");
        let after = ctrl.log_prob_of_actions(&actions).expect("after");
        assert!(
            after > before,
            "log-prob should increase: before {before}, after {after}"
        );
    }

    #[test]
    fn policy_decreases_after_negative_advantage() {
        let mut rng = LcgRng::new(41);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (actions, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        let before = ctrl.log_prob_of_actions(&actions).expect("before");
        // reward below baseline (0.0) → negative advantage → discourage this seq.
        let _loss = ctrl.reinforce_update(&log_probs, -1.0).expect("update");
        let after = ctrl.log_prob_of_actions(&actions).expect("after");
        assert!(
            after < before,
            "log-prob should decrease: before {before}, after {after}"
        );
    }

    #[test]
    fn baseline_ema_moves_toward_reward() {
        let mut rng = LcgRng::new(8);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        assert_eq!(ctrl.baseline(), 0.0);
        let (_, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        ctrl.reinforce_update(&log_probs, 1.0).expect("update");
        let b1 = ctrl.baseline();
        assert!(b1 > 0.0 && b1 < 1.0, "baseline = {b1}");
        // decay 0.9, reward 1.0 → 0.9*0 + 0.1*1 = 0.1.
        assert!((b1 - 0.1).abs() < 1e-6, "baseline = {b1}");
        let (_, lp2, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        ctrl.reinforce_update(&lp2, 1.0).expect("update");
        assert!(ctrl.baseline() > b1, "baseline should keep rising");
    }

    #[test]
    fn reinforce_rejects_length_mismatch() {
        let mut rng = LcgRng::new(2);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        ctrl.sample_architecture(&mut rng).expect("sample");
        let wrong = vec![0.0_f32; 2]; // n_steps is 3
        assert!(matches!(
            ctrl.reinforce_update(&wrong, 1.0),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn reinforce_rejects_without_sample() {
        let mut rng = LcgRng::new(2);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let log_probs = vec![0.0_f32; 3];
        assert_eq!(
            ctrl.reinforce_update(&log_probs, 1.0),
            Err(NasError::NoFeasibleArchitecture)
        );
    }

    #[test]
    fn reinforce_rejects_nonfinite_reward() {
        let mut rng = LcgRng::new(2);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (_, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        assert_eq!(
            ctrl.reinforce_update(&log_probs, f32::NAN),
            Err(NasError::NanInArchParams)
        );
    }

    #[test]
    fn log_prob_of_actions_validates() {
        let mut rng = LcgRng::new(2);
        let ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        assert!(matches!(
            ctrl.log_prob_of_actions(&[0, 1]),
            Err(NasError::DimensionMismatch { .. })
        ));
        assert_eq!(
            ctrl.log_prob_of_actions(&[0, 1, 99]),
            Err(NasError::InvalidArchEncoding)
        );
    }

    /// Finite-difference check of the analytic BPTT gradient on a tiny config.
    ///
    /// For every category of parameter we perturb one scalar by ±ε, recompute
    /// the loss over a *fixed* action sequence, and compare the central
    /// difference to the corresponding analytic gradient.
    #[test]
    fn finite_difference_gradient_matches_bptt() {
        let cfg = EnasConfig {
            hidden_dim: 3,
            n_steps: 3,
            n_choices: 4,
            learning_rate: 0.0, // we never step here
            ema_baseline_decay: 0.9,
            temperature: 0.8,
            entropy_weight: 0.1, // exercise the entropy gradient too
        };
        let mut rng = LcgRng::new(2025);
        let ctrl = EnasController::new(cfg, &mut rng).expect("ctrl");
        let actions = vec![1usize, 3, 0];
        let advantage = 0.7_f32;

        // Analytic gradient.
        let caches = ctrl.forward_actions(&actions);
        let (_loss0, grads) = ctrl.loss_and_grads(&caches, advantage);

        // Closure: loss for the current parameters over the fixed actions.
        let loss_of = |c: &EnasController| -> f32 {
            let caches = c.forward_actions(&actions);
            let mut sum_lp = 0.0_f32;
            let mut sum_ent = 0.0_f32;
            for cache in &caches {
                let p_a = cache.probs[cache.action].max(f32::MIN_POSITIVE);
                sum_lp += p_a.ln();
                sum_ent += entropy(&cache.probs);
            }
            -advantage * sum_lp - c.cfg.entropy_weight * sum_ent
        };

        let eps = 1e-3_f32;
        let tol = 2e-2_f32;

        // Helper that perturbs one scalar of a selected buffer and returns the
        // central-difference derivative.
        let h = ctrl.config().hidden_dim;
        let nc = ctrl.config().n_choices;

        // hidden_dim = 3, n_choices = 4. All flat indices below assume row-major
        // `[out_unit × hidden_dim]` layout and target distinct rows/cols.
        assert_eq!(h, 3);
        assert_eq!(nc, 4);

        // Check a representative scalar in each parameter group via a flat index.
        struct Probe {
            label: &'static str,
            // (gate, flat_index) for matrix/bias groups; gate ignored otherwise.
            get: fn(&mut EnasController) -> &mut f32,
            analytic: f32,
        }

        let probes = [
            Probe {
                // w_ih gate i, row 1 col 2 → flat 5
                label: "w_ih[i]",
                get: |c| &mut c.w_ih[GATE_I][5],
                analytic: grads.w_ih[GATE_I][5],
            },
            Probe {
                // w_hh gate o, row 0 col 1 → flat 1
                label: "w_hh[o]",
                get: |c| &mut c.w_hh[GATE_O][1],
                analytic: grads.w_hh[GATE_O][1],
            },
            Probe {
                // bias gate g, unit 2
                label: "bias[g]",
                get: |c| &mut c.bias[GATE_G][2],
                analytic: grads.bias[GATE_G][2],
            },
            Probe {
                // embedding row 1 (= action at step 0), col 0 → flat 3
                label: "embedding",
                get: |c| &mut c.embedding[3],
                analytic: grads.embedding[3],
            },
            Probe {
                // decoder weight row 3 col 1 → flat 10
                label: "w_dec",
                get: |c| &mut c.w_dec[10],
                analytic: grads.w_dec[10],
            },
            Probe {
                // decoder bias logit 0
                label: "b_dec",
                get: |c| &mut c.b_dec[0],
                analytic: grads.b_dec[0],
            },
        ];

        for probe in &probes {
            let mut cp = ctrl.clone();
            let orig = *(probe.get)(&mut cp);
            *(probe.get)(&mut cp) = orig + eps;
            let lp = loss_of(&cp);
            *(probe.get)(&mut cp) = orig - eps;
            let lm = loss_of(&cp);
            *(probe.get)(&mut cp) = orig;
            let fd = (lp - lm) / (2.0 * eps);
            let diff = (fd - probe.analytic).abs();
            let denom = probe.analytic.abs().max(1.0);
            assert!(
                diff / denom < tol,
                "{}: analytic {} vs finite-diff {} (|Δ| {})",
                probe.label,
                probe.analytic,
                fd,
                diff
            );
        }
    }

    #[test]
    fn loss_is_finite_and_returned() {
        let mut rng = LcgRng::new(64);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (_, log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        let loss = ctrl.reinforce_update(&log_probs, 0.5).expect("update");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    #[test]
    fn repeated_positive_updates_increase_logprob_monotonically() {
        let mut rng = LcgRng::new(321);
        let mut ctrl = EnasController::new(tiny_cfg(), &mut rng).expect("ctrl");
        let (actions, mut log_probs, _) = ctrl.sample_architecture(&mut rng).expect("sample");
        let mut prev = ctrl.log_prob_of_actions(&actions).expect("lp");
        for _ in 0..5 {
            // Keep last_actions == actions by re-deriving log_probs over them.
            ctrl.last_actions = actions.clone();
            ctrl.reinforce_update(&log_probs, 1.0).expect("update");
            let now = ctrl.log_prob_of_actions(&actions).expect("lp");
            assert!(now >= prev - 1e-6, "log-prob dropped: {prev} -> {now}");
            prev = now;
            // refresh log_probs to current policy for the next iteration
            log_probs = (0..ctrl.config().n_steps).map(|_| 0.0).collect();
        }
    }
}
