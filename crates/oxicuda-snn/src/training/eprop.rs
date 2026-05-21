//! e-prop online learning rule for spiking neural networks.
//!
//! Implements the e-prop learning rule (Bellec et al. 2020 Nature Commun. 11:3625)
//! and the DECOLLE variant for local credit assignment (Kaiser et al. 2020
//! Front. Neurosci. 14:424).
//!
//! # e-prop
//!
//! For each synapse (pre→post), an eligibility trace `e_ij` is maintained:
//!
//! ```text
//! e_ij(t+1) = (1 − dt/τ_e) · e_ij(t)  +  h_j(V_j(t)) · s_i(t)
//! ```
//!
//! where `h_j` is the piecewise-linear pseudo-derivative of the spike function:
//!
//! ```text
//! h(v, v_th) = (1/v_th) · max(0, 1 − |v − v_th| / v_th)
//! ```
//!
//! Weight updates are:
//!
//! ```text
//! ΔW_ij = η · l_j · e_ij  +  η · f_reg · (r_target − r_j) · h_j(V_j)
//! ```
//!
//! # DECOLLE
//!
//! Local credit assignment without a global error signal: each neuron has a
//! local linear readout `y = W_out · s`, and the learning signal is the local
//! reconstruction error backpropagated through `W_out`.

use crate::error::{SnnError, SnnResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// e-prop hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct EpropConfig {
    /// Learning rate η (default 0.01).
    pub learning_rate: f32,
    /// Eligibility trace time constant τ_e ms (default 20.0).
    pub tau_e: f32,
    /// Membrane time constant τ_m ms (default 20.0).
    pub tau_m: f32,
    /// Output trace time constant τ_o ms (default 30.0).
    pub tau_o: f32,
    /// Integration time step ms (default 1.0).
    pub dt: f32,
    /// Gradient clip norm — `None` disables clipping.
    pub clip_grad: Option<f32>,
    /// Target firing rate for regularisation Hz (default 0.01).
    pub target_rate: f32,
    /// Firing rate regularisation coefficient (default 1.0).
    pub f_reg: f32,
}

impl Default for EpropConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            tau_e: 20.0,
            tau_m: 20.0,
            tau_o: 30.0,
            dt: 1.0,
            clip_grad: None,
            target_rate: 0.01,
            f_reg: 1.0,
        }
    }
}

// ── Eligibility traces ────────────────────────────────────────────────────────

/// Per-synapse eligibility traces stored as an `n_pre × n_post` row-major matrix.
#[derive(Debug, Clone)]
pub struct EligibilityTraces {
    /// Eligibility trace values, length `n_pre * n_post`.
    pub e: Vec<f32>,
    /// Number of pre-synaptic neurons.
    pub n_pre: usize,
    /// Number of post-synaptic neurons.
    pub n_post: usize,
}

impl EligibilityTraces {
    /// Allocate an all-zero trace matrix of shape `n_pre × n_post`.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            e: vec![0.0_f32; n_pre * n_post],
            n_pre,
            n_post,
        }
    }

    /// Reset all traces to zero.
    pub fn reset(&mut self) {
        for x in self.e.iter_mut() {
            *x = 0.0;
        }
    }
}

// ── Learning signal accumulator ───────────────────────────────────────────────

/// Online learning signal for one output neuron.
#[derive(Debug, Clone)]
pub struct LearningSignal {
    /// Broadcast weights from the output neuron back to each hidden neuron,
    /// length `n_post` (i.e., the number of hidden neurons).
    pub b: Vec<f32>,
    /// Scalar learning signal (error at the output neuron).
    pub l: f32,
}

impl LearningSignal {
    /// Allocate with zero broadcast weights and zero signal.
    #[must_use]
    pub fn new(n_post: usize) -> Self {
        Self {
            b: vec![0.0_f32; n_post],
            l: 0.0,
        }
    }
}

// ── Pseudo-derivative ─────────────────────────────────────────────────────────

/// Piecewise-linear pseudo-derivative of the spike function:
/// `h(v, v_th) = (1/v_th) · max(0, 1 − |v − v_th| / v_th)`.
///
/// Returns 0 when `v_th ≤ 0`.
#[inline]
pub fn pseudo_derivative(v: f32, v_th: f32) -> f32 {
    if v_th <= 0.0 {
        return 0.0;
    }
    let x = 1.0 - (v - v_th).abs() / v_th;
    if x > 0.0 { x / v_th } else { 0.0 }
}

// ── e-prop core functions ─────────────────────────────────────────────────────

/// Update eligibility traces for one timestep.
///
/// `pre_spikes` and `post_spikes` are binary {0,1} spike vectors.
/// `post_v` is the post-synaptic membrane potential (used for pseudo-derivative).
///
/// Update rule:
/// ```text
/// e_ij(t+1) = (1 − dt/τ_e) · e_ij(t)  +  h_j(V_j) · s_i(t)
/// ```
pub fn update_eligibility_traces(
    traces: &mut EligibilityTraces,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    post_v: &[f32],
    v_th: f32,
    cfg: &EpropConfig,
) -> SnnResult<()> {
    let n_pre = traces.n_pre;
    let n_post = traces.n_post;
    if n_pre == 0 || n_post == 0 {
        return Err(SnnError::EmptyInput);
    }
    if pre_spikes.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: pre_spikes.len(),
        });
    }
    if post_spikes.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: post_spikes.len(),
        });
    }
    if post_v.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: post_v.len(),
        });
    }
    if cfg.tau_e <= 0.0 || !cfg.tau_e.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_e });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }

    let decay = 1.0 - cfg.dt / cfg.tau_e;
    // Pre-compute pseudo-derivatives for all post-synaptic neurons
    let h: Vec<f32> = post_v.iter().map(|&v| pseudo_derivative(v, v_th)).collect();

    for (i, &s_i) in pre_spikes.iter().enumerate() {
        let row_off = i * n_post;
        for (j, &h_j) in h.iter().enumerate() {
            traces.e[row_off + j] = decay * traces.e[row_off + j] + h_j * s_i;
        }
    }
    Ok(())
}

/// Compute weight updates: `ΔW_ij = η · l_j · e_ij + η · f_reg · (r_target − r_j) · h_j`.
///
/// Returns a flat row-major vector of length `n_pre * n_post`.
pub fn compute_weight_update(
    traces: &EligibilityTraces,
    learning_signals: &[f32],
    post_v: &[f32],
    running_rates: &[f32],
    v_th: f32,
    cfg: &EpropConfig,
) -> SnnResult<Vec<f32>> {
    let n_pre = traces.n_pre;
    let n_post = traces.n_post;
    if n_pre == 0 || n_post == 0 {
        return Err(SnnError::EmptyInput);
    }
    if learning_signals.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: learning_signals.len(),
        });
    }
    if post_v.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: post_v.len(),
        });
    }
    if running_rates.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: running_rates.len(),
        });
    }

    let eta = cfg.learning_rate;
    let f_reg = cfg.f_reg;
    let r_target = cfg.target_rate;

    // Pre-compute pseudo-derivatives and regularisation terms per post-neuron
    let h: Vec<f32> = post_v.iter().map(|&v| pseudo_derivative(v, v_th)).collect();
    let reg: Vec<f32> = (0..n_post)
        .map(|j| eta * f_reg * (r_target - running_rates[j]) * h[j])
        .collect();

    let mut dw = vec![0.0_f32; n_pre * n_post];
    for i in 0..n_pre {
        let row_off = i * n_post;
        for j in 0..n_post {
            let task_term = eta * learning_signals[j] * traces.e[row_off + j];
            dw[row_off + j] = task_term + reg[j];
        }
    }
    Ok(dw)
}

/// Apply `dw` to `weights` in-place, with optional gradient clipping.
///
/// If `cfg.clip_grad = Some(c)` and `‖dw‖₂ > c`, the update is scaled so
/// that `‖dw‖₂ = c` before application.
pub fn apply_weight_update(weights: &mut [f32], dw: &[f32], cfg: &EpropConfig) -> SnnResult<()> {
    if weights.len() != dw.len() {
        return Err(SnnError::IncompatibleLength {
            a: weights.len(),
            b: dw.len(),
        });
    }
    if weights.is_empty() {
        return Err(SnnError::EmptyInput);
    }

    // Compute norm for optional clipping
    let norm_sq: f32 = dw.iter().map(|&x| x * x).sum();
    let norm = norm_sq.sqrt();

    let scale = if let Some(clip) = cfg.clip_grad {
        if clip <= 0.0 || !clip.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "clip_grad".into(),
                val: clip,
            });
        }
        if norm > clip { clip / norm } else { 1.0 }
    } else {
        1.0
    };

    for (w, &d) in weights.iter_mut().zip(dw.iter()) {
        *w += scale * d;
    }
    Ok(())
}

/// Update running spike rates with an exponential moving average.
///
/// ```text
/// r_j(t+1) = (1 − dt/τ_o) · r_j(t)  +  (dt/τ_o) · s_j(t)
/// ```
pub fn update_running_rates(rates: &mut [f32], spikes: &[f32], cfg: &EpropConfig) -> SnnResult<()> {
    if rates.len() != spikes.len() {
        return Err(SnnError::IncompatibleLength {
            a: rates.len(),
            b: spikes.len(),
        });
    }
    if rates.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if cfg.tau_o <= 0.0 || !cfg.tau_o.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_o });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    let alpha = cfg.dt / cfg.tau_o;
    let decay = 1.0 - alpha;
    for (r, &s) in rates.iter_mut().zip(spikes.iter()) {
        *r = decay * *r + alpha * s;
    }
    Ok(())
}

// ── DECOLLE ───────────────────────────────────────────────────────────────────

/// DECOLLE: compute local learning signals without a global error signal.
///
/// Each neuron `j` has a local linear readout over `n_out` output targets.
/// The weight matrix `w_out` has shape `n_neurons × n_out` (row-major).
///
/// 1. Local readout: `y_k = Σ_j w_out[j,k] · spikes[j]`  (length `n_out`)
/// 2. Local error:   `l_readout[k] = targets[k] − y_k`   (length `n_out`)
/// 3. Learning signal per neuron: `l_j = Σ_k w_out[j,k] · l_readout[k]`
///
/// Returns `(learning_signals len n_neurons, readout len n_out)`.
pub fn decolle_learning_signals(
    spikes: &[f32],
    w_out: &[f32],
    targets: &[f32],
    n_out: usize,
) -> SnnResult<(Vec<f32>, Vec<f32>)> {
    let n_neurons = spikes.len();
    if n_neurons == 0 {
        return Err(SnnError::EmptyInput);
    }
    if n_out == 0 {
        return Err(SnnError::BadDim { got: n_out });
    }
    if targets.len() != n_out {
        return Err(SnnError::IncompatibleLength {
            a: n_out,
            b: targets.len(),
        });
    }
    let expected_w = n_neurons * n_out;
    if w_out.len() != expected_w {
        return Err(SnnError::BadShape {
            expected: expected_w,
            got: w_out.len(),
        });
    }

    // 1. Compute readout: y = W_out^T · spikes  (n_out outputs)
    let mut readout = vec![0.0_f32; n_out];
    for (j, &s_j) in spikes.iter().enumerate() {
        let row_off = j * n_out;
        for (k, y_k) in readout.iter_mut().enumerate() {
            *y_k += w_out[row_off + k] * s_j;
        }
    }

    // 2. Local errors: l_readout = targets − y
    let l_readout: Vec<f32> = targets
        .iter()
        .zip(readout.iter())
        .map(|(&t, &y)| t - y)
        .collect();

    // 3. Learning signals per neuron: l_j = W_out[j,:] · l_readout
    let learning_signals: Vec<f32> = (0..n_neurons)
        .map(|j| {
            let row_off = j * n_out;
            l_readout
                .iter()
                .enumerate()
                .map(|(k, &lr)| w_out[row_off + k] * lr)
                .sum()
        })
        .collect();

    Ok((learning_signals, readout))
}

// ── Full e-prop update step ───────────────────────────────────────────────────

/// Convenience wrapper executing a full e-prop update step:
///
/// 1. Update eligibility traces
/// 2. Update running spike rates
/// 3. Compute weight update (task loss + firing rate regularisation)
/// 4. Apply weight update (with optional gradient clipping)
pub fn eprop_step(
    weights: &mut [f32],
    traces: &mut EligibilityTraces,
    running_rates: &mut [f32],
    pre_spikes: &[f32],
    post_spikes: &[f32],
    post_v: &[f32],
    learning_signals: &[f32],
    v_th: f32,
    cfg: &EpropConfig,
) -> SnnResult<()> {
    update_eligibility_traces(traces, pre_spikes, post_spikes, post_v, v_th, cfg)?;
    update_running_rates(running_rates, post_spikes, cfg)?;
    let dw = compute_weight_update(traces, learning_signals, post_v, running_rates, v_th, cfg)?;
    apply_weight_update(weights, &dw, cfg)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> EpropConfig {
        EpropConfig::default()
    }

    // 1. New traces are all zero
    #[test]
    fn eligibility_trace_zero_at_init() {
        let traces = EligibilityTraces::new(4, 3);
        assert!(
            traces.e.iter().all(|&x| x == 0.0),
            "traces should be zero at init"
        );
    }

    // 2. Pre=1 post=1 → trace positive after 1 step
    #[test]
    fn eligibility_trace_increases_with_co_activity() {
        let mut traces = EligibilityTraces::new(1, 1);
        let cfg = default_cfg();
        let pre = vec![1.0_f32];
        let post = vec![1.0_f32];
        let post_v = vec![1.0_f32]; // v_th = 1.0
        let v_th = 1.0_f32;
        update_eligibility_traces(&mut traces, &pre, &post, &post_v, v_th, &cfg)
            .expect("trace update");
        assert!(traces.e[0] > 0.0, "trace should increase with co-activity");
    }

    // 3. Pre=0 post=0 → traces decay toward zero
    #[test]
    fn eligibility_trace_decays_without_spikes() {
        let mut traces = EligibilityTraces::new(2, 2);
        // Seed with non-zero values
        for x in traces.e.iter_mut() {
            *x = 1.0;
        }
        let cfg = default_cfg();
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 2];
        let post_v = vec![0.0_f32; 2];
        let v_th = 1.0_f32;
        for _ in 0..10 {
            update_eligibility_traces(&mut traces, &pre, &post, &post_v, v_th, &cfg)
                .expect("update");
        }
        // After 10 steps of decay, all traces should be < original value
        assert!(
            traces.e.iter().all(|&x| x < 1.0),
            "traces should decay without activity"
        );
    }

    // 4. Pseudo-derivative at threshold = 1/v_th
    #[test]
    fn pseudo_derivative_at_threshold() {
        let v_th = 2.0_f32;
        let h = pseudo_derivative(v_th, v_th);
        let expected = 1.0 / v_th;
        assert!(
            (h - expected).abs() < 1e-6,
            "h(v_th, v_th) = {}, expected {}",
            h,
            expected
        );
    }

    // 5. Pseudo-derivative far from threshold = 0
    #[test]
    fn pseudo_derivative_far_from_threshold() {
        let v_th = 1.0_f32;
        let h = pseudo_derivative(10.0 * v_th, v_th);
        assert_eq!(h, 0.0, "h far from threshold should be 0");
    }

    // 6. Pseudo-derivative below threshold = 0 (v=0, v_th=1.0 → |0−1|/1 = 1 → max(0,0) = 0)
    #[test]
    fn pseudo_derivative_below_threshold() {
        let v_th = 1.0_f32;
        let h = pseudo_derivative(0.0, v_th);
        assert_eq!(h, 0.0, "h(0, 1) should be 0");
    }

    // 7. Zero learning signal → ΔW task term = 0 (when rates at target, reg also 0)
    #[test]
    fn weight_update_zero_with_zero_learning_signal() {
        let mut traces = EligibilityTraces::new(3, 2);
        // Set traces to non-zero
        for x in traces.e.iter_mut() {
            *x = 0.5;
        }
        let mut cfg = default_cfg();
        cfg.f_reg = 0.0; // disable regularisation
        let l_signals = vec![0.0_f32; 2];
        let post_v = vec![1.0_f32; 2]; // at threshold
        let rates = vec![cfg.target_rate; 2]; // at target → reg term = 0
        let dw =
            compute_weight_update(&traces, &l_signals, &post_v, &rates, 1.0, &cfg).expect("dw");
        assert!(
            dw.iter().all(|&x| x.abs() < 1e-8),
            "dw should be zero with zero learning signal and zero reg"
        );
    }

    // 8. l_j=1 everywhere → ΔW ∝ e_ij (task term only, no reg)
    #[test]
    fn weight_update_proportional_to_eligibility() {
        let mut traces = EligibilityTraces::new(2, 2);
        traces.e[0] = 0.3;
        traces.e[1] = 0.7;
        traces.e[2] = 0.1;
        traces.e[3] = 0.9;
        let mut cfg = default_cfg();
        cfg.f_reg = 0.0;
        let l_signals = vec![1.0_f32; 2];
        let post_v = vec![0.0_f32; 2]; // zero pseudo-deriv → reg = 0
        let rates = vec![0.0_f32; 2];
        let dw =
            compute_weight_update(&traces, &l_signals, &post_v, &rates, 1.0, &cfg).expect("dw");
        let eta = cfg.learning_rate;
        // dw[i,j] = eta * 1.0 * e[i,j]
        assert!((dw[0] - eta * 0.3).abs() < 1e-6, "dw[0,0] mismatch");
        assert!((dw[1] - eta * 0.7).abs() < 1e-6, "dw[0,1] mismatch");
        assert!((dw[2] - eta * 0.1).abs() < 1e-6, "dw[1,0] mismatch");
        assert!((dw[3] - eta * 0.9).abs() < 1e-6, "dw[1,1] mismatch");
    }

    // 9. apply_weight_update changes weights by dW
    #[test]
    fn apply_weight_update_changes_weights() {
        let mut cfg = default_cfg();
        cfg.clip_grad = None;
        let mut weights = vec![1.0_f32; 4];
        let dw = vec![0.1_f32, 0.2, -0.1, -0.3];
        apply_weight_update(&mut weights, &dw, &cfg).expect("apply");
        let expected = [1.1_f32, 1.2, 0.9, 0.7];
        for (w, e) in weights.iter().zip(expected.iter()) {
            assert!((w - e).abs() < 1e-6, "weight mismatch: {} vs {}", w, e);
        }
    }

    // 10. Large dW + clip_grad=1.0 → ‖ΔW‖ capped at 1.0
    #[test]
    fn gradient_clip_caps_update_norm() {
        let cfg = EpropConfig {
            clip_grad: Some(1.0),
            ..default_cfg()
        };
        let mut weights = vec![0.0_f32; 4];
        let dw = vec![10.0_f32, 10.0, 10.0, 10.0];
        let norm_dw = dw.iter().map(|&x| x * x).sum::<f32>().sqrt();
        apply_weight_update(&mut weights, &dw, &cfg).expect("apply");
        let applied_norm: f32 = weights.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let expected_norm = 1.0_f32.min(norm_dw);
        assert!(
            (applied_norm - expected_norm).abs() < 1e-5,
            "applied norm {} should equal clip {}",
            applied_norm,
            expected_norm
        );
    }

    // 11. Small dW + clip_grad=100.0 → ΔW unchanged
    #[test]
    fn no_clip_when_within_bound() {
        let cfg = EpropConfig {
            clip_grad: Some(100.0),
            ..default_cfg()
        };
        let mut weights = vec![0.0_f32; 3];
        let dw = vec![0.1_f32, 0.2, 0.3];
        apply_weight_update(&mut weights, &dw, &cfg).expect("apply");
        for (w, &d) in weights.iter().zip(dw.iter()) {
            assert!(
                (w - d).abs() < 1e-6,
                "weight should equal dw when within clip"
            );
        }
    }

    // 12. Running rates decay toward zero without spikes
    #[test]
    fn running_rates_decay_without_spikes() {
        let cfg = default_cfg();
        let mut rates = vec![1.0_f32; 3];
        let spikes = vec![0.0_f32; 3];
        for _ in 0..100 {
            update_running_rates(&mut rates, &spikes, &cfg).expect("update");
        }
        assert!(
            rates.iter().all(|&r| r < 0.1),
            "rates should decay without spikes"
        );
    }

    // 13. Rates increase with constant spiking
    #[test]
    fn running_rates_increase_with_spikes() {
        let cfg = default_cfg();
        let mut rates = vec![0.0_f32; 2];
        let spikes = vec![1.0_f32; 2];
        for _ in 0..100 {
            update_running_rates(&mut rates, &spikes, &cfg).expect("update");
        }
        assert!(
            rates.iter().all(|&r| r > 0.5),
            "rates should increase toward 1.0 with constant spiking"
        );
    }

    // 14. DECOLLE: zero spikes → zero readout, zero learning signal
    #[test]
    fn decolle_zero_spikes_zero_readout() {
        let spikes = vec![0.0_f32; 3];
        let w_out = vec![1.0_f32; 3 * 2]; // 3 neurons × 2 outputs
        let targets = vec![0.5_f32; 2];
        let (ls, readout) =
            decolle_learning_signals(&spikes, &w_out, &targets, 2).expect("decolle");
        assert!(
            readout.iter().all(|&y| y.abs() < 1e-9),
            "readout should be zero with zero spikes"
        );
        assert!(
            ls.iter().all(|&l| l.abs() > 0.0),
            "learning signal = w_out @ (target - 0) should be non-zero when target ≠ 0"
        );
    }

    // 15. DECOLLE output shape correct
    #[test]
    fn decolle_output_shape_correct() {
        let n_neurons = 5_usize;
        let n_out = 3_usize;
        let spikes = vec![1.0_f32; n_neurons];
        let w_out = vec![0.1_f32; n_neurons * n_out];
        let targets = vec![0.0_f32; n_out];
        let (ls, readout) =
            decolle_learning_signals(&spikes, &w_out, &targets, n_out).expect("decolle");
        assert_eq!(ls.len(), n_neurons, "learning signals length mismatch");
        assert_eq!(readout.len(), n_out, "readout length mismatch");
    }

    // 16. DECOLLE: manual 2×2 w_out × spikes comparison
    #[test]
    fn decolle_learning_signal_correct() {
        // n_neurons=2, n_out=2
        // w_out = [[1, 0], [0, 1]] (identity)
        let spikes = vec![0.5_f32, 0.3];
        let w_out = vec![1.0_f32, 0.0, 0.0, 1.0]; // row-major 2×2 identity
        let targets = vec![1.0_f32, 1.0];
        let n_out = 2_usize;
        let (ls, readout) =
            decolle_learning_signals(&spikes, &w_out, &targets, n_out).expect("decolle");

        // readout = W^T · spikes = spikes (identity)
        assert!((readout[0] - 0.5).abs() < 1e-6, "readout[0]={}", readout[0]);
        assert!((readout[1] - 0.3).abs() < 1e-6, "readout[1]={}", readout[1]);

        // l_readout = targets - readout
        let lr0 = 1.0 - 0.5;
        let lr1 = 1.0 - 0.3;

        // ls[j] = sum_k w_out[j,k] * l_readout[k]
        // ls[0] = 1*lr0 + 0*lr1 = 0.5
        // ls[1] = 0*lr0 + 1*lr1 = 0.7
        assert!(
            (ls[0] - lr0).abs() < 1e-5,
            "ls[0]={} expected {}",
            ls[0],
            lr0
        );
        assert!(
            (ls[1] - lr1).abs() < 1e-5,
            "ls[1]={} expected {}",
            ls[1],
            lr1
        );
    }

    // 17. Full eprop_step end-to-end — weights change after non-zero signals
    #[test]
    fn eprop_step_end_to_end() {
        let cfg = EpropConfig {
            f_reg: 0.0, // isolate task learning signal
            ..default_cfg()
        };
        let n_pre = 3_usize;
        let n_post = 2_usize;
        let mut weights = vec![0.5_f32; n_pre * n_post];
        let weights_before = weights.clone();
        let mut traces = EligibilityTraces::new(n_pre, n_post);
        let mut running_rates = vec![0.0_f32; n_post];

        let pre_spikes = vec![1.0_f32, 0.0, 1.0];
        let post_spikes = vec![1.0_f32; n_post];
        let post_v = vec![1.0_f32; n_post]; // at threshold → h > 0
        let learning_signals = vec![1.0_f32; n_post];
        let v_th = 1.0_f32;

        eprop_step(
            &mut weights,
            &mut traces,
            &mut running_rates,
            &pre_spikes,
            &post_spikes,
            &post_v,
            &learning_signals,
            v_th,
            &cfg,
        )
        .expect("eprop_step");

        let any_changed = weights
            .iter()
            .zip(weights_before.iter())
            .any(|(&w, &wb)| (w - wb).abs() > 1e-9);
        assert!(
            any_changed,
            "weights should change after non-zero learning signals"
        );
    }

    // 18. Error on pre/post length mismatch
    #[test]
    fn err_length_mismatch_pre_post() {
        let mut traces = EligibilityTraces::new(3, 2);
        let cfg = default_cfg();
        let pre = vec![1.0_f32; 4]; // wrong — traces.n_pre = 3
        let post = vec![1.0_f32; 2];
        let post_v = vec![1.0_f32; 2];
        let err = update_eligibility_traces(&mut traces, &pre, &post, &post_v, 1.0, &cfg);
        assert!(
            matches!(err, Err(SnnError::IncompatibleLength { .. })),
            "expected IncompatibleLength error"
        );
    }
}
