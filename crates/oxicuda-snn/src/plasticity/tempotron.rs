//! Tempotron — a temporally-sensitive binary spike classifier.
//!
//! Reference: Gütig & Sompolinsky, "The tempotron: a neuron that learns spike
//! timing-based decisions", *Nature Neuroscience* 9, 420–428 (2006). A single
//! leaky-integrate-and-fire readout neuron learns to emit at least one spike in
//! response to "+" patterns and to stay silent for "−" patterns. Because the
//! decision depends on whether the *peak* sub-threshold voltage crosses the
//! firing threshold, learning reduces to gradient ascent/descent on that peak.
//!
//! The membrane voltage induced by the afferent spikes is
//!
//! ```text
//! V(t) = Σ_i w_i Σ_{t_i^f < t} K(t − t_i^f),
//! K(s) = V_0 · [exp(−s/τ_m) − exp(−s/τ_s)]   for s ≥ 0, else 0,
//! ```
//!
//! where `K` is the normalized post-synaptic potential kernel. The
//! normalisation constant `V_0` scales the kernel so that its peak equals one;
//! the peak occurs at the analytically derived time
//!
//! ```text
//! t_max = (τ_m·τ_s / (τ_m − τ_s)) · ln(τ_m / τ_s).
//! ```
//!
//! **Learning rule.** Present a pattern as a list of `(synapse_index,
//! spike_time)` events and let `t*` be the time of the maximum voltage. If the
//! pattern is "+" but the neuron failed to reach threshold, every synapse is
//! potentiated by `Δw_i = +λ · Σ_{t_i^f < t*} K(t* − t_i^f)`. If the pattern is
//! "−" but the neuron erroneously crossed threshold, the same sum is subtracted
//! (`Δw_i = −λ · …`). Correctly classified patterns trigger no update. This is
//! the classic gradient on the peak voltage with respect to each weight
//! (Gütig & Sompolinsky 2006, eqs. 1–3).

use crate::error::{SnnError, SnnResult};

/// Tempotron hyperparameters.
///
/// `tau_m` and `tau_s` must be strictly positive and **distinct** (the kernel
/// normalisation divides by their difference); `dt` and `t_max_window` must be
/// strictly positive.
#[derive(Debug, Clone, Copy)]
pub struct TempotronConfig {
    /// Membrane time constant `τ_m`.
    pub tau_m: f32,
    /// Synaptic time constant `τ_s` (typically `τ_m / 4`).
    pub tau_s: f32,
    /// Firing threshold `V_th`.
    pub v_th: f32,
    /// Rest potential `V_rest` (baseline of the voltage trace).
    pub v_rest: f32,
    /// Learning rate `λ`.
    pub learning_rate: f32,
    /// Integration step `dt` over which the voltage trace is sampled.
    pub dt: f32,
    /// Length of the time window `[0, t_max_window)` over which the trace is
    /// evaluated and the peak is searched.
    pub t_max_window: f32,
}

impl Default for TempotronConfig {
    /// Canonical settings: `τ_m = 15`, `τ_s = τ_m/4 = 3.75`, `V_th = 1`,
    /// `V_rest = 0`, `λ = 0.01`, `dt = 1`, window `= 100`.
    fn default() -> Self {
        let tau_m = 15.0;
        Self {
            tau_m,
            tau_s: tau_m / 4.0,
            v_th: 1.0,
            v_rest: 0.0,
            learning_rate: 0.01,
            dt: 1.0,
            t_max_window: 100.0,
        }
    }
}

/// Analytic peak time of the (unnormalized) double-exponential kernel.
///
/// `t_max = (τ_m·τ_s / (τ_m − τ_s)) · ln(τ_m / τ_s)`. Positive whenever
/// `τ_m > τ_s > 0`.
#[must_use]
pub fn kernel_peak_time(cfg: &TempotronConfig) -> f32 {
    let ratio = cfg.tau_m / cfg.tau_s;
    (cfg.tau_m * cfg.tau_s / (cfg.tau_m - cfg.tau_s)) * ratio.ln()
}

/// Normalisation constant `V_0 = 1 / max_s [exp(−s/τ_m) − exp(−s/τ_s)]`.
///
/// Evaluated at the analytic peak time so the normalized kernel peaks at one.
#[must_use]
pub fn kernel_norm(cfg: &TempotronConfig) -> f32 {
    let t_peak = kernel_peak_time(cfg);
    let peak_val = (-t_peak / cfg.tau_m).exp() - (-t_peak / cfg.tau_s).exp();
    1.0 / peak_val
}

/// Normalized post-synaptic potential kernel `K(s)` (peak ≈ 1).
///
/// Returns `0` for `s < 0`.
#[must_use]
pub fn psp_kernel(s: f32, cfg: &TempotronConfig) -> f32 {
    if s < 0.0 {
        return 0.0;
    }
    let v0 = kernel_norm(cfg);
    v0 * ((-s / cfg.tau_m).exp() - (-s / cfg.tau_s).exp())
}

/// Validate Tempotron configuration shared across all methods.
fn validate_cfg(cfg: &TempotronConfig) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_m });
    }
    if cfg.tau_s <= 0.0 || !cfg.tau_s.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_s });
    }
    if (cfg.tau_m - cfg.tau_s).abs() < f32::EPSILON {
        // Normalisation divides by (τ_m − τ_s); equal constants are invalid.
        return Err(SnnError::OutOfRange {
            name: "tau_m==tau_s".into(),
            val: cfg.tau_m,
        });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if cfg.t_max_window <= 0.0 || !cfg.t_max_window.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "t_max_window".into(),
            val: cfg.t_max_window,
        });
    }
    if !cfg.v_th.is_finite() {
        return Err(SnnError::BadThreshold { v_th: cfg.v_th });
    }
    Ok(())
}

/// Number of grid samples in `[0, t_max_window)` at resolution `dt`.
fn grid_len(cfg: &TempotronConfig) -> usize {
    (cfg.t_max_window / cfg.dt).ceil() as usize
}

/// A Tempotron classifier holding one weight per afferent synapse.
#[derive(Debug, Clone)]
pub struct Tempotron {
    /// Per-synapse weights, length `n_synapses`.
    pub weights: Vec<f32>,
    /// Hyperparameters.
    pub cfg: TempotronConfig,
}

/// Small positive weight initialiser.
///
/// Starting from exactly zero leaves the voltage trace flat at `v_rest`, so the
/// peak-voltage argmax is degenerate and the gradient vanishes (a cold-start
/// fixed point). A tiny uniform positive weight breaks this symmetry while
/// keeping the neuron silent for sub-threshold patterns.
const DEFAULT_INIT_WEIGHT: f32 = 0.01;

impl Tempotron {
    /// Create a Tempotron with `n_synapses` small positive initial weights.
    ///
    /// The weights are seeded to `DEFAULT_INIT_WEIGHT` rather than exactly
    /// zero so the peak-voltage gradient is non-degenerate from the first
    /// presentation; the value is small enough that the neuron stays silent for
    /// sub-threshold patterns. Returns [`SnnError::BadDim`] when
    /// `n_synapses == 0` and validates `cfg`.
    pub fn new(n_synapses: usize, cfg: TempotronConfig) -> SnnResult<Self> {
        if n_synapses == 0 {
            return Err(SnnError::BadDim { got: n_synapses });
        }
        validate_cfg(&cfg)?;
        Ok(Self {
            weights: vec![DEFAULT_INIT_WEIGHT; n_synapses],
            cfg,
        })
    }

    /// Create a Tempotron from explicit initial weights.
    ///
    /// Returns [`SnnError::BadDim`] for an empty slice and validates `cfg`.
    pub fn with_weights(weights: Vec<f32>, cfg: TempotronConfig) -> SnnResult<Self> {
        if weights.is_empty() {
            return Err(SnnError::BadDim { got: 0 });
        }
        validate_cfg(&cfg)?;
        Ok(Self { weights, cfg })
    }

    /// Number of afferent synapses.
    #[must_use]
    pub fn n_synapses(&self) -> usize {
        self.weights.len()
    }

    /// Validate that a pattern references in-range synapses and finite times.
    fn validate_pattern(&self, pattern: &[(usize, f32)]) -> SnnResult<()> {
        let n = self.weights.len();
        for &(idx, t) in pattern {
            if idx >= n {
                return Err(SnnError::OutOfRange {
                    name: "synapse_index".into(),
                    val: idx as f32,
                });
            }
            if !t.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "spike_time".into(),
                    val: t,
                });
            }
        }
        Ok(())
    }

    /// Compute the membrane voltage `V(t)` sampled over the time grid
    /// `0, dt, 2·dt, …` within `[0, t_max_window)`.
    pub fn voltage_trace(&self, pattern: &[(usize, f32)]) -> SnnResult<Vec<f32>> {
        validate_cfg(&self.cfg)?;
        self.validate_pattern(pattern)?;
        let len = grid_len(&self.cfg);
        let mut trace = vec![self.cfg.v_rest; len];
        for (k, v) in trace.iter_mut().enumerate() {
            let t = k as f32 * self.cfg.dt;
            let mut acc = 0.0_f32;
            for &(idx, t_f) in pattern {
                // Only spikes strictly before t contribute (causal PSP).
                if t_f < t {
                    acc += self.weights[idx] * psp_kernel(t - t_f, &self.cfg);
                }
            }
            *v += acc;
        }
        Ok(trace)
    }

    /// Return the peak voltage and the time at which it occurs.
    pub fn max_voltage(&self, pattern: &[(usize, f32)]) -> SnnResult<(f32, f32)> {
        let trace = self.voltage_trace(pattern)?;
        let mut best_val = f32::NEG_INFINITY;
        let mut best_t = 0.0_f32;
        for (k, &v) in trace.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_t = k as f32 * self.cfg.dt;
            }
        }
        Ok((best_val, best_t))
    }

    /// Classify a pattern: `true` if the peak voltage reaches `v_th`.
    pub fn classify(&self, pattern: &[(usize, f32)]) -> SnnResult<bool> {
        let (peak, _) = self.max_voltage(pattern)?;
        Ok(peak >= self.cfg.v_th)
    }

    /// Per-synapse contribution `Σ_{t_i^f < t*} K(t* − t_i^f)` at the peak time.
    fn synapse_contributions(&self, pattern: &[(usize, f32)], t_star: f32) -> Vec<f32> {
        let mut contrib = vec![0.0_f32; self.weights.len()];
        for &(idx, t_f) in pattern {
            if t_f < t_star {
                contrib[idx] += psp_kernel(t_star - t_f, &self.cfg);
            }
        }
        contrib
    }

    /// Present a labelled pattern and apply the Tempotron learning rule.
    ///
    /// `target == true` means the neuron should fire ("+"), `false` means it
    /// should stay silent ("−"). Returns `true` when an update was applied
    /// (i.e. the current classification was wrong), `false` otherwise.
    pub fn train_pattern(&mut self, pattern: &[(usize, f32)], target: bool) -> SnnResult<bool> {
        validate_cfg(&self.cfg)?;
        self.validate_pattern(pattern)?;
        let (peak, t_star) = self.max_voltage(pattern)?;
        let fired = peak >= self.cfg.v_th;
        if fired == target {
            // Correct classification — no weight change.
            return Ok(false);
        }
        // Wrong: potentiate for a missed "+", depress for a spurious "−".
        let sign = if target { 1.0_f32 } else { -1.0_f32 };
        let contrib = self.synapse_contributions(pattern, t_star);
        for (w, &c) in self.weights.iter_mut().zip(contrib.iter()) {
            *w += sign * self.cfg.learning_rate * c;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TempotronConfig {
        TempotronConfig::default()
    }

    #[test]
    fn kernel_zero_for_negative_s() {
        let c = cfg();
        assert_eq!(psp_kernel(-1.0, &c), 0.0);
        assert_eq!(psp_kernel(-50.0, &c), 0.0);
    }

    #[test]
    fn kernel_peak_is_unity_at_peak_time() {
        let c = cfg();
        let t_peak = kernel_peak_time(&c);
        let v = psp_kernel(t_peak, &c);
        assert!((v - 1.0).abs() < 1e-4, "peak kernel value={v}, expected 1");
    }

    #[test]
    fn peak_time_formula_positive() {
        let c = cfg();
        let t_peak = kernel_peak_time(&c);
        assert!(t_peak > 0.0, "t_peak={t_peak}");
        assert!(t_peak.is_finite());
    }

    #[test]
    fn new_rejects_zero_synapses() {
        let err = Tempotron::new(0, cfg());
        assert!(matches!(err, Err(SnnError::BadDim { .. })));
    }

    #[test]
    fn voltage_trace_length_matches_window() {
        let c = cfg();
        let neuron = Tempotron::new(4, c).expect("new");
        let trace = neuron.voltage_trace(&[(0, 5.0)]).expect("trace");
        let expected = (c.t_max_window / c.dt).ceil() as usize;
        assert_eq!(trace.len(), expected);
    }

    #[test]
    fn classify_silent_on_empty_pattern() {
        let neuron = Tempotron::new(3, cfg()).expect("new");
        let fired = neuron.classify(&[]).expect("classify");
        assert!(!fired, "empty pattern must stay silent");
        let (peak, _) = neuron.max_voltage(&[]).expect("max");
        assert!((peak - cfg().v_rest).abs() < 1e-6);
    }

    #[test]
    fn train_positive_pattern_eventually_fires() {
        // Several synapses spiking early should, after enough "+" updates,
        // drive the neuron above threshold.
        let c = cfg();
        let mut neuron = Tempotron::new(5, c).expect("new");
        let pattern: Vec<(usize, f32)> = vec![(0, 2.0), (1, 3.0), (2, 4.0), (3, 5.0), (4, 6.0)];
        let mut fired = neuron.classify(&pattern).expect("classify");
        assert!(!fired, "should start silent with small initial weights");
        for _ in 0..2000 {
            neuron.train_pattern(&pattern, true).expect("train");
            fired = neuron.classify(&pattern).expect("classify");
            if fired {
                break;
            }
        }
        assert!(fired, "tempotron failed to learn the + pattern");
    }

    #[test]
    fn train_negative_pattern_eventually_silent() {
        // Start with large positive weights so the neuron fires, then teach it
        // to stay silent for the "−" pattern.
        let c = cfg();
        let mut neuron = Tempotron::with_weights(vec![2.0_f32; 5], c).expect("new");
        let pattern: Vec<(usize, f32)> = vec![(0, 2.0), (1, 3.0), (2, 4.0), (3, 5.0), (4, 6.0)];
        let mut fired = neuron.classify(&pattern).expect("classify");
        assert!(fired, "should start firing with large weights");
        for _ in 0..5000 {
            neuron.train_pattern(&pattern, false).expect("train");
            fired = neuron.classify(&pattern).expect("classify");
            if !fired {
                break;
            }
        }
        assert!(!fired, "tempotron failed to silence the − pattern");
    }

    #[test]
    fn correct_classification_yields_no_update() {
        // Small initial weights keep the neuron silent. A correctly-classified
        // "−" pattern must not change weights and must report no update.
        let neuron_cfg = cfg();
        let mut neuron = Tempotron::new(3, neuron_cfg).expect("new");
        let before = neuron.weights.clone();
        let updated = neuron
            .train_pattern(&[(0, 1.0), (1, 2.0)], false)
            .expect("train");
        assert!(!updated, "no update expected for correct classification");
        assert_eq!(neuron.weights, before);
    }

    #[test]
    fn bad_synapse_index_rejected() {
        let mut neuron = Tempotron::new(2, cfg()).expect("new");
        let err = neuron.train_pattern(&[(5, 1.0)], true);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
        let err2 = neuron.voltage_trace(&[(2, 1.0)]);
        assert!(matches!(err2, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn equal_tau_rejected() {
        let c = TempotronConfig {
            tau_m: 10.0,
            tau_s: 10.0,
            ..cfg()
        };
        let err = Tempotron::new(3, c);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn two_pattern_separation_converges() {
        // One "+" pattern (early spikes) and one "−" pattern (different
        // synapses) must be linearly separated within a bounded epoch budget.
        let c = cfg();
        let mut neuron = Tempotron::new(6, c).expect("new");
        let pat_plus: Vec<(usize, f32)> = vec![(0, 2.0), (1, 4.0), (2, 6.0)];
        let pat_minus: Vec<(usize, f32)> = vec![(3, 2.0), (4, 4.0), (5, 6.0)];
        let mut ok = false;
        for _ in 0..5000 {
            neuron.train_pattern(&pat_plus, true).expect("train+");
            neuron.train_pattern(&pat_minus, false).expect("train-");
            let plus_fires = neuron.classify(&pat_plus).expect("c+");
            let minus_fires = neuron.classify(&pat_minus).expect("c-");
            if plus_fires && !minus_fires {
                ok = true;
                break;
            }
        }
        assert!(ok, "failed to separate + and − patterns");
    }
}
