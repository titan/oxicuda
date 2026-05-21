//! Heterogeneous Leaky Integrate-and-Fire population.
//!
//! Each neuron `i` in a population of `N` units carries its own membrane time
//! constant `τ_m[i]` and spike threshold `v_th[i]`. The rest of the dynamics
//! (resting potential, integration step, reset mode) is shared across the
//! population, matching the LIF discretisation used in
//! [`crate::neuron::lif`].
//!
//! Discrete-time update for neuron `i` (`β_i = exp(-dt / τ_m[i])`):
//!
//! ```text
//! v_i_{t+1} = β_i · v_i_t + I_i_t
//! s_i       = (v_i_{t+1} ≥ v_th[i])
//! v_i_{t+1} ← v_rest                       if s_i and Hard reset
//! v_i_{t+1} ← v_i_{t+1} − v_th[i]          if s_i and Soft reset
//! ```
//!
//! Heterogeneous parameters are biologically motivated — cortical pyramidal
//! cells exhibit a wide range of membrane time constants (≈10–40 ms; see
//! Tripathy et al., *Front. Neuroinformatics* 2014, *NeuroElectro*) and
//! diverse thresholds. Per-neuron `τ_m` has also been used as a learnable
//! parameter in works such as Perez-Nieves et al., *Nature Communications*
//! 2021 ("Neural heterogeneity promotes robust learning").

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::ResetMode;

/// Per-neuron heterogeneous LIF configuration.
///
/// `tau_m.len()` and `v_th.len()` define the population size `N` and **must**
/// match. Use [`HetLifConfig::validate`] to verify a configuration before any
/// step calls; the step routine re-validates on every invocation for safety.
#[derive(Debug, Clone)]
pub struct HetLifConfig {
    /// Per-neuron membrane time constants `τ_m[i] > 0`, length `N`.
    pub tau_m: Vec<f64>,
    /// Per-neuron spike thresholds `v_th[i] > 0`, length `N`.
    pub v_th: Vec<f64>,
    /// Shared resting potential.
    pub v_rest: f64,
    /// Shared integration step `dt > 0`.
    pub dt: f64,
    /// Reset mode applied after a spike (shared).
    pub reset: ResetMode,
}

impl HetLifConfig {
    /// Build a homogeneous configuration where every neuron shares `tau_m` and
    /// `v_th`. Useful as a sanity baseline against scalar [`crate::neuron::lif`].
    #[must_use]
    pub fn homogeneous(
        n: usize,
        tau_m: f64,
        v_th: f64,
        v_rest: f64,
        dt: f64,
        reset: ResetMode,
    ) -> Self {
        Self {
            tau_m: vec![tau_m; n],
            v_th: vec![v_th; n],
            v_rest,
            dt,
            reset,
        }
    }

    /// Population size `N = tau_m.len()`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.tau_m.len()
    }

    /// Validate the configuration. Returns:
    /// - [`SnnError::IncompatibleLength`] if `tau_m.len() != v_th.len()`,
    /// - [`SnnError::BadDt`] if `dt ≤ 0` or non-finite,
    /// - [`SnnError::BadTau`] for any non-positive / non-finite `τ_m[i]`,
    /// - [`SnnError::BadThreshold`] for any non-positive / non-finite `v_th[i]`,
    /// - [`SnnError::OutOfRange`] if `v_rest` is non-finite.
    pub fn validate(&self) -> SnnResult<()> {
        if self.tau_m.len() != self.v_th.len() {
            return Err(SnnError::IncompatibleLength {
                a: self.tau_m.len(),
                b: self.v_th.len(),
            });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SnnError::BadDt { dt: self.dt as f32 });
        }
        if !self.v_rest.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "v_rest".into(),
                val: self.v_rest as f32,
            });
        }
        for &t in &self.tau_m {
            if t <= 0.0 || !t.is_finite() {
                return Err(SnnError::BadTau { tau: t as f32 });
            }
        }
        for &th in &self.v_th {
            if th <= 0.0 || !th.is_finite() {
                return Err(SnnError::BadThreshold { v_th: th as f32 });
            }
        }
        Ok(())
    }
}

/// Mutable state for a heterogeneous LIF population.
#[derive(Debug, Clone)]
pub struct HetLifState {
    /// Per-neuron membrane potential, length `N`.
    pub v: Vec<f64>,
}

impl HetLifState {
    /// Allocate state for `n` neurons with `v` initialised to `v_init`.
    #[must_use]
    pub fn new(n: usize, v_init: f64) -> Self {
        Self { v: vec![v_init; n] }
    }

    /// Population size `N`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.v.len()
    }
}

/// Advance a heterogeneous LIF population by one timestep.
///
/// Returns a fresh `Vec<bool>` of length `N` indicating which neurons spiked.
/// The membrane potentials in `state.v` are updated in place.
pub fn het_lif_step(
    state: &mut HetLifState,
    input: &[f64],
    cfg: &HetLifConfig,
) -> SnnResult<Vec<bool>> {
    cfg.validate()?;
    let n = cfg.n();
    if state.v.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.v.len(),
        });
    }
    if input.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: input.len(),
        });
    }
    let mut spikes = vec![false; n];
    for i in 0..n {
        let v_i = state.v[i];
        let in_i = input[i];
        if !in_i.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "input".into(),
                val: in_i as f32,
            });
        }
        let beta_i = (-cfg.dt / cfg.tau_m[i]).exp();
        let v_new = beta_i * v_i + in_i;
        let spike = v_new >= cfg.v_th[i];
        let v_after = if spike {
            match cfg.reset {
                ResetMode::Hard => cfg.v_rest,
                ResetMode::Soft => v_new - cfg.v_th[i],
            }
        } else {
            v_new
        };
        state.v[i] = v_after;
        spikes[i] = spike;
    }
    Ok(spikes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_homo(n: usize) -> HetLifConfig {
        HetLifConfig::homogeneous(n, 20.0, 1.0, 0.0, 1.0, ResetMode::Hard)
    }

    #[test]
    fn n_zero_is_valid() {
        let cfg = HetLifConfig {
            tau_m: Vec::new(),
            v_th: Vec::new(),
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        cfg.validate().expect("empty population is valid");
        let mut state = HetLifState::new(0, 0.0);
        let spikes = het_lif_step(&mut state, &[], &cfg).expect("step");
        assert!(spikes.is_empty());
    }

    #[test]
    fn rejects_mismatched_tau_and_vth_lengths() {
        let cfg = HetLifConfig {
            tau_m: vec![20.0; 3],
            v_th: vec![1.0; 2],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let err = cfg.validate();
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn rejects_input_length_mismatch() {
        let cfg = cfg_homo(2);
        let mut state = HetLifState::new(2, 0.0);
        let input = vec![0.1_f64; 3];
        let err = het_lif_step(&mut state, &input, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn rejects_state_length_mismatch() {
        let cfg = cfg_homo(2);
        let mut state = HetLifState::new(3, 0.0);
        let input = vec![0.1_f64; 2];
        // input.len()=2 matches cfg, but state.v.len()=3 ≠ 2.
        let err = het_lif_step(&mut state, &input, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn rejects_non_positive_tau_m() {
        let cfg = HetLifConfig {
            tau_m: vec![20.0, 0.0, 30.0],
            v_th: vec![1.0; 3],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let err = cfg.validate();
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_non_positive_v_th() {
        let cfg = HetLifConfig {
            tau_m: vec![20.0; 3],
            v_th: vec![1.0, -0.1, 1.0],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let err = cfg.validate();
        assert!(matches!(err, Err(SnnError::BadThreshold { .. })));
    }

    #[test]
    fn rejects_zero_dt() {
        let cfg = HetLifConfig {
            tau_m: vec![20.0; 2],
            v_th: vec![1.0; 2],
            v_rest: 0.0,
            dt: 0.0,
            reset: ResetMode::Hard,
        };
        let err = cfg.validate();
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn homogeneous_matches_scalar_lif_50_steps() {
        use crate::neuron::lif::{LifConfig, LifState, lif_step};
        let n = 4;
        let cfg_h = HetLifConfig::homogeneous(n, 20.0, 1.0, 0.0, 1.0, ResetMode::Hard);
        let cfg_s = LifConfig {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state_h = HetLifState::new(n, 0.0);
        let mut state_s = LifState::new(n);
        // Deterministic input: a different fixed value per neuron.
        let input_f64: Vec<f64> = (0..n).map(|i| 0.1 + 0.05 * i as f64).collect();
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
        let mut spikes_s = vec![0.0_f32; n];
        for _ in 0..50 {
            let spikes_h = het_lif_step(&mut state_h, &input_f64, &cfg_h).expect("step");
            lif_step(&mut state_s, &input_f32, &cfg_s, &mut spikes_s).expect("step");
            for i in 0..n {
                let lif_spiked = spikes_s[i] == 1.0;
                assert_eq!(
                    spikes_h[i], lif_spiked,
                    "spike mismatch at neuron {i}: het={}, lif={}",
                    spikes_h[i], lif_spiked
                );
                let v_diff = (state_h.v[i] - state_s.v[i] as f64).abs();
                assert!(v_diff < 1e-4, "v mismatch at neuron {i}: diff={v_diff}");
            }
        }
    }

    #[test]
    fn slow_tau_spikes_later_than_fast_tau() {
        // Notation: "fast tau" = small τ (heavy leak); "slow tau" = large τ
        // (little leak).  β = exp(-dt/τ); the asymptotic membrane is
        // I/(1-β).  With larger τ → larger β → faster accumulation toward a
        // higher steady state → reaches v_th sooner.  Hence we expect the
        // *small*-τ neuron to lag the *large*-τ neuron in time-to-first-spike.
        let cfg = HetLifConfig {
            tau_m: vec![2.0, 50.0], // neuron 0: heavy leak; neuron 1: little leak
            v_th: vec![1.0, 1.0],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = HetLifState::new(2, 0.0);
        // Input must be large enough so the small-τ neuron (saturating at
        // I/(1-exp(-0.5)) ≈ I/0.3935) eventually crosses 1.0.
        let input = vec![0.5_f64, 0.5_f64];
        let mut first_spike = [None, None];
        for t in 0..400 {
            let spikes = het_lif_step(&mut state, &input, &cfg).expect("step");
            for i in 0..2 {
                if spikes[i] && first_spike[i].is_none() {
                    first_spike[i] = Some(t);
                }
            }
            if first_spike.iter().all(|x| x.is_some()) {
                break;
            }
        }
        let fast_leak = first_spike[0].expect("small-tau (fast-leak) neuron must spike");
        let slow_leak = first_spike[1].expect("large-tau (slow-leak) neuron must spike");
        assert!(
            slow_leak < fast_leak,
            "large-tau neuron should spike first: small_tau(t)={fast_leak}, large_tau(t)={slow_leak}"
        );
    }

    #[test]
    fn hard_reset_clears_to_v_rest() {
        let cfg = HetLifConfig {
            tau_m: vec![1e9; 2],
            v_th: vec![0.5, 0.5],
            v_rest: -0.25,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = HetLifState::new(2, 0.4);
        let input = vec![0.2_f64, 0.2];
        let spikes = het_lif_step(&mut state, &input, &cfg).expect("step");
        assert!(spikes[0] && spikes[1]);
        for &v in &state.v {
            assert!((v - cfg.v_rest).abs() < 1e-9);
        }
    }

    #[test]
    fn soft_reset_subtracts_per_neuron_v_th() {
        let cfg = HetLifConfig {
            tau_m: vec![1e9; 2],
            v_th: vec![1.0, 2.0],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Soft,
        };
        let mut state = HetLifState::new(2, 0.0);
        // Neuron 0: v_new = 1.3, spike, soft ⇒ 0.3.
        // Neuron 1: v_new = 2.3, spike, soft ⇒ 0.3.
        let input = vec![1.3_f64, 2.3];
        let spikes = het_lif_step(&mut state, &input, &cfg).expect("step");
        assert!(spikes[0] && spikes[1]);
        assert!((state.v[0] - 0.3).abs() < 1e-9);
        assert!((state.v[1] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn deterministic_over_repeated_runs() {
        let cfg = HetLifConfig {
            tau_m: vec![10.0, 25.0, 40.0],
            v_th: vec![1.0, 1.0, 1.0],
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let inputs = [
            [0.1_f64, 0.2, 0.3],
            [0.5, 0.6, 0.7],
            [0.0, 0.4, 1.1],
            [0.8, 0.2, 0.0],
        ];
        let mut state_a = HetLifState::new(3, 0.0);
        let mut state_b = HetLifState::new(3, 0.0);
        for input in &inputs {
            let sa = het_lif_step(&mut state_a, input, &cfg).expect("step");
            let sb = het_lif_step(&mut state_b, input, &cfg).expect("step");
            assert_eq!(sa, sb);
            assert_eq!(state_a.v, state_b.v);
        }
    }

    #[test]
    fn large_n_round_trip() {
        let n = 100usize;
        let tau_m: Vec<f64> = (0..n).map(|i| 5.0 + 0.5 * i as f64).collect();
        let v_th: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let cfg = HetLifConfig {
            tau_m,
            v_th,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        cfg.validate().expect("valid");
        let mut state = HetLifState::new(n, 0.0);
        let input: Vec<f64> = (0..n).map(|i| 0.2 + 0.001 * i as f64).collect();
        let mut total_spikes = 0usize;
        for _ in 0..100 {
            let spikes = het_lif_step(&mut state, &input, &cfg).expect("step");
            for s in spikes {
                if s {
                    total_spikes += 1;
                }
            }
        }
        // Ensure every neuron's state remained finite and that *some* spikes
        // occurred (population is well within the firing regime).
        assert!(total_spikes > 0, "expected non-zero population activity");
        for &v in &state.v {
            assert!(v.is_finite(), "v not finite: {v}");
        }
    }

    #[test]
    fn rejects_non_finite_input() {
        let cfg = cfg_homo(2);
        let mut state = HetLifState::new(2, 0.0);
        let input = vec![0.5_f64, f64::NAN];
        let err = het_lif_step(&mut state, &input, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn n_helpers_report_population_size() {
        let cfg = cfg_homo(5);
        assert_eq!(cfg.n(), 5);
        let state = HetLifState::new(5, 0.0);
        assert_eq!(state.n(), 5);
    }

    #[test]
    fn rejects_non_finite_v_rest() {
        let cfg = HetLifConfig {
            tau_m: vec![20.0; 2],
            v_th: vec![1.0; 2],
            v_rest: f64::NAN,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let err = cfg.validate();
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
