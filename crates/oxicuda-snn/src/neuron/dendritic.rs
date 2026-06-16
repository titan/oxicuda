//! Two-layer nonlinear dendritic neuron (Poirazi, Brannon & Mel 2003).
//!
//! Reference: Poirazi, Brannon & Mel — *"Pyramidal Neuron as Two-Layer Neural
//! Network"* (Neuron, 2003). A CA1 pyramidal cell is well-approximated by a
//! two-stage model: each thin dendritic *subunit* first integrates its own
//! synaptic inputs through a *sigmoidal* nonlinearity (local NMDA-spike-driven
//! supralinear summation), and the soma then performs a second, weighted
//! summation of the subunit outputs which drives spike generation.
//!
//! Subunit `k` over its `m_k` synapses with weights `w_k` and presynaptic
//! activations `x`:
//!
//! ```text
//! a_k   = Σ_j w_k[j] · x[idx_k[j]]                       # local linear drive
//! g_k   = s · σ(α·(a_k − β))                             # sigmoid subunit, gain s
//! σ(z)  = 1 / (1 + exp(−z))
//! ```
//!
//! Soma (a leaky integrate-and-fire membrane integrating the subunit currents):
//!
//! ```text
//! I_soma   = Σ_k w_soma[k] · g_k + I_ext
//! v_{t+1}  = ρ · v_t + I_soma,     ρ = exp(-dt / τ_m)
//! s_{t+1}  = (v_{t+1} ≥ v_th);  reset to v_rest on spike (hard reset)
//! ```
//!
//! Disabling the dendritic nonlinearity (`α → 0` with appropriate gain, or a
//! single subunit spanning all synapses) recovers a point-neuron LIF; the
//! supralinear sigmoid is what gives the cell its enhanced pattern-storage
//! capacity reported by Poirazi et al.

use crate::error::{SnnError, SnnResult};

/// Numerically-stable logistic sigmoid `σ(z) = 1/(1+e^{−z})`.
#[must_use]
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// A single nonlinear dendritic subunit: a weighted set of synapses feeding a
/// sigmoidal nonlinearity.
#[derive(Debug, Clone)]
pub struct DendriticSubunit {
    /// Indices into the presynaptic activation vector for this subunit's
    /// synapses, length `m_k`.
    pub input_indices: Vec<usize>,
    /// Synaptic weights aligned with `input_indices`, length `m_k`.
    pub weights: Vec<f64>,
    /// Output gain `s` scaling the sigmoid (max subunit current).
    pub gain: f64,
    /// Sigmoid slope `α` (steepness of the supralinear transition).
    pub slope: f64,
    /// Sigmoid offset `β` (local-drive value at the half-activation point).
    pub offset: f64,
}

impl DendriticSubunit {
    /// Build a subunit; `input_indices` and `weights` must be the same length.
    #[must_use]
    pub fn new(
        input_indices: Vec<usize>,
        weights: Vec<f64>,
        gain: f64,
        slope: f64,
        offset: f64,
    ) -> Self {
        Self {
            input_indices,
            weights,
            gain,
            slope,
            offset,
        }
    }

    /// Evaluate the subunit output `g_k = gain · σ(slope·(aₖ − offset))` for the
    /// supplied presynaptic activation vector.
    fn evaluate(&self, x: &[f64]) -> SnnResult<f64> {
        if self.input_indices.len() != self.weights.len() {
            return Err(SnnError::IncompatibleLength {
                a: self.input_indices.len(),
                b: self.weights.len(),
            });
        }
        let mut drive = 0.0_f64;
        for (&idx, &w) in self.input_indices.iter().zip(self.weights.iter()) {
            if idx >= x.len() {
                return Err(SnnError::OutOfRange {
                    name: "input_index".into(),
                    val: idx as f32,
                });
            }
            drive += w * x[idx];
        }
        Ok(self.gain * sigmoid(self.slope * (drive - self.offset)))
    }
}

/// A two-layer dendritic neuron: a bank of nonlinear subunits and an LIF soma.
#[derive(Debug, Clone)]
pub struct DendriticNeuron {
    /// Dendritic subunits (first layer).
    pub subunits: Vec<DendriticSubunit>,
    /// Somatic weights `w_soma[k]` applied to each subunit output, length =
    /// `subunits.len()`.
    pub soma_weights: Vec<f64>,
    /// Membrane time constant `τ_m` of the soma (`> 0`).
    pub tau_m: f64,
    /// Somatic spike threshold `v_th`.
    pub v_th: f64,
    /// Somatic resting / reset potential `v_rest`.
    pub v_rest: f64,
    /// Integration step `dt` (`> 0`).
    pub dt: f64,
    /// Somatic membrane potential `v` (mutable state).
    pub v: f64,
}

impl DendriticNeuron {
    /// Construct a dendritic neuron from its subunits and somatic weights.
    ///
    /// The membrane potential is initialised to `v_rest`.
    #[must_use]
    pub fn new(
        subunits: Vec<DendriticSubunit>,
        soma_weights: Vec<f64>,
        tau_m: f64,
        v_th: f64,
        v_rest: f64,
        dt: f64,
    ) -> Self {
        Self {
            subunits,
            soma_weights,
            tau_m,
            v_th,
            v_rest,
            dt,
            v: v_rest,
        }
    }

    /// Membrane decay factor `ρ = exp(-dt / τ_m)`.
    #[must_use]
    pub fn rho(&self) -> f64 {
        (-self.dt / self.tau_m).exp()
    }

    /// Validate structural and numerical parameters.
    fn validate(&self) -> SnnResult<()> {
        if self.subunits.is_empty() {
            return Err(SnnError::BadDim { got: 0 });
        }
        if self.soma_weights.len() != self.subunits.len() {
            return Err(SnnError::IncompatibleLength {
                a: self.subunits.len(),
                b: self.soma_weights.len(),
            });
        }
        if self.tau_m <= 0.0 || !self.tau_m.is_finite() {
            return Err(SnnError::BadTau {
                tau: self.tau_m as f32,
            });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SnnError::BadDt { dt: self.dt as f32 });
        }
        if !self.v_th.is_finite() {
            return Err(SnnError::BadThreshold {
                v_th: self.v_th as f32,
            });
        }
        if !self.v_rest.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "v_rest".into(),
                val: self.v_rest as f32,
            });
        }
        for w in &self.soma_weights {
            if !w.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "soma_weight".into(),
                    val: *w as f32,
                });
            }
        }
        Ok(())
    }

    /// Evaluate the dendritic subunit outputs `g_k` for a presynaptic
    /// activation vector `x`, returning a length-`subunits.len()` vector.
    ///
    /// # Errors
    /// [`SnnError::BadDim`], [`SnnError::IncompatibleLength`],
    /// [`SnnError::OutOfRange`] for invalid structure / out-of-range indices.
    pub fn dendritic_outputs(&self, x: &[f64]) -> SnnResult<Vec<f64>> {
        if self.subunits.is_empty() {
            return Err(SnnError::BadDim { got: 0 });
        }
        for &xi in x {
            if !xi.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "x".into(),
                    val: xi as f32,
                });
            }
        }
        let mut out = Vec::with_capacity(self.subunits.len());
        for su in &self.subunits {
            out.push(su.evaluate(x)?);
        }
        Ok(out)
    }

    /// Somatic input current `I_soma = Σ_k w_soma[k]·g_k + I_ext`.
    ///
    /// # Errors
    /// As [`DendriticNeuron::dendritic_outputs`].
    pub fn soma_current(&self, x: &[f64], i_ext: f64) -> SnnResult<f64> {
        let g = self.dendritic_outputs(x)?;
        let mut i_soma = i_ext;
        for (&w, &gk) in self.soma_weights.iter().zip(g.iter()) {
            i_soma += w * gk;
        }
        Ok(i_soma)
    }

    /// Advance the neuron by one timestep given presynaptic activations `x` and
    /// an external somatic current `i_ext`. Returns the boolean spike indicator.
    ///
    /// The two-layer cascade is evaluated (dendrites → soma), the LIF membrane
    /// is integrated, and a hard reset to `v_rest` is applied on a spike.
    ///
    /// # Errors
    /// [`SnnError::BadTau`], [`SnnError::BadDt`], [`SnnError::BadThreshold`],
    /// [`SnnError::IncompatibleLength`], [`SnnError::OutOfRange`].
    pub fn step(&mut self, x: &[f64], i_ext: f64) -> SnnResult<bool> {
        self.validate()?;
        if !i_ext.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "i_ext".into(),
                val: i_ext as f32,
            });
        }
        let i_soma = self.soma_current(x, i_ext)?;
        let v_new = self.rho() * self.v + i_soma;
        let spike = v_new >= self.v_th;
        self.v = if spike { self.v_rest } else { v_new };
        Ok(spike)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two subunits, each over two distinct synapses, equal somatic weights.
    fn two_subunit_neuron() -> DendriticNeuron {
        let su0 = DendriticSubunit::new(vec![0, 1], vec![1.0, 1.0], 1.0, 4.0, 0.5);
        let su1 = DendriticSubunit::new(vec![2, 3], vec![1.0, 1.0], 1.0, 4.0, 0.5);
        DendriticNeuron::new(vec![su0, su1], vec![1.0, 1.0], 20.0, 1.0, 0.0, 1.0)
    }

    #[test]
    fn sigmoid_is_monotone_and_bounded() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(-50.0) >= 0.0 && sigmoid(-50.0) < 1e-10);
        assert!(sigmoid(50.0) <= 1.0 && sigmoid(50.0) > 1.0 - 1e-10);
        assert!(sigmoid(1.0) > sigmoid(-1.0));
        // No overflow / NaN at extreme arguments.
        assert!(sigmoid(1e6).is_finite());
        assert!(sigmoid(-1e6).is_finite());
    }

    #[test]
    fn rejects_empty_subunits() {
        let mut n = DendriticNeuron::new(Vec::new(), Vec::new(), 20.0, 1.0, 0.0, 1.0);
        assert!(matches!(
            n.step(&[0.0; 4], 0.0),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn rejects_soma_weight_length_mismatch() {
        let su = DendriticSubunit::new(vec![0], vec![1.0], 1.0, 1.0, 0.0);
        let mut n = DendriticNeuron::new(vec![su], vec![1.0, 1.0], 20.0, 1.0, 0.0, 1.0);
        assert!(matches!(
            n.step(&[1.0], 0.0),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn rejects_zero_tau() {
        let mut n = two_subunit_neuron();
        n.tau_m = 0.0;
        assert!(matches!(
            n.step(&[1.0; 4], 0.0),
            Err(SnnError::BadTau { .. })
        ));
    }

    #[test]
    fn rejects_zero_dt() {
        let mut n = two_subunit_neuron();
        n.dt = 0.0;
        assert!(matches!(
            n.step(&[1.0; 4], 0.0),
            Err(SnnError::BadDt { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_index() {
        // Subunit references synapse index 5 but x has length 4.
        let su = DendriticSubunit::new(vec![5], vec![1.0], 1.0, 1.0, 0.0);
        let n = DendriticNeuron::new(vec![su], vec![1.0], 20.0, 1.0, 0.0, 1.0);
        assert!(matches!(
            n.dendritic_outputs(&[0.0; 4]),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_input() {
        let n = two_subunit_neuron();
        assert!(matches!(
            n.dendritic_outputs(&[f64::NAN, 0.0, 0.0, 0.0]),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rho_matches_formula() {
        let mut n = two_subunit_neuron();
        n.tau_m = 10.0;
        n.dt = 1.0;
        assert!((n.rho() - (-0.1_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn dendritic_outputs_bounded_by_gain() {
        let n = two_subunit_neuron();
        // Drive both subunits very hard ⇒ outputs saturate near gain (=1).
        let g = n.dendritic_outputs(&[10.0, 10.0, 10.0, 10.0]).expect("g");
        assert_eq!(g.len(), 2);
        for &gk in &g {
            assert!((0.0..=1.0).contains(&gk), "gk out of [0,gain]: {gk}");
            assert!(gk > 0.99, "saturated subunit should approach gain: {gk}");
        }
        // Drive both very negative ⇒ outputs near zero.
        let g_lo = n
            .dendritic_outputs(&[-10.0, -10.0, -10.0, -10.0])
            .expect("g");
        for &gk in &g_lo {
            assert!(gk < 0.01, "starved subunit should approach 0: {gk}");
        }
    }

    #[test]
    fn supralinear_dendritic_integration() {
        // Poirazi 2003 key property: clustered input to ONE subunit produces a
        // larger somatic drive than the same total input spread across subunits,
        // because of the per-subunit sigmoid supralinearity below half-max.
        let su0 = DendriticSubunit::new(vec![0, 1], vec![1.0, 1.0], 2.0, 6.0, 1.0);
        let su1 = DendriticSubunit::new(vec![2, 3], vec![1.0, 1.0], 2.0, 6.0, 1.0);
        let n = DendriticNeuron::new(vec![su0, su1], vec![1.0, 1.0], 20.0, 1.0, 0.0, 1.0);
        // Clustered: both active synapses in subunit 0.
        let clustered = n.soma_current(&[1.0, 1.0, 0.0, 0.0], 0.0).expect("c");
        // Distributed: one active synapse per subunit (same total input).
        let distributed = n.soma_current(&[1.0, 0.0, 1.0, 0.0], 0.0).expect("d");
        assert!(
            clustered > distributed,
            "clustered drive {clustered} should exceed distributed {distributed}"
        );
    }

    #[test]
    fn subthreshold_does_not_spike() {
        let mut n = two_subunit_neuron();
        n.tau_m = 1e9; // no leak
        n.v_th = 100.0; // unreachable
        let spike = n.step(&[1.0, 1.0, 1.0, 1.0], 0.0).expect("step");
        assert!(!spike);
        assert!(n.v < n.v_th);
    }

    #[test]
    fn strong_drive_spikes_and_resets() {
        let mut n = two_subunit_neuron();
        n.tau_m = 1e9;
        n.v_th = 1.0;
        n.v_rest = 0.0;
        // Saturated subunits give ~1 each, somatic weight 1 each ⇒ I_soma ≈ 2.
        let spike = n.step(&[10.0, 10.0, 10.0, 10.0], 0.0).expect("step");
        assert!(spike);
        assert!(
            (n.v - n.v_rest).abs() < 1e-12,
            "must reset to v_rest, v={}",
            n.v
        );
    }

    #[test]
    fn external_current_contributes_to_soma() {
        let mut n = two_subunit_neuron();
        n.tau_m = 1e9;
        n.v_th = 5.0;
        // Subunits alone give ≤ 2; add a large external current to cross v_th.
        let spike = n.step(&[0.0, 0.0, 0.0, 0.0], 10.0).expect("step");
        assert!(spike, "external current should drive a spike");
    }

    #[test]
    fn membrane_finite_over_long_run() {
        let mut n = two_subunit_neuron();
        for t in 0..500 {
            let phase = (t as f64 * 0.2).sin();
            let x = [0.5 + 0.5 * phase, 0.5, 0.5 - 0.5 * phase, 0.5];
            let _ = n.step(&x, 0.1).expect("step");
            assert!(n.v.is_finite(), "membrane not finite at t={t}: {}", n.v);
        }
    }

    #[test]
    fn rejects_non_finite_external_current() {
        let mut n = two_subunit_neuron();
        assert!(matches!(
            n.step(&[0.0; 4], f64::INFINITY),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn deterministic_under_same_input() {
        let inputs = [
            [0.3_f64, 0.5, 0.7, 0.1],
            [0.9, 0.2, 0.4, 0.6],
            [0.1, 0.8, 0.3, 0.5],
        ];
        let mut a = two_subunit_neuron();
        let mut b = two_subunit_neuron();
        for x in &inputs {
            let sa = a.step(x, 0.2).expect("step");
            let sb = b.step(x, 0.2).expect("step");
            assert_eq!(sa, sb);
            assert!((a.v - b.v).abs() < 1e-15);
        }
    }
}
