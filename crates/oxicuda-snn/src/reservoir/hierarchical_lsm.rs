#![allow(clippy::needless_range_loop)]
//! Hierarchical / deep Liquid State Machine — a stack of spiking reservoirs.
//!
//! Deep reservoir computing (Gallicchio, Micheli & Pedrelli 2017, "Deep
//! Reservoir Computing: A Critical Experimental Analysis"; Gallicchio & Micheli
//! 2017, "Echo State Property of Deep Reservoir Networks") stacks several
//! untrained recurrent reservoirs so that successively higher layers develop
//! progressively longer effective time-scales and more abstract dynamics. Only
//! a single linear readout — fitted on the *concatenation* of all layer states —
//! is trained.
//!
//! Here the construction is adapted to **spiking** reservoirs ([`crate::reservoir::lsm::Lsm`]): layer
//! `0` is driven by the external input, and each subsequent layer `L` receives
//! the *spike train* emitted by layer `L − 1`:
//!
//! ```text
//! s^(0)_t = LSM_0( u_t )                      (external input → layer 0)
//! s^(L)_t = LSM_L( s^(L-1)_t )    for L ≥ 1   (spikes cascade upward)
//! ```
//!
//! The deep-reservoir feature vector handed to the readout is the concatenation
//! of every layer's persistent membrane state:
//!
//! ```text
//! φ_t = [ x^(0)_t ‖ x^(1)_t ‖ … ‖ x^(L-1)_t ] ∈ ℝ^(Σ Nₗ)
//! ```
//!
//! Dimensional chaining is enforced at construction: `cfg.in_dim` must equal the
//! external input width, and each higher layer's input width must equal the
//! neuron count of the layer below it.

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::LifConfig;
use crate::reservoir::lsm::{Lsm, LsmConfig};

/// Configuration of a [`HierarchicalLsm`].
#[derive(Debug, Clone)]
pub struct HierarchicalLsmConfig {
    /// Per-layer reservoir configurations, ordered from the input layer upward.
    pub layer_configs: Vec<LsmConfig>,
    /// Dimensionality of the external input feeding layer `0`.
    pub in_dim: usize,
}

impl Default for HierarchicalLsmConfig {
    fn default() -> Self {
        // A 3-layer pyramid of shrinking reservoirs driven by a 4-d input.
        let mk = |n: usize, seed: u64| LsmConfig {
            n_neurons: n,
            density: 0.1,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed,
        };
        Self {
            layer_configs: vec![mk(120, 1), mk(80, 2), mk(60, 3)],
            in_dim: 4,
        }
    }
}

impl HierarchicalLsmConfig {
    /// Validate the stack: at least one layer and a positive external input
    /// dimension. (Per-layer dimensional chaining is verified in
    /// [`HierarchicalLsm::new`], where the constructed reservoirs expose their
    /// neuron counts.)
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadDim`] if there are no layers or `in_dim == 0`.
    pub fn validate(&self) -> SnnResult<()> {
        if self.layer_configs.is_empty() {
            return Err(SnnError::BadDim { got: 0 });
        }
        if self.in_dim == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        Ok(())
    }
}

/// A vertical stack of spiking reservoirs forming a deep liquid state machine.
#[derive(Debug, Clone)]
pub struct HierarchicalLsm {
    /// Reservoirs ordered from the input layer (index `0`) upward.
    layers: Vec<Lsm>,
    /// LIF dynamics shared by every layer.
    lif_cfg: LifConfig,
    /// External input dimensionality (= `layers[0].in_dim`).
    in_dim: usize,
    /// Cached per-layer neuron counts, for cheap state-length queries.
    layer_sizes: Vec<usize>,
}

impl HierarchicalLsm {
    /// Build a deep LSM, constructing each layer in turn and verifying that the
    /// input width of layer `L` equals the neuron count of layer `L − 1`.
    ///
    /// Layer `0` is built with input width `cfg.in_dim`. Each reservoir is seeded
    /// independently from its own [`LsmConfig::seed`], so the whole stack is
    /// deterministic given the configuration.
    ///
    /// # Errors
    ///
    /// * Propagates [`HierarchicalLsmConfig::validate`] errors.
    /// * [`SnnError::BadDim`] if any layer reports zero neurons.
    /// * Propagates any error from [`Lsm::new`].
    pub fn new(cfg: &HierarchicalLsmConfig, lif_cfg: &LifConfig) -> SnnResult<Self> {
        cfg.validate()?;

        let mut layers: Vec<Lsm> = Vec::with_capacity(cfg.layer_configs.len());
        let mut layer_sizes: Vec<usize> = Vec::with_capacity(cfg.layer_configs.len());

        // The input width of the next layer to build: external `in_dim` for the
        // first layer, then the neuron count of the layer just constructed.
        let mut next_in_dim = cfg.in_dim;
        for layer_cfg in &cfg.layer_configs {
            let lsm = Lsm::new(next_in_dim, layer_cfg, lif_cfg)?;
            let n = lsm.n;
            if n == 0 {
                return Err(SnnError::BadDim { got: 0 });
            }
            next_in_dim = n;
            layer_sizes.push(n);
            layers.push(lsm);
        }

        Ok(Self {
            layers,
            lif_cfg: *lif_cfg,
            in_dim: cfg.in_dim,
            layer_sizes,
        })
    }

    /// Number of stacked reservoirs.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// External input dimensionality.
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Neuron count of layer `idx`.
    ///
    /// # Errors
    ///
    /// [`SnnError::LayerOutOfRange`] if `idx >= num_layers`.
    pub fn layer_size(&self, idx: usize) -> SnnResult<usize> {
        self.layer_sizes
            .get(idx)
            .copied()
            .ok_or(SnnError::LayerOutOfRange {
                idx,
                num_layers: self.layers.len(),
            })
    }

    /// Total length of the concatenated deep-reservoir feature vector,
    /// `Σ Nₗ`.
    #[must_use]
    pub fn feature_len(&self) -> usize {
        self.layer_sizes.iter().sum()
    }

    /// Advance every layer by one timestep, cascading spikes upward.
    ///
    /// Returns one spike vector per layer (in layer order); spikes from layer
    /// `L` become the input to layer `L + 1` within the same call.
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadShape`] if `input.len() != in_dim`.
    /// * Propagates any error from [`Lsm::forward_step`].
    pub fn forward_step(&mut self, input: &[f32]) -> SnnResult<Vec<Vec<f32>>> {
        if input.len() != self.in_dim {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: input.len(),
            });
        }

        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        // `current` is the input to the layer being processed. It starts as a
        // copy of the external input and is replaced by each layer's spikes.
        let mut current: Vec<f32> = input.to_vec();
        for layer in &mut self.layers {
            let mut spikes = vec![0.0_f32; layer.n];
            layer.forward_step(&current, &self.lif_cfg, &mut spikes)?;
            current = spikes.clone();
            outputs.push(spikes);
        }
        Ok(outputs)
    }

    /// Concatenated persistent membrane state across all layers — the deep
    /// reservoir readout feature vector `φ_t` of length [`feature_len`](Self::feature_len).
    #[must_use]
    pub fn collected_state(&self) -> Vec<f32> {
        let mut feat = Vec::with_capacity(self.feature_len());
        for layer in &self.layers {
            feat.extend_from_slice(&layer.state.v);
        }
        feat
    }

    /// Concatenated *last spike pattern* across all layers — an alternative
    /// binary deep feature vector of length [`feature_len`](Self::feature_len).
    #[must_use]
    pub fn collected_spikes(&self) -> Vec<f32> {
        let mut feat = Vec::with_capacity(self.feature_len());
        for layer in &self.layers {
            feat.extend_from_slice(&layer.last_spikes);
        }
        feat
    }

    /// Reset every layer's membrane potential and recurrent spike feedback to
    /// zero, leaving the (fixed) weights intact.
    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            for v in &mut layer.state.v {
                *v = 0.0;
            }
            for s in &mut layer.last_spikes {
                *s = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three chained layers with explicit, distinct sizes.
    fn three_layer_cfg() -> HierarchicalLsmConfig {
        let mk = |n: usize, seed: u64| LsmConfig {
            n_neurons: n,
            density: 0.2,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed,
        };
        HierarchicalLsmConfig {
            layer_configs: vec![mk(30, 1), mk(20, 2), mk(10, 3)],
            in_dim: 4,
        }
    }

    #[test]
    fn builds_and_chains_dimensions() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        assert_eq!(net.num_layers(), 3);
        assert_eq!(net.layer_size(0).expect("size0"), 30);
        assert_eq!(net.layer_size(1).expect("size1"), 20);
        assert_eq!(net.layer_size(2).expect("size2"), 10);
        // Layer 0 takes external input; higher layers take the layer below.
        assert_eq!(net.layers[0].in_dim, 4);
        assert_eq!(net.layers[1].in_dim, 30);
        assert_eq!(net.layers[2].in_dim, 20);
    }

    #[test]
    fn forward_step_per_layer_sizes() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let mut net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        let input = vec![0.5_f32; 4];
        let outs = net.forward_step(&input).expect("forward");
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].len(), 30);
        assert_eq!(outs[1].len(), 20);
        assert_eq!(outs[2].len(), 10);
        // Spike vectors are binary {0,1}.
        for layer_out in &outs {
            for &s in layer_out {
                assert!(s == 0.0 || s == 1.0, "non-binary spike {s}");
            }
        }
    }

    #[test]
    fn concatenated_feature_length() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let mut net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        assert_eq!(net.feature_len(), 30 + 20 + 10);
        let input = vec![1.0_f32; 4];
        net.forward_step(&input).expect("forward");
        assert_eq!(net.collected_state().len(), 60);
        assert_eq!(net.collected_spikes().len(), 60);
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let mut a = HierarchicalLsm::new(&cfg, &lif).expect("build a");
        let mut b = HierarchicalLsm::new(&cfg, &lif).expect("build b");
        let inputs = [
            vec![0.2_f32, -0.1, 0.4, 0.0],
            vec![0.5_f32, 0.5, -0.3, 0.1],
            vec![-0.2_f32, 0.3, 0.1, 0.6],
        ];
        for inp in &inputs {
            let oa = a.forward_step(inp).expect("step a");
            let ob = b.forward_step(inp).expect("step b");
            assert_eq!(oa, ob, "stacks diverged for identical seed");
        }
        // Concatenated states must match bit-for-bit too.
        assert_eq!(a.collected_state(), b.collected_state());
    }

    #[test]
    fn reset_zeros_all_state() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let mut net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        // Drive it so some membranes/spikes become non-zero.
        for _ in 0..20 {
            net.forward_step(&[2.0_f32; 4]).expect("forward");
        }
        net.reset_state();
        assert!(net.collected_state().iter().all(|&v| v == 0.0));
        assert!(net.collected_spikes().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn input_dim_mismatch_errors() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let mut net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        assert!(matches!(
            net.forward_step(&[0.0_f32; 3]),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn empty_stack_is_error() {
        let cfg = HierarchicalLsmConfig {
            layer_configs: vec![],
            in_dim: 4,
        };
        let lif = LifConfig::default();
        assert!(matches!(
            HierarchicalLsm::new(&cfg, &lif),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn zero_input_dim_is_error() {
        let mk = |n: usize, seed: u64| LsmConfig {
            n_neurons: n,
            density: 0.2,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed,
        };
        let cfg = HierarchicalLsmConfig {
            layer_configs: vec![mk(10, 1)],
            in_dim: 0,
        };
        let lif = LifConfig::default();
        assert!(matches!(
            HierarchicalLsm::new(&cfg, &lif),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn bad_layer_config_propagates() {
        // A layer with zero neurons must surface as an error from Lsm::new.
        let mk = |n: usize, seed: u64| LsmConfig {
            n_neurons: n,
            density: 0.2,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed,
        };
        let cfg = HierarchicalLsmConfig {
            layer_configs: vec![mk(10, 1), mk(0, 2)],
            in_dim: 3,
        };
        let lif = LifConfig::default();
        assert!(matches!(
            HierarchicalLsm::new(&cfg, &lif),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn layer_size_out_of_range_errors() {
        let cfg = three_layer_cfg();
        let lif = LifConfig::default();
        let net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        assert!(matches!(
            net.layer_size(3),
            Err(SnnError::LayerOutOfRange { idx: 3, .. })
        ));
    }

    #[test]
    fn single_layer_stack_works() {
        let mk = |n: usize, seed: u64| LsmConfig {
            n_neurons: n,
            density: 0.2,
            spectral_radius: 0.9,
            w_in_scale: 1.0,
            seed,
        };
        let cfg = HierarchicalLsmConfig {
            layer_configs: vec![mk(16, 5)],
            in_dim: 2,
        };
        let lif = LifConfig::default();
        let mut net = HierarchicalLsm::new(&cfg, &lif).expect("build");
        let outs = net.forward_step(&[0.3_f32; 2]).expect("forward");
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].len(), 16);
        assert_eq!(net.feature_len(), 16);
    }
}
