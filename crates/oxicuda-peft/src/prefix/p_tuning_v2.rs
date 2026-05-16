use crate::handle::LcgRng;
use crate::prefix::prefix_tuning::{PrefixConfig, PrefixModule};

/// P-Tuning v2: a stack of per-layer `PrefixModule` instances, one per transformer layer.
///
/// Each layer independently learns its own key and value prefix vectors.
#[derive(Debug, Clone)]
pub struct PTuningV2 {
    /// One `PrefixModule` per transformer layer.
    pub layers: Vec<PrefixModule>,
}

impl PTuningV2 {
    /// Construct a `PTuningV2` with `cfg.num_layers` independent `PrefixModule` instances.
    ///
    /// All prefix tensors are initialised from N(0, 0.02).
    #[must_use]
    pub fn new(cfg: PrefixConfig, rng: &mut LcgRng) -> Self {
        let num_layers = cfg.num_layers;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Each layer gets its own clone of the config so that PrefixModule.num_layers==1
            // but we store the full stack here at the PTuningV2 level.
            let layer_cfg = PrefixConfig {
                num_virtual_tokens: cfg.num_virtual_tokens,
                prefix_dim: cfg.prefix_dim,
                num_layers: 1,
                num_heads: cfg.num_heads,
                head_dim: cfg.head_dim,
            };
            layers.push(PrefixModule::new(layer_cfg, rng));
        }
        Self { layers }
    }

    /// Return references to the key and value prefix slices for a given transformer layer.
    #[must_use]
    pub fn layer_prefix(&self, layer: usize) -> (&[f32], &[f32]) {
        let m = &self.layers[layer];
        (m.get_key_prefix(0), m.get_value_prefix(0))
    }

    /// Count total trainable parameters across all layers.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|m| m.num_params()).sum()
    }
}
