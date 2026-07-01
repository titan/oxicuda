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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::prefix::prefix_tuning::PrefixConfig;

    /// Small reusable config: 3 layers, 4 virtual tokens, 2 heads, head_dim=8.
    fn small_cfg() -> PrefixConfig {
        PrefixConfig {
            num_virtual_tokens: 4,
            prefix_dim: 32,
            num_layers: 3,
            num_heads: 2,
            head_dim: 8,
        }
    }

    // ──── total_params ─────────────────────────────────────────────────────

    #[test]
    fn total_params_exact_arithmetic() {
        // PTuningV2::new creates num_layers PrefixModules, each with num_layers=1.
        // Each module's num_params() = 2 · 1 · num_virtual_tokens · num_heads · head_dim
        //                            = 2 · 1 · 4 · 2 · 8 = 128
        // Total across 3 modules = 3 · 128 = 384
        // Equivalently: 2 · num_layers · num_virtual_tokens · num_heads · head_dim
        //             = 2 · 3 · 4 · 2 · 8 = 384
        let mut rng = LcgRng::new(42);
        let pt = PTuningV2::new(small_cfg(), &mut rng);
        assert_eq!(pt.total_params(), 384, "expected 2·3·4·2·8 = 384");
    }

    #[test]
    fn total_params_different_config() {
        // num_layers=2, num_virtual_tokens=5, num_heads=3, head_dim=6
        // 2 · 2 · 5 · 3 · 6 = 360
        let cfg = PrefixConfig {
            num_virtual_tokens: 5,
            prefix_dim: 48,
            num_layers: 2,
            num_heads: 3,
            head_dim: 6,
        };
        let mut rng = LcgRng::new(7);
        let pt = PTuningV2::new(cfg, &mut rng);
        assert_eq!(pt.total_params(), 360, "expected 2·2·5·3·6 = 360");
    }

    // ──── layer_prefix ─────────────────────────────────────────────────────

    #[test]
    fn layer_prefix_returns_correct_slice_length() {
        // prefix length per layer = num_virtual_tokens · num_heads · head_dim = 4·2·8 = 64
        let expected_len = 4 * 2 * 8;
        let mut rng = LcgRng::new(1);
        let pt = PTuningV2::new(small_cfg(), &mut rng);
        for layer in 0..3 {
            let (k, v) = pt.layer_prefix(layer);
            assert_eq!(k.len(), expected_len, "key prefix len at layer {layer}");
            assert_eq!(v.len(), expected_len, "value prefix len at layer {layer}");
        }
    }

    #[test]
    fn layer_prefix_aliases_inner_module() {
        // layer_prefix(l) must return the same data as layers[l].get_key_prefix(0)
        // and layers[l].get_value_prefix(0), because each inner PrefixModule has num_layers=1.
        let mut rng = LcgRng::new(2);
        let pt = PTuningV2::new(small_cfg(), &mut rng);
        for layer in 0..3 {
            let (k, v) = pt.layer_prefix(layer);
            assert_eq!(
                k,
                pt.layers[layer].get_key_prefix(0),
                "key prefix mismatch at layer {layer}"
            );
            assert_eq!(
                v,
                pt.layers[layer].get_value_prefix(0),
                "value prefix mismatch at layer {layer}"
            );
        }
    }

    #[test]
    fn layer_prefixes_differ_across_layers() {
        // Each layer is independently initialised → distinct prefix values
        let mut rng = LcgRng::new(31415);
        let pt = PTuningV2::new(small_cfg(), &mut rng);
        let (k0, _) = pt.layer_prefix(0);
        let (k1, _) = pt.layer_prefix(1);
        assert_ne!(
            k0, k1,
            "independently-seeded layers must have different key prefixes"
        );
    }

    // ──── structure ────────────────────────────────────────────────────────

    #[test]
    fn layer_count_matches_config_num_layers() {
        let mut rng = LcgRng::new(0);
        let pt = PTuningV2::new(small_cfg(), &mut rng);
        assert_eq!(
            pt.layers.len(),
            3,
            "PTuningV2 must have exactly one PrefixModule per layer"
        );
    }

    // ──── determinism ──────────────────────────────────────────────────────

    #[test]
    fn determinism_same_seed_identical_prefixes() {
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let pt_a = PTuningV2::new(small_cfg(), &mut rng_a);
        let pt_b = PTuningV2::new(small_cfg(), &mut rng_b);
        for layer in 0..3 {
            let (ka, va) = pt_a.layer_prefix(layer);
            let (kb, vb) = pt_b.layer_prefix(layer);
            assert_eq!(ka, kb, "key prefix differs at layer {layer}");
            assert_eq!(va, vb, "value prefix differs at layer {layer}");
        }
    }
}
