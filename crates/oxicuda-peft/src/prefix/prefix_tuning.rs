use crate::handle::LcgRng;

/// Configuration for a prefix-tuning module.
#[derive(Debug, Clone)]
pub struct PrefixConfig {
    /// Number of virtual (soft) tokens prepended to each layer's key and value.
    pub num_virtual_tokens: usize,
    /// Total dimension of the prefix representation (unused directly — kept for API symmetry).
    pub prefix_dim: usize,
    /// Number of transformer layers this module serves.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
}

/// A single-layer prefix-tuning module holding key and value prefix tensors.
///
/// The key prefix and value prefix each have shape
/// `[num_virtual_tokens × num_heads × head_dim]` stored as a flat `f32` vector.
#[derive(Debug, Clone)]
pub struct PrefixModule {
    /// Configuration shared with the parent PTuningV2 stack.
    pub config: PrefixConfig,
    /// Key prefixes per layer; each element has length `num_virtual_tokens × num_heads × head_dim`.
    pub k_prefixes: Vec<Vec<f32>>,
    /// Value prefixes per layer; same shape as `k_prefixes`.
    pub v_prefixes: Vec<Vec<f32>>,
}

impl PrefixModule {
    /// Construct a new `PrefixModule`.
    ///
    /// Each key and value prefix is initialised with samples from N(0, 0.02).
    #[must_use]
    pub fn new(cfg: PrefixConfig, rng: &mut LcgRng) -> Self {
        let prefix_len = cfg.num_virtual_tokens * cfg.num_heads * cfg.head_dim;
        let mut k_prefixes = Vec::with_capacity(cfg.num_layers);
        let mut v_prefixes = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let mut k = vec![0.0_f32; prefix_len];
            rng.fill_normal(&mut k);
            for v in k.iter_mut() {
                *v *= 0.02;
            }
            let mut v = vec![0.0_f32; prefix_len];
            rng.fill_normal(&mut v);
            for val in v.iter_mut() {
                *val *= 0.02;
            }
            k_prefixes.push(k);
            v_prefixes.push(v);
        }
        Self {
            config: cfg,
            k_prefixes,
            v_prefixes,
        }
    }

    /// Return the key prefix for a given layer index.
    #[must_use]
    pub fn get_key_prefix(&self, layer: usize) -> &[f32] {
        &self.k_prefixes[layer]
    }

    /// Return the value prefix for a given layer index.
    #[must_use]
    pub fn get_value_prefix(&self, layer: usize) -> &[f32] {
        &self.v_prefixes[layer]
    }

    /// Return the shape of a single prefix tensor as `(num_virtual_tokens, num_heads, head_dim)`.
    #[must_use]
    pub fn prefix_shape(&self) -> (usize, usize, usize) {
        (
            self.config.num_virtual_tokens,
            self.config.num_heads,
            self.config.head_dim,
        )
    }

    /// Count total trainable parameters: `2 × num_layers × num_virtual_tokens × num_heads × head_dim`.
    #[must_use]
    pub fn num_params(&self) -> usize {
        2 * self.config.num_layers
            * self.config.num_virtual_tokens
            * self.config.num_heads
            * self.config.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Reusable small config: 3 layers, 4 virtual tokens, 2 heads, head_dim=8.
    fn small_config() -> PrefixConfig {
        PrefixConfig {
            num_virtual_tokens: 4,
            prefix_dim: 32,
            num_layers: 3,
            num_heads: 2,
            head_dim: 8,
        }
    }

    // ──── Shape / structure ────────────────────────────────────────────────

    #[test]
    fn prefix_shape_matches_config() {
        let mut rng = LcgRng::new(42);
        let module = PrefixModule::new(small_config(), &mut rng);
        let (vtok, heads, hdim) = module.prefix_shape();
        assert_eq!(vtok, 4, "num_virtual_tokens");
        assert_eq!(heads, 2, "num_heads");
        assert_eq!(hdim, 8, "head_dim");
    }

    #[test]
    fn num_params_exact_arithmetic() {
        // 2 (K+V) × num_layers × num_virtual_tokens × num_heads × head_dim
        // = 2 × 3 × 4 × 2 × 8 = 384
        let mut rng = LcgRng::new(1);
        let module = PrefixModule::new(small_config(), &mut rng);
        assert_eq!(module.num_params(), 384, "expected 2×3×4×2×8 = 384");
    }

    #[test]
    fn prefix_vector_length_matches_config() {
        // Each prefix slice per layer must have num_virtual_tokens×num_heads×head_dim elements.
        let cfg = small_config();
        let expected_len = cfg.num_virtual_tokens * cfg.num_heads * cfg.head_dim; // 4×2×8 = 64
        let mut rng = LcgRng::new(7);
        let module = PrefixModule::new(cfg, &mut rng);
        assert_eq!(module.k_prefixes.len(), 3, "k_prefixes layer count");
        assert_eq!(module.v_prefixes.len(), 3, "v_prefixes layer count");
        for layer in 0..3 {
            assert_eq!(
                module.get_key_prefix(layer).len(),
                expected_len,
                "key prefix len layer {layer}"
            );
            assert_eq!(
                module.get_value_prefix(layer).len(),
                expected_len,
                "value prefix len layer {layer}"
            );
        }
    }

    #[test]
    fn get_prefix_returns_correct_layer_slice() {
        // get_key_prefix(l) must alias k_prefixes[l]; same for v.
        let mut rng = LcgRng::new(55);
        let module = PrefixModule::new(small_config(), &mut rng);
        for layer in 0..3 {
            assert_eq!(
                module.get_key_prefix(layer),
                module.k_prefixes[layer].as_slice(),
                "get_key_prefix({layer}) mismatch"
            );
            assert_eq!(
                module.get_value_prefix(layer),
                module.v_prefixes[layer].as_slice(),
                "get_value_prefix({layer}) mismatch"
            );
        }
    }

    // ──── Determinism ──────────────────────────────────────────────────────

    #[test]
    fn determinism_same_seed_identical_prefixes() {
        // Two modules constructed with the same seed must be bit-for-bit identical.
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let mod_a = PrefixModule::new(small_config(), &mut rng_a);
        let mod_b = PrefixModule::new(small_config(), &mut rng_b);
        for layer in 0..3 {
            assert_eq!(
                mod_a.get_key_prefix(layer),
                mod_b.get_key_prefix(layer),
                "key prefix differs at layer {layer}"
            );
            assert_eq!(
                mod_a.get_value_prefix(layer),
                mod_b.get_value_prefix(layer),
                "value prefix differs at layer {layer}"
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_prefixes() {
        let mut rng_a = LcgRng::new(12345);
        let mut rng_b = LcgRng::new(99999);
        let mod_a = PrefixModule::new(small_config(), &mut rng_a);
        let mod_b = PrefixModule::new(small_config(), &mut rng_b);
        let any_diff = (0..3).any(|layer| {
            mod_a.get_key_prefix(layer) != mod_b.get_key_prefix(layer)
                || mod_a.get_value_prefix(layer) != mod_b.get_value_prefix(layer)
        });
        assert!(
            any_diff,
            "different seeds must produce different prefix values"
        );
    }

    // ──── Initialisation quality ───────────────────────────────────────────

    #[test]
    fn prefix_values_finite_and_small_scale() {
        // Initialised from N(0, 0.02): all values must be finite and within ±1.0
        // (99.7 % of N(0,0.02) lies in ±0.06; ±1.0 is a very loose safety guard).
        let mut rng = LcgRng::new(31415);
        let module = PrefixModule::new(small_config(), &mut rng);
        for layer in 0..3 {
            for &v in module.get_key_prefix(layer) {
                assert!(v.is_finite(), "key prefix value is not finite: {v}");
                assert!(v.abs() < 1.0, "key prefix value unexpectedly large: {v}");
            }
            for &v in module.get_value_prefix(layer) {
                assert!(v.is_finite(), "value prefix value is not finite: {v}");
                assert!(v.abs() < 1.0, "value prefix value unexpectedly large: {v}");
            }
        }
    }

    // ──── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn zero_virtual_tokens_gives_empty_prefix_slices() {
        // num_virtual_tokens = 0 → each prefix slice has length 0 and num_params = 0.
        let cfg = PrefixConfig {
            num_virtual_tokens: 0,
            prefix_dim: 32,
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
        };
        let mut rng = LcgRng::new(0);
        let module = PrefixModule::new(cfg, &mut rng);
        assert_eq!(module.num_params(), 0, "zero virtual tokens → zero params");
        for layer in 0..2 {
            assert_eq!(
                module.get_key_prefix(layer).len(),
                0,
                "key prefix for layer {layer} should be empty"
            );
            assert_eq!(
                module.get_value_prefix(layer).len(),
                0,
                "value prefix for layer {layer} should be empty"
            );
        }
    }
}
