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
