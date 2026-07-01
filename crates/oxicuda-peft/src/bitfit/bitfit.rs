/// The type of layer whose bias parameters are tracked by BitFit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Standard fully-connected / projection layer.
    Linear,
    /// Convolutional layer.
    Conv,
    /// Token embedding layer.
    Embedding,
    /// Layer normalisation.
    LayerNorm,
}

/// Descriptor for the bias parameters of a single layer.
#[derive(Debug, Clone)]
pub struct BitFitLayerInfo {
    /// Kind of layer this bias belongs to.
    pub layer_type: LayerType,
    /// Number of elements in this bias vector.
    pub bias_size: usize,
}

/// BitFit parameter mask: enumerates all bias terms that are trained while everything else is frozen.
///
/// Selecting only bias parameters is an extremely parameter-efficient approach;
/// for BERT-base there are roughly 100 K bias parameters out of 110 M total parameters.
#[derive(Debug, Clone)]
pub struct BitFitMask {
    /// Ordered list of all trainable bias descriptors.
    pub layers: Vec<BitFitLayerInfo>,
}

impl BitFitMask {
    /// Build the standard BERT-like bias mask for a transformer encoder.
    ///
    /// For each of the `num_layers` transformer layers this generates entries for:
    /// Q/K/V projection biases (`hidden_dim` each), output projection bias,
    /// FFN first-layer bias (`ffn_dim`), FFN second-layer bias (`hidden_dim`),
    /// and two LayerNorm biases (`hidden_dim` each).
    /// The head dimension is used for attention output projection.
    #[must_use]
    pub fn for_transformer(
        num_layers: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        _num_heads: usize,
    ) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            // Q, K, V projection biases
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: hidden_dim,
            });
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: hidden_dim,
            });
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: hidden_dim,
            });
            // Attention output projection bias
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: hidden_dim,
            });
            // FFN layer 1 bias
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: ffn_dim,
            });
            // FFN layer 2 bias
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::Linear,
                bias_size: hidden_dim,
            });
            // LayerNorm 1 bias (after attention)
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::LayerNorm,
                bias_size: hidden_dim,
            });
            // LayerNorm 2 bias (after FFN)
            layers.push(BitFitLayerInfo {
                layer_type: LayerType::LayerNorm,
                bias_size: hidden_dim,
            });
        }
        Self { layers }
    }

    /// Sum of all bias parameter counts across all tracked layers.
    #[must_use]
    pub fn total_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.bias_size).sum()
    }

    /// Returns `true` if `name` looks like a bias parameter name.
    ///
    /// A name is considered a bias if it ends with `"bias"` or `"b"`.
    #[must_use]
    pub fn is_bias_param(name: &str) -> bool {
        name.ends_with("bias") || name.ends_with('b')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──── is_bias_param ────────────────────────────────────────────────────

    #[test]
    fn is_bias_param_bias_suffix_true() {
        // Any name that ends literally with "bias" must be recognised.
        assert!(BitFitMask::is_bias_param("bias"));
        assert!(BitFitMask::is_bias_param("encoder.layer.0.bias"));
        assert!(BitFitMask::is_bias_param("decoder.self_attn.k_proj.bias"));
        assert!(BitFitMask::is_bias_param("transformer.h.11.mlp.c_fc.bias"));
        assert!(BitFitMask::is_bias_param("layer_norm.bias"));
    }

    #[test]
    fn is_bias_param_b_suffix_true() {
        // Single-letter shorthand "b" and names ending with "_b" are also biases.
        assert!(BitFitMask::is_bias_param("b"));
        assert!(BitFitMask::is_bias_param("attn_b"));
        assert!(BitFitMask::is_bias_param("ffn_b"));
        assert!(BitFitMask::is_bias_param("qkv_b"));
    }

    #[test]
    fn is_bias_param_weight_names_false() {
        // Standard weight / kernel / norm-scale names must not be matched.
        assert!(!BitFitMask::is_bias_param("weight"));
        assert!(!BitFitMask::is_bias_param("kernel"));
        assert!(!BitFitMask::is_bias_param("gamma"));
        assert!(!BitFitMask::is_bias_param("embedding_weight"));
        // "bias_weight" ends with "weight", NOT "bias" — must be false.
        assert!(!BitFitMask::is_bias_param("bias_weight"));
    }

    #[test]
    fn is_bias_param_edge_cases() {
        // Empty string → false.
        assert!(!BitFitMask::is_bias_param(""));
        // "biased" ends with 'd', not "bias" → false.
        assert!(!BitFitMask::is_bias_param("biased"));
        // "biasness" ends with 's' → false.
        assert!(!BitFitMask::is_bias_param("biasness"));
        // "ab" ends with 'b' → true (single-char-'b' rule).
        assert!(BitFitMask::is_bias_param("ab"));
        // "running_mean" → false.
        assert!(!BitFitMask::is_bias_param("running_mean"));
    }

    // ──── for_transformer + total_trainable_params ─────────────────────────

    #[test]
    fn total_trainable_params_exact_arithmetic() {
        // Per-layer breakdown (hidden=H, ffn=F):
        //   Q(H) + K(H) + V(H) + O(H) + FFN1(F) + FFN2(H) + LN1(H) + LN2(H)
        //   = 7·H + F
        //
        // Config A: 1 layer, H=4, F=8  → 7·4+8 = 36
        let mask_a = BitFitMask::for_transformer(1, 4, 8, 2);
        assert_eq!(
            mask_a.total_trainable_params(),
            36,
            "1 layer h=4 f=8 expected 36"
        );

        // Config B: 2 layers, same dims → 2·36 = 72
        let mask_b = BitFitMask::for_transformer(2, 4, 8, 2);
        assert_eq!(
            mask_b.total_trainable_params(),
            72,
            "2 layers h=4 f=8 expected 72"
        );

        // Config C: 3 layers, H=8, F=32 → 7·8+32 = 88 per layer → 264
        let mask_c = BitFitMask::for_transformer(3, 8, 32, 4);
        assert_eq!(
            mask_c.total_trainable_params(),
            264,
            "3 layers h=8 f=32 expected 264"
        );
    }

    #[test]
    fn for_transformer_layer_count_and_types() {
        // 2 transformer layers → 8 entries each → 16 total.
        let mask = BitFitMask::for_transformer(2, 8, 16, 4);
        assert_eq!(mask.layers.len(), 16, "expected 8 entries × 2 layers = 16");
        // Within the first layer (indices 0..8):
        //   0=Q, 1=K, 2=V, 3=O — all Linear
        //   4=FFN1, 5=FFN2        — both Linear
        //   6=LN1, 7=LN2          — both LayerNorm
        assert_eq!(mask.layers[0].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[1].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[2].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[3].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[4].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[5].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[6].layer_type, LayerType::LayerNorm);
        assert_eq!(mask.layers[7].layer_type, LayerType::LayerNorm);
        // Second layer repeats the same pattern at offset 8.
        assert_eq!(mask.layers[8].layer_type, LayerType::Linear);
        assert_eq!(mask.layers[14].layer_type, LayerType::LayerNorm);
        assert_eq!(mask.layers[15].layer_type, LayerType::LayerNorm);
    }

    #[test]
    fn trainable_params_small_fraction_of_full_model() {
        // BitFit's key claim (Zaken 2021): only ~0.1% of params are trained.
        // Small but representative config: 4 layers, H=64, F=256, 8 heads.
        //   Bias params = 4 × (7·64 + 256) = 4 × 704 = 2 816
        //   Approx full-model weight params: Q+K+V+O = 4·H² per layer,
        //     FFN = 2·H·F per layer → 4·(4·64² + 2·64·256) = 4·(16 384+32 768) = 196 608
        //   Bias / weights < 2 % → bias_params × 50 < full_params
        let mask = BitFitMask::for_transformer(4, 64, 256, 8);
        let bias_params = mask.total_trainable_params();
        assert_eq!(bias_params, 2816, "exact bias param count mismatch");
        let approx_full = 4 * (4 * 64 * 64 + 2 * 64 * 256usize);
        assert!(
            bias_params * 50 < approx_full,
            "bias ({bias_params}) is not <2 % of full model (~{approx_full})"
        );
    }

    #[test]
    fn for_transformer_zero_layers_is_empty() {
        let mask = BitFitMask::for_transformer(0, 64, 256, 8);
        assert_eq!(mask.layers.len(), 0, "zero layers → zero entries");
        assert_eq!(
            mask.total_trainable_params(),
            0,
            "zero layers → zero trainable params"
        );
    }
}
