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
