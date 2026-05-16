/// Indicates where in the attention or FFN block an IA³ vector is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ia3Placement {
    /// Applied to the key projection output.
    Key,
    /// Applied to the value projection output.
    Value,
    /// Applied inside the feed-forward network.
    FeedForward,
}

/// An IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) scaling vector.
///
/// Stores a learned element-wise scale that is multiplied into the activations.
/// Initialised to ones so that the adapter is an identity at the start of training.
#[derive(Debug, Clone)]
pub struct Ia3Vector {
    /// Dimension of the activation this vector scales.
    pub size: usize,
    /// Learned scale factors, initialised to `1.0`.
    pub scale: Vec<f32>,
    /// Which position in the transformer block this vector belongs to.
    pub placement: Ia3Placement,
}

impl Ia3Vector {
    /// Create a new `Ia3Vector` with `scale` initialised to all-ones.
    #[must_use]
    pub fn new(size: usize, placement: Ia3Placement) -> Self {
        let scale = vec![1.0_f32; size];
        Self {
            size,
            scale,
            placement,
        }
    }

    /// Apply the IA³ scale element-wise: `out[i] = x[i] * scale[i]`.
    ///
    /// `x` and the internal scale vector must have equal length (`size`).
    #[must_use]
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(self.scale.iter())
            .map(|(xi, si)| xi * si)
            .collect()
    }

    /// Return the number of trainable parameters (equal to `size`).
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.size
    }
}
