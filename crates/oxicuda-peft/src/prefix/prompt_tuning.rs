use crate::handle::LcgRng;

/// A soft-prompt module for prompt tuning.
///
/// Stores `num_tokens` learnable embeddings of dimension `embed_dim`, each initialised
/// from N(0, 0.02). These embeddings are prepended to the input sequence before the
/// first transformer layer.
#[derive(Debug, Clone)]
pub struct SoftPrompt {
    /// Number of virtual prompt tokens.
    pub num_tokens: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Flat embedding matrix, shape `[num_tokens × embed_dim]`.
    pub embeddings: Vec<f32>,
}

impl SoftPrompt {
    /// Create a new `SoftPrompt` with embeddings initialised from N(0, 0.02).
    #[must_use]
    pub fn new(num_tokens: usize, embed_dim: usize, rng: &mut LcgRng) -> Self {
        let mut embeddings = vec![0.0_f32; num_tokens * embed_dim];
        rng.fill_normal(&mut embeddings);
        for v in embeddings.iter_mut() {
            *v *= 0.02;
        }
        Self {
            num_tokens,
            embed_dim,
            embeddings,
        }
    }

    /// Prepend the soft prompt to a sequence of token embeddings.
    ///
    /// `seq_embeddings` must have length `seq_len * embed_dim`.
    /// Returns a flat vector of shape `[(num_tokens + seq_len) × embed_dim]`.
    #[must_use]
    pub fn prepend_to_sequence(&self, seq_embeddings: &[f32], seq_len: usize) -> Vec<f32> {
        let total_len = (self.num_tokens + seq_len) * self.embed_dim;
        let mut result = Vec::with_capacity(total_len);
        result.extend_from_slice(&self.embeddings);
        result.extend_from_slice(&seq_embeddings[..seq_len * self.embed_dim]);
        result
    }

    /// Count the number of trainable parameters: `num_tokens × embed_dim`.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.num_tokens * self.embed_dim
    }
}
