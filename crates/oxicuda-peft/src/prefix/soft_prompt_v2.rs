use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a `SoftPromptV2` module.
#[derive(Debug, Clone)]
pub struct SoftPromptConfig {
    /// Number of virtual prompt tokens; 0 is valid (empty prompt).
    pub n_prompt_tokens: usize,
    /// Hidden/model dimension; must be > 0.
    pub d_model: usize,
}

/// Soft-prompt module with explicit gradient-update support.
///
/// Stores `n_prompt_tokens × d_model` learnable embeddings initialised from
/// N(0, 0.02). Unlike [`crate::prefix::prompt_tuning::SoftPrompt`] this struct
/// exposes an SGD [`update`](SoftPromptV2::update) method so that the caller can
/// apply gradients directly, making it suitable for custom training loops without
/// any external framework dependency.
#[derive(Debug, Clone)]
pub struct SoftPromptV2 {
    /// Flat embedding matrix, shape `[n_prompt_tokens × d_model]`.
    tokens: Vec<f32>,
    /// Module configuration.
    config: SoftPromptConfig,
}

impl SoftPromptV2 {
    /// Create a new `SoftPromptV2` with embeddings initialised from N(0, 0.02).
    ///
    /// `config.n_prompt_tokens == 0` is valid and produces an empty prompt.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch { expected: 1, got: 0 }` when
    /// `config.d_model == 0`.
    pub fn new(config: SoftPromptConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        if config.d_model == 0 {
            return Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let total = config.n_prompt_tokens * config.d_model;
        let mut tokens = vec![0.0_f32; total];
        // fill_normal gives N(0,1); multiplying by 0.02 gives N(0, 0.02).
        rng.fill_normal(&mut tokens);
        for v in tokens.iter_mut() {
            *v *= 0.02;
        }
        Ok(Self { tokens, config })
    }

    /// Prepend the learned prompt embeddings to `input_embeds`.
    ///
    /// `input_embeds` must be a flat buffer of shape `[n_input_tokens × d_model]`.
    /// Returns a flat buffer of shape `[(n_prompt_tokens + n_input_tokens) × d_model]`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch` when
    /// `input_embeds.len() != n_input_tokens * d_model`.
    pub fn prepend(&self, input_embeds: &[f32], n_input_tokens: usize) -> PeftResult<Vec<f32>> {
        let expected_len = n_input_tokens * self.config.d_model;
        if input_embeds.len() != expected_len {
            return Err(PeftError::DimensionMismatch {
                expected: expected_len,
                got: input_embeds.len(),
            });
        }
        let total = self.tokens.len() + input_embeds.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&self.tokens);
        out.extend_from_slice(input_embeds);
        Ok(out)
    }

    /// Total number of trainable parameters: `n_prompt_tokens × d_model`.
    #[must_use]
    #[inline]
    pub fn n_params(&self) -> usize {
        self.config.n_prompt_tokens * self.config.d_model
    }

    /// Return a read-only view of the flat token-embedding buffer.
    #[must_use]
    #[inline]
    pub fn token_embeds(&self) -> &[f32] {
        &self.tokens
    }

    /// Apply one step of plain SGD: `tokens[i] -= lr * grad[i]`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch` when `grad.len() != tokens.len()`.
    pub fn update(&mut self, grad: &[f32], lr: f32) -> PeftResult<()> {
        if grad.len() != self.tokens.len() {
            return Err(PeftError::DimensionMismatch {
                expected: self.tokens.len(),
                got: grad.len(),
            });
        }
        for (t, g) in self.tokens.iter_mut().zip(grad.iter()) {
            *t -= lr * g;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_prompt(n_prompt: usize, d_model: usize) -> SoftPromptV2 {
        let cfg = SoftPromptConfig {
            n_prompt_tokens: n_prompt,
            d_model,
        };
        let mut rng = LcgRng::new(7);
        SoftPromptV2::new(cfg, &mut rng).expect("valid config")
    }

    // 1. prepend_shape: output len == (n_prompt + n_input) * d_model
    #[test]
    fn prepend_shape() {
        let prompt = make_prompt(4, 16);
        let n_input = 8usize;
        let input: Vec<f32> = vec![0.0; n_input * 16];
        let out = prompt.prepend(&input, n_input).expect("prepend");
        assert_eq!(out.len(), (4 + n_input) * 16);
    }

    // 2. prepend_finite: all output values are finite
    #[test]
    fn prepend_finite() {
        let prompt = make_prompt(3, 8);
        let n_input = 5usize;
        let input: Vec<f32> = (0..n_input * 8).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let out = prompt.prepend(&input, n_input).expect("prepend");
        for &v in &out {
            assert!(v.is_finite(), "non-finite value in prepend output: {v}");
        }
    }

    // 3. n_params_correct: n_params() == n_prompt_tokens * d_model
    #[test]
    fn n_params_correct() {
        let prompt = make_prompt(6, 32);
        assert_eq!(prompt.n_params(), 6 * 32);
    }

    // 4. token_embeds_shape: token_embeds().len() == n_prompt_tokens * d_model
    #[test]
    fn token_embeds_shape() {
        let prompt = make_prompt(5, 20);
        assert_eq!(prompt.token_embeds().len(), 5 * 20);
    }

    // 5. update_changes_tokens: after update with non-zero grad, tokens differ
    #[test]
    fn update_changes_tokens() {
        let mut prompt = make_prompt(4, 8);
        let before: Vec<f32> = prompt.token_embeds().to_vec();
        let grad: Vec<f32> = vec![1.0_f32; 4 * 8];
        prompt.update(&grad, 0.1).expect("update");
        let after = prompt.token_embeds();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .any(|(b, a)| (b - a).abs() > 1e-9),
            "tokens should have changed after update"
        );
    }

    // 6. n_prompt_0_ok: n_prompt_tokens=0 prepend returns just the input
    #[test]
    fn n_prompt_0_ok() {
        let prompt = make_prompt(0, 8);
        let n_input = 4usize;
        let input: Vec<f32> = (0..n_input * 8).map(|i| i as f32).collect();
        let out = prompt.prepend(&input, n_input).expect("prepend");
        assert_eq!(out.len(), n_input * 8);
        assert_eq!(out, input);
    }

    // 7. d_model_0_error: SoftPromptConfig{n_prompt_tokens:1, d_model:0} returns Err
    #[test]
    fn d_model_0_error() {
        let cfg = SoftPromptConfig {
            n_prompt_tokens: 1,
            d_model: 0,
        };
        let mut rng = LcgRng::new(0);
        let result = SoftPromptV2::new(cfg, &mut rng);
        match result {
            Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // 8. multiple_prepend_consistent: two prepend calls with same input return identical results
    #[test]
    fn multiple_prepend_consistent() {
        let prompt = make_prompt(3, 12);
        let n_input = 6usize;
        let input: Vec<f32> = (0..n_input * 12).map(|i| (i as f32) * 0.05).collect();
        let out1 = prompt.prepend(&input, n_input).expect("first prepend");
        let out2 = prompt.prepend(&input, n_input).expect("second prepend");
        assert_eq!(out1, out2);
    }
}
