//! FT-Transformer (feature-tokenizer variant), compact scalar-output form.
//!
//! Gorishniy et al., *"Revisiting Deep Learning Models for Tabular Data"*,
//! NeurIPS 2021. This is a self-contained reference of the feature-tokenizer
//! transformer with a single scalar regression/score head, complementary to
//! the multi-class [`super::ft_transformer::FtTransformer`].
//!
//! Pipeline:
//! 1. **Feature tokenizer** — each numeric feature `x_i` is mapped to a
//!    `d_token`-dimensional token `x_i · W_i + b_i`; each categorical feature is
//!    mapped through a per-feature embedding table.
//! 2. **CLS token** — a learnable `[CLS]` token is prepended to the sequence.
//! 3. **Transformer layers** — token-wise feed-forward residual blocks with
//!    layer normalisation acting over the `(n_features + 1)` tokens.
//! 4. **Head** — the final `[CLS]` representation is projected to a scalar.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

/// Configuration for the [`FtTransformer`].
#[derive(Debug, Clone)]
pub struct FtTransformerConfig {
    /// Number of numeric (continuous) features.
    pub n_num_features: usize,
    /// Number of categorical features.
    pub n_cat_features: usize,
    /// Per-categorical-feature cardinality (number of categories).
    pub cat_cardinalities: Vec<usize>,
    /// Token embedding dimension `d_token`.
    pub d_token: usize,
    /// Number of attention heads (used to validate `d_token` divisibility).
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
}

/// Compact FT-Transformer with a scalar output head.
#[derive(Debug, Clone)]
pub struct FtTransformer {
    /// Per-numeric-feature scale, `[n_num_features × d_token]` row-major.
    num_emb_w: Vec<f32>,
    /// Per-numeric-feature bias, `[n_num_features × d_token]` row-major.
    num_emb_b: Vec<f32>,
    /// Per-categorical-feature embedding table, `[n_cat_features][cardinality × d_token]`.
    cat_emb: Vec<Vec<f32>>,
    /// Learnable `[CLS]` token, `[d_token]`.
    cls_token: Vec<f32>,
    /// Per-layer feed-forward weight, `[n_layers][d_token × d_token]` row-major.
    layer_w: Vec<Vec<f32>>,
    /// Per-layer feed-forward bias, `[n_layers][d_token]`.
    layer_b: Vec<Vec<f32>>,
    /// Output head projecting the `[CLS]` token to a scalar, `[d_token]`.
    head_w: Vec<f32>,
    /// Configuration this model was built from.
    config: FtTransformerConfig,
}

/// Layer normalisation over a single `d_token`-dimensional token.
fn layer_norm(token: &[f32]) -> Vec<f32> {
    let n = token.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = token.iter().sum::<f32>() / n as f32;
    let var = token.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    token.iter().map(|&v| (v - mean) * inv_std).collect()
}

impl FtTransformer {
    /// Construct a new model with Xavier-style normal initialisation.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidEmbedDim`] if `d_token == 0`,
    /// [`TabularError::InvalidAttentionDim`] if `n_heads == 0` or `d_token` is
    /// not a multiple of `n_heads`, and [`TabularError::DimensionMismatch`] if
    /// `cat_cardinalities.len() != n_cat_features`.
    pub fn new(config: FtTransformerConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.d_token == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if config.n_heads == 0 || !config.d_token.is_multiple_of(config.n_heads) {
            return Err(TabularError::InvalidAttentionDim {
                dim: config.d_token,
            });
        }
        if config.cat_cardinalities.len() != config.n_cat_features {
            return Err(TabularError::DimensionMismatch {
                expected: config.n_cat_features,
                got: config.cat_cardinalities.len(),
            });
        }

        let dt = config.d_token;
        let std = (2.0_f32 / (dt as f32 + 1.0)).sqrt();

        let mut num_emb_w = vec![0.0_f32; config.n_num_features * dt];
        rng.fill_normal_scaled(&mut num_emb_w, std);
        let num_emb_b = vec![0.0_f32; config.n_num_features * dt];

        let mut cat_emb = Vec::with_capacity(config.n_cat_features);
        for &card in &config.cat_cardinalities {
            // Guard against zero cardinality to keep tables non-empty.
            let rows = card.max(1);
            let mut table = vec![0.0_f32; rows * dt];
            rng.fill_normal_scaled(&mut table, std);
            cat_emb.push(table);
        }

        let mut cls_token = vec![0.0_f32; dt];
        rng.fill_normal_scaled(&mut cls_token, std);

        let std_layer = (2.0_f32 / (2.0 * dt as f32)).sqrt();
        let mut layer_w = Vec::with_capacity(config.n_layers);
        let mut layer_b = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let mut w = vec![0.0_f32; dt * dt];
            rng.fill_normal_scaled(&mut w, std_layer);
            layer_w.push(w);
            layer_b.push(vec![0.0_f32; dt]);
        }

        let mut head_w = vec![0.0_f32; dt];
        rng.fill_normal_scaled(&mut head_w, std);

        Ok(Self {
            num_emb_w,
            num_emb_b,
            cat_emb,
            cls_token,
            layer_w,
            layer_b,
            head_w,
            config,
        })
    }

    /// Tokenise the raw features into the input token sequence.
    ///
    /// Each numeric feature `x_i` becomes a `d_token` token `x_i·W_i + b_i`,
    /// each categorical feature is looked up in its embedding table, and the
    /// learnable `[CLS]` token is prepended. The result is row-major with shape
    /// `[(n_features + 1) × d_token]` (the `[CLS]` token occupies row 0).
    ///
    /// # Errors
    /// Returns [`TabularError::DimensionMismatch`] if the input counts do not
    /// match the configuration, and [`TabularError::CategoricalOutOfRange`] if
    /// a categorical index exceeds its cardinality.
    pub fn tokenize(&self, num_feats: &[f32], cat_feats: &[usize]) -> TabularResult<Vec<f32>> {
        let cfg = &self.config;
        let dt = cfg.d_token;
        if num_feats.len() != cfg.n_num_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_num_features,
                got: num_feats.len(),
            });
        }
        if cat_feats.len() != cfg.n_cat_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_cat_features,
                got: cat_feats.len(),
            });
        }

        let n_features = cfg.n_num_features + cfg.n_cat_features;
        let seq_len = n_features + 1;
        let mut tokens = vec![0.0_f32; seq_len * dt];

        // Row 0 is the CLS token.
        tokens[0..dt].copy_from_slice(&self.cls_token);

        // Numeric tokens occupy rows 1..=n_num_features.
        for (i, &x) in num_feats.iter().enumerate() {
            let dst = (1 + i) * dt;
            for d in 0..dt {
                tokens[dst + d] = x * self.num_emb_w[i * dt + d] + self.num_emb_b[i * dt + d];
            }
        }

        // Categorical tokens follow the numeric block.
        for (j, &cat) in cat_feats.iter().enumerate() {
            let card = cfg.cat_cardinalities[j];
            if cat >= card {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: j,
                    val: cat,
                    n: card,
                });
            }
            let dst = (1 + cfg.n_num_features + j) * dt;
            let src = cat * dt;
            tokens[dst..dst + dt].copy_from_slice(&self.cat_emb[j][src..src + dt]);
        }

        Ok(tokens)
    }

    /// Forward pass producing a scalar score.
    ///
    /// Tokenises the inputs, applies the stack of transformer layers — each a
    /// permutation-invariant token-mixing residual followed by a pre-LN
    /// position-wise feed-forward residual — then projects the `[CLS]` token
    /// through the linear head.
    ///
    /// # Errors
    /// Propagates the errors of [`FtTransformer::tokenize`].
    pub fn forward(&self, num_feats: &[f32], cat_feats: &[usize]) -> TabularResult<f32> {
        let cfg = &self.config;
        let dt = cfg.d_token;
        let mut tokens = self.tokenize(num_feats, cat_feats)?;
        let seq_len = tokens.len() / dt;

        for layer in 0..cfg.n_layers {
            let w = &self.layer_w[layer];
            let b = &self.layer_b[layer];

            // ── Token-mixing block ────────────────────────────────────────────
            // A permutation-invariant attention surrogate: every token attends
            // uniformly to the whole sequence via the mean token, added back as
            // a residual. This lets the `[CLS]` representation incorporate the
            // feature tokens (without it, a purely token-wise FFN would leave
            // CLS independent of the inputs).
            let inv_len = 1.0 / seq_len as f32;
            let mut mean_tok = vec![0.0_f32; dt];
            for s in 0..seq_len {
                let tok = &tokens[s * dt..(s + 1) * dt];
                for d in 0..dt {
                    mean_tok[d] += tok[d] * inv_len;
                }
            }
            for s in 0..seq_len {
                for d in 0..dt {
                    tokens[s * dt + d] += mean_tok[d];
                }
            }

            // ── Position-wise feed-forward block ─────────────────────────────
            let mut next = vec![0.0_f32; tokens.len()];
            for s in 0..seq_len {
                let tok = &tokens[s * dt..(s + 1) * dt];
                // Pre-LN.
                let normed = layer_norm(tok);
                // Feed-forward with ReLU then residual.
                for o in 0..dt {
                    let mut acc = b[o];
                    for (i, &nv) in normed.iter().enumerate() {
                        acc += w[o * dt + i] * nv;
                    }
                    let activated = acc.max(0.0);
                    next[s * dt + o] = tok[o] + activated;
                }
            }
            tokens = next;
        }

        // Project the CLS token (row 0) to a scalar.
        let cls = &tokens[0..dt];
        let mut out = 0.0_f32;
        for (d, &cv) in cls.iter().enumerate() {
            out += self.head_w[d] * cv;
        }
        Ok(out)
    }

    /// Return the token embedding dimension.
    #[must_use]
    pub fn d_token(&self) -> usize {
        self.config.d_token
    }

    /// Return the total number of input features (numeric + categorical).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_num_features + self.config.n_cat_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(n_num: usize, n_cat: usize, cards: Vec<usize>) -> FtTransformer {
        let mut rng = LcgRng::new(42);
        let cfg = FtTransformerConfig {
            n_num_features: n_num,
            n_cat_features: n_cat,
            cat_cardinalities: cards,
            d_token: 8,
            n_heads: 2,
            n_layers: 2,
        };
        FtTransformer::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn tokenize_shape() {
        let model = make_model(3, 2, vec![4, 5]);
        let toks = model
            .tokenize(&[0.1, 0.2, 0.3], &[1, 2])
            .expect("tokenize should succeed");
        // (3 + 2 + 1) tokens × d_token=8.
        assert_eq!(toks.len(), 6 * 8);
    }

    #[test]
    fn forward_finite() {
        let model = make_model(4, 2, vec![3, 3]);
        let out = model
            .forward(&[0.5, -1.0, 2.0, 0.0], &[0, 2])
            .expect("forward should succeed");
        assert!(out.is_finite());
    }

    #[test]
    fn cls_token_present() {
        // The first d_token entries of the tokenised sequence must equal the
        // learned CLS token regardless of the inputs.
        let model = make_model(2, 1, vec![4]);
        let a = model
            .tokenize(&[1.0, 2.0], &[0])
            .expect("tokenize should succeed");
        let b = model
            .tokenize(&[-9.0, 5.0], &[3])
            .expect("tokenize should succeed");
        assert_eq!(&a[0..8], &b[0..8]);
    }

    #[test]
    fn cat_out_of_range_error() {
        let model = make_model(1, 1, vec![3]);
        let res = model.forward(&[0.0], &[3]); // valid indices are 0..3
        assert!(matches!(
            res,
            Err(TabularError::CategoricalOutOfRange { .. })
        ));
    }

    #[test]
    fn different_inputs_different_outputs() {
        let model = make_model(3, 1, vec![4]);
        let a = model
            .forward(&[0.0, 0.0, 0.0], &[0])
            .expect("forward should succeed");
        let b = model
            .forward(&[5.0, -3.0, 2.0], &[2])
            .expect("forward should succeed");
        assert!((a - b).abs() > 1e-6, "outputs identical: {a} vs {b}");
    }

    #[test]
    fn n_num_0_only_cat() {
        let model = make_model(0, 3, vec![4, 5, 6]);
        let out = model
            .forward(&[], &[1, 2, 3])
            .expect("forward should succeed");
        assert!(out.is_finite());
        let toks = model
            .tokenize(&[], &[0, 0, 0])
            .expect("tokenize should succeed");
        assert_eq!(toks.len(), 4 * 8); // 3 cat + CLS.
    }

    #[test]
    fn n_cat_0_only_num() {
        let model = make_model(4, 0, vec![]);
        let out = model
            .forward(&[0.1, 0.2, 0.3, 0.4], &[])
            .expect("forward should succeed");
        assert!(out.is_finite());
        let toks = model
            .tokenize(&[1.0, 2.0, 3.0, 4.0], &[])
            .expect("tokenize should succeed");
        assert_eq!(toks.len(), 5 * 8); // 4 num + CLS.
    }

    #[test]
    fn d_token_0_error() {
        let mut rng = LcgRng::new(1);
        let cfg = FtTransformerConfig {
            n_num_features: 2,
            n_cat_features: 0,
            cat_cardinalities: vec![],
            d_token: 0,
            n_heads: 1,
            n_layers: 1,
        };
        assert!(matches!(
            FtTransformer::new(cfg, &mut rng),
            Err(TabularError::InvalidEmbedDim { .. })
        ));
    }

    #[test]
    fn num_feats_mismatch_error() {
        let model = make_model(3, 1, vec![4]);
        let res = model.forward(&[0.1, 0.2], &[0]); // expects 3 numeric features
        assert!(matches!(res, Err(TabularError::DimensionMismatch { .. })));
    }

    #[test]
    fn cat_feats_mismatch_error() {
        let model = make_model(2, 2, vec![4, 4]);
        let res = model.tokenize(&[0.1, 0.2], &[0]); // expects 2 categoricals
        assert!(matches!(res, Err(TabularError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_deterministic() {
        let model = make_model(3, 2, vec![4, 5]);
        let a = model
            .forward(&[0.5, 0.6, 0.7], &[1, 2])
            .expect("forward should succeed");
        let b = model
            .forward(&[0.5, 0.6, 0.7], &[1, 2])
            .expect("forward should succeed");
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn heads_divisibility_error() {
        let mut rng = LcgRng::new(1);
        let cfg = FtTransformerConfig {
            n_num_features: 2,
            n_cat_features: 0,
            cat_cardinalities: vec![],
            d_token: 7, // not divisible by 2
            n_heads: 2,
            n_layers: 1,
        };
        assert!(matches!(
            FtTransformer::new(cfg, &mut rng),
            Err(TabularError::InvalidAttentionDim { .. })
        ));
    }
}
