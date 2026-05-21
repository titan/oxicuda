//! AdaMerging — unsupervised adaptive coefficient learning for model merging.
//!
//! Reference: Yang E, Wang Z, Shen L, Liu S, Guo G, Wang X, Tao D (2024)
//! "AdaMerging: Adaptive Model Merging for Multi-Task Learning", ICLR.
//! <https://arxiv.org/abs/2310.02575>
//!
//! AdaMerging extends task-arithmetic style merging by learning the mixing
//! coefficients `{λ_k}` from *unlabeled* data via entropy minimisation. Given
//! a base model `θ_0` and `K` task vectors `{τ_k = θ_k − θ_0}`, the merged
//! parameters are
//!
//! ```text
//! θ̄  =  θ_0  +  Σ_k softmax_k(λ) · τ_k                   (task-wise)
//! ```
//!
//! For the *layer-wise* variant the coefficients are blocked along the
//! parameter axis using the user-supplied `layer_offsets`:
//!
//! ```text
//! θ̄_p  =  θ_{0,p}  +  Σ_k softmax_k(λ_{ℓ(p), ·}) · τ_{k,p}
//! ```
//!
//! where `ℓ(p)` is the layer index containing parameter `p`. There are
//! `K · L` coefficients laid out as `λ[ℓ * K + k]`.
//!
//! Since the crate is dependency-free and has no autograd, we approximate the
//! forward-pass entropy with a *logit proxy*: the caller supplies, for each
//! task `k`, the unlabeled-batch logit vector that task would produce. The
//! merged-model proxy is `logits_pred = Σ_k softmax_k(λ̄) · unlabeled_logits[k]`
//! where `λ̄` collapses to the per-task softmax (for layer-wise mode we
//! arithmetic-average the per-layer softmax weights over layers, since each
//! layer contributes uniformly to the proxy classifier). Entropy is taken over
//! the temperature-scaled softmax of the proxy logits. Gradients are computed
//! via central finite differences; updates project back onto the simplex by
//! applying softmax after the step.

use crate::error::{PeftError, PeftResult};

/// Configuration controlling the entropy-minimisation procedure.
#[derive(Debug, Clone)]
pub struct AdaMergingConfig {
    /// Gradient-descent learning rate; must be strictly positive. Typical 1e-3.
    pub learning_rate: f32,
    /// Number of gradient-descent iterations; must be ≥ 1. Typical 200.
    pub n_iters: usize,
    /// Softmax temperature for the entropy proxy; must be strictly positive.
    /// Typical 1.0.
    pub temperature: f32,
    /// When `true`, coefficients are learned per task *and* per layer.
    pub layer_wise: bool,
    /// Cumulative parameter offsets per layer (length `L`). Each entry is the
    /// *exclusive* upper bound of a layer (so the implicit lower bound of layer
    /// 0 is zero, and the final entry must equal `base.len()`). Only consulted
    /// when `layer_wise == true`.
    pub layer_offsets: Vec<usize>,
}

impl AdaMergingConfig {
    /// Construct a per-task (i.e. non-layer-wise) configuration.
    #[must_use]
    pub fn per_task(learning_rate: f32, n_iters: usize, temperature: f32) -> Self {
        Self {
            learning_rate,
            n_iters,
            temperature,
            layer_wise: false,
            layer_offsets: Vec::new(),
        }
    }

    /// Construct a layer-wise configuration with the given partitioning.
    #[must_use]
    pub fn layer_wise(
        learning_rate: f32,
        n_iters: usize,
        temperature: f32,
        layer_offsets: Vec<usize>,
    ) -> Self {
        Self {
            learning_rate,
            n_iters,
            temperature,
            layer_wise: true,
            layer_offsets,
        }
    }
}

/// Result of an [`AdaMerging::merge`] run.
#[derive(Debug, Clone)]
pub struct AdaMergingResult {
    /// Merged parameter vector (same length as `base`).
    pub merged: Vec<f32>,
    /// Final softmax-normalised mixing coefficients. Length is `K` for the
    /// per-task variant and `K · L` for the layer-wise variant (laid out as
    /// `coefficients[ℓ * K + k]`).
    pub coefficients: Vec<f32>,
    /// Entropy after the final iteration.
    pub final_entropy: f32,
    /// Per-iteration entropy values (length `n_iters`).
    pub iter_history: Vec<f32>,
}

/// AdaMerging algorithm namespace.
pub struct AdaMerging;

impl AdaMerging {
    /// Run AdaMerging entropy-minimisation and return the merged model along
    /// with the learnt coefficients.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] when any input or configuration field is
    /// invalid (see module-level documentation for the full list of checks).
    pub fn merge(
        base: &[f32],
        task_vectors: &[Vec<f32>],
        unlabeled_logits: &[Vec<f32>],
        cfg: &AdaMergingConfig,
    ) -> PeftResult<AdaMergingResult> {
        validate(base, task_vectors, unlabeled_logits, cfg)?;

        let k = task_vectors.len();
        let n_layers = if cfg.layer_wise {
            cfg.layer_offsets.len()
        } else {
            1
        };
        let n_coef = k * n_layers;

        // Initialise pre-softmax logits to zero (so initial softmax is uniform).
        let mut lambda = vec![0.0_f32; n_coef];

        let mut history = Vec::with_capacity(cfg.n_iters);
        let lr = cfg.learning_rate;
        let h = 1e-4_f32;

        for _ in 0..cfg.n_iters {
            let entropy = compute_entropy(&lambda, unlabeled_logits, cfg);
            history.push(entropy);

            // Central finite-difference gradient.
            let mut grad = vec![0.0_f32; n_coef];
            for i in 0..n_coef {
                let saved = lambda[i];
                lambda[i] = saved + h;
                let h_plus = compute_entropy(&lambda, unlabeled_logits, cfg);
                lambda[i] = saved - h;
                let h_minus = compute_entropy(&lambda, unlabeled_logits, cfg);
                lambda[i] = saved;
                grad[i] = (h_plus - h_minus) / (2.0 * h);
            }

            // Gradient-descent step.
            for (l, g) in lambda.iter_mut().zip(grad.iter()) {
                *l -= lr * g;
            }

            // Project onto the simplex per (layer-)group via softmax of the
            // pre-activation logits, then convert back to logit-space by taking
            // the log so the subsequent finite-difference perturbation is well
            // conditioned. This keeps the optimisation in the same parameter
            // space the simplex constraint demands.
            for layer in 0..n_layers {
                let off = layer * k;
                let block = &mut lambda[off..off + k];
                let sm = softmax(block, 1.0);
                for (slot, &p) in block.iter_mut().zip(sm.iter()) {
                    // log p, with eps guard to avoid −∞ for entries the
                    // optimiser has driven to zero.
                    *slot = (p.max(1e-12)).ln();
                }
            }
        }

        // Final coefficients: softmax of the lambda logits.
        let mut coefficients = Vec::with_capacity(n_coef);
        for layer in 0..n_layers {
            let off = layer * k;
            let sm = softmax(&lambda[off..off + k], 1.0);
            coefficients.extend_from_slice(&sm);
        }

        let merged = compose(base, task_vectors, &coefficients, cfg);
        let final_entropy = compute_entropy(&lambda, unlabeled_logits, cfg);

        Ok(AdaMergingResult {
            merged,
            coefficients,
            final_entropy,
            iter_history: history,
        })
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn validate(
    base: &[f32],
    task_vectors: &[Vec<f32>],
    unlabeled_logits: &[Vec<f32>],
    cfg: &AdaMergingConfig,
) -> PeftResult<()> {
    if base.is_empty() {
        return Err(PeftError::Internal {
            msg: "AdaMerging requires a non-empty base".to_string(),
        });
    }
    if task_vectors.is_empty() {
        return Err(PeftError::Internal {
            msg: "AdaMerging requires at least one task vector".to_string(),
        });
    }
    if unlabeled_logits.is_empty() {
        return Err(PeftError::Internal {
            msg: "AdaMerging requires at least one unlabeled-logit sample".to_string(),
        });
    }
    if cfg.learning_rate.is_nan() || cfg.learning_rate <= 0.0 {
        return Err(PeftError::Internal {
            msg: format!(
                "AdaMerging learning_rate must be > 0, got {}",
                cfg.learning_rate
            ),
        });
    }
    if cfg.n_iters == 0 {
        return Err(PeftError::Internal {
            msg: "AdaMerging n_iters must be ≥ 1".to_string(),
        });
    }
    if cfg.temperature.is_nan() || cfg.temperature <= 0.0 {
        return Err(PeftError::Internal {
            msg: format!(
                "AdaMerging temperature must be > 0, got {}",
                cfg.temperature
            ),
        });
    }
    let n = base.len();
    for (i, tv) in task_vectors.iter().enumerate() {
        if tv.len() != n {
            return Err(PeftError::Internal {
                msg: format!(
                    "AdaMerging task_vectors[{i}] length {} != base length {n}",
                    tv.len()
                ),
            });
        }
    }
    let c = unlabeled_logits[0].len();
    if c == 0 {
        return Err(PeftError::Internal {
            msg: "AdaMerging unlabeled_logits[0] must be non-empty".to_string(),
        });
    }
    for (i, ul) in unlabeled_logits.iter().enumerate() {
        if ul.len() != c {
            return Err(PeftError::Internal {
                msg: format!(
                    "AdaMerging unlabeled_logits[{i}] length {} != logits[0] length {c}",
                    ul.len()
                ),
            });
        }
    }
    if cfg.layer_wise {
        if cfg.layer_offsets.is_empty() {
            return Err(PeftError::Internal {
                msg: "AdaMerging layer-wise mode requires non-empty layer_offsets".to_string(),
            });
        }
        let mut prev: i64 = 0;
        for (i, &off) in cfg.layer_offsets.iter().enumerate() {
            if off == 0 {
                return Err(PeftError::Internal {
                    msg: format!("AdaMerging layer_offsets[{i}] must be > 0"),
                });
            }
            if off > n {
                return Err(PeftError::Internal {
                    msg: format!("AdaMerging layer_offsets[{i}]={off} exceeds base length {n}"),
                });
            }
            if (off as i64) <= prev {
                return Err(PeftError::Internal {
                    msg: format!(
                        "AdaMerging layer_offsets must be strictly increasing at index {i}"
                    ),
                });
            }
            prev = off as i64;
        }
        if let Some(&last) = cfg.layer_offsets.last()
            && last != n
        {
            return Err(PeftError::Internal {
                msg: format!("AdaMerging final layer_offset {last} must equal base length {n}"),
            });
        }
    }
    Ok(())
}

/// Numerically stable softmax with the given temperature applied to `xs`.
fn softmax(xs: &[f32], temperature: f32) -> Vec<f32> {
    if xs.is_empty() {
        return Vec::new();
    }
    let inv_t = 1.0 / temperature;
    let mut shifted: Vec<f32> = xs.iter().map(|&x| x * inv_t).collect();
    let mut m = shifted[0];
    for &v in &shifted[1..] {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0_f32;
    for s in &mut shifted {
        *s = (*s - m).exp();
        sum += *s;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    shifted.iter().map(|&v| v * inv).collect()
}

/// Average per-layer task coefficients into a single K-vector by mean of the
/// per-layer softmax weights. The result is itself a valid simplex point
/// because each layer's softmax sums to 1 and the mean of K non-negative
/// numbers that each sum to one over the layer dimension still sums to 1.
fn task_mean_coefs(lambda: &[f32], k: usize, layer_wise: bool, n_layers: usize) -> Vec<f32> {
    if !layer_wise {
        return softmax(lambda, 1.0);
    }
    let mut acc = vec![0.0_f32; k];
    for layer in 0..n_layers {
        let off = layer * k;
        let sm = softmax(&lambda[off..off + k], 1.0);
        for (a, &s) in acc.iter_mut().zip(sm.iter()) {
            *a += s;
        }
    }
    let inv = 1.0_f32 / (n_layers as f32);
    for a in &mut acc {
        *a *= inv;
    }
    acc
}

/// Compute the entropy proxy `H(softmax_T(Σ_k w_k · logits_k))` where the
/// `w_k` are the (layer-averaged) softmax weights derived from `lambda`.
fn compute_entropy(lambda: &[f32], unlabeled_logits: &[Vec<f32>], cfg: &AdaMergingConfig) -> f32 {
    let k = unlabeled_logits.len();
    let n_layers = if cfg.layer_wise {
        cfg.layer_offsets.len()
    } else {
        1
    };
    let weights = task_mean_coefs(lambda, k, cfg.layer_wise, n_layers);
    let c = unlabeled_logits[0].len();
    let mut blended = vec![0.0_f32; c];
    for (w, logits) in weights.iter().zip(unlabeled_logits.iter()) {
        for (b, &v) in blended.iter_mut().zip(logits.iter()) {
            *b += *w * v;
        }
    }
    let probs = softmax(&blended, cfg.temperature);
    let mut h = 0.0_f32;
    for &p in &probs {
        if p > 0.0 {
            h -= p * (p.max(1e-12)).ln();
        }
    }
    h
}

/// Compose the merged parameter vector from the base and the task vectors
/// weighted by the (per-task or per-layer-per-task) coefficients.
fn compose(
    base: &[f32],
    task_vectors: &[Vec<f32>],
    coefficients: &[f32],
    cfg: &AdaMergingConfig,
) -> Vec<f32> {
    let mut merged = base.to_vec();
    let k = task_vectors.len();
    if !cfg.layer_wise {
        for (kdx, tv) in task_vectors.iter().enumerate() {
            let w = coefficients[kdx];
            for (m, &t) in merged.iter_mut().zip(tv.iter()) {
                *m += w * t;
            }
        }
        return merged;
    }
    // Layer-wise composition.
    let mut start = 0_usize;
    for (layer, &end) in cfg.layer_offsets.iter().enumerate() {
        let coef_off = layer * k;
        for kdx in 0..k {
            let w = coefficients[coef_off + kdx];
            let tv = &task_vectors[kdx];
            for p in start..end {
                merged[p] += w * tv[p];
            }
        }
        start = end;
    }
    merged
}
