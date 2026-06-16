//! SigLIP — sigmoid loss for language-image pre-training (Zhai et al., 2023).
//!
//! Unlike CLIP's softmax InfoNCE (which normalises across a whole batch row /
//! column), SigLIP treats every image–text pair *independently* as a binary
//! classification problem: the matched diagonal pairs are positives, every
//! off-diagonal pair is a negative, and the loss is a sum of element-wise
//! sigmoid cross-entropies. This removes the need for a global softmax (and the
//! associated all-gather over the batch), which is what makes SigLIP scale.
//!
//! Given L2-normalised image embeddings `Z_img ∈ ℝ^{n×d}` and text embeddings
//! `Z_txt ∈ ℝ^{n×d}`, a learnable temperature `t` (stored in log space,
//! `t = exp(t')`) and a learnable bias `b`, the pairwise logit is
//!
//! ```text
//! ℓ_ij = t · (z_img_i · z_txt_j) + b
//! ```
//!
//! with labels `y_ij = +1` when `i == j` and `y_ij = −1` otherwise. The loss is
//!
//! ```text
//! L = −(1/n) Σ_i Σ_j log σ(y_ij · ℓ_ij).
//! ```
//!
//! `σ` is the logistic sigmoid. `−log σ(z) = softplus(−z)` is evaluated with a
//! numerically stable `softplus` so that large `|z|` never overflows.

use crate::error::{MmResult, MultiModalError};

// ─── SigLIP configuration ──────────────────────────────────────────────────────

/// Learnable parameters of the SigLIP sigmoid-contrastive head.
///
/// The temperature is stored in log space (`log_t`) exactly as in the paper, so
/// `t = exp(log_t)` is guaranteed positive without any constraint. The bias `b`
/// is initialised to a large negative value in practice (so that early in
/// training most negatives are already classified correctly); the default here
/// reproduces the paper's `log_t = log(10)`, `b = −10`.
#[derive(Debug, Clone)]
pub struct SigLipConfig {
    /// Log-temperature `t' = log(t)`. The effective temperature is `exp(t')`.
    pub log_t: f32,
    /// Additive logit bias `b`.
    pub bias: f32,
}

impl SigLipConfig {
    /// Paper default: `t = 10` (`log_t = ln 10`), `b = −10`.
    #[must_use]
    pub fn paper_default() -> Self {
        Self {
            log_t: std::f32::consts::LN_10,
            bias: -10.0,
        }
    }

    /// Construct directly from a (linear) temperature and bias.
    ///
    /// Returns [`MultiModalError::InvalidTemperature`] if `t` is not strictly
    /// positive and finite.
    pub fn from_temperature(t: f32, bias: f32) -> MmResult<Self> {
        if t <= 0.0 || !t.is_finite() {
            return Err(MultiModalError::InvalidTemperature { temp: t });
        }
        Ok(Self {
            log_t: t.ln(),
            bias,
        })
    }

    /// Effective (linear) temperature `t = exp(log_t)`.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.log_t.exp()
    }
}

// ─── Numerically stable primitives ─────────────────────────────────────────────

/// `softplus(x) = log(1 + exp(x))`, evaluated without overflow.
///
/// For `x ≥ 0` this is `x + log(1 + exp(−x))`; for `x < 0` it is
/// `log(1 + exp(x))`. Both branches keep the `exp` argument `≤ 0`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (1.0 + (-x).exp()).ln()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `−log σ(z) = softplus(−z)` — the binary cross-entropy of a `+1` target
/// against logit `z`.
#[inline]
fn neg_log_sigmoid(z: f32) -> f32 {
    softplus(-z)
}

// ─── Similarity matrix ──────────────────────────────────────────────────────────

/// Pairwise cosine-similarity matrix `S ∈ ℝ^{n×n}`, `S[i,j] = z_img_i · z_txt_j`.
///
/// The embeddings are expected to be **already L2-normalised** (as in SigLIP);
/// no re-normalisation is performed here so that callers retain full control.
fn similarity_matrix(z_img: &[f32], z_txt: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut sim = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0_f32;
            for k in 0..d {
                dot += z_img[i * d + k] * z_txt[j * d + k];
            }
            sim[i * n + j] = dot;
        }
    }
    sim
}

/// Build the SigLIP label matrix `Y ∈ {−1, +1}^{n×n}`: `+1` on the diagonal
/// (matched pairs), `−1` everywhere else.
#[must_use]
pub fn siglip_labels(n: usize) -> Vec<f32> {
    let mut y = vec![-1.0_f32; n * n];
    for i in 0..n {
        y[i * n + i] = 1.0;
    }
    y
}

// ─── Logits & loss ──────────────────────────────────────────────────────────────

/// Pairwise logits `ℓ_ij = t · S[i,j] + b` from a pre-computed similarity matrix.
fn logits_from_sim(sim: &[f32], cfg: &SigLipConfig) -> Vec<f32> {
    let t = cfg.temperature();
    let b = cfg.bias;
    sim.iter().map(|&s| t * s + b).collect()
}

/// Validate that two embedding matrices share the `[n × d]` shape with `n ≥ 1`,
/// `d ≥ 1`.
fn check_shapes(z_img: &[f32], z_txt: &[f32], n: usize, d: usize) -> MmResult<()> {
    if n == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    if d == 0 {
        return Err(MultiModalError::InvalidFeatureDim);
    }
    if z_img.len() != n * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: n * d,
            got: z_img.len(),
        });
    }
    if z_txt.len() != n * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: n * d,
            got: z_txt.len(),
        });
    }
    Ok(())
}

/// Compute the SigLIP sigmoid-contrastive loss.
///
/// `z_img`, `z_txt`: row-major `[n × d]`, expected L2-normalised. `cfg` carries
/// the learnable log-temperature and bias.
///
/// Returns the mean over all `n²` pairs of `−log σ(y_ij · ℓ_ij)`, which is
/// always `≥ 0`.
pub fn siglip_loss(
    z_img: &[f32],
    z_txt: &[f32],
    n: usize,
    d: usize,
    cfg: &SigLipConfig,
) -> MmResult<f32> {
    check_shapes(z_img, z_txt, n, d)?;

    let sim = similarity_matrix(z_img, z_txt, n, d);
    let logits = logits_from_sim(&sim, cfg);
    let labels = siglip_labels(n);

    let mut total = 0.0_f32;
    for idx in 0..n * n {
        // y · ℓ, then −log σ(·) = softplus(−yℓ).
        total += neg_log_sigmoid(labels[idx] * logits[idx]);
    }
    let loss = total / (n * n) as f32;

    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "siglip_loss",
        });
    }
    Ok(loss)
}

/// Same as [`siglip_loss`] but takes a caller-supplied similarity matrix
/// `sim ∈ ℝ^{n×n}` directly (e.g. when the cosine similarities have already
/// been materialised, or for the symmetry test that transposes them).
pub fn siglip_loss_from_sim(sim: &[f32], n: usize, cfg: &SigLipConfig) -> MmResult<f32> {
    if n == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    if sim.len() != n * n {
        return Err(MultiModalError::DimensionMismatch {
            expected: n * n,
            got: sim.len(),
        });
    }
    let logits = logits_from_sim(sim, cfg);
    let labels = siglip_labels(n);
    let mut total = 0.0_f32;
    for idx in 0..n * n {
        total += neg_log_sigmoid(labels[idx] * logits[idx]);
    }
    let loss = total / (n * n) as f32;
    if !loss.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "siglip_loss_from_sim",
        });
    }
    Ok(loss)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `[n × d]` embedding where row `i` is the one-hot unit vector
    /// `e_{i mod d}` (already L2-normalised). Used to make a perfectly-aligned
    /// image/text pair (diagonal sim = 1).
    fn one_hot_rows(n: usize, d: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n * d];
        for i in 0..n {
            v[i * d + (i % d)] = 1.0;
        }
        v
    }

    #[test]
    fn loss_is_non_negative() {
        let n = 6;
        let d = 8;
        let img = one_hot_rows(n, d);
        let txt = one_hot_rows(n, d);
        let cfg = SigLipConfig::paper_default();
        let loss = siglip_loss(&img, &txt, n, d, &cfg).expect("siglip_loss should succeed");
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        assert!(loss.is_finite());
    }

    #[test]
    fn label_matrix_diagonal_plus_offdiag_minus() {
        let n = 4;
        let y = siglip_labels(n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(y[i * n + j], 1.0, "diagonal ({i},{j}) must be +1");
                } else {
                    assert_eq!(y[i * n + j], -1.0, "off-diagonal ({i},{j}) must be -1");
                }
            }
        }
    }

    #[test]
    fn perfect_alignment_is_near_minimal() {
        // Construct a similarity matrix with diagonal sim = 1, off-diagonal = -1.
        // With a large temperature and zero bias, every y·ℓ is large & positive,
        // so the loss is close to its minimum (0).
        let n = 5;
        let mut sim = vec![-1.0_f32; n * n];
        for i in 0..n {
            sim[i * n + i] = 1.0;
        }
        let cfg =
            SigLipConfig::from_temperature(10.0, 0.0).expect("from_temperature should succeed");
        let loss =
            siglip_loss_from_sim(&sim, n, &cfg).expect("siglip_loss_from_sim should succeed");
        // y·ℓ = 10 on every entry → −log σ(10) ≈ 4.5e-5.
        assert!(
            loss < 1e-3,
            "perfectly aligned loss should be tiny, got {loss}"
        );
    }

    #[test]
    fn anti_aligned_has_larger_loss_than_aligned() {
        let n = 5;
        // Aligned: diagonal +1, off-diagonal -1.
        let mut aligned = vec![-1.0_f32; n * n];
        for i in 0..n {
            aligned[i * n + i] = 1.0;
        }
        // Anti-aligned: diagonal LOW (-1), off-diagonal high (+1) — the worst case.
        let mut anti = vec![1.0_f32; n * n];
        for i in 0..n {
            anti[i * n + i] = -1.0;
        }
        let cfg =
            SigLipConfig::from_temperature(10.0, 0.0).expect("from_temperature should succeed");
        let loss_aligned =
            siglip_loss_from_sim(&aligned, n, &cfg).expect("siglip_loss_from_sim should succeed");
        let loss_anti =
            siglip_loss_from_sim(&anti, n, &cfg).expect("siglip_loss_from_sim should succeed");
        assert!(
            loss_anti > loss_aligned,
            "anti-aligned loss {loss_anti} must exceed aligned loss {loss_aligned}"
        );
    }

    #[test]
    fn symmetry_under_image_text_swap() {
        // SigLIP loss must be invariant to swapping image/text roles, which
        // corresponds to transposing the similarity matrix (S[i,j] -> S[j,i])
        // because the label matrix is symmetric.
        let n = 4;
        let d = 6;
        let img: Vec<f32> = (0..n * d).map(|k| ((k as f32) * 0.17).sin()).collect();
        let txt: Vec<f32> = (0..n * d).map(|k| ((k as f32) * 0.09).cos()).collect();
        let cfg = SigLipConfig::paper_default();

        let sim = super::similarity_matrix(&img, &txt, n, d);
        let mut sim_t = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                sim_t[i * n + j] = sim[j * n + i];
            }
        }
        let loss =
            siglip_loss_from_sim(&sim, n, &cfg).expect("siglip_loss_from_sim should succeed");
        let loss_t =
            siglip_loss_from_sim(&sim_t, n, &cfg).expect("siglip_loss_from_sim should succeed");
        assert!(
            (loss - loss_t).abs() < 1e-5,
            "loss {loss} and transposed loss {loss_t} must match"
        );
    }

    #[test]
    fn higher_temperature_sharpens_confident_logits() {
        // For a confidently-correct example (all y·sim > 0), raising the
        // temperature scales every correct margin up, so the loss decreases
        // monotonically (the sigmoid saturates harder towards 1).
        let n = 4;
        let mut sim = vec![-0.5_f32; n * n];
        for i in 0..n {
            sim[i * n + i] = 0.5;
        }
        let lo = SigLipConfig::from_temperature(2.0, 0.0).expect("from_temperature should succeed");
        let hi = SigLipConfig::from_temperature(8.0, 0.0).expect("from_temperature should succeed");
        let loss_lo =
            siglip_loss_from_sim(&sim, n, &lo).expect("siglip_loss_from_sim should succeed");
        let loss_hi =
            siglip_loss_from_sim(&sim, n, &hi).expect("siglip_loss_from_sim should succeed");
        assert!(
            loss_hi < loss_lo,
            "higher t should sharpen correct logits: lo={loss_lo}, hi={loss_hi}"
        );
    }

    #[test]
    fn finite_difference_matched_similarity_decreases_loss() {
        // Gradient sign check: increasing a matched (diagonal) similarity should
        // DECREASE the loss, because that pair becomes more confidently positive.
        let n = 4;
        let d = 5;
        let img: Vec<f32> = (0..n * d)
            .map(|k| ((k as f32) * 0.11).sin() * 0.3)
            .collect();
        let txt = img.clone();
        let cfg =
            SigLipConfig::from_temperature(5.0, 0.0).expect("from_temperature should succeed");

        let base = super::similarity_matrix(&img, &txt, n, d);
        let loss0 =
            siglip_loss_from_sim(&base, n, &cfg).expect("siglip_loss_from_sim should succeed");

        // Perturb the (0,0) matched similarity upward by a small epsilon.
        let mut bumped = base.clone();
        bumped[0] += 0.01;
        let loss1 =
            siglip_loss_from_sim(&bumped, n, &cfg).expect("siglip_loss_from_sim should succeed");

        assert!(
            loss1 < loss0,
            "raising matched sim should reduce loss: before={loss0}, after={loss1}"
        );
    }

    #[test]
    fn config_temperature_roundtrip() {
        let cfg =
            SigLipConfig::from_temperature(7.5, -3.0).expect("from_temperature should succeed");
        assert!((cfg.temperature() - 7.5).abs() < 1e-4);
        assert_eq!(cfg.bias, -3.0);
    }

    #[test]
    fn config_rejects_non_positive_temperature() {
        let err = SigLipConfig::from_temperature(0.0, 0.0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidTemperature { .. }));
        let err2 = SigLipConfig::from_temperature(-1.0, 0.0).unwrap_err();
        assert!(matches!(err2, MultiModalError::InvalidTemperature { .. }));
    }

    #[test]
    fn loss_rejects_bad_shapes() {
        let cfg = SigLipConfig::paper_default();
        let img = vec![0.0_f32; 3 * 4];
        let txt = vec![0.0_f32; 2 * 4]; // wrong n
        let err = siglip_loss(&img, &txt, 3, 4, &cfg).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));

        let err_n = siglip_loss(&[], &[], 0, 4, &cfg).unwrap_err();
        assert!(matches!(err_n, MultiModalError::InvalidBatchSize));

        let err_d = siglip_loss(&[], &[], 2, 0, &cfg).unwrap_err();
        assert!(matches!(err_d, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn softplus_is_stable_for_large_magnitude() {
        // softplus(+1000) ≈ 1000 (not inf); softplus(-1000) ≈ 0.
        assert!((softplus(1000.0) - 1000.0).abs() < 1e-2);
        assert!(softplus(-1000.0).abs() < 1e-6);
        assert!(softplus(1000.0).is_finite());
    }

    #[test]
    fn bias_shifts_loss_consistently() {
        // A more negative bias makes positives harder and negatives easier; on a
        // mostly-negative matrix (n² entries, n positives) a large negative bias
        // should keep the loss finite and the function must respond to b.
        let n = 4;
        let mut sim = vec![-0.5_f32; n * n];
        for i in 0..n {
            sim[i * n + i] = 0.8;
        }
        let cfg_a =
            SigLipConfig::from_temperature(4.0, 0.0).expect("from_temperature should succeed");
        let cfg_b =
            SigLipConfig::from_temperature(4.0, -5.0).expect("from_temperature should succeed");
        let la =
            siglip_loss_from_sim(&sim, n, &cfg_a).expect("siglip_loss_from_sim should succeed");
        let lb =
            siglip_loss_from_sim(&sim, n, &cfg_b).expect("siglip_loss_from_sim should succeed");
        assert!(la.is_finite() && lb.is_finite());
        assert!(
            (la - lb).abs() > 1e-4,
            "bias must change the loss: {la} vs {lb}"
        );
    }
}
