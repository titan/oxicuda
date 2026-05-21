//! MGD — Masked Generative Distillation (Yang et al. 2022 ECCV).
//!
//! Instead of forcing the student to *mimic* the teacher's full feature map, MGD randomly
//! **masks** a large fraction of the (channel-aligned) student feature pixels and trains a
//! small **generator** to **reconstruct** the teacher's complete feature map from the masked
//! student feature. Using the partial student feature to recover the full teacher feature
//! pushes the student towards more representative, generative features rather than rote
//! pixel-by-pixel imitation.
//!
//! Pipeline (per spatial location, mask shared across channels):
//! 1. Align student to `C_t` channels via a 1×1 convolution (skipped when `C_s == C_t`
//!    and no align weights are present).
//! 2. Draw a binary spatial mask `M ∈ {0,1}^{H×W}` shared across all channels: a location
//!    is masked (`0`) when `rng.next_f32() < mask_ratio`, otherwise kept (`1`).
//! 3. Multiply the aligned student feature by the broadcast mask.
//! 4. Run the generator: `conv3×3(C_t→C_t) → ReLU → conv3×3(C_t→C_t)` (same-pad, zero pad).
//! 5. Return `alpha_mgd · MSE(recon, teacher)` averaged over all `C_t·H·W` elements.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// Default masking fraction recommended by the MGD paper.
pub const DEFAULT_MASK_RATIO: f32 = 0.65;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Masked Generative Distillation.
#[derive(Debug, Clone)]
pub struct MgdConfig {
    /// Number of student feature channels `C_s`.
    pub n_channels_s: usize,
    /// Number of teacher feature channels `C_t`.
    pub n_channels_t: usize,
    /// Spatial height `H` of both feature maps.
    pub height: usize,
    /// Spatial width `W` of both feature maps.
    pub width: usize,
    /// Fraction of spatial locations to mask, in `[0, 1]` (paper default `0.65`).
    pub mask_ratio: f32,
    /// Loss weight `α_MGD` applied to the reconstruction MSE.
    pub alpha_mgd: f32,
    /// Seed used when initializing generator weights.
    pub seed: u64,
}

impl Default for MgdConfig {
    fn default() -> Self {
        Self {
            n_channels_s: 0,
            n_channels_t: 0,
            height: 0,
            width: 0,
            mask_ratio: DEFAULT_MASK_RATIO,
            alpha_mgd: 1.0,
            seed: 0,
        }
    }
}

impl MgdConfig {
    /// Validate the configuration's structural fields.
    fn validate(&self) -> DistillResult<()> {
        if self.n_channels_s == 0 || self.n_channels_t == 0 || self.height == 0 || self.width == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "MgdConfig: n_channels_s, n_channels_t, height, width must all be > 0".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.mask_ratio) {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "MgdConfig: mask_ratio must be in [0, 1], got {}",
                    self.mask_ratio
                ),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generator
// ─────────────────────────────────────────────────────────────────────────────

/// The MGD generator with an optional 1×1 channel-alignment convolution.
///
/// Weight layouts (all row-major):
/// * `w_align`: optional `[C_t × C_s]` 1×1 convolution applied per spatial location to lift
///   the student feature from `C_s` to `C_t` channels. `None` when `C_s == C_t`.
/// * `w1`: `[C_t × C_t × 3 × 3]` first 3×3 same-pad convolution (out, in, kh, kw).
/// * `w2`: `[C_t × C_t × 3 × 3]` second 3×3 same-pad convolution.
///
/// The forward pass is `conv3×3(w1) → ReLU → conv3×3(w2)`.
#[derive(Debug, Clone)]
pub struct MgdGenerator {
    /// Optional `[C_t × C_s]` 1×1 alignment weights.
    pub w_align: Option<Vec<f32>>,
    /// First 3×3 convolution weights `[C_t × C_t × 3 × 3]`.
    pub w1: Vec<f32>,
    /// Second 3×3 convolution weights `[C_t × C_t × 3 × 3]`.
    pub w2: Vec<f32>,
    /// Student channel count `C_s`.
    pub n_channels_s: usize,
    /// Teacher channel count `C_t`.
    pub n_channels_t: usize,
}

impl MgdGenerator {
    /// Construct a generator with randomly initialized weights.
    ///
    /// The 1×1 alignment convolution is created only when `n_channels_s != n_channels_t`.
    /// All weights are drawn from a scaled normal distribution (Kaiming-style fan-in scaling)
    /// using the crate's [`LcgRng::next_normal`].
    pub fn new(n_channels_s: usize, n_channels_t: usize, rng: &mut LcgRng) -> DistillResult<Self> {
        if n_channels_s == 0 || n_channels_t == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "MgdGenerator: n_channels_s and n_channels_t must be > 0".into(),
            });
        }
        let w_align = if n_channels_s != n_channels_t {
            // 1×1 conv fan_in = C_s.
            let scale = (2.0_f32 / n_channels_s as f32).sqrt();
            Some(
                (0..n_channels_t * n_channels_s)
                    .map(|_| rng.next_normal() * scale)
                    .collect(),
            )
        } else {
            None
        };
        // 3×3 conv fan_in = C_t * 9.
        let fan_in = n_channels_t * 9;
        let scale = (2.0_f32 / fan_in as f32).sqrt();
        let w_len = n_channels_t * n_channels_t * 9;
        let w1 = (0..w_len).map(|_| rng.next_normal() * scale).collect();
        let w2 = (0..w_len).map(|_| rng.next_normal() * scale).collect();
        Ok(Self {
            w_align,
            w1,
            w2,
            n_channels_s,
            n_channels_t,
        })
    }

    /// Build a generator from explicit weights for deterministic tests.
    ///
    /// Validates that the supplied weight slices have the expected lengths:
    /// `w_align` (if `Some`) must be `C_t·C_s`, and both `w1`/`w2` must be `C_t·C_t·9`.
    pub fn from_weights(
        w_align: Option<Vec<f32>>,
        w1: Vec<f32>,
        w2: Vec<f32>,
        n_channels_s: usize,
        n_channels_t: usize,
    ) -> DistillResult<Self> {
        if n_channels_s == 0 || n_channels_t == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "MgdGenerator: n_channels_s and n_channels_t must be > 0".into(),
            });
        }
        let conv_len = n_channels_t * n_channels_t * 9;
        if w1.len() != conv_len {
            return Err(DistillError::DimensionMismatch {
                expected: conv_len,
                got: w1.len(),
            });
        }
        if w2.len() != conv_len {
            return Err(DistillError::DimensionMismatch {
                expected: conv_len,
                got: w2.len(),
            });
        }
        if let Some(wa) = &w_align {
            let align_len = n_channels_t * n_channels_s;
            if wa.len() != align_len {
                return Err(DistillError::DimensionMismatch {
                    expected: align_len,
                    got: wa.len(),
                });
            }
        }
        Ok(Self {
            w_align,
            w1,
            w2,
            n_channels_s,
            n_channels_t,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core operations
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a binary spatial mask `M ∈ {0,1}^{H·W}` (row-major, length `h·w`).
///
/// A location is masked (`0.0`) when `rng.next_f32() < mask_ratio`, otherwise kept (`1.0`).
/// With `mask_ratio == 0.0` every location is `1.0`; with `mask_ratio == 1.0` every
/// location is `0.0` (`next_f32()` lies in `[0, 1)`, so it is always `< 1.0`).
#[must_use]
pub fn generate_mask(h: usize, w: usize, mask_ratio: f32, rng: &mut LcgRng) -> Vec<f32> {
    let n = h * w;
    let mut mask = Vec::with_capacity(n);
    for _ in 0..n {
        let keep = if rng.next_f32() < mask_ratio {
            0.0
        } else {
            1.0
        };
        mask.push(keep);
    }
    mask
}

/// Apply the 1×1 alignment convolution lifting `C_s` channels to `C_t` channels.
///
/// `student` is `[C_s × H × W]` (channel-major). Output is `[C_t × H × W]`.
/// `out[co, p] = Σ_ci w_align[co·C_s + ci] · student[ci·HW + p]`.
fn align_student(
    student: &[f32],
    w_align: &[f32],
    n_channels_s: usize,
    n_channels_t: usize,
    hw: usize,
) -> DistillResult<Vec<f32>> {
    let mut out = vec![0.0_f32; n_channels_t * hw];
    for co in 0..n_channels_t {
        for ci in 0..n_channels_s {
            let weight =
                *w_align
                    .get(co * n_channels_s + ci)
                    .ok_or_else(|| DistillError::Internal {
                        msg: "align_student: w_align index out of bounds".into(),
                    })?;
            let in_base = ci * hw;
            let out_base = co * hw;
            for p in 0..hw {
                let sv = *student
                    .get(in_base + p)
                    .ok_or_else(|| DistillError::Internal {
                        msg: "align_student: student index out of bounds".into(),
                    })?;
                let dst = out
                    .get_mut(out_base + p)
                    .ok_or_else(|| DistillError::Internal {
                        msg: "align_student: output index out of bounds".into(),
                    })?;
                *dst += weight * sv;
            }
        }
    }
    Ok(out)
}

/// Single 3×3 same-pad (zero-pad) convolution `[C × H × W] → [C × H × W]`.
///
/// `weights` is `[C_out × C_in × 3 × 3]` row-major (here `C_out == C_in == channels`).
/// Output pixel `(co, y, x)` sums over input channels and the 3×3 neighbourhood with
/// zero padding outside the spatial bounds. All indexing is `.get()`-guarded.
fn conv3x3_same(
    input: &[f32],
    weights: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> DistillResult<Vec<f32>> {
    let hw = h * w;
    let expected_in = channels * hw;
    if input.len() != expected_in {
        return Err(DistillError::DimensionMismatch {
            expected: expected_in,
            got: input.len(),
        });
    }
    let expected_w = channels * channels * 9;
    if weights.len() != expected_w {
        return Err(DistillError::DimensionMismatch {
            expected: expected_w,
            got: weights.len(),
        });
    }
    let mut out = vec![0.0_f32; channels * hw];
    for co in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for ci in 0..channels {
                    let in_ch_base = ci * hw;
                    let w_base = (co * channels + ci) * 9;
                    for ky in 0..3usize {
                        // Source row with zero padding: kernel center at offset 1.
                        let sy = y as isize + ky as isize - 1;
                        if sy < 0 || sy >= h as isize {
                            continue;
                        }
                        let sy = sy as usize;
                        for kx in 0..3usize {
                            let sx = x as isize + kx as isize - 1;
                            if sx < 0 || sx >= w as isize {
                                continue;
                            }
                            let sx = sx as usize;
                            let wv = *weights.get(w_base + ky * 3 + kx).ok_or_else(|| {
                                DistillError::Internal {
                                    msg: "conv3x3_same: weight index out of bounds".into(),
                                }
                            })?;
                            let iv = *input.get(in_ch_base + sy * w + sx).ok_or_else(|| {
                                DistillError::Internal {
                                    msg: "conv3x3_same: input index out of bounds".into(),
                                }
                            })?;
                            acc += wv * iv;
                        }
                    }
                }
                let dst =
                    out.get_mut(co * hw + y * w + x)
                        .ok_or_else(|| DistillError::Internal {
                            msg: "conv3x3_same: output index out of bounds".into(),
                        })?;
                *dst = acc;
            }
        }
    }
    Ok(out)
}

/// Run the generator on a masked feature map: `conv3×3(w1) → ReLU → conv3×3(w2)`.
///
/// `masked` must be `[C_t × H × W]`. Returns the reconstruction `[C_t × H × W]`.
pub fn forward_generator(
    generator: &MgdGenerator,
    masked: &[f32],
    h: usize,
    w: usize,
) -> DistillResult<Vec<f32>> {
    if masked.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let c = generator.n_channels_t;
    let mut hidden = conv3x3_same(masked, &generator.w1, c, h, w)?;
    for v in hidden.iter_mut() {
        if *v < 0.0 {
            *v = 0.0; // ReLU
        }
    }
    let recon = conv3x3_same(&hidden, &generator.w2, c, h, w)?;
    Ok(recon)
}

/// Mean-squared error between two equal-length slices.
fn mse(a: &[f32], b: &[f32]) -> DistillResult<f32> {
    if a.len() != b.len() {
        return Err(DistillError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    Ok(sum / a.len() as f32)
}

/// Compute the Masked Generative Distillation loss.
///
/// # Arguments
/// * `student_feat` — `[C_s · H · W]` row-major (channel-major, then `H`, then `W`).
/// * `teacher_feat` — `[C_t · H · W]` row-major.
/// * `gen` — the generator (with optional 1×1 alignment).
/// * `cfg` — structural + hyper-parameter configuration.
/// * `rng` — RNG used to draw the spatial mask.
///
/// Returns `alpha_mgd · MSE(recon, teacher)` averaged over all `C_t·H·W` elements.
pub fn mgd_loss(
    student_feat: &[f32],
    teacher_feat: &[f32],
    generator: &MgdGenerator,
    cfg: &MgdConfig,
    rng: &mut LcgRng,
) -> DistillResult<f32> {
    cfg.validate()?;
    if student_feat.is_empty() || teacher_feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if generator.n_channels_s != cfg.n_channels_s || generator.n_channels_t != cfg.n_channels_t {
        return Err(DistillError::InvalidConfig {
            msg: "mgd_loss: generator channel counts must match config".into(),
        });
    }
    let hw = cfg.height * cfg.width;
    let expected_s = cfg.n_channels_s * hw;
    let expected_t = cfg.n_channels_t * hw;
    if student_feat.len() != expected_s {
        return Err(DistillError::DimensionMismatch {
            expected: expected_s,
            got: student_feat.len(),
        });
    }
    if teacher_feat.len() != expected_t {
        return Err(DistillError::DimensionMismatch {
            expected: expected_t,
            got: teacher_feat.len(),
        });
    }

    // 1. Align student → C_t channels via 1×1 conv (skip when C_s == C_t and no weights).
    let aligned = match &generator.w_align {
        Some(wa) => align_student(student_feat, wa, cfg.n_channels_s, cfg.n_channels_t, hw)?,
        None => {
            if cfg.n_channels_s != cfg.n_channels_t {
                return Err(DistillError::InvalidConfig {
                    msg: "mgd_loss: C_s != C_t requires alignment weights in the generator".into(),
                });
            }
            student_feat.to_vec()
        }
    };

    // 2. Spatial binary mask shared across channels.
    let mask = generate_mask(cfg.height, cfg.width, cfg.mask_ratio, rng);

    // 3. masked = aligned ⊙ mask (broadcast over channels).
    let mut masked = vec![0.0_f32; aligned.len()];
    for co in 0..cfg.n_channels_t {
        let base = co * hw;
        for p in 0..hw {
            let mv = *mask.get(p).ok_or_else(|| DistillError::Internal {
                msg: "mgd_loss: mask index out of bounds".into(),
            })?;
            let av = *aligned
                .get(base + p)
                .ok_or_else(|| DistillError::Internal {
                    msg: "mgd_loss: aligned index out of bounds".into(),
                })?;
            let dst = masked
                .get_mut(base + p)
                .ok_or_else(|| DistillError::Internal {
                    msg: "mgd_loss: masked index out of bounds".into(),
                })?;
            *dst = av * mv;
        }
    }

    // 4. Reconstruct via the generator.
    let recon = forward_generator(generator, &masked, cfg.height, cfg.width)?;
    if recon.len() != teacher_feat.len() {
        return Err(DistillError::DimensionMismatch {
            expected: teacher_feat.len(),
            got: recon.len(),
        });
    }

    // 5. Weighted MSE against the full teacher feature.
    let loss = mse(&recon, teacher_feat)?;
    Ok(cfg.alpha_mgd * loss)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    /// Identity-conv weights: each output channel copies the same input channel's center tap.
    fn identity_conv(channels: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; channels * channels * 9];
        for co in 0..channels {
            // center tap (ky=1, kx=1 → offset 4) of the (co, co) filter = 1.
            let idx = (co * channels + co) * 9 + 4;
            w[idx] = 1.0;
        }
        w
    }

    fn base_cfg(cs: usize, ct: usize, h: usize, w: usize, ratio: f32) -> MgdConfig {
        MgdConfig {
            n_channels_s: cs,
            n_channels_t: ct,
            height: h,
            width: w,
            mask_ratio: ratio,
            alpha_mgd: 1.0,
            seed: 7,
        }
    }

    // ── 1. mask values are strictly binary ──────────────────────────────────
    #[test]
    fn mask_values_binary() {
        let mut rng = make_rng();
        let mask = generate_mask(8, 8, 0.5, &mut rng);
        for &m in &mask {
            assert!(m == 0.0 || m == 1.0, "mask value not binary: {m}");
        }
    }

    // ── 2. masked fraction ≈ mask_ratio over large H·W ──────────────────────
    #[test]
    fn mask_fraction_approx_ratio() {
        let mut rng = make_rng();
        let ratio = 0.65_f32;
        let h = 100usize;
        let w = 100usize;
        let mask = generate_mask(h, w, ratio, &mut rng);
        let masked = mask.iter().filter(|&&m| m == 0.0).count();
        let frac = masked as f32 / (h * w) as f32;
        assert!(
            (frac - ratio).abs() < 0.03,
            "masked fraction {frac} far from ratio {ratio}"
        );
    }

    // ── 3. deterministic given seed ─────────────────────────────────────────
    #[test]
    fn mask_deterministic_seed() {
        let mut a = LcgRng::new(123);
        let mut b = LcgRng::new(123);
        let ma = generate_mask(16, 16, 0.6, &mut a);
        let mb = generate_mask(16, 16, 0.6, &mut b);
        assert_eq!(ma, mb, "same seed must yield identical masks");
    }

    // ── 4. mask_ratio = 0 → all-ones mask ───────────────────────────────────
    #[test]
    fn mask_ratio_zero_all_ones() {
        let mut rng = make_rng();
        let mask = generate_mask(20, 20, 0.0, &mut rng);
        assert!(mask.iter().all(|&m| m == 1.0), "ratio 0 must keep all");
    }

    // ── 5. mask_ratio = 1 → all-zeros mask ──────────────────────────────────
    #[test]
    fn mask_ratio_one_all_zeros() {
        let mut rng = make_rng();
        let mask = generate_mask(20, 20, 1.0, &mut rng);
        assert!(mask.iter().all(|&m| m == 0.0), "ratio 1 must mask all");
    }

    // ── 6. loss is non-negative ─────────────────────────────────────────────
    #[test]
    fn loss_nonneg() {
        let mut rng = make_rng();
        let cfg = base_cfg(4, 4, 5, 6, 0.65);
        let generator = MgdGenerator::new(4, 4, &mut rng).unwrap();
        let student: Vec<f32> = (0..4 * 5 * 6).map(|i| (i as f32) * 0.01).collect();
        let teacher: Vec<f32> = (0..4 * 5 * 6).map(|i| (i as f32) * 0.02).collect();
        let loss = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng).unwrap();
        assert!(loss >= 0.0 && loss.is_finite(), "loss={loss}");
    }

    // ── 7. mask_ratio = 0 → loss == MSE(gen(student), teacher) ───────────────
    #[test]
    fn ratio_zero_equals_full_recon_mse() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 3, 4, 4, 0.0);
        let mut wrng = LcgRng::new(99);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.03 - 0.5).collect();
        let teacher: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.01).collect();

        // mask_ratio=0 → mask all ones → masked == aligned == student (C_s==C_t, no align).
        let recon = forward_generator(&generator, &student, 4, 4).unwrap();
        let expected = mse(&recon, &teacher).unwrap();

        let loss = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng).unwrap();
        assert!(
            (loss - expected).abs() < 1e-5,
            "ratio0 loss {loss} != MSE(gen(student),teacher) {expected}"
        );
    }

    // ── 7b. mask_ratio = 1 → student fully zeroed → recon == gen(zeros) ──────
    #[test]
    fn ratio_one_zeroes_student() {
        let mut rng = make_rng();
        let cfg = base_cfg(2, 2, 4, 4, 1.0);
        let mut wrng = LcgRng::new(5);
        let generator = MgdGenerator::new(2, 2, &mut wrng).unwrap();
        let student: Vec<f32> = (0..2 * 4 * 4).map(|i| (i as f32) + 1.0).collect();
        let teacher: Vec<f32> = vec![0.3_f32; 2 * 4 * 4];

        let zeros = vec![0.0_f32; 2 * 4 * 4];
        let recon_zero = forward_generator(&generator, &zeros, 4, 4).unwrap();
        let expected = mse(&recon_zero, &teacher).unwrap();

        let loss = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng).unwrap();
        assert!(
            (loss - expected).abs() < 1e-5,
            "ratio1 loss {loss} != MSE(gen(0),teacher) {expected}"
        );
    }

    // ── 8. identity generator, student==teacher, ratio 0 → loss 0 ────────────
    #[test]
    fn identity_gen_student_eq_teacher_zero_loss() {
        let mut rng = make_rng();
        let cfg = base_cfg(2, 2, 5, 5, 0.0);
        // w1 = identity, w2 = identity ⇒ generator is identity (ReLU only clips negatives).
        let id = identity_conv(2);
        let generator = MgdGenerator::from_weights(None, id.clone(), id, 2, 2).unwrap();
        // Use non-negative features so ReLU is a no-op and identity holds exactly.
        let feat: Vec<f32> = (0..2 * 5 * 5).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let loss = mgd_loss(&feat, &feat, &generator, &cfg, &mut rng).unwrap();
        assert!(
            loss.abs() < 1e-5,
            "identity gen, s==t, ratio0 → 0, got {loss}"
        );
    }

    // ── 9. C_s != C_t alignment path → recon has C_t channels & shape ────────
    #[test]
    fn align_path_recon_shape() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 5, 4, 4, 0.5);
        let mut wrng = LcgRng::new(17);
        let generator = MgdGenerator::new(3, 5, &mut wrng).unwrap();
        assert!(
            generator.w_align.is_some(),
            "C_s!=C_t must create align weights"
        );
        let student: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.05).collect();
        let teacher: Vec<f32> = (0..5 * 4 * 4).map(|i| (i as f32) * 0.02).collect();
        // The internal recon length must equal teacher length (C_t·H·W) or mgd_loss errs.
        let loss = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(loss.is_ok(), "align path should succeed: {:?}", loss.err());
        assert!(loss.unwrap().is_finite());
    }

    // ── 10. recon shape == teacher shape (forward_generator directly) ────────
    #[test]
    fn recon_shape_matches_teacher() {
        let mut wrng = LcgRng::new(3);
        let generator = MgdGenerator::new(4, 4, &mut wrng).unwrap();
        let masked: Vec<f32> = vec![0.1_f32; 4 * 6 * 7];
        let recon = forward_generator(&generator, &masked, 6, 7).unwrap();
        assert_eq!(recon.len(), 4 * 6 * 7, "recon shape must be C_t·H·W");
    }

    // ── 11. alpha scales loss linearly ──────────────────────────────────────
    #[test]
    fn alpha_scales_loss() {
        let mut cfg1 = base_cfg(3, 3, 5, 5, 0.5);
        cfg1.alpha_mgd = 1.0;
        let mut cfg2 = cfg1.clone();
        cfg2.alpha_mgd = 3.0;
        let mut wrng = LcgRng::new(21);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student: Vec<f32> = (0..3 * 5 * 5).map(|i| (i as f32) * 0.04).collect();
        let teacher: Vec<f32> = (0..3 * 5 * 5).map(|i| (i as f32) * 0.03).collect();
        // Use independent RNG copies seeded identically so the mask is the same.
        let mut r1 = LcgRng::new(1000);
        let mut r2 = LcgRng::new(1000);
        let l1 = mgd_loss(&student, &teacher, &generator, &cfg1, &mut r1).unwrap();
        let l2 = mgd_loss(&student, &teacher, &generator, &cfg2, &mut r2).unwrap();
        assert!(
            (l2 - 3.0 * l1).abs() < 1e-4,
            "alpha must scale: l1={l1} l2={l2}"
        );
    }

    // ── 12. deterministic generator forward ─────────────────────────────────
    #[test]
    fn deterministic_forward() {
        let mut wrng = LcgRng::new(8);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let masked: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.1).collect();
        let a = forward_generator(&generator, &masked, 4, 4).unwrap();
        let b = forward_generator(&generator, &masked, 4, 4).unwrap();
        assert_eq!(a, b, "generator forward must be deterministic");
    }

    // ── 13. full pipeline deterministic given seeded rng ────────────────────
    #[test]
    fn loss_deterministic_with_seed() {
        let cfg = base_cfg(3, 3, 6, 6, 0.65);
        let mut wrng = LcgRng::new(50);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student: Vec<f32> = (0..3 * 6 * 6).map(|i| (i as f32) * 0.02).collect();
        let teacher: Vec<f32> = (0..3 * 6 * 6).map(|i| (i as f32) * 0.025).collect();
        let mut r1 = LcgRng::new(77);
        let mut r2 = LcgRng::new(77);
        let l1 = mgd_loss(&student, &teacher, &generator, &cfg, &mut r1).unwrap();
        let l2 = mgd_loss(&student, &teacher, &generator, &cfg, &mut r2).unwrap();
        assert!((l1 - l2).abs() < 1e-9, "same seed → same loss: {l1} {l2}");
    }

    // ── 14. conv3x3 same-pad preserves spatial shape ────────────────────────
    #[test]
    fn conv_same_pad_shape() {
        let id = identity_conv(2);
        let input: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
        let out = conv3x3_same(&input, &id, 2, 3, 4).unwrap();
        assert_eq!(out.len(), 2 * 3 * 4);
        // Identity conv must reproduce the input exactly (zero-pad doesn't matter for center tap).
        assert_eq!(out, input, "identity conv must reproduce input");
    }

    // ── 15. err: student/teacher C/H/W mismatch ─────────────────────────────
    #[test]
    fn err_student_size_mismatch() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 3, 4, 4, 0.5);
        let mut wrng = LcgRng::new(2);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student = vec![0.0_f32; 3 * 4 * 4 - 1]; // wrong size
        let teacher = vec![0.0_f32; 3 * 4 * 4];
        let r = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 16. err: teacher size mismatch ──────────────────────────────────────
    #[test]
    fn err_teacher_size_mismatch() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 3, 4, 4, 0.5);
        let mut wrng = LcgRng::new(2);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student = vec![0.0_f32; 3 * 4 * 4];
        let teacher = vec![0.0_f32; 3 * 4 * 4 + 5]; // wrong size
        let r = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 17. err: mask_ratio out of [0,1] ────────────────────────────────────
    #[test]
    fn err_mask_ratio_out_of_range() {
        let mut rng = make_rng();
        let mut cfg = base_cfg(3, 3, 4, 4, 1.5);
        let mut wrng = LcgRng::new(2);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student = vec![0.0_f32; 3 * 4 * 4];
        let teacher = vec![0.0_f32; 3 * 4 * 4];
        let r = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
        // negative ratio too
        cfg.mask_ratio = -0.1;
        let r2 = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r2, Err(DistillError::InvalidConfig { .. })));
    }

    // ── 18. err: empty input ────────────────────────────────────────────────
    #[test]
    fn err_empty_input() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 3, 4, 4, 0.5);
        let mut wrng = LcgRng::new(2);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let teacher = vec![0.0_f32; 3 * 4 * 4];
        let r = mgd_loss(&[], &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::EmptyInput)));
    }

    // ── 19. err: zero dims in config ────────────────────────────────────────
    #[test]
    fn err_zero_dims() {
        let mut rng = make_rng();
        let cfg = base_cfg(3, 3, 0, 4, 0.5); // height = 0
        let mut wrng = LcgRng::new(2);
        let generator = MgdGenerator::new(3, 3, &mut wrng).unwrap();
        let student = vec![0.0_f32; 12];
        let teacher = vec![0.0_f32; 12];
        let r = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }

    // ── 20. err: generator weight-shape mismatch ────────────────────────────
    #[test]
    fn err_generator_weight_shape() {
        // w1 wrong length for C_t=3 (should be 3*3*9 = 81).
        let bad_w1 = vec![0.0_f32; 10];
        let good_w2 = vec![0.0_f32; 3 * 3 * 9];
        let r = MgdGenerator::from_weights(None, bad_w1, good_w2, 3, 3);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 21. err: align weight-shape mismatch in from_weights ────────────────
    #[test]
    fn err_align_weight_shape() {
        let good = vec![0.0_f32; 3 * 3 * 9];
        // C_s=2, C_t=3 → align must be 3*2=6; supply wrong length.
        let bad_align = Some(vec![0.0_f32; 5]);
        let r = MgdGenerator::from_weights(bad_align, good.clone(), good, 2, 3);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 22. err: C_s != C_t without align weights in mgd_loss ───────────────
    #[test]
    fn err_no_align_for_mismatched_channels() {
        let mut rng = make_rng();
        let cfg = base_cfg(2, 3, 4, 4, 0.5);
        // Construct a generator claiming C_s=2,C_t=3 but with NO align weights.
        let conv = vec![0.0_f32; 3 * 3 * 9];
        let generator = MgdGenerator::from_weights(None, conv.clone(), conv, 2, 3).unwrap();
        let student = vec![0.0_f32; 2 * 4 * 4];
        let teacher = vec![0.0_f32; 3 * 4 * 4];
        let r = mgd_loss(&student, &teacher, &generator, &cfg, &mut rng);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }
}
