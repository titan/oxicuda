//! Long-context RoPE frequency scaling: Linear Position-Interpolation,
//! NTK-aware scaling, and YaRN.
//!
//! This module generalises the base [`RotaryEmbedding`](crate::layer::RotaryEmbedding)
//! by rescaling the per-dimension inverse frequencies so that a model trained
//! with a short context window can attend over much longer sequences without
//! retraining.
//!
//! # References
//!
//! - Peng, Quesnelle, Fan & Shippole, *"YaRN: Efficient Context Window
//!   Extension of Large Language Models"* (2023), arXiv:2309.00071.
//! - Chen, Wong, Chen & Tian, *"Extending Context Window of Large Language
//!   Models via Positional Interpolation"* (2023), arXiv:2306.15595.
//! - bloc97, *"NTK-Aware Scaled RoPE"* (2023), reddit/LocalLLaMA writeup.
//!
//! # Convention
//!
//! Frequencies, table layout and rotation match
//! [`RotaryEmbedding`](crate::layer::RotaryEmbedding) **exactly**:
//!
//! - base frequency `θ_d = base^(−2d/dim)` for `d = 0..dim/2`,
//! - `cos`/`sin` tables are laid out `[n_positions × dim/2]` row-major with
//!   `angle = pos · θ_d`,
//! - rotation acts on interleaved pairs `(x[2d], x[2d+1])`:
//!
//! ```text
//! x_out[2d]   = x[2d]·cos − x[2d+1]·sin
//! x_out[2d+1] = x[2d]·sin + x[2d+1]·cos
//! ```
//!
//! With [`RopeScalingKind::None`] the produced `inv_freqs()` and `cos_sin`
//! tables are bit-for-bit identical to `RotaryEmbedding`.

use crate::error::{LmError, LmResult};

use core::f32::consts::PI;

// ─── RopeScalingKind ───────────────────────────────────────────────────────────

/// Which long-context scaling scheme to apply to the RoPE frequencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScalingKind {
    /// No scaling — identical to base [`RotaryEmbedding`](crate::layer::RotaryEmbedding).
    None,
    /// Linear Position-Interpolation (Chen et al. 2023): positions `m → m/scale`,
    /// equivalently every inverse frequency is divided by `scale`.
    Linear {
        /// Context-extension factor (`new_max / original_max`), must be > 0.
        scale: f32,
    },
    /// NTK-aware scaling (bloc97 2023): the base is raised so high-frequency
    /// dimensions are interpolated less than low-frequency ones.
    NtkAware {
        /// Context-extension factor, must be > 0.
        scale: f32,
    },
    /// YaRN (Peng et al. 2023): a per-dimension ramp blends NTK-by-parts
    /// interpolation between the original and the linearly-interpolated
    /// frequency, plus an attention temperature `mscale`.
    Yarn {
        /// Context-extension factor, must be > 0.
        scale: f32,
        /// The original training context length, must be > 0.
        original_max_pos: usize,
        /// High-frequency boundary (rotations); dims with more rotations than
        /// this keep their original frequency. Must be > `beta_slow`.
        beta_fast: f32,
        /// Low-frequency boundary (rotations); dims with fewer rotations than
        /// this are fully interpolated.
        beta_slow: f32,
    },
}

// ─── RopeScaling ───────────────────────────────────────────────────────────────

/// Precomputed scaled RoPE inverse frequencies and attention temperature.
///
/// Construct with [`RopeScaling::new`]; query [`RopeScaling::inv_freqs`] and
/// [`RopeScaling::mscale`], build tables with [`RopeScaling::cos_sin`], or
/// rotate a buffer in place with [`RopeScaling::apply`].
#[derive(Debug, Clone)]
pub struct RopeScaling {
    /// Rotary dimension (must be even and > 0).
    dim: usize,
    /// Base frequency (typically 10 000 / 500 000).
    base: f32,
    /// Which scaling scheme produced [`Self::inv_freqs`].
    kind: RopeScalingKind,
    /// Scaled inverse frequencies `θ'_d`, length `dim/2`.
    inv_freqs: Vec<f32>,
    /// Attention temperature factor folded into the cos/sin amplitude.
    mscale: f32,
}

impl RopeScaling {
    /// Build a scaled RoPE configuration.
    ///
    /// `dim` is the rotary (head) dimension and must be even and positive;
    /// `base` must be positive. Each scaling kind additionally validates its
    /// own parameters.
    pub fn new(dim: usize, base: f32, kind: RopeScalingKind) -> LmResult<Self> {
        if dim == 0 || dim % 2 != 0 {
            return Err(LmError::InvalidConfig {
                msg: format!("RopeScaling: dim={dim} must be even and > 0"),
            });
        }
        if base <= 0.0 {
            return Err(LmError::InvalidConfig {
                msg: "RopeScaling: base must be > 0".into(),
            });
        }

        let half = dim / 2;
        // Base frequencies θ_d = base^(−2d/dim), matching `RotaryEmbedding`.
        let base_inv_freqs: Vec<f32> = (0..half)
            .map(|d| base.powf(-((2 * d) as f32) / dim as f32))
            .collect();

        let (inv_freqs, mscale) = match kind {
            RopeScalingKind::None => (base_inv_freqs, 1.0_f32),

            RopeScalingKind::Linear { scale } => {
                Self::check_scale(scale)?;
                // Position-interpolation: m → m/scale ⇔ θ_d → θ_d/scale.
                let f = base_inv_freqs.iter().map(|&w| w / scale).collect();
                (f, 1.0_f32)
            }

            RopeScalingKind::NtkAware { scale } => {
                Self::check_scale(scale)?;
                // base' = base · scale^(dim/(dim−2)); recompute θ_d with base'.
                let exponent = dim as f32 / (dim as f32 - 2.0);
                let new_base = base * scale.powf(exponent);
                let f = (0..half)
                    .map(|d| new_base.powf(-((2 * d) as f32) / dim as f32))
                    .collect();
                (f, 1.0_f32)
            }

            RopeScalingKind::Yarn {
                scale,
                original_max_pos,
                beta_fast,
                beta_slow,
            } => {
                Self::check_scale(scale)?;
                if original_max_pos == 0 {
                    return Err(LmError::InvalidConfig {
                        msg: "RopeScaling::Yarn: original_max_pos must be > 0".into(),
                    });
                }
                if beta_fast <= beta_slow {
                    return Err(LmError::InvalidConfig {
                        msg: format!(
                            "RopeScaling::Yarn: beta_fast={beta_fast} must be > beta_slow={beta_slow}"
                        ),
                    });
                }
                let orig = original_max_pos as f32;
                let denom = beta_fast - beta_slow;
                let f = base_inv_freqs
                    .iter()
                    .map(|&w| {
                        // Per-dim wavelength λ_d = 2π/θ_d and rotation count over
                        // the original context: r_d = original_max_pos / λ_d.
                        let wavelength = 2.0 * PI / w;
                        let rotations = orig / wavelength;
                        // Ramp γ_d ∈ [0, 1].
                        let ramp = ((rotations - beta_slow) / denom).clamp(0.0, 1.0);
                        // θ'_d = (1−γ)·(θ_d/scale) + γ·θ_d.
                        (1.0 - ramp) * (w / scale) + ramp * w
                    })
                    .collect();
                // Attention temperature mscale = 0.1·ln(scale) + 1.
                let mscale = 0.1 * scale.ln() + 1.0;
                (f, mscale)
            }
        };

        Ok(Self {
            dim,
            base,
            kind,
            inv_freqs,
            mscale,
        })
    }

    /// Validate a context-extension factor (`> 0`).
    fn check_scale(scale: f32) -> LmResult<()> {
        if scale <= 0.0 {
            return Err(LmError::InvalidConfig {
                msg: "RopeScaling: scale must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Rotary dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Base frequency this configuration was built with.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Which scaling scheme is in effect.
    pub fn kind(&self) -> RopeScalingKind {
        self.kind
    }

    /// Scaled inverse frequencies `θ'_d`, length `dim/2`.
    pub fn inv_freqs(&self) -> &[f32] {
        &self.inv_freqs
    }

    /// Attention temperature factor (`1.0` for every non-YaRN kind).
    pub fn mscale(&self) -> f32 {
        self.mscale
    }

    /// Build `cos`/`sin` tables of shape `[n_positions × dim/2]` (row-major),
    /// using the scaled frequencies. The YaRN attention temperature
    /// [`Self::mscale`] is folded into the amplitude so that the tables can be
    /// consumed by the same apply path as [`RotaryEmbedding`](crate::layer::RotaryEmbedding)
    /// (`mscale == 1.0` for the non-YaRN kinds leaves the tables identical).
    pub fn cos_sin(&self, n_positions: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.dim / 2;
        let n = n_positions * half;
        let mut cos_table = Vec::with_capacity(n);
        let mut sin_table = Vec::with_capacity(n);
        for pos in 0..n_positions {
            for &freq in &self.inv_freqs {
                let angle = pos as f32 * freq;
                cos_table.push(self.mscale * angle.cos());
                sin_table.push(self.mscale * angle.sin());
            }
        }
        (cos_table, sin_table)
    }

    /// Apply scaled RoPE in place to `x`, shaped `[n_tokens × n_heads × dim]`
    /// row-major, where `positions[t]` is the absolute position of token `t`.
    ///
    /// Rotation acts on interleaved pairs and folds in [`Self::mscale`], exactly
    /// matching [`RotaryEmbedding::apply`](crate::layer::RotaryEmbedding::apply)
    /// when `mscale == 1.0`.
    pub fn apply(&self, x: &mut [f32], n_heads: usize, positions: &[usize]) -> LmResult<()> {
        if positions.is_empty() {
            return Err(LmError::EmptyInput {
                context: "RopeScaling::apply positions",
            });
        }
        if n_heads == 0 {
            return Err(LmError::InvalidConfig {
                msg: "RopeScaling::apply: n_heads must be > 0".into(),
            });
        }
        let n_tokens = positions.len();
        let expected = n_tokens * n_heads * self.dim;
        if x.len() != expected {
            return Err(LmError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let half = self.dim / 2;
        for (t, &abs_pos) in positions.iter().enumerate() {
            for h in 0..n_heads {
                let base = (t * n_heads + h) * self.dim;
                for i in 0..half {
                    let freq = *self.inv_freqs.get(i).ok_or_else(|| LmError::Internal {
                        msg: "RopeScaling::apply: inv_freqs index out of range".into(),
                    })?;
                    let angle = abs_pos as f32 * freq;
                    let cos = self.mscale * angle.cos();
                    let sin = self.mscale * angle.sin();
                    let i0 = base + 2 * i;
                    let i1 = base + 2 * i + 1;
                    let x0 = *x.get(i0).ok_or_else(|| LmError::Internal {
                        msg: "RopeScaling::apply: x index out of range".into(),
                    })?;
                    let x1 = *x.get(i1).ok_or_else(|| LmError::Internal {
                        msg: "RopeScaling::apply: x index out of range".into(),
                    })?;
                    *x.get_mut(i0).ok_or_else(|| LmError::Internal {
                        msg: "RopeScaling::apply: x index out of range".into(),
                    })? = x0 * cos - x1 * sin;
                    *x.get_mut(i1).ok_or_else(|| LmError::Internal {
                        msg: "RopeScaling::apply: x index out of range".into(),
                    })? = x0 * sin + x1 * cos;
                }
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::embedding::RotaryEmbedding;

    const DIM: usize = 8;
    const BASE: f32 = 10_000.0;

    fn reference_inv_freqs(dim: usize, base: f32) -> Vec<f32> {
        (0..dim / 2)
            .map(|d| base.powf(-((2 * d) as f32) / dim as f32))
            .collect()
    }

    #[test]
    fn none_reproduces_rotary_embedding_inv_freqs_exactly() {
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None)
            .expect("None with even dim and positive base is valid");
        // Reproduce the exact embedding.rs frequency formula element-by-element.
        let want = reference_inv_freqs(DIM, BASE);
        assert_eq!(rs.inv_freqs().len(), want.len());
        for (got, exp) in rs.inv_freqs().iter().zip(want.iter()) {
            assert_eq!(
                *got, *exp,
                "inv_freq must match RotaryEmbedding bit-for-bit"
            );
        }
    }

    #[test]
    fn none_reproduces_rotary_embedding_cos_sin_exactly() {
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("None config is valid");
        let rope = RotaryEmbedding::new(DIM, 16, BASE).expect("RotaryEmbedding is valid");
        let (cos, sin) = rs.cos_sin(16);
        let half = DIM / 2;
        for pos in 0..16 {
            for i in 0..half {
                let idx = pos * half + i;
                assert_eq!(cos[idx], rope.cos_at(pos, i), "cos at ({pos},{i})");
                assert_eq!(sin[idx], rope.sin_at(pos, i), "sin at ({pos},{i})");
            }
        }
    }

    #[test]
    fn linear_scale_one_equals_none() {
        let none = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let lin = RopeScaling::new(DIM, BASE, RopeScalingKind::Linear { scale: 1.0 })
            .expect("linear scale=1");
        for (a, b) in none.inv_freqs().iter().zip(lin.inv_freqs().iter()) {
            assert_eq!(*a, *b, "Linear(1.0) must equal None");
        }
        assert_eq!(lin.mscale(), 1.0);
    }

    #[test]
    fn ntk_scale_one_equals_none() {
        let none = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let ntk = RopeScaling::new(DIM, BASE, RopeScalingKind::NtkAware { scale: 1.0 })
            .expect("ntk scale=1");
        // base' = base · 1^k = base ⇒ identical frequencies (within fp rounding).
        for (a, b) in none.inv_freqs().iter().zip(ntk.inv_freqs().iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "NtkAware(1.0) must equal None: {a} vs {b}"
            );
        }
    }

    #[test]
    fn linear_scale_shifts_position() {
        // Linear scale s: cos/sin at position p must equal the base RoPE at p/s.
        let scale = 4.0_f32;
        let lin = RopeScaling::new(DIM, BASE, RopeScalingKind::Linear { scale }).expect("linear");
        let base_inv = reference_inv_freqs(DIM, BASE);
        let half = DIM / 2;
        let (cos, sin) = lin.cos_sin(16);
        let p = 12usize;
        for (i, &w) in base_inv.iter().enumerate() {
            // Scaled angle at position p with θ/s == base angle at p/s with θ.
            let want_angle = (p as f32 / scale) * w;
            let idx = p * half + i;
            assert!((cos[idx] - want_angle.cos()).abs() < 1e-5);
            assert!((sin[idx] - want_angle.sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn ntk_base_matches_hand_computation() {
        let scale = 8.0_f32;
        let ntk = RopeScaling::new(DIM, BASE, RopeScalingKind::NtkAware { scale }).expect("ntk");
        let exponent = DIM as f32 / (DIM as f32 - 2.0);
        let new_base = BASE * scale.powf(exponent);
        let want: Vec<f32> = (0..DIM / 2)
            .map(|d| new_base.powf(-((2 * d) as f32) / DIM as f32))
            .collect();
        for (got, exp) in ntk.inv_freqs().iter().zip(want.iter()) {
            assert!((got - exp).abs() < 1e-9, "{got} vs {exp}");
        }
    }

    #[test]
    fn yarn_ramp_within_unit_interval() {
        // The blended frequency must lie between the fully-interpolated value
        // (θ/scale) and the original (θ), which is exactly the γ ∈ [0,1] range.
        let scale = 16.0_f32;
        let yarn = RopeScaling::new(
            DIM,
            BASE,
            RopeScalingKind::Yarn {
                scale,
                original_max_pos: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        )
        .expect("yarn");
        let base_inv = reference_inv_freqs(DIM, BASE);
        for (got, &w) in yarn.inv_freqs().iter().zip(base_inv.iter()) {
            let lo = (w / scale).min(w);
            let hi = (w / scale).max(w);
            assert!(
                *got >= lo - 1e-6 && *got <= hi + 1e-6,
                "{got} not in [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn yarn_scale_one_is_near_identity() {
        let none = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let yarn = RopeScaling::new(
            DIM,
            BASE,
            RopeScalingKind::Yarn {
                scale: 1.0,
                original_max_pos: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        )
        .expect("yarn scale=1");
        // scale=1 ⇒ θ/scale == θ, so blending is the identity regardless of γ.
        for (a, b) in none.inv_freqs().iter().zip(yarn.inv_freqs().iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // mscale = 0.1·ln(1) + 1 = 1.
        assert!((yarn.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mscale_formula() {
        let scale = 32.0_f32;
        let yarn = RopeScaling::new(
            DIM,
            BASE,
            RopeScalingKind::Yarn {
                scale,
                original_max_pos: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        )
        .expect("yarn");
        let want = 0.1 * scale.ln() + 1.0;
        assert!(
            (yarn.mscale() - want).abs() < 1e-6,
            "{} vs {want}",
            yarn.mscale()
        );
    }

    #[test]
    fn non_yarn_mscale_is_one() {
        for kind in [
            RopeScalingKind::None,
            RopeScalingKind::Linear { scale: 4.0 },
            RopeScalingKind::NtkAware { scale: 4.0 },
        ] {
            let rs = RopeScaling::new(DIM, BASE, kind).expect("valid kind");
            assert_eq!(rs.mscale(), 1.0, "non-YaRN mscale must be 1.0");
        }
    }

    #[test]
    fn large_position_no_nan_inf() {
        for kind in [
            RopeScalingKind::None,
            RopeScalingKind::Linear { scale: 8.0 },
            RopeScalingKind::NtkAware { scale: 8.0 },
            RopeScalingKind::Yarn {
                scale: 8.0,
                original_max_pos: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        ] {
            let rs = RopeScaling::new(DIM, BASE, kind).expect("valid");
            let (cos, sin) = rs.cos_sin(8193);
            assert!(cos.iter().all(|v| v.is_finite()), "cos finite for {kind:?}");
            assert!(sin.iter().all(|v| v.is_finite()), "sin finite for {kind:?}");
        }
    }

    #[test]
    fn inv_freqs_length_is_half_dim() {
        for d in [2usize, 4, 8, 16, 64, 128] {
            let rs = RopeScaling::new(d, BASE, RopeScalingKind::None).expect("valid");
            assert_eq!(rs.inv_freqs().len(), d / 2);
        }
    }

    #[test]
    fn freqs_monotone_decreasing() {
        for kind in [
            RopeScalingKind::None,
            RopeScalingKind::Linear { scale: 4.0 },
            RopeScalingKind::NtkAware { scale: 4.0 },
            RopeScalingKind::Yarn {
                scale: 4.0,
                original_max_pos: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        ] {
            let rs = RopeScaling::new(64, BASE, kind).expect("valid");
            let f = rs.inv_freqs();
            for w in f.windows(2) {
                assert!(w[0] >= w[1], "freqs must be non-increasing for {kind:?}");
            }
        }
    }

    #[test]
    fn apply_then_inverse_recovers_input() {
        let rs =
            RopeScaling::new(DIM, BASE, RopeScalingKind::Linear { scale: 2.0 }).expect("linear");
        let original = vec![0.3_f32, -1.2, 0.7, 2.1, -0.5, 1.1, 0.9, -0.2];
        let mut x = original.clone();
        let positions = [5usize];
        rs.apply(&mut x, 1, &positions).expect("apply forward");
        // Inverse rotation: negate every angle by flipping sin sign in a hand
        // rotation (since mscale==1 for Linear, this is an exact orthogonal inverse).
        let half = DIM / 2;
        let inv = rs.inv_freqs();
        for i in 0..half {
            let angle = positions[0] as f32 * inv[i];
            let cos = angle.cos();
            let sin = angle.sin();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            // Apply R(−θ): [cos, sin; −sin, cos].
            x[2 * i] = x0 * cos + x1 * sin;
            x[2 * i + 1] = -x0 * sin + x1 * cos;
        }
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn apply_matches_cos_sin_tables() {
        // apply() and cos_sin() must produce the same rotation.
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let positions = [7usize];
        rs.apply(&mut x, 1, &positions).expect("apply");
        let (cos, sin) = rs.cos_sin(8);
        let half = DIM / 2;
        for i in 0..half {
            let idx = positions[0] * half + i;
            let x0 = original[2 * i];
            let x1 = original[2 * i + 1];
            let want0 = x0 * cos[idx] - x1 * sin[idx];
            let want1 = x0 * sin[idx] + x1 * cos[idx];
            assert!((x[2 * i] - want0).abs() < 1e-5);
            assert!((x[2 * i + 1] - want1).abs() < 1e-5);
        }
    }

    #[test]
    fn apply_matches_rotary_embedding_for_none() {
        // The None kind's apply must coincide with RotaryEmbedding::apply.
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let rope = RotaryEmbedding::new(DIM, 32, BASE).expect("rope");
        let original = vec![0.5_f32, -0.3, 1.7, 0.2, -2.1, 0.8, 1.0, -1.0];
        let mut a = original.clone();
        let mut b = original.clone();
        rs.apply(&mut a, 1, &[9]).expect("rope_scaling apply");
        rope.apply(&mut b, 1, 1, 9).expect("rotary apply");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn apply_multi_token_multi_head() {
        let rs =
            RopeScaling::new(DIM, BASE, RopeScalingKind::NtkAware { scale: 2.0 }).expect("ntk");
        let mut x = vec![0.5_f32; 3 * 2 * DIM]; // 3 tokens, 2 heads
        rs.apply(&mut x, 2, &[0, 1, 2]).expect("apply");
        assert_eq!(x.len(), 3 * 2 * DIM);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn err_dim_odd() {
        assert!(RopeScaling::new(7, BASE, RopeScalingKind::None).is_err());
    }

    #[test]
    fn err_dim_zero() {
        assert!(RopeScaling::new(0, BASE, RopeScalingKind::None).is_err());
    }

    #[test]
    fn err_base_non_positive() {
        assert!(RopeScaling::new(DIM, 0.0, RopeScalingKind::None).is_err());
        assert!(RopeScaling::new(DIM, -10.0, RopeScalingKind::None).is_err());
    }

    #[test]
    fn err_scale_non_positive() {
        assert!(RopeScaling::new(DIM, BASE, RopeScalingKind::Linear { scale: 0.0 }).is_err());
        assert!(RopeScaling::new(DIM, BASE, RopeScalingKind::NtkAware { scale: -1.0 }).is_err());
        assert!(
            RopeScaling::new(
                DIM,
                BASE,
                RopeScalingKind::Yarn {
                    scale: 0.0,
                    original_max_pos: 2048,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn err_yarn_beta_order() {
        assert!(
            RopeScaling::new(
                DIM,
                BASE,
                RopeScalingKind::Yarn {
                    scale: 4.0,
                    original_max_pos: 2048,
                    beta_fast: 1.0,
                    beta_slow: 32.0,
                }
            )
            .is_err()
        );
        // Equal betas also rejected (division by zero otherwise).
        assert!(
            RopeScaling::new(
                DIM,
                BASE,
                RopeScalingKind::Yarn {
                    scale: 4.0,
                    original_max_pos: 2048,
                    beta_fast: 8.0,
                    beta_slow: 8.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn err_yarn_original_max_pos_zero() {
        assert!(
            RopeScaling::new(
                DIM,
                BASE,
                RopeScalingKind::Yarn {
                    scale: 4.0,
                    original_max_pos: 0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn err_apply_empty_positions() {
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let mut x = vec![0.0_f32; DIM];
        assert!(matches!(
            rs.apply(&mut x, 1, &[]),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn err_apply_dim_mismatch() {
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let mut x = vec![0.0_f32; DIM + 1];
        assert!(matches!(
            rs.apply(&mut x, 1, &[0]),
            Err(LmError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_apply_zero_heads() {
        let rs = RopeScaling::new(DIM, BASE, RopeScalingKind::None).expect("none");
        let mut x = vec![0.0_f32; DIM];
        assert!(rs.apply(&mut x, 0, &[0]).is_err());
    }
}
