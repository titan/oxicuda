//! MoRA — High-Rank Updating with a Square Matrix.
//!
//! Reference: Jiang, T., Huang, S., Luo, S., Zhang, Z., Huang, H., Wei, F., Deng, W., Sun, F.,
//! Zhang, Q., Wang, D., & Zhuang, F. (2024). *MoRA: High-Rank Updating for Parameter-Efficient
//! Fine-Tuning*. <https://arxiv.org/abs/2405.12130>
//!
//! LoRA's update `ΔW = B·A` is inherently low-rank (`rank ≤ r`), which limits its capacity to
//! memorise new knowledge. MoRA keeps the **same trainable-parameter budget** as LoRA but
//! spends it on a *square* matrix `M ∈ ℝ^{r̂ × r̂}` (so `r̂ ≈ √(r · (in + out))`), whose update
//! can reach rank `r̂` — much higher than LoRA's `r`.
//!
//! Because `M` is square (`r̂ × r̂`) but the layer maps `in_features → out_features`, MoRA uses
//! **non-parametric** compression and decompression operators (no extra trainable weights):
//!
//! * **Compression** `f_comp : ℝ^{in} → ℝ^{r̂}` — reshape `x` into `r̂` contiguous groups and
//!   sum within each group (a fixed "group-sum" pooling). If `in` is not divisible by `r̂` the
//!   trailing elements are zero-padded.
//!
//! * **Decompression** `f_decomp : ℝ^{r̂} → ℝ^{out}` — replicate each of the `r̂` outputs across
//!   a contiguous block of `out / r̂` positions (truncating any overhang). This is the
//!   transpose-style inverse of the group-sum compression.
//!
//! The MoRA delta applied to the frozen base is therefore
//!
//! ```text
//!   y = W₀·x  +  scale · f_decomp( M · f_comp(x) )
//! ```
//!
//! with `scale = alpha / r̂`. `M` is zero-initialised so the adapter is an exact identity at the
//! start of training (the output equals `W₀·x`), and it can later be **merged** into `W₀` by
//! materialising the equivalent dense `[out × in]` delta.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a [`MoraLinear`] adapter.
#[derive(Debug, Clone)]
pub struct MoraConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Side length `r̂` of the square trainable matrix `M ∈ ℝ^{r̂ × r̂}`.
    ///
    /// Must be `> 0` and `≤ min(in_features, out_features)`.
    pub square_rank: usize,
    /// Scaling factor α; the effective scale is `α / r̂`.
    pub alpha: f32,
}

/// Suggest a square side length `r̂` that matches a target LoRA rank's parameter budget.
///
/// A LoRA adapter of rank `r` has `r · (in + out)` trainable parameters, whereas MoRA's square
/// matrix has `r̂²`. Equating the two gives `r̂ = round(√(r · (in + out)))`, clamped to the
/// valid range `[1, min(in, out)]`.
#[must_use]
pub fn suggest_square_rank(in_features: usize, out_features: usize, lora_rank: usize) -> usize {
    let budget = (lora_rank * (in_features + out_features)) as f32;
    let r_hat = budget.sqrt().round() as usize;
    let upper = in_features.min(out_features);
    r_hat.clamp(1, upper.max(1))
}

/// MoRA adapter for a single linear layer with a square trainable matrix.
#[derive(Debug, Clone)]
pub struct MoraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Square matrix side `r̂`.
    pub square_rank: usize,
    /// Effective scale `α / r̂`.
    pub scale: f32,
    /// Frozen base weight `W₀`, shape `[out_features × in_features]` (row-major).
    pub w: Vec<f32>,
    /// Trainable square matrix `M`, shape `[r̂ × r̂]` (row-major). Zero-initialised.
    pub m: Vec<f32>,
}

impl MoraLinear {
    /// Construct a new MoRA adapter.
    ///
    /// `W₀` is zero-initialised and `M` is zero-initialised (so the initial delta is zero).
    ///
    /// # Errors
    /// - [`PeftError::ZeroBlockSize`] if `square_rank == 0`.
    /// - [`PeftError::RankTooLarge`] if `square_rank > min(in_features, out_features)`.
    pub fn new(cfg: &MoraConfig, _rng: &mut LcgRng) -> PeftResult<Self> {
        if cfg.square_rank == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        let upper = cfg.in_features.min(cfg.out_features);
        if cfg.square_rank > upper {
            return Err(PeftError::RankTooLarge {
                rank: cfg.square_rank,
                dim: upper,
            });
        }
        let scale = cfg.alpha / cfg.square_rank as f32;
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            square_rank: cfg.square_rank,
            scale,
            w: vec![0.0_f32; cfg.out_features * cfg.in_features],
            m: vec![0.0_f32; cfg.square_rank * cfg.square_rank],
        })
    }

    /// Compress an `in_features`-vector into an `r̂`-vector by contiguous group-sum pooling.
    ///
    /// Group `g` aggregates input indices `[g·g_size, (g+1)·g_size)` where `g_size = ⌈in/r̂⌉`;
    /// indices beyond `in_features` contribute zero (implicit zero-padding).
    #[must_use]
    pub fn compress(&self, x: &[f32]) -> Vec<f32> {
        let r = self.square_rank;
        let group = self.in_features.div_ceil(r).max(1);
        let mut out = vec![0.0_f32; r];
        for (g, slot) in out.iter_mut().enumerate() {
            let start = g * group;
            let end = (start + group).min(self.in_features);
            if start >= self.in_features {
                break;
            }
            let mut acc = 0.0_f32;
            for &xi in &x[start..end] {
                acc += xi;
            }
            *slot = acc;
        }
        out
    }

    /// Decompress an `r̂`-vector into an `out_features`-vector by block replication.
    ///
    /// Output index `o` copies compressed value `o / block` where `block = ⌈out/r̂⌉`; any block
    /// index `≥ r̂` yields zero.
    #[must_use]
    pub fn decompress(&self, y: &[f32]) -> Vec<f32> {
        let r = self.square_rank;
        let block = self.out_features.div_ceil(r).max(1);
        let mut out = vec![0.0_f32; self.out_features];
        for (o, slot) in out.iter_mut().enumerate() {
            let g = o / block;
            if g < r {
                *slot = y[g];
            }
        }
        out
    }

    /// Forward pass `y = W₀·x + scale · decompress(M · compress(x))`.
    ///
    /// # Errors
    /// [`PeftError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        // Base path.
        let mut out = mat_vec(&self.w, x, self.out_features, self.in_features);
        // MoRA path.
        let compressed = self.compress(x);
        let m_out = mat_vec(&self.m, &compressed, self.square_rank, self.square_rank);
        let delta = self.decompress(&m_out);
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += self.scale * d;
        }
        Ok(out)
    }

    /// Materialise the equivalent dense delta `[out_features × in_features]`.
    ///
    /// Column `j` of the delta equals `scale · decompress(M · compress(e_j))` where `e_j` is the
    /// `j`-th standard basis vector. Because compression is group-sum and decompression is block
    /// replication, this is computed in closed form without per-column passes.
    #[must_use]
    pub fn dense_delta(&self) -> Vec<f32> {
        let r = self.square_rank;
        let in_group = self.in_features.div_ceil(r).max(1);
        let out_block = self.out_features.div_ceil(r).max(1);
        let mut delta = vec![0.0_f32; self.out_features * self.in_features];
        for o in 0..self.out_features {
            let go = o / out_block;
            if go >= r {
                continue;
            }
            for j in 0..self.in_features {
                let gj = j / in_group;
                if gj >= r {
                    continue;
                }
                // (M · compress)[go] picks up M[go, gj] from input column j.
                delta[o * self.in_features + j] = self.scale * self.m[go * r + gj];
            }
        }
        delta
    }

    /// Merge the MoRA delta into the frozen base weight: `W₀ += dense_delta()`.
    pub fn merge_into_w(&mut self) {
        let delta = self.dense_delta();
        for (w, d) in self.w.iter_mut().zip(delta.iter()) {
            *w += d;
        }
    }

    /// Number of trainable parameters (the square matrix only): `r̂²`.
    #[must_use]
    #[inline]
    pub fn num_trainable(&self) -> usize {
        self.square_rank * self.square_rank
    }
}

/// Multiply matrix `m` (`[rows × cols]`, row-major) by vector `v` (length `cols`).
fn mat_vec(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            let start = i * cols;
            m[start..start + cols]
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(in_f: usize, out_f: usize, r: usize, alpha: f32) -> MoraConfig {
        MoraConfig {
            in_features: in_f,
            out_features: out_f,
            square_rank: r,
            alpha,
        }
    }

    #[test]
    fn new_zero_init_delta_is_zero() {
        let mut rng = LcgRng::new(1);
        let m = MoraLinear::new(&cfg(8, 8, 4, 8.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        let x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let out = m
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        // W₀=0 and M=0 → output all zeros.
        for &v in &out {
            assert!(v.abs() < 1e-6, "zero-init forward must be zero, got {v}");
        }
    }

    #[test]
    fn new_zero_block_size_errors() {
        let mut rng = LcgRng::new(2);
        assert!(matches!(
            MoraLinear::new(&cfg(8, 8, 0, 8.0), &mut rng),
            Err(PeftError::ZeroBlockSize)
        ));
    }

    #[test]
    fn new_rank_too_large_errors() {
        let mut rng = LcgRng::new(3);
        assert!(matches!(
            MoraLinear::new(&cfg(4, 8, 5, 8.0), &mut rng),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn scale_equals_alpha_over_square_rank() {
        let mut rng = LcgRng::new(4);
        let m = MoraLinear::new(&cfg(16, 16, 4, 12.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        assert!((m.scale - 3.0).abs() < 1e-6, "scale={}", m.scale);
    }

    #[test]
    fn compress_group_sum_correct() {
        let mut rng = LcgRng::new(5);
        let m = MoraLinear::new(&cfg(8, 8, 4, 8.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        // group size = 8/4 = 2; groups: [0,1],[2,3],[4,5],[6,7]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = m.compress(&x);
        assert_eq!(c, vec![3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn compress_handles_indivisible_in_features() {
        let mut rng = LcgRng::new(6);
        // in=7, r=3 → group=ceil(7/3)=3; groups: [0..3]=3, [3..6]=3, [6..7]=1
        let m = MoraLinear::new(&cfg(7, 9, 3, 3.0), &mut rng)
            .expect("MoraLinear::new should succeed with indivisible in_features");
        let x = vec![1.0_f32, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0];
        let c = m.compress(&x);
        assert_eq!(c.len(), 3);
        assert_eq!(c, vec![3.0, 6.0, 5.0]);
    }

    #[test]
    fn decompress_block_replication_correct() {
        let mut rng = LcgRng::new(7);
        // out=8, r=4 → block=2; output blocks each repeat one compressed value
        let m = MoraLinear::new(&cfg(8, 8, 4, 8.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        let y = vec![10.0_f32, 20.0, 30.0, 40.0];
        let d = m.decompress(&y);
        assert_eq!(d, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]);
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(8);
        let m = MoraLinear::new(&cfg(8, 8, 4, 8.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        assert!(matches!(
            m.forward(&[1.0, 2.0, 3.0]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_matches_explicit_path() {
        let mut rng = LcgRng::new(9);
        let mut m = MoraLinear::new(&cfg(8, 8, 4, 4.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        // Set a non-trivial M.
        for (i, v) in m.m.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 0.5;
        }
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let out = m
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        // Reconstruct manually: scale·decompress(M·compress(x)) since W₀=0.
        let c = m.compress(&x);
        let mo = mat_vec(&m.m, &c, 4, 4);
        let manual = m.decompress(&mo);
        for (o, mref) in out.iter().zip(manual.iter()) {
            assert!(
                (o - m.scale * mref).abs() < 1e-5,
                "{o} vs {}",
                m.scale * mref
            );
        }
    }

    #[test]
    fn dense_delta_matches_column_probes() {
        let mut rng = LcgRng::new(10);
        let mut m = MoraLinear::new(&cfg(8, 8, 4, 4.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        for (i, v) in m.m.iter_mut().enumerate() {
            *v = ((i * 7 + 3) % 11) as f32 * 0.2 - 1.0;
        }
        let delta = m.dense_delta();
        // Probe each column with a basis vector through the MoRA path.
        for j in 0..m.in_features {
            let mut e = vec![0.0_f32; m.in_features];
            e[j] = 1.0;
            let c = m.compress(&e);
            let mo = mat_vec(&m.m, &c, 4, 4);
            let probe = m.decompress(&mo);
            for o in 0..m.out_features {
                let got = delta[o * m.in_features + j];
                let want = m.scale * probe[o];
                assert!(
                    (got - want).abs() < 1e-5,
                    "col {j} row {o}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn merge_then_forward_equals_unmerged_plus_base() {
        let mut rng = LcgRng::new(11);
        let mut m = MoraLinear::new(&cfg(8, 8, 4, 4.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        for (i, v) in m.m.iter_mut().enumerate() {
            *v = (i as f32) * 0.05;
        }
        // Give the base weight some content.
        for (i, w) in m.w.iter_mut().enumerate() {
            *w = (i as f32) * 0.001;
        }
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let before = m
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        m.merge_into_w();
        // After merge, the MoRA path is *still* active, so forward differs; instead verify the
        // pure-base forward of the merged weight equals the original full forward.
        let merged_base = mat_vec(&m.w, &x, m.out_features, m.in_features);
        for (b, mb) in before.iter().zip(merged_base.iter()) {
            assert!((b - mb).abs() < 1e-4, "merge mismatch: {b} vs {mb}");
        }
    }

    #[test]
    fn num_trainable_is_square() {
        let mut rng = LcgRng::new(12);
        let m = MoraLinear::new(&cfg(16, 16, 5, 8.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        assert_eq!(m.num_trainable(), 25);
    }

    #[test]
    fn suggest_square_rank_matches_budget() {
        // LoRA rank 8 over 64+64 → budget 1024 → sqrt = 32.
        let r = suggest_square_rank(64, 64, 8);
        assert_eq!(r, 32);
    }

    #[test]
    fn suggest_square_rank_clamped_to_min_dim() {
        // Huge LoRA rank but small dims → clamp to min(in,out).
        let r = suggest_square_rank(8, 16, 100);
        assert_eq!(r, 8);
    }

    #[test]
    fn suggest_square_rank_never_zero() {
        let r = suggest_square_rank(0, 0, 0);
        assert!(r >= 1, "square rank must be ≥ 1, got {r}");
    }

    #[test]
    fn high_rank_delta_can_exceed_lora_rank() {
        // MoRA's dense delta can have rank up to r̂ = 4, demonstrably higher than a rank-1 LoRA.
        let mut rng = LcgRng::new(13);
        let mut m = MoraLinear::new(&cfg(8, 8, 4, 4.0), &mut rng)
            .expect("MoraLinear::new should succeed with valid config");
        // Identity-like M → delta has 4 distinct replicated rows/cols → rank 4.
        for d in 0..4 {
            m.m[d * 4 + d] = 1.0;
        }
        let delta = m.dense_delta();
        // Count distinct non-zero output blocks to confirm > 1 effective rank.
        let block = m.out_features.div_ceil(4);
        let mut nonzero_blocks = 0;
        for g in 0..4 {
            let o = g * block;
            let row = &delta[o * m.in_features..(o + 1) * m.in_features];
            if row.iter().any(|&v| v.abs() > 1e-9) {
                nonzero_blocks += 1;
            }
        }
        assert!(
            nonzero_blocks >= 2,
            "MoRA delta should have rank > 1, blocks={nonzero_blocks}"
        );
    }
}
