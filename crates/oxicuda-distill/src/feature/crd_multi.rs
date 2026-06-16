//! CRD multi-positive contrastive representation distillation (Tian et al. 2020 ICLR +
//! multi-positive extension).
//!
//! Maintains a teacher-side memory bank of L2-normalized feature embeddings updated via EMA.
//! Loss = multi-positive InfoNCE: anchor vs N positives and M negatives from the bank.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CRD multi-positive distillation.
#[derive(Debug, Clone)]
pub struct CrdMultiConfig {
    /// Embedding dimension (both student and teacher).
    pub feat_dim: usize,
    /// Number of negative samples M per call.
    pub n_negatives: usize,
    /// NCE temperature τ (e.g. 0.07).
    pub temperature: f32,
    /// EMA momentum m for bank update (e.g. 0.5).
    pub momentum: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory bank
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalized feature memory bank (teacher-side).
///
/// All stored embeddings are maintained on the L2 unit sphere via EMA updates.
#[derive(Debug, Clone)]
pub struct CrdMemoryBank {
    /// Flat `[n_samples × feat_dim]` row-major array of L2-normalized embeddings.
    pub embeddings: Vec<f32>,
    /// Number of samples in the bank.
    pub n_samples: usize,
    /// Feature dimension.
    pub feat_dim: usize,
}

impl CrdMemoryBank {
    /// Initialize with random unit-sphere samples drawn from normal distribution.
    pub fn new(n_samples: usize, feat_dim: usize, rng: &mut LcgRng) -> DistillResult<Self> {
        if n_samples == 0 || feat_dim == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "CrdMemoryBank: n_samples and feat_dim must be > 0".into(),
            });
        }
        let total = n_samples * feat_dim;
        let mut embeddings = vec![0.0_f32; total];
        // Fill with normal samples
        rng.fill_normal(&mut embeddings);
        // L2-normalize each row
        for i in 0..n_samples {
            let row = &mut embeddings[i * feat_dim..(i + 1) * feat_dim];
            let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
        Ok(Self {
            embeddings,
            n_samples,
            feat_dim,
        })
    }

    /// EMA update for one sample: `emb[idx] = normalize(m * emb[idx] + (1-m) * new_feat)`.
    pub fn update(&mut self, idx: usize, new_feat: &[f32], momentum: f32) -> DistillResult<()> {
        if idx >= self.n_samples {
            return Err(DistillError::DimensionMismatch {
                expected: self.n_samples,
                got: idx + 1,
            });
        }
        if new_feat.len() != self.feat_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.feat_dim,
                got: new_feat.len(),
            });
        }
        let row = &mut self.embeddings[idx * self.feat_dim..(idx + 1) * self.feat_dim];
        // EMA blend
        for (s, &n) in row.iter_mut().zip(new_feat.iter()) {
            *s = momentum * *s + (1.0 - momentum) * n;
        }
        // Re-normalize
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= norm;
        }
        Ok(())
    }

    /// Return slice for sample at `idx`.
    pub fn lookup(&self, idx: usize) -> DistillResult<&[f32]> {
        if idx >= self.n_samples {
            return Err(DistillError::DimensionMismatch {
                expected: self.n_samples,
                got: idx + 1,
            });
        }
        Ok(&self.embeddings[idx * self.feat_dim..(idx + 1) * self.feat_dim])
    }

    /// Sample `n` random negative indices, avoiding any index in `exclude`.
    ///
    /// Uses partial Fisher-Yates on valid candidates. Returns `InvalidConfig` if
    /// there are not enough non-excluded samples.
    pub fn sample_negatives(
        &self,
        exclude: &[usize],
        n: usize,
        rng: &mut LcgRng,
    ) -> DistillResult<Vec<usize>> {
        // Build candidate list: all indices not in exclude set
        let exclude_set: std::collections::HashSet<usize> = exclude.iter().copied().collect();
        let mut candidates: Vec<usize> = (0..self.n_samples)
            .filter(|i| !exclude_set.contains(i))
            .collect();
        let valid_len = candidates.len();
        if valid_len < n {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "not enough samples: need {n} negatives but only {valid_len} available after exclusion"
                ),
            });
        }
        // Partial Fisher-Yates: select first n
        for i in 0..n {
            let remaining = valid_len - i;
            let j = i + (rng.next_u32() as usize) % remaining;
            candidates.swap(i, j);
        }
        Ok(candidates[..n].to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss computation
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-positive InfoNCE loss and related utilities.
pub struct CrdMultiLoss;

impl CrdMultiLoss {
    /// L2 normalize a vector in-place (divide by norm, clamp to avoid divide-by-zero).
    pub fn l2_normalize(v: &mut [f32]) {
        let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    /// Compute dot product between two equal-length slices.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Numerically stable log-sum-exp over a slice of values.
    fn log_sum_exp(vals: &[f32]) -> f32 {
        if vals.is_empty() {
            return f32::NEG_INFINITY;
        }
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return f32::NEG_INFINITY;
        }
        let sum: f32 = vals.iter().map(|&v| (v - max).exp()).sum();
        max + sum.ln()
    }

    /// Multi-positive InfoNCE loss.
    ///
    /// `anchor`: `[feat_dim]` (student embedding, L2-normalized)
    /// `positives`: `[n_pos × feat_dim]` row-major
    /// `negatives`: `[n_neg × feat_dim]` row-major
    ///
    /// `loss = -lse_pos + lse_all`
    /// where `lse_pos = log(Σ_j exp(dot(anchor, pos_j) / τ))`
    /// and   `lse_all = log(Σ_j exp(dot(anchor, pos_j) / τ) + Σ_k exp(dot(anchor, neg_k) / τ))`
    pub fn nce_loss(
        anchor: &[f32],
        positives: &[f32],
        n_positives: usize,
        negatives: &[f32],
        n_negatives: usize,
        temperature: f32,
    ) -> DistillResult<f32> {
        if anchor.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let feat_dim = anchor.len();
        if n_positives == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_positives must be > 0".into(),
            });
        }
        if positives.len() != n_positives * feat_dim {
            return Err(DistillError::DimensionMismatch {
                expected: n_positives * feat_dim,
                got: positives.len(),
            });
        }
        if n_negatives > 0 && negatives.len() != n_negatives * feat_dim {
            return Err(DistillError::DimensionMismatch {
                expected: n_negatives * feat_dim,
                got: negatives.len(),
            });
        }

        let tau = temperature.max(1e-12);

        // Positive dot products / τ
        let pos_scaled: Vec<f32> = (0..n_positives)
            .map(|j| {
                let pos_row = &positives[j * feat_dim..(j + 1) * feat_dim];
                Self::dot(anchor, pos_row) / tau
            })
            .collect();

        // Negative dot products / τ
        let neg_scaled: Vec<f32> = (0..n_negatives)
            .map(|k| {
                let neg_row = &negatives[k * feat_dim..(k + 1) * feat_dim];
                Self::dot(anchor, neg_row) / tau
            })
            .collect();

        // Numerically stable log-sum-exp of positives
        let lse_pos = Self::log_sum_exp(&pos_scaled);

        // Numerically stable log-sum-exp of all (pos + neg)
        let mut all_scaled = pos_scaled;
        all_scaled.extend_from_slice(&neg_scaled);
        let lse_all = Self::log_sum_exp(&all_scaled);

        // loss = -lse_pos + lse_all
        let loss = -lse_pos + lse_all;
        Ok(loss)
    }

    /// Convenience: look up embeddings from memory bank, then call `nce_loss`.
    pub fn nce_loss_from_bank(
        anchor: &[f32],
        pos_indices: &[usize],
        neg_indices: &[usize],
        bank: &CrdMemoryBank,
        temperature: f32,
    ) -> DistillResult<f32> {
        if anchor.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if pos_indices.is_empty() {
            return Err(DistillError::InvalidConfig {
                msg: "pos_indices must be non-empty".into(),
            });
        }
        let feat_dim = bank.feat_dim;

        // Collect positive embeddings
        let mut positives = Vec::with_capacity(pos_indices.len() * feat_dim);
        for &pi in pos_indices {
            positives.extend_from_slice(bank.lookup(pi)?);
        }

        // Collect negative embeddings
        let mut negatives = Vec::with_capacity(neg_indices.len() * feat_dim);
        for &ni in neg_indices {
            negatives.extend_from_slice(bank.lookup(ni)?);
        }

        Self::nce_loss(
            anchor,
            &positives,
            pos_indices.len(),
            &negatives,
            neg_indices.len(),
            temperature,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    fn unit_vec(v: Vec<f32>) -> Vec<f32> {
        let mut v = v;
        CrdMultiLoss::l2_normalize(&mut v);
        v
    }

    // ── 1. loss_positive ────────────────────────────────────────────────────

    #[test]
    fn loss_positive() {
        let feat_dim = 8usize;
        let mut rng = make_rng();
        let bank = CrdMemoryBank::new(10, feat_dim, &mut rng).expect("new should succeed");
        let anchor = unit_vec((0..feat_dim).map(|i| i as f32 * 0.1).collect());
        let loss = CrdMultiLoss::nce_loss_from_bank(&anchor, &[0], &[1, 2, 3], &bank, 0.07)
            .expect("nce_loss_from_bank should succeed");
        assert!(loss >= 0.0, "NCE loss must be >= 0, got {loss}");
        assert!(loss.is_finite(), "NCE loss must be finite");
    }

    // ── 2. loss_single_positive ──────────────────────────────────────────────

    #[test]
    fn loss_single_positive() {
        // With n_positives=1, multi-positive InfoNCE collapses to classic InfoNCE
        let feat_dim = 4usize;
        let pos = unit_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let neg = unit_vec(vec![0.0_f32, 1.0, 0.0, 0.0]);
        let anchor = unit_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let loss = CrdMultiLoss::nce_loss(&anchor, &pos, 1, &neg, 1, 0.07)
            .expect("nce_loss should succeed");
        // anchor == pos → high sim → small positive loss
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        // single-pos classic InfoNCE: -dot/τ + log(exp(dot/τ) + exp(neg_dot/τ))
        // should approach log(1 + exp(neg-pos)/τ) ≈ log(1 + tiny) ≈ 0 for perfect alignment
        assert!(
            loss < 1.0,
            "single-pos with perfect match should be small, got {loss}"
        );
        _ = feat_dim; // used implicitly
    }

    // ── 3. loss_zero_when_perfect ────────────────────────────────────────────

    #[test]
    fn loss_zero_when_perfect() {
        // anchor == positive, negative is orthogonal → low loss
        let anchor = unit_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let pos = anchor.clone();
        let neg = unit_vec(vec![0.0_f32, 1.0, 0.0, 0.0]);
        let loss = CrdMultiLoss::nce_loss(&anchor, &pos, 1, &neg, 1, 0.07)
            .expect("nce_loss should succeed");
        // dot(anchor, neg) = 0, dot(anchor, pos) = 1
        // loss ≈ -1/τ + log(exp(1/τ) + exp(0)) = log(1 + exp(-1/τ))
        assert!(loss >= 0.0 && loss.is_finite());
        // With τ=0.07, loss ≈ log(1 + exp(-14.3)) ≈ tiny
        assert!(
            loss < 0.1,
            "near-perfect alignment → very small loss, got {loss}"
        );
    }

    // ── 4. loss_increases_with_more_negatives ────────────────────────────────

    #[test]
    fn loss_increases_with_more_negatives() {
        let mut rng = make_rng();
        let feat_dim = 16usize;
        let bank = CrdMemoryBank::new(50, feat_dim, &mut rng).expect("new should succeed");
        let anchor = bank.lookup(0).expect("lookup should succeed").to_vec();
        let loss_few = CrdMultiLoss::nce_loss_from_bank(&anchor, &[0], &[1, 2], &bank, 0.07)
            .expect("nce_loss_from_bank should succeed");
        let loss_more =
            CrdMultiLoss::nce_loss_from_bank(&anchor, &[0], &[1, 2, 3, 4, 5, 6, 7, 8], &bank, 0.07)
                .expect("value should be present");
        // More negatives → larger denominator → larger loss
        assert!(
            loss_more >= loss_few,
            "more negatives should not decrease loss: few={loss_few} more={loss_more}"
        );
    }

    // ── 5. temperature_effect ────────────────────────────────────────────────

    #[test]
    fn temperature_effect() {
        let feat_dim = 8usize;
        let anchor = unit_vec((0..feat_dim).map(|i| i as f32).collect());
        let pos = unit_vec((0..feat_dim).map(|i| (feat_dim - i) as f32).collect());
        let neg_flat: Vec<f32> = (0..feat_dim).map(|i| -(i as f32)).collect();
        let neg = unit_vec(neg_flat);

        let loss_cold = CrdMultiLoss::nce_loss(&anchor, &pos, 1, &neg, 1, 0.1)
            .expect("nce_loss should succeed");
        let loss_warm = CrdMultiLoss::nce_loss(&anchor, &pos, 1, &neg, 1, 1.0)
            .expect("nce_loss should succeed");
        // Different temperatures produce different loss values
        assert!(
            (loss_cold - loss_warm).abs() > 1e-4,
            "τ=0.1 ({loss_cold}) should differ from τ=1.0 ({loss_warm})"
        );
    }

    // ── 6. l2_normalize_unit_length ──────────────────────────────────────────

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0_f32, 4.0, 0.0];
        CrdMultiLoss::l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "normalized vector should have unit norm, got {norm}"
        );
    }

    // ── 7. l2_normalize_zero_safe ────────────────────────────────────────────

    #[test]
    fn l2_normalize_zero_safe() {
        let mut v = vec![0.0_f32; 4];
        // Should not panic
        CrdMultiLoss::l2_normalize(&mut v);
        // Result is well-defined (all entries scaled by 1/1e-12)
        assert!(v.iter().all(|x| x.is_finite()));
    }

    // ── 8. bank_init_unit_norm ───────────────────────────────────────────────

    #[test]
    fn bank_init_unit_norm() {
        let mut rng = make_rng();
        let bank = CrdMemoryBank::new(20, 8, &mut rng).expect("new should succeed");
        for i in 0..bank.n_samples {
            let row = bank.lookup(i).expect("lookup should succeed");
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "bank row {i} norm should be 1.0, got {norm}"
            );
        }
    }

    // ── 9. bank_lookup_valid ─────────────────────────────────────────────────

    #[test]
    fn bank_lookup_valid() {
        let mut rng = make_rng();
        let feat_dim = 16usize;
        let bank = CrdMemoryBank::new(5, feat_dim, &mut rng).expect("new should succeed");
        let slice = bank.lookup(0).expect("lookup should succeed");
        assert_eq!(slice.len(), feat_dim);
    }

    // ── 10. bank_lookup_oob ──────────────────────────────────────────────────

    #[test]
    fn bank_lookup_oob() {
        let mut rng = make_rng();
        let bank = CrdMemoryBank::new(5, 8, &mut rng).expect("new should succeed");
        let result = bank.lookup(5); // index == n_samples → OOB
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "OOB lookup should yield DimensionMismatch"
        );
    }

    // ── 11. bank_update_normalizes ───────────────────────────────────────────

    #[test]
    fn bank_update_normalizes() {
        let mut rng = make_rng();
        let feat_dim = 8usize;
        let mut bank = CrdMemoryBank::new(5, feat_dim, &mut rng).expect("new should succeed");
        // Update with a non-unit vector
        let new_feat: Vec<f32> = vec![5.0_f32; feat_dim];
        bank.update(0, &new_feat, 0.5)
            .expect("update should succeed");
        let row = bank.lookup(0).expect("lookup should succeed");
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "after update, bank row must still have unit norm, got {norm}"
        );
    }

    // ── 12. bank_update_momentum ─────────────────────────────────────────────

    #[test]
    fn bank_update_momentum() {
        let mut rng = make_rng();
        let feat_dim = 8usize;
        let mut bank = CrdMemoryBank::new(5, feat_dim, &mut rng).expect("new should succeed");
        let original = bank.lookup(0).expect("lookup should succeed").to_vec();
        // With momentum=1.0, EMA = 1.0*old + 0.0*new = old (then renormalized = old)
        let new_feat = vec![0.1_f32; feat_dim]; // different direction
        bank.update(0, &new_feat, 1.0)
            .expect("update should succeed");
        let updated = bank.lookup(0).expect("lookup should succeed");
        for (a, &b) in original.iter().zip(updated.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "momentum=1.0 → embedding should be unchanged"
            );
        }
    }

    // ── 13. sample_negatives_count ───────────────────────────────────────────

    #[test]
    fn sample_negatives_count() {
        let mut rng = make_rng();
        let bank = CrdMemoryBank::new(100, 8, &mut rng).expect("new should succeed");
        let negs = bank
            .sample_negatives(&[0], 10, &mut rng)
            .expect("sample_negatives should succeed");
        assert_eq!(negs.len(), 10, "should return exactly 10 negatives");
    }

    // ── 14. sample_negatives_excludes ────────────────────────────────────────

    #[test]
    fn sample_negatives_excludes() {
        let mut rng = make_rng();
        let bank = CrdMemoryBank::new(50, 8, &mut rng).expect("new should succeed");
        let exclude = vec![0usize, 1, 2, 3];
        let negs = bank
            .sample_negatives(&exclude, 5, &mut rng)
            .expect("sample_negatives should succeed");
        for &ni in &negs {
            assert!(
                !exclude.contains(&ni),
                "excluded index {ni} appeared in negatives"
            );
        }
    }

    // ── 15. nce_from_bank_runs ────────────────────────────────────────────────

    #[test]
    fn nce_from_bank_runs() {
        let mut rng = make_rng();
        let feat_dim = 16usize;
        let bank = CrdMemoryBank::new(20, feat_dim, &mut rng).expect("new should succeed");
        let anchor = bank.lookup(0).expect("lookup should succeed").to_vec();
        let result = CrdMultiLoss::nce_loss_from_bank(
            &anchor,
            &[1, 2],    // 2 positives
            &[3, 4, 5], // 3 negatives
            &bank,
            0.07,
        );
        assert!(
            result.is_ok(),
            "nce_loss_from_bank should succeed: {:?}",
            result.err()
        );
        assert!(result.expect("result should be present").is_finite());
    }

    // ── 16. empty_anchor_err ─────────────────────────────────────────────────

    #[test]
    fn empty_anchor_err() {
        let pos = vec![1.0_f32, 0.0, 0.0];
        let neg = vec![0.0_f32, 1.0, 0.0];
        let result = CrdMultiLoss::nce_loss(&[], &pos, 1, &neg, 1, 0.07);
        assert!(
            matches!(result, Err(DistillError::EmptyInput)),
            "empty anchor should yield EmptyInput"
        );
    }

    // ── 17. dim_mismatch_pos_err ──────────────────────────────────────────────

    #[test]
    fn dim_mismatch_pos_err() {
        // anchor is 4-dim but positive row has 3 elements (wrong)
        let anchor = vec![1.0_f32, 0.0, 0.0, 0.0];
        let pos_wrong = vec![1.0_f32, 0.0, 0.0]; // should be 1 × 4 = 4 elements
        let neg = vec![0.0_f32, 1.0, 0.0, 0.0];
        let result = CrdMultiLoss::nce_loss(&anchor, &pos_wrong, 1, &neg, 1, 0.07);
        assert!(
            matches!(result, Err(DistillError::DimensionMismatch { .. })),
            "wrong positive length should yield DimensionMismatch"
        );
    }

    // ── 18. insufficient_negatives_err ───────────────────────────────────────

    #[test]
    fn insufficient_negatives_err() {
        let mut rng = make_rng();
        // Bank of 7 samples, exclude 5, request 5 → only 2 valid → error
        let bank = CrdMemoryBank::new(7, 8, &mut rng).expect("new should succeed");
        let exclude: Vec<usize> = (0..5).collect(); // excludes 0..5, leaves 5,6
        let result = bank.sample_negatives(&exclude, 5, &mut rng);
        assert!(
            matches!(result, Err(DistillError::InvalidConfig { .. })),
            "insufficient negatives should yield InvalidConfig"
        );
    }
}
