//! DARE-TIES — composite DARE pruning + TIES sign-consensus model merging.
//!
//! Reference: Yu L, Yu B, Yu H, Huang F, Li Y (2024) "Language Models are
//! Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch",
//! ICML.
//! <https://arxiv.org/abs/2311.03099>
//!
//! Yu et al. observe that fine-tuned delta vectors can be aggressively *DARE*-
//! pruned (Drop And REscale: keep each entry with probability `p`, rescale
//! retained entries by `1/p`) without harming downstream accuracy. They then
//! propose stacking DARE on top of [`crate::merge::merge::ties_merge`] to obtain a
//! single per-task fingerprint that survives multi-task merging:
//!
//! 1. Per-task DARE prune with density `p₁`.
//! 2. Per-task TIES trim — keep the top `p₂` fraction by magnitude.
//! 3. Sign consensus across tasks — for each coordinate, collapse to the
//!    majority sign (zero on ties).
//! 4. Disjoint mean — average only the contributors that share the elected
//!    sign.
//!
//! This module performs the four steps over plain task-vector slices using the
//! same [`crate::handle::LcgRng`] seed pattern as
//! [`crate::merge::arithmetic::dare_prune`], so callers can reproduce a merge bit-for-
//! bit by re-running with the same seed.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for the [`DareTies::merge`] pipeline.
#[derive(Debug, Clone, Copy)]
pub struct DareTiesConfig {
    /// DARE keep probability in `(0, 1]`. Each entry of each task vector is
    /// retained with this probability and rescaled by `1/density` to preserve
    /// expected magnitude.
    pub density: f32,
    /// TIES top-`p` keep fraction in `(0, 1]`. After DARE, only the largest
    /// `ceil(trim_density · n)` entries of each task vector survive.
    pub trim_density: f32,
    /// Seed used to derive the per-task Bernoulli sequences. Task `i` uses
    /// `LcgRng::new(seed.wrapping_add(i as u64))` so re-runs with the same
    /// seed reproduce the same masks.
    pub seed: u64,
}

impl Default for DareTiesConfig {
    fn default() -> Self {
        Self {
            density: 0.5,
            trim_density: 0.2,
            seed: 0,
        }
    }
}

/// DARE-TIES algorithm namespace.
pub struct DareTies;

impl DareTies {
    /// Run the DARE → TIES → sign-consensus → disjoint-mean pipeline on the
    /// given task vectors and return the merged delta.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] when the configuration is malformed
    /// (`density` or `trim_density` outside their accepted ranges), when
    /// `task_vectors` is empty, the first task vector is empty, or any
    /// subsequent task vector disagrees in length with the first.
    pub fn merge(task_vectors: &[Vec<f32>], cfg: &DareTiesConfig) -> PeftResult<Vec<f32>> {
        validate(task_vectors, cfg)?;

        let n = task_vectors[0].len();
        let k = task_vectors.len();

        // Step 1+2: DARE prune (Bernoulli + rescale) then TIES trim.
        let mut pruned: Vec<Vec<f32>> = Vec::with_capacity(k);
        for (i, tv) in task_vectors.iter().enumerate() {
            let mut rng = LcgRng::new(cfg.seed.wrapping_add(i as u64));
            let after_dare = dare_prune_local(tv, cfg.density, &mut rng);
            let trimmed = ties_trim(&after_dare, cfg.trim_density);
            pruned.push(trimmed);
        }

        // Step 3: elect the majority sign per coordinate.
        let mut sign_sum = vec![0.0_f32; n];
        for delta in &pruned {
            for (s, &v) in sign_sum.iter_mut().zip(delta.iter()) {
                *s += v;
            }
        }
        let elected: Vec<i8> = sign_sum
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    1_i8
                } else if s < 0.0 {
                    -1_i8
                } else {
                    0_i8
                }
            })
            .collect();

        // Step 4: disjoint mean — only count contributors whose sign agrees
        // with the elected sign for that coordinate.
        let mut merged = vec![0.0_f32; n];
        let mut counts = vec![0_u32; n];
        for delta in &pruned {
            for ((m, cnt), (&v, &sign)) in merged
                .iter_mut()
                .zip(counts.iter_mut())
                .zip(delta.iter().zip(elected.iter()))
            {
                if v == 0.0 {
                    continue;
                }
                let agrees = sign == 0 || (sign > 0 && v > 0.0) || (sign < 0 && v < 0.0);
                if agrees {
                    *m += v;
                    *cnt += 1;
                }
            }
        }
        for (m, &c) in merged.iter_mut().zip(counts.iter()) {
            let denom = c.max(1) as f32;
            *m /= denom;
        }
        Ok(merged)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn validate(task_vectors: &[Vec<f32>], cfg: &DareTiesConfig) -> PeftResult<()> {
    if task_vectors.is_empty() {
        return Err(PeftError::Internal {
            msg: "DARE-TIES requires at least one task vector".to_string(),
        });
    }
    let n = task_vectors[0].len();
    if n == 0 {
        return Err(PeftError::Internal {
            msg: "DARE-TIES task_vectors[0] must be non-empty".to_string(),
        });
    }
    for (i, tv) in task_vectors.iter().enumerate() {
        if tv.len() != n {
            return Err(PeftError::Internal {
                msg: format!(
                    "DARE-TIES task_vectors[{i}] length {} != task_vectors[0] length {n}",
                    tv.len()
                ),
            });
        }
    }
    if cfg.density.is_nan() || cfg.density <= 0.0 || cfg.density > 1.0 {
        return Err(PeftError::Internal {
            msg: format!("DARE-TIES density must be in (0, 1], got {}", cfg.density),
        });
    }
    if cfg.trim_density.is_nan() || cfg.trim_density <= 0.0 || cfg.trim_density > 1.0 {
        return Err(PeftError::Internal {
            msg: format!(
                "DARE-TIES trim_density must be in (0, 1], got {}",
                cfg.trim_density
            ),
        });
    }
    Ok(())
}

/// Local DARE prune that *does not* clamp `density` (the workspace
/// [`super::arithmetic::dare_prune`] clamps to `[1e-6, 1.0]` for safety, which
/// would silently widen our validated range). Caller has already guaranteed
/// `density ∈ (0, 1]`.
fn dare_prune_local(delta: &[f32], density: f32, rng: &mut LcgRng) -> Vec<f32> {
    let scale = 1.0_f32 / density;
    delta
        .iter()
        .map(|&v| {
            let u = rng.next_f32();
            if u < density { v * scale } else { 0.0 }
        })
        .collect()
}

/// TIES trim — keep the top `density · n` (ceil-rounded, at least one) entries
/// by absolute magnitude; zero the rest.
fn ties_trim(delta: &[f32], density: f32) -> Vec<f32> {
    let n = delta.len();
    if n == 0 {
        return Vec::new();
    }
    let keep = (((n as f32) * density).ceil() as usize).clamp(1, n);

    // Index-and-magnitude pairs, then keep the top-`keep` by |value|.
    let mut idxmag: Vec<(usize, f32)> = delta
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    idxmag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut mask = vec![false; n];
    for &(idx, _) in idxmag.iter().take(keep) {
        mask[idx] = true;
    }
    delta
        .iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] { v } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn single_task_vector_returns_trimmed_scaled_version() {
        let tv = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cfg = DareTiesConfig {
            density: 1.0,
            trim_density: 0.5,
            seed: 1,
        };
        let merged = DareTies::merge(std::slice::from_ref(&tv), &cfg).expect("merge");
        // density=1 → identity DARE; trim_density=0.5 keeps the top half by |·|.
        // Top 2 are 3 and 4 → others zeroed → with k=1, mean trivially = self.
        let expected = vec![0.0_f32, 0.0, 3.0, 4.0];
        assert!(approx_eq_slice(&merged, &expected, 1e-6));
    }

    #[test]
    fn empty_task_vectors_errors() {
        let res = DareTies::merge(&[], &DareTiesConfig::default());
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn density_one_reduces_to_ties_only() {
        // Two task vectors with same sign pattern.
        let a = vec![3.0_f32, 1.0, -1.0, 0.5];
        let b = vec![2.0_f32, 0.5, -0.5, 0.1];
        let cfg = DareTiesConfig {
            density: 1.0,
            trim_density: 0.5,
            seed: 7,
        };
        let merged = DareTies::merge(&[a.clone(), b.clone()], &cfg).expect("merge");
        assert_eq!(merged.len(), a.len());
        // The pipeline with density=1 should produce a finite, non-trivial vector.
        let nonzero = merged.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "expected at least one non-zero coordinate");
    }

    #[test]
    fn trim_density_one_keeps_all_entries() {
        // trim_density=1 → keep every entry after DARE; reduces to DARE-only
        // (plus the sign-consensus + disjoint-mean step).
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let cfg = DareTiesConfig {
            density: 1.0,
            trim_density: 1.0,
            seed: 0,
        };
        let merged = DareTies::merge(&[a.clone(), b.clone()], &cfg).expect("merge");
        // All-positive vectors agree in sign → disjoint mean is the mean.
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x + y) * 0.5).collect();
        assert!(approx_eq_slice(&merged, &expected, 1e-5));
    }

    #[test]
    fn invalid_density_errors() {
        let tv = vec![1.0_f32, 2.0];
        let cfg_zero = DareTiesConfig {
            density: 0.0,
            trim_density: 0.5,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(std::slice::from_ref(&tv), &cfg_zero),
            Err(PeftError::Internal { .. })
        ));
        let cfg_too_high = DareTiesConfig {
            density: 1.5,
            trim_density: 0.5,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(&[tv], &cfg_too_high),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn invalid_trim_density_errors() {
        let tv = vec![1.0_f32, 2.0];
        let cfg_zero = DareTiesConfig {
            density: 0.5,
            trim_density: 0.0,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(std::slice::from_ref(&tv), &cfg_zero),
            Err(PeftError::Internal { .. })
        ));
        let cfg_too_high = DareTiesConfig {
            density: 0.5,
            trim_density: 1.5,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(&[tv], &cfg_too_high),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn length_mismatch_errors() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let cfg = DareTiesConfig::default();
        assert!(matches!(
            DareTies::merge(&[a, b], &cfg),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn deterministic_same_seed() {
        let a = vec![1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let b = vec![0.5_f32, -1.0, 0.0, 2.0, -3.0];
        let cfg = DareTiesConfig {
            density: 0.5,
            trim_density: 0.5,
            seed: 42,
        };
        let r1 = DareTies::merge(&[a.clone(), b.clone()], &cfg).expect("merge");
        let r2 = DareTies::merge(&[a, b], &cfg).expect("merge");
        assert_eq!(r1, r2);
    }

    #[test]
    fn different_seeds_produce_different_outputs() {
        let a = vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let b = vec![0.5_f32, -1.0, 0.0, 2.0, -3.0, 1.5, -2.5, 4.0];
        let cfg_a = DareTiesConfig {
            density: 0.5,
            trim_density: 0.5,
            seed: 1,
        };
        let cfg_b = DareTiesConfig {
            density: 0.5,
            trim_density: 0.5,
            seed: 999,
        };
        let r1 = DareTies::merge(&[a.clone(), b.clone()], &cfg_a).expect("merge");
        let r2 = DareTies::merge(&[a, b], &cfg_b).expect("merge");
        // At least one coordinate should differ under different RNG streams.
        assert!(r1 != r2);
    }

    #[test]
    fn opposing_signs_zero_conflicts() {
        // Two task vectors with exactly opposing signs in every entry.
        // With density=1 and trim_density=1, sign consensus is 0 everywhere →
        // disjoint mean treats sign=0 as "any sign agrees", so contributors are
        // included; the sum cancels exactly so each coordinate is 0.
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![-1.0_f32, -2.0, -3.0];
        let cfg = DareTiesConfig {
            density: 1.0,
            trim_density: 1.0,
            seed: 0,
        };
        let merged = DareTies::merge(&[a, b], &cfg).expect("merge");
        for v in merged {
            assert!(v.abs() < 1e-5, "expected ≈0, got {v}");
        }
    }

    #[test]
    fn all_same_sign_preserved() {
        // Three vectors with identical signs → consensus = +1 everywhere,
        // every contributor is kept, and the merged delta is the mean.
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.5_f32, 2.5, 3.5];
        let c = vec![2.0_f32, 3.0, 4.0];
        let cfg = DareTiesConfig {
            density: 1.0,
            trim_density: 1.0,
            seed: 0,
        };
        let merged = DareTies::merge(&[a.clone(), b.clone(), c.clone()], &cfg).expect("merge");
        let expected: Vec<f32> = (0..3).map(|i| (a[i] + b[i] + c[i]) / 3.0).collect();
        assert!(approx_eq_slice(&merged, &expected, 1e-5));
    }

    #[test]
    fn output_length_matches_input() {
        let n = 17_usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) - 8.0).collect();
        let b: Vec<f32> = (0..n).map(|i| 8.0 - (i as f32)).collect();
        let cfg = DareTiesConfig {
            density: 0.4,
            trim_density: 0.3,
            seed: 11,
        };
        let merged = DareTies::merge(&[a, b], &cfg).expect("merge");
        assert_eq!(merged.len(), n);
    }

    #[test]
    fn empty_first_task_vector_errors() {
        let cfg = DareTiesConfig::default();
        let res = DareTies::merge(&[Vec::<f32>::new()], &cfg);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn density_negative_errors() {
        let tv = vec![1.0_f32, 2.0];
        let cfg = DareTiesConfig {
            density: -0.1,
            trim_density: 0.5,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(&[tv], &cfg),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn trim_density_negative_errors() {
        let tv = vec![1.0_f32, 2.0];
        let cfg = DareTiesConfig {
            density: 0.5,
            trim_density: -0.1,
            seed: 0,
        };
        assert!(matches!(
            DareTies::merge(&[tv], &cfg),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn merged_is_finite() {
        let a = vec![10.0_f32, -20.0, 30.0, -40.0];
        let b = vec![-5.0_f32, 15.0, -25.0, 35.0];
        let cfg = DareTiesConfig {
            density: 0.5,
            trim_density: 0.5,
            seed: 12345,
        };
        let merged = DareTies::merge(&[a, b], &cfg).expect("merge");
        for v in merged {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
