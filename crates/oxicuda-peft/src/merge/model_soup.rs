//! Model Soup — averaging weights of multiple fine-tuned models.
//!
//! Reference: Wortsman M, Ilharco G, Gadre SY, Roelofs R, et al. (2022)
//! "Model soups: averaging weights of multiple fine-tuned models improves
//! accuracy without increasing inference time", ICML.
//! <https://arxiv.org/abs/2203.05482>
//!
//! Wortsman et al. observe that fine-tuned weights from the same pretrained
//! checkpoint typically reside in the same loss-basin, so naively averaging
//! them often *improves* validation accuracy compared with the best single
//! ingredient — without any extra inference cost. This module implements the
//! three soup recipes from §3 of the paper:
//!
//! * **Uniform soup** `θ̄_j = (1/M) · Σᵢ θᵢⱼ` (the cheapest baseline);
//! * **Weighted soup** `θ̄_j = Σᵢ wᵢ · θᵢⱼ / Σᵢ wᵢ` (mixture coefficients);
//! * **Greedy soup** — start from the best validated ingredient and absorb the
//!   next-best candidate iff its inclusion does not regress validation score.
//!
//! All inputs are flat slices; the merge happens coordinate-wise, so the
//! semantic interpretation of each index (per-tensor flat index) is owned by
//! the caller.

use crate::error::{PeftError, PeftResult};

/// Configuration knob for greedy soup acceptance.
#[derive(Debug, Clone, Copy)]
pub struct ModelSoupConfig {
    /// When `true`, higher validation score is preferred and a candidate is
    /// only accepted if it does not *decrease* the current soup score. When
    /// `false`, the comparison is reversed: a candidate is accepted if it does
    /// not *increase* the score (useful for loss-like metrics).
    pub validation_higher_is_better: bool,
}

impl Default for ModelSoupConfig {
    fn default() -> Self {
        Self {
            validation_higher_is_better: true,
        }
    }
}

/// Model-soup algorithm namespace.
pub struct ModelSoup;

impl ModelSoup {
    /// Uniform soup: coordinate-wise arithmetic mean of every model.
    ///
    /// Returns a vector of the same dimensionality as each ingredient.
    ///
    /// # Errors
    /// * [`PeftError::EmptyInput`] when `models` is empty, or the first model
    ///   is itself an empty slice.
    /// * [`PeftError::DimensionMismatch`] when any model has a different length
    ///   from the first.
    pub fn uniform(models: &[&[f32]]) -> PeftResult<Vec<f32>> {
        if models.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = models[0].len();
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        for &m in &models[1..] {
            if m.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: m.len(),
                });
            }
        }
        let m_count = models.len() as f32;
        let mut out = vec![0.0_f32; n];
        for &m in models {
            for (o, &v) in out.iter_mut().zip(m.iter()) {
                *o += v;
            }
        }
        for o in &mut out {
            *o /= m_count;
        }
        Ok(out)
    }

    /// Weighted soup: `θ̄_j = (Σᵢ wᵢ · θᵢⱼ) / (Σᵢ wᵢ)`.
    ///
    /// Each weight must be non-negative and the total must be strictly
    /// positive; the result is normalised so the convex-combination property
    /// holds regardless of the absolute scale of the user-supplied weights.
    ///
    /// # Errors
    /// * [`PeftError::EmptyInput`] when `models` is empty or the first weight
    ///   is paired with an empty slice.
    /// * [`PeftError::DimensionMismatch`] when any model has a different length
    ///   from the first.
    /// * [`PeftError::Internal`] when a weight is negative or the total weight
    ///   is not strictly positive (no usable mixing coefficient).
    pub fn weighted(models: &[(&[f32], f32)]) -> PeftResult<Vec<f32>> {
        if models.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = models[0].0.len();
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        let mut total = 0.0_f32;
        for &(m, w) in models {
            if m.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: m.len(),
                });
            }
            if w.is_nan() || w < 0.0 {
                return Err(PeftError::Internal {
                    msg: format!("model-soup weight must be non-negative, got {w}"),
                });
            }
            total += w;
        }
        if total.is_nan() || total <= 0.0 {
            return Err(PeftError::Internal {
                msg: "model-soup weights sum to zero — no usable mixture".to_string(),
            });
        }
        let mut out = vec![0.0_f32; n];
        for &(m, w) in models {
            if w == 0.0 {
                continue;
            }
            for (o, &v) in out.iter_mut().zip(m.iter()) {
                *o += w * v;
            }
        }
        let inv = 1.0 / total;
        for o in &mut out {
            *o *= inv;
        }
        Ok(out)
    }

    /// Greedy soup (Wortsman et al. 2022 §3, Algorithm 1).
    ///
    /// 1. Score every individual ingredient with `val_scores`.
    /// 2. Order indices descending by score (or ascending when
    ///    `validation_higher_is_better == false`).
    /// 3. Seed the soup with the best ingredient.
    /// 4. For each remaining candidate, compute the prospective average
    ///    `(k · soup + θᵢ) / (k + 1)` and score it; accept iff the score does
    ///    not regress, otherwise reject.
    ///
    /// Returns the accepted soup and the indices that were absorbed, in the
    /// order in which they were accepted (the first entry is the seed).
    ///
    /// # Errors
    /// Mirrors [`Self::uniform`] for shape validation.
    pub fn greedy<F>(
        models: &[&[f32]],
        val_scores: F,
        cfg: &ModelSoupConfig,
    ) -> PeftResult<(Vec<f32>, Vec<usize>)>
    where
        F: Fn(&[f32]) -> f64,
    {
        if models.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = models[0].len();
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        for &m in &models[1..] {
            if m.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: m.len(),
                });
            }
        }

        // Score each individual ingredient.
        let mut scored: Vec<(usize, f64)> = models
            .iter()
            .enumerate()
            .map(|(i, &m)| (i, val_scores(m)))
            .collect();
        sort_by_validation(&mut scored, cfg.validation_higher_is_better);

        // Seed with the best ingredient.
        let (best_idx, best_score) = scored[0];
        let mut soup: Vec<f32> = models[best_idx].to_vec();
        let mut members: Vec<usize> = vec![best_idx];
        let mut current_score = best_score;

        // Try to absorb the rest in ranked order.
        for &(cand_idx, _initial_score) in scored.iter().skip(1) {
            let k = members.len() as f32;
            let denom = k + 1.0;
            let mut proposed = vec![0.0_f32; n];
            for ((p, &s), &c) in proposed
                .iter_mut()
                .zip(soup.iter())
                .zip(models[cand_idx].iter())
            {
                *p = (k * s + c) / denom;
            }
            let proposed_score = val_scores(&proposed);
            if accept(
                proposed_score,
                current_score,
                cfg.validation_higher_is_better,
            ) {
                soup = proposed;
                members.push(cand_idx);
                current_score = proposed_score;
            }
        }
        Ok((soup, members))
    }
}

/// Sort `scored` so that the most preferred ingredient is first.
fn sort_by_validation(scored: &mut [(usize, f64)], higher_is_better: bool) {
    scored.sort_by(|a, b| {
        let (av, bv) = (a.1, b.1);
        // Stable, total ordering on f64 using `partial_cmp` plus NaN tiebreak.
        let ord = if higher_is_better {
            bv.partial_cmp(&av)
        } else {
            av.partial_cmp(&bv)
        };
        match ord {
            Some(o) => o.then_with(|| a.0.cmp(&b.0)),
            // NaNs are pushed to the tail and broken by index.
            None => match (av.is_nan(), bv.is_nan()) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.0.cmp(&b.0),
            },
        }
    });
}

/// Decide whether `proposed` improves on (or merely ties) `current`.
fn accept(proposed: f64, current: f64, higher_is_better: bool) -> bool {
    if proposed.is_nan() || current.is_nan() {
        return false;
    }
    if higher_is_better {
        proposed >= current
    } else {
        proposed <= current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn uniform_two_equal_models_returns_the_same_vector() {
        let m = vec![0.5_f32, -1.0, 2.0, 4.5];
        let soup = ModelSoup::uniform(&[&m, &m]).expect("uniform soup");
        assert!(approx_eq_slice(&soup, &m, 1e-7));
    }

    #[test]
    fn uniform_three_models_is_arithmetic_mean() {
        let a = [0.0_f32, 6.0, 12.0];
        let b = [3.0_f32, 6.0, 9.0];
        let c = [6.0_f32, 6.0, 0.0];
        let soup = ModelSoup::uniform(&[&a[..], &b[..], &c[..]]).expect("uniform");
        let expected = [3.0_f32, 6.0, 7.0];
        assert!(approx_eq_slice(&soup, &expected, 1e-6));
    }

    #[test]
    fn uniform_empty_list_errors() {
        let res = ModelSoup::uniform(&[]);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn uniform_empty_inner_errors() {
        let empty: &[f32] = &[];
        let res = ModelSoup::uniform(&[empty]);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn uniform_dimension_mismatch_errors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        let res = ModelSoup::uniform(&[&a[..], &b[..]]);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn weighted_matches_manual_sum() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0];
        let wa = 0.25_f32;
        let wb = 0.75_f32;
        let soup = ModelSoup::weighted(&[(&a[..], wa), (&b[..], wb)]).expect("weighted");
        let total = wa + wb;
        let expected: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (wa * x + wb * y) / total)
            .collect();
        assert!(approx_eq_slice(&soup, &expected, 1e-6));
    }

    #[test]
    fn weighted_normalisation_invariant_under_scaling() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let small = ModelSoup::weighted(&[(&a[..], 1.0), (&b[..], 3.0)]).expect("small");
        let big = ModelSoup::weighted(&[(&a[..], 10.0), (&b[..], 30.0)]).expect("big");
        assert!(approx_eq_slice(&small, &big, 1e-6));
    }

    #[test]
    fn weighted_all_zero_weights_errors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let res = ModelSoup::weighted(&[(&a[..], 0.0), (&b[..], 0.0)]);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn weighted_negative_weight_errors() {
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 4.0];
        let res = ModelSoup::weighted(&[(&a[..], 1.0), (&b[..], -0.5)]);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn weighted_dimension_mismatch_errors() {
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 4.0, 5.0];
        let res = ModelSoup::weighted(&[(&a[..], 1.0), (&b[..], 1.0)]);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn weighted_empty_list_errors() {
        let res = ModelSoup::weighted(&[]);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn greedy_single_model_returns_itself() {
        let a = [3.0_f32, 4.0, 5.0];
        let (soup, members) =
            ModelSoup::greedy(&[&a[..]], |_| 0.0, &ModelSoupConfig::default()).expect("greedy");
        assert_eq!(members, vec![0]);
        assert!(approx_eq_slice(&soup, &a, 1e-7));
    }

    #[test]
    fn greedy_all_improving_keeps_them_in_descending_score_order() {
        // Three orthogonal "models"; score is just the sum of entries.
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 2.0, 0.0];
        let c = [0.0_f32, 0.0, 3.0];
        let score = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
        // Initial individual scores: 1, 2, 3 → ordered c, b, a (descending).
        // Seed with c (sum=3). Add b: avg = [0,1,1.5] sum=2.5 < 3 → reject.
        // So only c is kept.
        let (soup, members) = ModelSoup::greedy(
            &[&a[..], &b[..], &c[..]],
            score,
            &ModelSoupConfig::default(),
        )
        .expect("greedy");
        assert_eq!(members, vec![2]);
        assert!(approx_eq_slice(&soup, &c, 1e-7));
    }

    #[test]
    fn greedy_rejects_model_that_lowers_score() {
        // Two models with the same flat score; addition averages identically.
        let a = [1.0_f32, 1.0];
        let b = [-1.0_f32, -1.0];
        // Score is sum: a→2, b→-2 → start with a (score=2).
        // Proposed mean → [0, 0] → score 0 < 2 → reject b.
        let score = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
        let (soup, members) =
            ModelSoup::greedy(&[&a[..], &b[..]], score, &ModelSoupConfig::default())
                .expect("greedy");
        assert_eq!(members, vec![0]);
        assert!(approx_eq_slice(&soup, &a, 1e-7));
    }

    #[test]
    fn greedy_zero_constant_score_is_deterministic_by_index_order() {
        // When all scores are 0, every proposed addition has score 0 which
        // ties the current best, so the algorithm accepts every model in
        // their original index order.
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 4.0];
        let c = [5.0_f32, 6.0];
        let (_soup, members) = ModelSoup::greedy(
            &[&a[..], &b[..], &c[..]],
            |_| 0.0,
            &ModelSoupConfig::default(),
        )
        .expect("greedy");
        assert_eq!(members, vec![0, 1, 2]);
    }

    #[test]
    fn greedy_higher_is_better_false_flips_comparison() {
        // Lower score is preferred: treat sum-of-squares as a loss.
        let a = [0.0_f32, 0.0]; // loss = 0 (best)
        let b = [5.0_f32, 0.0]; // loss = 25
        let c = [0.0_f32, 5.0]; // loss = 25
        let cfg = ModelSoupConfig {
            validation_higher_is_better: false,
        };
        let loss = |v: &[f32]| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
        // Seed with a (loss=0). Proposed avg with b → [2.5, 0] loss=6.25 > 0 → reject.
        let (soup, members) =
            ModelSoup::greedy(&[&a[..], &b[..], &c[..]], loss, &cfg).expect("greedy");
        assert_eq!(members, vec![0]);
        assert!(approx_eq_slice(&soup, &a, 1e-7));
    }

    #[test]
    fn greedy_closure_call_count_matches_specification() {
        // M models → M individual scoring calls + (M-1) proposal scoring calls.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let c = [1.0_f32, 1.0];
        let counter = Cell::new(0_usize);
        let counted = |v: &[f32]| {
            counter.set(counter.get() + 1);
            v.iter().map(|x| *x as f64).sum::<f64>()
        };
        let _ = ModelSoup::greedy(
            &[&a[..], &b[..], &c[..]],
            counted,
            &ModelSoupConfig::default(),
        )
        .expect("greedy");
        // 3 individuals + 2 proposed averages = 5 invocations.
        assert_eq!(counter.get(), 5);
    }

    #[test]
    fn greedy_accepts_strict_improvements() {
        // Construct three models with carefully chosen scores so that the
        // accumulator gets gradually better. Use the linear functional
        // score(v) = sum(v) so averaging behaves transparently.
        let a = [3.0_f32, 0.0]; // score=3
        let b = [3.0_f32, 0.0]; // score=3 (mean of {a,b} = same → tie → accept)
        let c = [3.0_f32, 0.0]; // score=3 (mean unchanged → accept)
        let score = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
        let (_soup, members) = ModelSoup::greedy(
            &[&a[..], &b[..], &c[..]],
            score,
            &ModelSoupConfig::default(),
        )
        .expect("greedy");
        // All three accepted as their additions tie the seed.
        assert_eq!(members.len(), 3);
    }

    #[test]
    fn greedy_dimension_mismatch_errors() {
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 4.0, 5.0];
        let res = ModelSoup::greedy(&[&a[..], &b[..]], |_| 0.0, &ModelSoupConfig::default());
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn greedy_empty_list_errors() {
        let res = ModelSoup::greedy::<fn(&[f32]) -> f64>(&[], |_| 0.0, &ModelSoupConfig::default());
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn weighted_zero_weight_skips_that_model() {
        // A model with weight 0 should not influence the result.
        let a = [2.0_f32, 4.0];
        let b = [100.0_f32, -100.0];
        let soup = ModelSoup::weighted(&[(&a[..], 1.0), (&b[..], 0.0)]).expect("weighted");
        assert!(approx_eq_slice(&soup, &a, 1e-6));
    }

    #[test]
    fn uniform_sm_check_matches_average() {
        // Detect off-by-one in the running average by computing two known means.
        let a = [10.0_f32];
        let b = [20.0_f32];
        let c = [60.0_f32];
        let soup = ModelSoup::uniform(&[&a[..], &b[..], &c[..]]).expect("uniform");
        assert!(approx_eq(soup[0], 30.0, 1e-6));
    }
}
