//! Resonator Network with attention-based (soft) unbinding.
//!
//! This is a *soft* relaxation of the hard-argmax VSA Resonator Network
//! (Frady-Kent-Olshausen-Sommer, NeurIPS 2020; see [`crate::memory::resonator`]).
//! The hard resonator snaps each per-role estimate to the single best codebook
//! item every iteration (a hard nearest-neighbour cleanup). The *attention*
//! variant instead forms a temperature-scaled softmax over the similarities of
//! the unbound probe to **every** codebook item and sets the new estimate to the
//! attention-weighted *superposition* of the codebook hypervectors — a soft
//! cleanup. This mirrors attention as soft key-value retrieval (queries = the
//! unbound probe, keys = codebook HVs, values = the same codebook HVs), and is
//! the continuous-relaxation view discussed by Kent et al. (2020) and
//! Frady-Kent-Olshausen-Sommer (2020).
//!
//! ## Algorithm (each iteration, for each role `i`)
//!
//! ```text
//! residual = composite - Σ_{j ≠ i} bind(estimates[j], roles[j])
//! probe    = unbind(roles[i], residual)              // circular correlation
//! score_m  = cosine(probe, codebook[i][m].hv)        // for every item m
//! w        = softmax(score / temperature)            // numerically stable
//! estimate_i = normalize( Σ_m w_m · codebook[i][m].hv )   // soft superposition
//! id_i     = argmax_m w_m                             // readout only
//! ```
//!
//! Convergence is declared when the per-role argmax id vector is unchanged
//! across two consecutive iterations (identical criterion to the hard version).
//!
//! ## Temperature → 0 limit (honest framing)
//!
//! This file implements a genuine soft attention update: `estimate_i` is a true
//! weighted sum over the codebook, not a disguised argmax. As `temperature → 0⁺`
//! the softmax becomes a one-hot vector on the highest-scoring item, so the soft
//! superposition collapses onto that single codebook HV and the update
//! *recovers* the hard resonator (modulo the final L2 renormalisation, which the
//! hard version does not need because stored HVs are already unit-norm). As
//! `temperature → ∞` the weights become uniform and the estimate tends to the
//! (normalised) mean of the codebook — the least informative cleanup. Practical
//! values sit in between; smaller temperatures sharpen the cleanup and behave
//! more like the hard resonator.
//!
//! ## Representation
//!
//! Operates on real-valued (HRR) hypervectors `Vec<f32>` of even length, reusing
//! [`crate::vector::hrr`] primitives (`hrr_bind`, `hrr_unbind`, `hrr_cosine`,
//! `hrr_normalize`). Because the public [`crate::vector::hrr::HrrItemMemory`]
//! exposes no iterator over its stored items (only nearest-neighbour queries),
//! attention — which must score *all* items — takes the codebooks as explicit
//! per-role lists `&[Vec<(usize, Vec<f32>)>]` of `(id, hv)` pairs. This keeps the
//! implementation independent of any private memory internals.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::hrr::{hrr_bind, hrr_cosine, hrr_normalize, hrr_unbind};

// ── Softmax helper ──────────────────────────────────────────────────────────

/// Numerically stable temperature-scaled softmax.
///
/// Returns `w_m = exp((s_m - max_s) / temperature) / Σ_k exp((s_k - max_s) / temperature)`.
/// Subtracting `max_s` before exponentiating prevents overflow for large scores
/// without changing the result. The returned weights are non-negative and sum to
/// 1 (up to floating-point rounding). Smaller `temperature` sharpens the
/// distribution towards a one-hot at the maximum score; larger values flatten it
/// towards uniform.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `scores` is empty.
/// - [`HdcError::InvalidProbability`] if `temperature` is not finite or not `> 0`.
pub fn softmax_stable(scores: &[f32], temperature: f32) -> HdcResult<Vec<f32>> {
    if scores.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(HdcError::InvalidProbability(temperature as f64));
    }

    // Stable shift by the maximum score.
    let mut max_score = f32::NEG_INFINITY;
    for &s in scores {
        if s > max_score {
            max_score = s;
        }
    }

    let inv_temp = 1.0f32 / temperature;
    let mut exps: Vec<f32> = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &s in scores {
        let e = (((s - max_score) * inv_temp) as f64).exp() as f32;
        exps.push(e);
        sum += e;
    }

    // `sum` is at least `exp(0) == 1` from the max element, so it is strictly
    // positive and the division below is safe. Guard defensively regardless.
    if sum < 1e-30 {
        // Degenerate (should be unreachable): fall back to a uniform distribution.
        let uniform = 1.0f32 / scores.len() as f32;
        return Ok(vec![uniform; scores.len()]);
    }

    let inv_sum = 1.0f32 / sum;
    for w in exps.iter_mut() {
        *w *= inv_sum;
    }
    Ok(exps)
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for an [`AttentionResonator::decompose`] run.
#[derive(Debug, Clone)]
pub struct AttentionResonatorConfig {
    /// Number of roles (and fillers) in the composite HV. Must be `> 0`.
    pub n_roles: usize,
    /// Maximum number of fixed-point iterations.
    pub max_iter: usize,
    /// Softmax temperature (`> 0`). Smaller values sharpen the attention towards
    /// the hard-argmax cleanup; as `temperature → 0⁺` the update recovers the
    /// hard resonator. Larger values flatten attention towards the codebook mean.
    pub temperature: f32,
}

impl Default for AttentionResonatorConfig {
    fn default() -> Self {
        Self {
            n_roles: 2,
            max_iter: 100,
            temperature: 0.1,
        }
    }
}

impl AttentionResonatorConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `n_roles == 0`.
    /// - [`HdcError::InvalidProbability`] if `temperature` is not finite or not `> 0`.
    pub fn validate(&self) -> HdcResult<()> {
        if self.n_roles == 0 {
            return Err(HdcError::EmptyInput);
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(HdcError::InvalidProbability(self.temperature as f64));
        }
        Ok(())
    }
}

// ── Result ──────────────────────────────────────────────────────────────────

/// Result of an [`AttentionResonator::decompose`] call.
#[derive(Debug, Clone)]
pub struct AttentionResonatorResult {
    /// Filler IDs (one per role, in role order): the argmax id of each role's
    /// final attention distribution, for discrete readout.
    pub filler_ids: Vec<usize>,
    /// Final soft filler estimates (one per role, in role order): the
    /// attention-weighted, L2-normalised codebook superpositions.
    pub filler_hvs: Vec<Vec<f32>>,
    /// Number of iterations actually executed.
    pub n_iter: usize,
    /// Whether the network converged (argmax ids stable) before `max_iter`.
    pub converged: bool,
}

// ── Attention Resonator ─────────────────────────────────────────────────────

/// Stateless attention-based resonator network.
///
/// All state is local to each [`AttentionResonator::decompose`] call. The soft
/// cleanup replaces the hard resonator's per-role argmax with a temperature-scaled
/// softmax superposition over the whole codebook; see the module docs for the
/// soft-vs-hard relationship and the `temperature → 0` limit.
pub struct AttentionResonator;

impl AttentionResonator {
    /// Decompose a composite HV into role-filler pairs via soft (attention) cleanup.
    ///
    /// # Parameters
    ///
    /// - `composite`: the superposition `Σ bind(filler_i, role_i)`, length `dim`.
    /// - `roles`: `n_roles` role HVs, each of length `dim`.
    /// - `codebooks`: `n_roles` explicit candidate lists; `codebooks[i]` is a
    ///   non-empty `Vec<(id, hv)>` of candidate fillers for role `i`, each `hv`
    ///   of length `dim`. Explicit lists are used (rather than `HrrItemMemory`)
    ///   so attention can score *every* item — the public memory API exposes no
    ///   such iterator.
    /// - `cfg`: configuration (`n_roles`, `max_iter`, `temperature`).
    /// - `rng`: used to seed the initial soft estimates with random HVs.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `cfg.n_roles == 0`.
    /// - [`HdcError::InvalidProbability`] if `cfg.temperature` is not finite / not `> 0`.
    /// - [`HdcError::ZeroDimension`] if `composite` is empty.
    /// - [`HdcError::DimensionMismatch`] if `roles.len()` or `codebooks.len()`
    ///   differ from `cfg.n_roles`, or if any role / codebook HV has the wrong length.
    /// - [`HdcError::EmptyItemMemory`] if any codebook is empty.
    pub fn decompose(
        composite: &[f32],
        roles: &[Vec<f32>],
        codebooks: &[Vec<(usize, Vec<f32>)>],
        cfg: &AttentionResonatorConfig,
        rng: &mut LcgRng,
    ) -> HdcResult<AttentionResonatorResult> {
        // ── Validate config ───────────────────────────────────────────────────
        cfg.validate()?;

        if composite.is_empty() {
            return Err(HdcError::ZeroDimension);
        }
        let dim = composite.len();

        if roles.len() != cfg.n_roles {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.n_roles,
                got: roles.len(),
            });
        }
        if codebooks.len() != cfg.n_roles {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.n_roles,
                got: codebooks.len(),
            });
        }

        // ── Validate roles and codebooks ──────────────────────────────────────
        for role in roles.iter() {
            if role.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: role.len(),
                });
            }
        }
        for cb in codebooks.iter() {
            if cb.is_empty() {
                return Err(HdcError::EmptyItemMemory);
            }
            for (_, hv) in cb.iter() {
                if hv.len() != dim {
                    return Err(HdcError::DimensionMismatch {
                        expected: dim,
                        got: hv.len(),
                    });
                }
            }
        }

        // ── Initialise estimates with random HVs ──────────────────────────────
        let mut estimates: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_roles);
        for _ in 0..cfg.n_roles {
            estimates.push(Self::random_estimate(dim, rng));
        }

        // ── Initial argmax-id snapshot (one soft sweep) ───────────────────────
        let mut prev_ids: Vec<usize> = Vec::with_capacity(cfg.n_roles);
        for i in 0..cfg.n_roles {
            let (id, hv) =
                Self::update_role_soft(composite, i, roles, &estimates, &codebooks[i], cfg)?;
            estimates[i] = hv;
            prev_ids.push(id);
        }

        // Early exit when no iteration budget remains.
        if cfg.max_iter == 0 {
            return Ok(AttentionResonatorResult {
                filler_ids: prev_ids,
                filler_hvs: estimates,
                n_iter: 0,
                converged: false,
            });
        }

        // ── Fixed-point iteration ─────────────────────────────────────────────
        let mut n_iter = 0usize;
        let mut converged = false;

        for _iter in 0..cfg.max_iter {
            let mut new_ids: Vec<usize> = Vec::with_capacity(cfg.n_roles);

            for i in 0..cfg.n_roles {
                let (id, hv) =
                    Self::update_role_soft(composite, i, roles, &estimates, &codebooks[i], cfg)?;
                estimates[i] = hv;
                new_ids.push(id);
            }

            n_iter += 1;

            if Self::ids_converged(&prev_ids, &new_ids) {
                converged = true;
                prev_ids = new_ids;
                break;
            }
            prev_ids = new_ids;
        }

        Ok(AttentionResonatorResult {
            filler_ids: prev_ids,
            filler_hvs: estimates,
            n_iter,
            converged,
        })
    }

    /// Perform one soft (attention) cleanup for role `role_idx`.
    ///
    /// Returns `(argmax_id, soft_estimate)` where `argmax_id` is the id of the
    /// codebook item carrying the largest attention weight (for readout) and
    /// `soft_estimate` is the L2-normalised attention-weighted superposition of
    /// the codebook hypervectors.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if any HV has the wrong length.
    /// - [`HdcError::EmptyItemMemory`] if `codebook` is empty.
    /// - [`HdcError::InvalidProbability`] propagated from the softmax on a bad temperature.
    pub fn update_role_soft(
        composite: &[f32],
        role_idx: usize,
        all_roles: &[Vec<f32>],
        current_estimates: &[Vec<f32>],
        codebook: &[(usize, Vec<f32>)],
        cfg: &AttentionResonatorConfig,
    ) -> HdcResult<(usize, Vec<f32>)> {
        if codebook.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let dim = composite.len();

        // 1) residual = composite - Σ_{j ≠ role_idx} bind(estimates[j], roles[j]).
        let residual = Self::residual_probe(composite, role_idx, all_roles, current_estimates)?;
        // 2) probe = unbind(roles[role_idx], residual).
        let probe = hrr_unbind(&all_roles[role_idx], &residual)?;

        // 3) score every codebook item by cosine similarity to the probe.
        let mut scores: Vec<f32> = Vec::with_capacity(codebook.len());
        for (_, hv) in codebook.iter() {
            if hv.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: hv.len(),
                });
            }
            scores.push(hrr_cosine(&probe, hv)?);
        }

        // 4) attention weights via numerically stable temperature softmax.
        let weights = softmax_stable(&scores, cfg.temperature)?;

        // 5) soft estimate = Σ_m w_m · hv_m, then L2-normalise.
        let mut estimate = vec![0.0f32; dim];
        for ((_, hv), &w) in codebook.iter().zip(weights.iter()) {
            for (e, &h) in estimate.iter_mut().zip(hv.iter()) {
                *e += w * h;
            }
        }
        // Normalise the soft superposition so its scale matches a unit-norm HV
        // (and so the next residual subtraction is well-conditioned). If every
        // weight collapsed onto a single unit-norm item the norm is already ~1.
        // A zero-norm superposition is essentially unreachable here (weights sum
        // to 1 over unit-norm HVs); fall back to the raw estimate if normalise
        // reports a degenerate norm rather than erroring out.
        if hrr_normalize(&mut estimate).is_err() {
            // Leave `estimate` as the (un-normalised) weighted sum.
        }

        // 6) argmax id = id of the item with the largest attention weight.
        let mut best_idx = 0usize;
        let mut best_w = f32::NEG_INFINITY;
        for (idx, &w) in weights.iter().enumerate() {
            if w > best_w {
                best_w = w;
                best_idx = idx;
            }
        }
        let best_id = codebook[best_idx].0;

        Ok((best_id, estimate))
    }

    /// Check whether two argmax-id vectors are identical (convergence criterion).
    pub fn ids_converged(ids_a: &[usize], ids_b: &[usize]) -> bool {
        ids_a.len() == ids_b.len() && ids_a.iter().zip(ids_b.iter()).all(|(a, b)| a == b)
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Compute `residual = composite - Σ_{j ≠ role_idx} bind(estimates[j], roles[j])`.
    ///
    /// Mirrors the hard resonator's `residual_probe`.
    fn residual_probe(
        composite: &[f32],
        role_idx: usize,
        all_roles: &[Vec<f32>],
        estimates: &[Vec<f32>],
    ) -> HdcResult<Vec<f32>> {
        let dim = composite.len();
        let mut residual = composite.to_vec();

        for (j, (est, role)) in estimates.iter().zip(all_roles.iter()).enumerate() {
            if j == role_idx {
                continue;
            }
            if est.len() != dim || role.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: if est.len() != dim {
                        est.len()
                    } else {
                        role.len()
                    },
                });
            }
            let contribution = hrr_bind(est, role)?;
            for (r, c) in residual.iter_mut().zip(contribution.iter()) {
                *r -= c;
            }
        }
        Ok(residual)
    }

    /// Generate a random estimate vector (uniform components in `[-1, 1)`).
    fn random_estimate(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(rng.next_f32() * 2.0 - 1.0);
        }
        v
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::hrr::{hrr_bind, random_hrr};

    fn rng() -> LcgRng {
        LcgRng::new(0x1234_5678_ABCD_EF01)
    }

    /// Build a composite HV `s = Σ bind(filler_i, role_i)`.
    fn build_composite(fillers: &[Vec<f32>], roles: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
        let dim = fillers[0].len();
        let mut acc = vec![0f32; dim];
        for (f, r) in fillers.iter().zip(roles.iter()) {
            let bound = hrr_bind(f, r)?;
            for (a, b) in acc.iter_mut().zip(bound.iter()) {
                *a += b;
            }
        }
        Ok(acc)
    }

    // ── softmax_stable ───────────────────────────────────────────────────────

    #[test]
    fn softmax_stable_sums_to_one() {
        let scores = vec![1.0f32, 2.0, 3.0, 0.5, -4.0];
        let w = softmax_stable(&scores, 0.7).expect("softmax");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        for &p in &w {
            assert!(p >= 0.0, "weight must be non-negative: {p}");
        }
    }

    #[test]
    fn softmax_stable_uniform_for_equal_scores() {
        let scores = vec![2.5f32; 6];
        let w = softmax_stable(&scores, 1.0).expect("softmax");
        let expected = 1.0f32 / 6.0;
        for &p in &w {
            assert!(
                (p - expected).abs() < 1e-6,
                "expected uniform {expected}, got {p}"
            );
        }
    }

    #[test]
    fn softmax_stable_sharpens_as_temperature_shrinks() {
        let scores = vec![1.0f32, 2.0, 3.0];
        let warm = softmax_stable(&scores, 5.0).expect("warm");
        let cold = softmax_stable(&scores, 0.05).expect("cold");
        // Index 2 is the max; a smaller temperature must concentrate more mass there.
        assert!(
            cold[2] > warm[2],
            "cold max weight {} should exceed warm {}",
            cold[2],
            warm[2]
        );
        // In the cold limit the max weight should be essentially 1.
        assert!(cold[2] > 0.99, "cold max weight not sharp: {}", cold[2]);
        // The warm distribution must remain meaningfully spread.
        assert!(warm[0] > 0.1, "warm distribution too peaked: {}", warm[0]);
    }

    #[test]
    fn softmax_stable_bad_temperature_error() {
        let scores = vec![1.0f32, 2.0];
        assert!(matches!(
            softmax_stable(&scores, 0.0),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            softmax_stable(&scores, -1.0),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            softmax_stable(&scores, f32::NAN),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            softmax_stable(&scores, f32::INFINITY),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn softmax_stable_empty_error() {
        let scores: Vec<f32> = vec![];
        assert!(matches!(
            softmax_stable(&scores, 1.0),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn softmax_stable_large_scores_no_overflow() {
        // Without the max-subtraction these would overflow to inf/NaN.
        let scores = vec![1.0e3f32, 2.0e3, 3.0e3];
        let w = softmax_stable(&scores, 1.0).expect("softmax");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        assert!(w.iter().all(|p| p.is_finite()), "weights must be finite");
        assert!(w[2] > 0.99, "max score should dominate: {}", w[2]);
    }

    // ── Config validation ────────────────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        let cfg = AttentionResonatorConfig::default();
        assert_eq!(cfg.n_roles, 2);
        assert_eq!(cfg.max_iter, 100);
        assert!(cfg.temperature > 0.0);
        cfg.validate().expect("default config must validate");
    }

    #[test]
    fn config_zero_roles_error() {
        let cfg = AttentionResonatorConfig {
            n_roles: 0,
            max_iter: 10,
            temperature: 0.1,
        };
        assert!(matches!(cfg.validate(), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn config_bad_temperature_error() {
        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            temperature: 0.0,
        };
        assert!(matches!(
            cfg.validate(),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    // ── 1-role single-item decomposition ─────────────────────────────────────

    #[test]
    fn decompose_one_role_single_item_returns_id() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let codebook = vec![vec![(42usize, filler.clone())]];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(99);
        let result = AttentionResonator::decompose(&composite, &[role], &codebook, &cfg, &mut rng2)
            .expect("decompose");

        assert_eq!(result.filler_ids.len(), 1);
        assert_eq!(result.filler_ids[0], 42);
        assert_eq!(result.filler_hvs.len(), 1);
        assert_eq!(result.filler_hvs[0].len(), dim);
    }

    // ── 2-role small-temperature recovery (the key test) ─────────────────────

    #[test]
    fn decompose_two_roles_small_temperature_recovers_fillers() {
        let mut rng = rng();
        let dim = 256;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");
        // Distractors to make the cleanup non-trivial.
        let other0 = random_hrr(dim, &mut rng).expect("o0");
        let other1 = random_hrr(dim, &mut rng).expect("o1");

        let composite = build_composite(
            &[filler0.clone(), filler1.clone()],
            &[role0.clone(), role1.clone()],
        )
        .expect("composite");

        let cb0 = vec![(0usize, filler0.clone()), (10usize, other0.clone())];
        let cb1 = vec![(1usize, filler1.clone()), (11usize, other1.clone())];

        let cfg = AttentionResonatorConfig {
            n_roles: 2,
            max_iter: 100,
            temperature: 0.02, // small → close to hard argmax
        };
        let mut rng2 = LcgRng::new(7);
        let result = AttentionResonator::decompose(
            &composite,
            &[role0, role1],
            &[cb0, cb1],
            &cfg,
            &mut rng2,
        )
        .expect("decompose");

        assert_eq!(result.filler_ids[0], 0, "role 0 should recover filler id 0");
        assert_eq!(result.filler_ids[1], 1, "role 1 should recover filler id 1");
        assert!(result.converged, "clean problem should converge");
    }

    // ── Converged flag on a clean problem ────────────────────────────────────

    #[test]
    fn decompose_converged_true_on_clean_problem() {
        let mut rng = rng();
        let dim = 128;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");

        let composite = build_composite(
            &[filler0.clone(), filler1.clone()],
            &[role0.clone(), role1.clone()],
        )
        .expect("composite");

        let cb0 = vec![(0usize, filler0.clone())];
        let cb1 = vec![(1usize, filler1.clone())];

        let cfg = AttentionResonatorConfig {
            n_roles: 2,
            max_iter: 100,
            temperature: 0.05,
        };
        let mut rng2 = LcgRng::new(13);
        let result = AttentionResonator::decompose(
            &composite,
            &[role0, role1],
            &[cb0, cb1],
            &cfg,
            &mut rng2,
        )
        .expect("decompose");

        assert!(result.converged);
    }

    // ── n_iter ≤ max_iter ────────────────────────────────────────────────────

    #[test]
    fn decompose_n_iter_le_max_iter() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let codebook = vec![vec![(0usize, filler.clone())]];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 30,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(1);
        let result = AttentionResonator::decompose(&composite, &[role], &codebook, &cfg, &mut rng2)
            .expect("decompose");

        assert!(
            result.n_iter <= 30,
            "n_iter {} exceeds max_iter",
            result.n_iter
        );
    }

    // ── max_iter = 0 returns early ───────────────────────────────────────────

    #[test]
    fn decompose_max_iter_zero_returns_early() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");
        let codebook = vec![vec![(0usize, filler.clone())]];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 0,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(6);
        let result = AttentionResonator::decompose(&composite, &[role], &codebook, &cfg, &mut rng2)
            .expect("decompose");

        assert_eq!(result.n_iter, 0);
        assert!(!result.converged);
    }

    // ── Error: roles.len() mismatch ──────────────────────────────────────────

    #[test]
    fn decompose_roles_len_mismatch_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");
        let cb = vec![vec![(0usize, filler.clone())]];

        let cfg = AttentionResonatorConfig {
            n_roles: 2, // mismatch: only 1 role supplied
            max_iter: 10,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(2);
        let res = AttentionResonator::decompose(&composite, &[role], &cb, &cfg, &mut rng2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Error: codebooks.len() mismatch ──────────────────────────────────────

    #[test]
    fn decompose_codebooks_len_mismatch_error() {
        let mut rng = rng();
        let dim = 64;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role0).expect("bind");
        let cb = vec![vec![(0usize, filler.clone())]]; // only 1 codebook

        let cfg = AttentionResonatorConfig {
            n_roles: 2,
            max_iter: 10,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(3);
        let res = AttentionResonator::decompose(&composite, &[role0, role1], &cb, &cfg, &mut rng2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Error: empty codebook ────────────────────────────────────────────────

    #[test]
    fn decompose_empty_codebook_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");
        let empty_cb: Vec<Vec<(usize, Vec<f32>)>> = vec![vec![]];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(4);
        let res = AttentionResonator::decompose(&composite, &[role], &empty_cb, &cfg, &mut rng2);
        assert!(matches!(res, Err(HdcError::EmptyItemMemory)));
    }

    // ── Error: wrong composite dim ───────────────────────────────────────────

    #[test]
    fn decompose_wrong_composite_dim_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let cb = vec![vec![(0usize, filler.clone())]];
        let composite_wrong = vec![0.0f32; 128]; // mismatched dim

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(5);
        let res = AttentionResonator::decompose(&composite_wrong, &[role], &cb, &cfg, &mut rng2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Error: empty composite ───────────────────────────────────────────────

    #[test]
    fn decompose_empty_composite_error() {
        let role = vec![0.0f32; 64];
        let cb = vec![vec![(0usize, vec![0.0f32; 64])]];
        let composite: Vec<f32> = vec![];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            temperature: 0.1,
        };
        let mut rng2 = LcgRng::new(5);
        let res = AttentionResonator::decompose(&composite, &[role], &cb, &cfg, &mut rng2);
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    // ── ids_converged ────────────────────────────────────────────────────────

    #[test]
    fn ids_converged_equal_and_different() {
        assert!(AttentionResonator::ids_converged(&[0, 1], &[0, 1]));
        assert!(!AttentionResonator::ids_converged(&[0, 1], &[1, 0]));
        assert!(!AttentionResonator::ids_converged(&[0], &[0, 1]));
    }

    // ── Determinism for a fixed seed ─────────────────────────────────────────

    #[test]
    fn decompose_deterministic_for_fixed_seed() {
        let mut rng = rng();
        let dim = 128;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");

        let composite = build_composite(
            &[filler0.clone(), filler1.clone()],
            &[role0.clone(), role1.clone()],
        )
        .expect("composite");

        let cb0 = vec![(0usize, filler0.clone()), (10usize, role0.clone())];
        let cb1 = vec![(1usize, filler1.clone()), (11usize, role1.clone())];

        let cfg = AttentionResonatorConfig {
            n_roles: 2,
            max_iter: 50,
            temperature: 0.1,
        };

        let mut rng_a = LcgRng::new(0xABCD);
        let res_a = AttentionResonator::decompose(
            &composite,
            &[role0.clone(), role1.clone()],
            &[cb0.clone(), cb1.clone()],
            &cfg,
            &mut rng_a,
        )
        .expect("decompose a");

        let mut rng_b = LcgRng::new(0xABCD);
        let res_b = AttentionResonator::decompose(
            &composite,
            &[role0, role1],
            &[cb0, cb1],
            &cfg,
            &mut rng_b,
        )
        .expect("decompose b");

        assert_eq!(res_a.filler_ids, res_b.filler_ids);
        assert_eq!(res_a.n_iter, res_b.n_iter);
        assert_eq!(res_a.converged, res_b.converged);
        for (a, b) in res_a.filler_hvs.iter().zip(res_b.filler_hvs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-9, "non-deterministic HV: {x} vs {y}");
            }
        }
    }

    // ── Soft estimate is unit-norm (normalisation actually applied) ──────────

    #[test]
    fn decompose_soft_estimate_is_unit_norm() {
        let mut rng = rng();
        let dim = 128;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");
        let composite = hrr_bind(&filler0, &role).expect("bind");

        // Multi-item codebook with a moderate temperature so the estimate is a
        // genuine blend, not a one-hot — the normalisation must still hold.
        let cb = vec![vec![(0usize, filler0.clone()), (1usize, filler1.clone())]];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 20,
            temperature: 1.0,
        };
        let mut rng2 = LcgRng::new(21);
        let result = AttentionResonator::decompose(&composite, &[role], &cb, &cfg, &mut rng2)
            .expect("decompose");

        let norm: f32 = result.filler_hvs[0]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "soft estimate not unit-norm: {norm}"
        );
    }

    // ── update_role_soft argmax matches the planted filler ───────────────────

    #[test]
    fn update_role_soft_picks_correct_item() {
        let mut rng = rng();
        let dim = 128;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let distractor = random_hrr(dim, &mut rng).expect("distractor");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let codebook = vec![(7usize, filler.clone()), (9usize, distractor.clone())];
        let estimates = vec![filler.clone()];

        let cfg = AttentionResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            temperature: 0.05,
        };
        let (id, hv) = AttentionResonator::update_role_soft(
            &composite,
            0,
            std::slice::from_ref(&role),
            &estimates,
            &codebook,
            &cfg,
        )
        .expect("update_role_soft");

        assert_eq!(id, 7, "soft cleanup should pick the planted filler id 7");
        assert_eq!(hv.len(), dim);
    }
}
