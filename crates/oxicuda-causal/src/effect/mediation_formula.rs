//! Pearl's mediation formula — nonparametric identification of natural
//! direct and indirect effects for a discrete mediator.
//!
//! Reference: Pearl, J. (2001). "Direct and indirect effects." In *Proceedings
//! of the Seventeenth Conference on Uncertainty in Artificial Intelligence*
//! (UAI 2001), 411-420. See also Pearl, J. (2009). *Causality: Models,
//! Reasoning, and Inference* (2nd ed.), §4.5, and VanderWeele, T. J. (2015).
//! *Explanation in Causal Inference*, Chapter 2.
//!
//! # Overview
//!
//! Whereas the Imai-Keele-Tingley estimator ([`super::mediation`]) fits *linear*
//! outcome/mediator models, Pearl's **mediation formula** identifies the natural
//! effects *nonparametrically* by directly plugging in the empirical
//! conditional distributions `P(M = m | T = t)` and conditional means
//! `E[Y | T = t, M = m]`. It is the canonical g-formula expression for
//! mediation and makes the mediator's discreteness explicit by summing over its
//! support.
//!
//! Under the standard sequential-ignorability assumptions (no unmeasured
//! treatment-outcome, treatment-mediator, or mediator-outcome confounding, and
//! no treatment-induced mediator-outcome confounding), Pearl's formula gives the
//! **natural direct effect** (NDE) and **natural indirect effect** (NIE):
//!
//! ```text
//!   NDE = Σ_m [ E[Y | T=1, M=m] − E[Y | T=0, M=m] ] · P(M=m | T=0)
//!
//!   NIE = Σ_m   E[Y | T=1, M=m] · [ P(M=m | T=0) − P(M=m | T=1) ]
//! ```
//!
//! Wait — the standard decomposition (VanderWeele 2015, eq. 2.4) fixes the
//! *reference* arm consistently:
//!
//! ```text
//!   NDE = Σ_m [ E[Y|T=1,M=m] − E[Y|T=0,M=m] ] · P(M=m | T=0)   (effect of T with M at its T=0 distribution)
//!   NIE = Σ_m   E[Y|T=1,M=m] · [ P(M=m|T=1) − P(M=m|T=0) ]     (effect of shifting M from its T=0 to T=1 dist, outcome fixed at T=1)
//!   TE  = NDE + NIE = Σ_m E[Y|T=1,M=m]·P(M=m|T=1) − Σ_m E[Y|T=0,M=m]·P(M=m|T=0)
//! ```
//!
//! This module estimates every conditional ingredient by **stratified empirical
//! averages / frequencies** over a sample of `(T, M, Y)` triples, where `M` is a
//! categorical mediator taking one of `n_levels` discrete values.
//!
//! The **proportion mediated** is `NIE / TE` (returned as `0.0` when the total
//! effect is numerically zero).

use crate::error::{CausalError, CausalResult};

/// Configuration for [`mediation_formula`].
#[derive(Debug, Clone)]
pub struct MediationFormulaConfig {
    /// Number of discrete levels the mediator `M` can take (`M ∈ {0,…,L−1}`).
    /// Must be `≥ 2`.
    pub n_levels: usize,
}

impl Default for MediationFormulaConfig {
    fn default() -> Self {
        Self { n_levels: 2 }
    }
}

/// Output of [`mediation_formula`].
#[derive(Debug, Clone, PartialEq)]
pub struct MediationFormulaResult {
    /// Natural direct effect (effect of `T` not operating through `M`).
    pub nde: f64,
    /// Natural indirect effect (effect of `T` operating through `M`).
    pub nie: f64,
    /// Total effect `= NDE + NIE`.
    pub total_effect: f64,
    /// Proportion of the total effect that is mediated, `NIE / TE`
    /// (reported as `0.0` when `|TE|` is below `1e-12`).
    pub proportion_mediated: f64,
    /// Estimated `E[Y | T=t, M=m]` as a flat `2 × n_levels` matrix in row-major
    /// order `[t · n_levels + m]`. Strata with no observations are filled with
    /// `0.0` (and the cell is treated as contributing nothing through its zero
    /// mediator probability mass).
    pub outcome_means: Vec<f64>,
    /// Estimated `P(M=m | T=t)` as a flat `2 × n_levels` matrix `[t·L + m]`.
    pub mediator_dist: Vec<f64>,
}

/// Identify natural direct/indirect effects via Pearl's mediation formula.
///
/// # Parameters
/// - `treatment`: length `n`; each entry in `{0.0, 1.0}`.
/// - `mediator`: length `n`; each entry a mediator level in `0..cfg.n_levels`.
/// - `outcome`: length `n`; the observed outcome `Y`.
/// - `n`: number of samples. Must be `> 0`.
/// - `cfg`: see [`MediationFormulaConfig`].
///
/// # Errors
/// - [`CausalError::EmptyInput`] if `n == 0` or any slice is empty.
/// - [`CausalError::DimensionMismatch`] if the slices' lengths disagree with `n`.
/// - [`CausalError::InvalidParameter`] if `cfg.n_levels < 2`, any
///   `treatment[i] ∉ {0,1}`, or any `mediator[i] ≥ cfg.n_levels`.
pub fn mediation_formula(
    treatment: &[f64],
    mediator: &[usize],
    outcome: &[f64],
    n: usize,
    cfg: &MediationFormulaConfig,
) -> CausalResult<MediationFormulaResult> {
    let levels = cfg.n_levels;
    if n == 0 || treatment.is_empty() || mediator.is_empty() || outcome.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    if treatment.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: treatment.len(),
        });
    }
    if mediator.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: mediator.len(),
        });
    }
    if outcome.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: outcome.len(),
        });
    }
    if levels < 2 {
        return Err(CausalError::InvalidParameter {
            reason: format!("n_levels must be >= 2, got {levels}"),
        });
    }
    for &ti in treatment {
        if !(ti == 0.0 || ti == 1.0) {
            return Err(CausalError::InvalidParameter {
                reason: "treatment must be binary {0,1}".to_string(),
            });
        }
    }
    for &mi in mediator {
        if mi >= levels {
            return Err(CausalError::InvalidParameter {
                reason: format!("mediator level {mi} >= n_levels {levels}"),
            });
        }
    }

    // ---- stratified accumulation -----------------------------------------
    // outcome_sum[t][m], stratum_count[t][m], arm_count[t]
    let cells = 2 * levels;
    let mut outcome_sum = vec![0.0_f64; cells];
    let mut stratum_count = vec![0_usize; cells];
    let mut arm_count = [0_usize; 2];

    for i in 0..n {
        let t = if treatment[i] == 1.0 { 1usize } else { 0usize };
        let m = mediator[i];
        let idx = t * levels + m;
        outcome_sum[idx] += outcome[i];
        stratum_count[idx] += 1;
        arm_count[t] += 1;
    }

    // E[Y | T=t, M=m]
    let mut outcome_means = vec![0.0_f64; cells];
    for idx in 0..cells {
        if stratum_count[idx] > 0 {
            outcome_means[idx] = outcome_sum[idx] / stratum_count[idx] as f64;
        }
    }

    // P(M=m | T=t)
    let mut mediator_dist = vec![0.0_f64; cells];
    for (t, &n_arm) in arm_count.iter().enumerate() {
        if n_arm > 0 {
            for m in 0..levels {
                let idx = t * levels + m;
                mediator_dist[idx] = stratum_count[idx] as f64 / n_arm as f64;
            }
        }
    }

    // ---- Pearl mediation formula -----------------------------------------
    // NDE = Σ_m [ E[Y|1,m] − E[Y|0,m] ] · P(M=m | T=0)
    // NIE = Σ_m   E[Y|1,m] · [ P(M=m|1) − P(M=m|0) ]
    let mut nde = 0.0_f64;
    let mut nie = 0.0_f64;
    for m in 0..levels {
        let y1m = outcome_means[levels + m]; // T=1
        let y0m = outcome_means[m]; // T=0
        let p_m_t0 = mediator_dist[m]; // P(M=m|0)
        let p_m_t1 = mediator_dist[levels + m]; // P(M=m|1)
        nde += (y1m - y0m) * p_m_t0;
        nie += y1m * (p_m_t1 - p_m_t0);
    }

    let total_effect = nde + nie;
    let proportion_mediated = if total_effect.abs() < 1e-12 {
        0.0
    } else {
        nie / total_effect
    };

    Ok(MediationFormulaResult {
        nde,
        nie,
        total_effect,
        proportion_mediated,
        outcome_means,
        mediator_dist,
    })
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(levels: usize) -> MediationFormulaConfig {
        MediationFormulaConfig { n_levels: levels }
    }

    // -------------------- input validation ---------------------------------

    #[test]
    fn n_0_error() {
        let r = mediation_formula(&[], &[], &[], 0, &cfg(2));
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn dim_mismatch_treatment_error() {
        let r = mediation_formula(&[1.0, 0.0], &[0, 1, 0], &[1.0, 2.0, 3.0], 3, &cfg(2));
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn dim_mismatch_mediator_error() {
        let r = mediation_formula(&[1.0, 0.0, 1.0], &[0, 1], &[1.0, 2.0, 3.0], 3, &cfg(2));
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn levels_too_small_error() {
        let r = mediation_formula(&[1.0, 0.0], &[0, 0], &[1.0, 2.0], 2, &cfg(1));
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn non_binary_treatment_error() {
        let r = mediation_formula(&[0.5, 0.0], &[0, 1], &[1.0, 2.0], 2, &cfg(2));
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn mediator_out_of_range_error() {
        // level 2 with n_levels=2 (valid levels are {0,1}).
        let r = mediation_formula(&[1.0, 0.0], &[2, 0], &[1.0, 2.0], 2, &cfg(2));
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    // -------------------- correctness --------------------------------------

    /// No mediation: mediator distribution is identical across treatment arms,
    /// so NIE = 0 and TE = NDE.
    #[test]
    fn no_mediation_nie_zero() {
        // Construct a balanced dataset: for each arm, M is 50/50, and the
        // outcome means differ only through a constant direct effect of T.
        // T=0: M=0→y=1, M=1→y=3 ; T=1: M=0→y=6, M=1→y=8  (direct effect = +5).
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        // 100 controls, 100 treated, each split 50/50 over M.
        for arm in [0.0_f64, 1.0] {
            for level in 0..2usize {
                for _ in 0..50 {
                    t.push(arm);
                    m.push(level);
                    let base = if arm == 0.0 { 0.0 } else { 5.0 };
                    let med = if level == 0 { 1.0 } else { 3.0 };
                    y.push(base + med);
                }
            }
        }
        let n = t.len();
        let res = mediation_formula(&t, &m, &y, n, &cfg(2)).expect("value should be present");
        assert!(res.nie.abs() < 1e-9, "NIE should be 0, got {}", res.nie);
        assert!(
            (res.nde - 5.0).abs() < 1e-9,
            "NDE should be 5, got {}",
            res.nde
        );
        assert!((res.total_effect - 5.0).abs() < 1e-9);
        assert!(res.proportion_mediated.abs() < 1e-9);
    }

    /// Pure mediation: the outcome depends ONLY on the mediator (no direct
    /// effect), but T shifts the mediator distribution → NDE = 0, NIE = TE.
    #[test]
    fn pure_mediation_nde_zero() {
        // Outcome: M=0 → 0, M=1 → 10. No direct effect of T.
        // T=0 → all M=0 ; T=1 → all M=1. So NIE = 10·(1−0) = 10, NDE = 0.
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        for _ in 0..50 {
            t.push(0.0);
            m.push(0usize);
            y.push(0.0);
        }
        for _ in 0..50 {
            t.push(1.0);
            m.push(1usize);
            y.push(10.0);
        }
        let n = t.len();
        let res = mediation_formula(&t, &m, &y, n, &cfg(2)).expect("value should be present");
        assert!(res.nde.abs() < 1e-9, "NDE should be 0, got {}", res.nde);
        assert!(
            (res.nie - 10.0).abs() < 1e-9,
            "NIE should be 10, got {}",
            res.nie
        );
        assert!((res.proportion_mediated - 1.0).abs() < 1e-9);
    }

    /// The decomposition identity NDE + NIE = TE holds exactly, and TE matches
    /// the marginal contrast Σ_m E[Y|1,m]P(M=m|1) − Σ_m E[Y|0,m]P(M=m|0).
    #[test]
    fn decomposition_identity() {
        // Mixed design with partial mediation.
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        // T=0: 70% M=0 (y=2), 30% M=1 (y=5)
        for _ in 0..70 {
            t.push(0.0);
            m.push(0usize);
            y.push(2.0);
        }
        for _ in 0..30 {
            t.push(0.0);
            m.push(1usize);
            y.push(5.0);
        }
        // T=1: 20% M=0 (y=4), 80% M=1 (y=9)  (direct + indirect both present)
        for _ in 0..20 {
            t.push(1.0);
            m.push(0usize);
            y.push(4.0);
        }
        for _ in 0..80 {
            t.push(1.0);
            m.push(1usize);
            y.push(9.0);
        }
        let n = t.len();
        let res = mediation_formula(&t, &m, &y, n, &cfg(2)).expect("value should be present");
        assert!((res.nde + res.nie - res.total_effect).abs() < 1e-9);

        // Marginal contrast directly.
        let e_y1 = 0.2 * 4.0 + 0.8 * 9.0;
        let e_y0 = 0.7 * 2.0 + 0.3 * 5.0;
        let te_marginal = e_y1 - e_y0;
        assert!(
            (res.total_effect - te_marginal).abs() < 1e-9,
            "TE {} vs marginal {}",
            res.total_effect,
            te_marginal
        );
    }

    /// Outputs are finite and the distribution rows sum to 1 within each arm.
    #[test]
    fn outputs_finite_and_dist_normalised() {
        let t = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let m = vec![0usize, 1, 2, 0, 1, 2];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let res = mediation_formula(&t, &m, &y, 6, &cfg(3)).expect("value should be present");
        assert!(res.nde.is_finite());
        assert!(res.nie.is_finite());
        assert!(res.total_effect.is_finite());
        assert!(res.proportion_mediated.is_finite());
        // Each arm's mediator distribution sums to 1.
        for arm in 0..2usize {
            let s: f64 = (0..3).map(|m| res.mediator_dist[arm * 3 + m]).sum();
            assert!((s - 1.0).abs() < 1e-9, "arm {arm} dist sums to {s}");
        }
    }

    /// Output shapes match `2 × n_levels`.
    #[test]
    fn output_shapes() {
        let levels = 4;
        let n = 40;
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let m: Vec<usize> = (0..n).map(|i| i % levels).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let res = mediation_formula(&t, &m, &y, n, &cfg(levels)).expect("value should be present");
        assert_eq!(res.outcome_means.len(), 2 * levels);
        assert_eq!(res.mediator_dist.len(), 2 * levels);
    }

    /// Three-level mediator runs and yields a coherent decomposition.
    #[test]
    fn three_level_mediator() {
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        // T=0 spread across 3 levels, T=1 shifted toward higher levels.
        let t0_levels = [0usize, 0, 1, 1, 2];
        let t1_levels = [1usize, 2, 2, 2, 0];
        for &lvl in &t0_levels {
            for _ in 0..10 {
                t.push(0.0);
                m.push(lvl);
                y.push(lvl as f64); // outcome increases with mediator level
            }
        }
        for &lvl in &t1_levels {
            for _ in 0..10 {
                t.push(1.0);
                m.push(lvl);
                y.push(lvl as f64 + 1.0); // direct bump of +1
            }
        }
        let n = t.len();
        let res = mediation_formula(&t, &m, &y, n, &cfg(3)).expect("value should be present");
        assert!((res.nde + res.nie - res.total_effect).abs() < 1e-9);
        assert!(res.total_effect.is_finite());
    }

    /// Zero total effect → proportion mediated reported as 0 (no NaN).
    #[test]
    fn zero_total_effect_proportion_zero() {
        // T has no effect at all: identical outcome means and mediator dists.
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        for arm in [0.0_f64, 1.0] {
            for level in 0..2usize {
                for _ in 0..25 {
                    t.push(arm);
                    m.push(level);
                    y.push(if level == 0 { 1.0 } else { 2.0 });
                }
            }
        }
        let n = t.len();
        let res = mediation_formula(&t, &m, &y, n, &cfg(2)).expect("value should be present");
        assert!(res.total_effect.abs() < 1e-9);
        assert_eq!(res.proportion_mediated, 0.0);
    }

    /// Deterministic: same inputs → identical results.
    #[test]
    fn deterministic() {
        let t = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let m = vec![0usize, 1, 1, 0, 0, 1, 1, 0];
        let y = vec![3.0, 1.0, 4.0, 2.0, 5.0, 1.5, 4.5, 2.5];
        let r1 = mediation_formula(&t, &m, &y, 8, &cfg(2)).expect("value should be present");
        let r2 = mediation_formula(&t, &m, &y, 8, &cfg(2)).expect("value should be present");
        assert_eq!(r1, r2);
    }

    #[test]
    fn config_default_is_sane() {
        let c = MediationFormulaConfig::default();
        assert_eq!(c.n_levels, 2);
    }
}
