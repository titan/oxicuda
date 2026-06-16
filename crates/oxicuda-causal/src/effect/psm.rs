//! Propensity-score matching (PSM) for the Average Treatment effect on the
//! Treated (ATT).
//!
//! PSM (Rosenbaum & Rubin 1983 *Biometrika* 70:41) reduces selection bias by
//! pairing each treated unit with one or more control units that have a similar
//! estimated propensity score `ê(x) = P(T=1 | X=x)`. Conditioning on the scalar
//! propensity score is sufficient to balance the (high-dimensional) covariates
//! under the strong-ignorability assumption.
//!
//! This module implements **greedy 1:k nearest-neighbour matching** on the
//! propensity score (the canonical variant), with:
//!
//! * **with / without replacement** control reuse,
//! * an optional **caliper** that discards matches farther than
//!   `caliper · sd(logit ê)` apart (Austin 2011's recommended 0.2 default scale),
//! * matching on either the raw score or the **logit** of the score (the latter
//!   being the statistically preferred metric).
//!
//! The ATT estimate is the mean over matched treated units of
//! `Y_i − mean_k(Y_{matched control})`, and a standardised-mean-difference
//! (SMD) balance diagnostic on the matched score is returned alongside.

use crate::error::{CausalError, CausalResult};

/// Distance metric for propensity matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMetric {
    /// Match on the raw propensity score `ê(x)`.
    Score,
    /// Match on the logit `log(ê / (1−ê))` (recommended; more uniform spacing).
    Logit,
}

/// Configuration for greedy nearest-neighbour propensity matching.
#[derive(Debug, Clone)]
pub struct PsmConfig {
    /// Number of control neighbours matched to each treated unit (`k ≥ 1`).
    pub k_neighbors: usize,
    /// Match with replacement (a control may serve multiple treated units).
    pub with_replacement: bool,
    /// Distance metric.
    pub metric: MatchMetric,
    /// Optional caliper as a multiple of the pooled SD of the matching metric.
    /// `None` disables caliper filtering; a treated unit with no neighbour
    /// inside the caliper is left unmatched.
    pub caliper_sd: Option<f32>,
}

impl Default for PsmConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 1,
            with_replacement: true,
            metric: MatchMetric::Logit,
            caliper_sd: Some(0.2),
        }
    }
}

/// A single treated unit's match record.
#[derive(Debug, Clone)]
pub struct MatchedPair {
    /// Index of the treated unit.
    pub treated_idx: usize,
    /// Indices of the matched control units (length ≤ `k_neighbors`).
    pub control_idxs: Vec<usize>,
    /// Per-unit treatment-effect contribution `Y_t − mean(Y_controls)`.
    pub effect: f32,
}

/// Result of propensity-score matching.
#[derive(Debug, Clone)]
pub struct PsmResult {
    /// Estimated ATT (mean of matched-pair effects).
    pub att: f32,
    /// Matched pairs (one per *successfully matched* treated unit).
    pub matches: Vec<MatchedPair>,
    /// Number of treated units that found ≥1 admissible match.
    pub n_matched: usize,
    /// Number of treated units discarded (no neighbour inside the caliper).
    pub n_unmatched: usize,
    /// Absolute standardised mean difference of the matching metric *after*
    /// matching (a balance diagnostic; smaller is better, < 0.1 is "balanced").
    pub smd_after: f32,
}

#[inline]
fn logit(p: f32) -> f32 {
    let pc = p.clamp(1e-6, 1.0 - 1e-6);
    (pc / (1.0 - pc)).ln()
}

/// Map a propensity score to the chosen matching metric value.
#[inline]
fn metric_value(p: f32, metric: MatchMetric) -> f32 {
    match metric {
        MatchMetric::Score => p,
        MatchMetric::Logit => logit(p),
    }
}

/// Pooled (population) standard deviation of a slice; returns `0` for < 2 items.
fn pooled_sd(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / n as f32;
    let var = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    var.sqrt()
}

/// Greedy 1:k nearest-neighbour propensity-score matching, returning the ATT.
///
/// # Arguments
/// * `y` — outcomes, length `n`.
/// * `t` — binary treatment indicators (`1.0` treated / `0.0` control), length `n`.
/// * `propensity` — estimated propensity scores `ê(x_i) ∈ (0, 1)`, length `n`.
/// * `config` — matching configuration.
///
/// # Errors
/// Returns [`CausalError::EmptyInput`] for empty inputs,
/// [`CausalError::DimensionMismatch`] on length mismatch,
/// [`CausalError::InvalidParameter`] for `k_neighbors == 0`, and
/// [`CausalError::NotFitted`] if no treated/control units exist or no treated
/// unit could be matched.
pub fn psm_att(
    y: &[f32],
    t: &[f32],
    propensity: &[f32],
    config: &PsmConfig,
) -> CausalResult<PsmResult> {
    let n = y.len();
    if n == 0 {
        return Err(CausalError::EmptyInput);
    }
    if t.len() != n || propensity.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len().min(propensity.len()),
        });
    }
    if config.k_neighbors == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "k_neighbors must be ≥ 1".to_string(),
        });
    }

    // Partition into treated / control index sets.
    let treated: Vec<usize> = (0..n).filter(|&i| t[i] > 0.5).collect();
    let controls: Vec<usize> = (0..n).filter(|&i| t[i] <= 0.5).collect();
    if treated.is_empty() || controls.is_empty() {
        return Err(CausalError::NotFitted);
    }

    // Pooled SD of the matching metric (over all units) for caliper scaling.
    let all_metric: Vec<f32> = (0..n)
        .map(|i| metric_value(propensity[i], config.metric))
        .collect();
    let sd = pooled_sd(&all_metric);
    let caliper = config.caliper_sd.map(|c| c * sd);

    // Track control usage when matching without replacement.
    let mut used = vec![false; n];

    let mut matches: Vec<MatchedPair> = Vec::with_capacity(treated.len());
    let mut n_unmatched = 0usize;

    for &ti in &treated {
        let tm = all_metric[ti];
        // Collect candidate controls with their distances.
        let mut candidates: Vec<(f32, usize)> = controls
            .iter()
            .filter(|&&ci| config.with_replacement || !used[ci])
            .map(|&ci| ((all_metric[ci] - tm).abs(), ci))
            .filter(|&(d, _)| caliper.is_none_or(|c| d <= c))
            .collect();

        if candidates.is_empty() {
            n_unmatched += 1;
            continue;
        }
        // Partial selection of the k nearest by distance.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = config.k_neighbors.min(candidates.len());
        let chosen: Vec<usize> = candidates[..k].iter().map(|&(_, ci)| ci).collect();
        if !config.with_replacement {
            for &ci in &chosen {
                used[ci] = true;
            }
        }
        let mean_ctrl_y = chosen.iter().map(|&ci| y[ci]).sum::<f32>() / chosen.len() as f32;
        let effect = y[ti] - mean_ctrl_y;
        matches.push(MatchedPair {
            treated_idx: ti,
            control_idxs: chosen,
            effect,
        });
    }

    let n_matched = matches.len();
    if n_matched == 0 {
        return Err(CausalError::NotFitted);
    }

    let att = matches.iter().map(|m| m.effect).sum::<f32>() / n_matched as f32;

    // Balance diagnostic: |SMD| of the matching metric between treated and
    // their matched controls, after matching.
    let treated_metric_mean = matches
        .iter()
        .map(|m| all_metric[m.treated_idx])
        .sum::<f32>()
        / n_matched as f32;
    let mut matched_ctrl_metric: Vec<f32> = Vec::new();
    for m in &matches {
        for &ci in &m.control_idxs {
            matched_ctrl_metric.push(all_metric[ci]);
        }
    }
    let ctrl_metric_mean =
        matched_ctrl_metric.iter().sum::<f32>() / matched_ctrl_metric.len().max(1) as f32;
    let treated_metric_vals: Vec<f32> = matches.iter().map(|m| all_metric[m.treated_idx]).collect();
    let pooled = {
        let s1 = pooled_sd(&treated_metric_vals);
        let s2 = pooled_sd(&matched_ctrl_metric);
        (0.5 * (s1 * s1 + s2 * s2)).sqrt()
    };
    let smd_after = if pooled > 1e-12 {
        ((treated_metric_mean - ctrl_metric_mean) / pooled).abs()
    } else {
        0.0
    };

    Ok(PsmResult {
        att,
        matches,
        n_matched,
        n_unmatched,
        smd_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a simple dataset with a known additive treatment effect `tau`,
    /// returning `(y, t, propensity)`. Treated and control share overlapping
    /// propensity ranges so matching is feasible.
    fn make_overlap_data(
        n_per: usize,
        tau: f32,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut y = Vec::new();
        let mut t = Vec::new();
        let mut p = Vec::new();
        for i in 0..n_per {
            // Treated unit.
            let score = 0.3 + 0.4 * (i as f32 / n_per as f32); // 0.3..0.7
            let base = 2.0 * score + 0.05 * rng.next_normal();
            y.push(base + tau);
            t.push(1.0);
            p.push(score);
            // Control unit at a nearby score.
            let cscore = 0.3 + 0.4 * (i as f32 / n_per as f32) + 0.01;
            let cbase = 2.0 * cscore + 0.05 * rng.next_normal();
            y.push(cbase);
            t.push(0.0);
            p.push(cscore.min(0.95));
        }
        (y, t, p)
    }

    #[test]
    fn psm_recovers_additive_effect() {
        let mut rng = LcgRng::new(1);
        let (y, t, p) = make_overlap_data(40, 3.0, &mut rng);
        let cfg = PsmConfig {
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        assert!(
            (r.att - 3.0).abs() < 0.5,
            "ATT {} should be near 3.0",
            r.att
        );
    }

    #[test]
    fn psm_att_finite() {
        let mut rng = LcgRng::new(2);
        let (y, t, p) = make_overlap_data(20, 1.5, &mut rng);
        let r = psm_att(&y, &t, &p, &PsmConfig::default()).expect("ok");
        assert!(r.att.is_finite());
    }

    #[test]
    fn psm_empty_errors() {
        let cfg = PsmConfig::default();
        assert!(matches!(
            psm_att(&[], &[], &[], &cfg),
            Err(CausalError::EmptyInput)
        ));
    }

    #[test]
    fn psm_length_mismatch_errors() {
        let cfg = PsmConfig::default();
        let r = psm_att(&[1.0, 2.0], &[1.0], &[0.5, 0.5], &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn psm_zero_k_errors() {
        let cfg = PsmConfig {
            k_neighbors: 0,
            ..PsmConfig::default()
        };
        let r = psm_att(&[1.0, 2.0], &[1.0, 0.0], &[0.6, 0.4], &cfg);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn psm_no_controls_errors() {
        let cfg = PsmConfig::default();
        let y = vec![1.0, 2.0, 3.0];
        let t = vec![1.0, 1.0, 1.0]; // all treated
        let p = vec![0.5, 0.6, 0.7];
        assert!(matches!(
            psm_att(&y, &t, &p, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn psm_no_treated_errors() {
        let cfg = PsmConfig::default();
        let y = vec![1.0, 2.0, 3.0];
        let t = vec![0.0, 0.0, 0.0]; // all control
        let p = vec![0.5, 0.6, 0.7];
        assert!(matches!(
            psm_att(&y, &t, &p, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn psm_exact_match_zero_effect() {
        // Treated and control with identical outcomes and scores → ATT ≈ 0.
        let y = vec![5.0, 5.0, 7.0, 7.0];
        let t = vec![1.0, 0.0, 1.0, 0.0];
        let p = vec![0.5, 0.5, 0.6, 0.6];
        let cfg = PsmConfig {
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        assert!(r.att.abs() < 1e-4, "ATT {} should be ~0", r.att);
    }

    #[test]
    fn psm_caliper_can_leave_unmatched() {
        // Treated score 0.9, controls clustered near 0.1 → tight caliper rejects.
        let y = vec![10.0, 1.0, 1.0, 1.0];
        let t = vec![1.0, 0.0, 0.0, 0.0];
        let p = vec![0.9, 0.1, 0.12, 0.11];
        let cfg = PsmConfig {
            k_neighbors: 1,
            with_replacement: true,
            metric: MatchMetric::Logit,
            caliper_sd: Some(0.01), // extremely tight
        };
        let r = psm_att(&y, &t, &p, &cfg);
        // Either everyone is unmatched (NotFitted) or the single treated is unmatched.
        match r {
            Err(CausalError::NotFitted) => {}
            Ok(res) => assert!(res.n_unmatched >= 1, "expected an unmatched treated unit"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn psm_without_replacement_unique_controls() {
        // 3 treated, 3 controls; without replacement every control used once.
        let y = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let t = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let p = vec![0.4, 0.5, 0.6, 0.41, 0.51, 0.61];
        let cfg = PsmConfig {
            k_neighbors: 1,
            with_replacement: false,
            metric: MatchMetric::Score,
            caliper_sd: None,
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        let mut all_ctrls: Vec<usize> = r
            .matches
            .iter()
            .flat_map(|m| m.control_idxs.clone())
            .collect();
        all_ctrls.sort_unstable();
        let unique = all_ctrls.clone();
        let mut dedup = unique.clone();
        dedup.dedup();
        assert_eq!(
            all_ctrls.len(),
            dedup.len(),
            "controls reused without replacement"
        );
    }

    #[test]
    fn psm_knn_averages_controls() {
        // 1 treated, 2 controls; k=2 averages both control outcomes.
        let y = vec![10.0, 2.0, 4.0];
        let t = vec![1.0, 0.0, 0.0];
        let p = vec![0.5, 0.49, 0.51];
        let cfg = PsmConfig {
            k_neighbors: 2,
            with_replacement: true,
            metric: MatchMetric::Score,
            caliper_sd: None,
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        // effect = 10 - mean(2, 4) = 10 - 3 = 7.
        assert!((r.att - 7.0).abs() < 1e-4, "ATT {} expected 7", r.att);
        assert_eq!(r.matches[0].control_idxs.len(), 2);
    }

    #[test]
    fn psm_logit_vs_score_both_run() {
        let mut rng = LcgRng::new(3);
        let (y, t, p) = make_overlap_data(15, 2.0, &mut rng);
        let cfg_s = PsmConfig {
            metric: MatchMetric::Score,
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let cfg_l = PsmConfig {
            metric: MatchMetric::Logit,
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let rs = psm_att(&y, &t, &p, &cfg_s).expect("ok");
        let rl = psm_att(&y, &t, &p, &cfg_l).expect("ok");
        assert!(rs.att.is_finite() && rl.att.is_finite());
    }

    #[test]
    fn psm_smd_after_small_with_good_overlap() {
        let mut rng = LcgRng::new(4);
        let (y, t, p) = make_overlap_data(50, 1.0, &mut rng);
        let cfg = PsmConfig {
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        assert!(r.smd_after.is_finite() && r.smd_after >= 0.0);
        assert!(
            r.smd_after < 0.5,
            "SMD after matching unexpectedly large: {}",
            r.smd_after
        );
    }

    #[test]
    fn psm_match_counts_consistent() {
        let mut rng = LcgRng::new(5);
        let (y, t, p) = make_overlap_data(30, 2.0, &mut rng);
        let n_treated = t.iter().filter(|&&v| v > 0.5).count();
        let cfg = PsmConfig {
            caliper_sd: None,
            ..PsmConfig::default()
        };
        let r = psm_att(&y, &t, &p, &cfg).expect("ok");
        assert_eq!(r.n_matched + r.n_unmatched, n_treated);
        assert_eq!(r.matches.len(), r.n_matched);
    }

    #[test]
    fn psm_default_config_values() {
        let cfg = PsmConfig::default();
        assert_eq!(cfg.k_neighbors, 1);
        assert!(cfg.with_replacement);
        assert_eq!(cfg.metric, MatchMetric::Logit);
        assert_eq!(cfg.caliper_sd, Some(0.2));
    }
}
