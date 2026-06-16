//! Staggered Difference-in-Differences — Callaway & Sant'Anna (2021).
//!
//! Reference: Callaway, B. & Sant'Anna, P. H. C. (2021). "Difference-in-Differences
//! with multiple time periods." *Journal of Econometrics*, 225(2), 200-230.
//!
//! # Overview
//!
//! Classical two-way fixed-effects (TWFE) difference-in-differences can be
//! severely biased when units adopt treatment at *different* times ("staggered
//! adoption") and treatment effects are heterogeneous across adoption cohorts
//! or over time — already-treated units enter the comparison group of the TWFE
//! regression with *negative* weights (Goodman-Bacon 2021). Callaway &
//! Sant'Anna sidestep this by building the estimator out of clean
//! **group-time average treatment effects** `ATT(g, t)`, each of which is a
//! transparent `2 × 2` difference-in-differences that *only* uses units that
//! are not-yet-treated as the comparison group.
//!
//! ## Notation
//!
//! - A *group* `g` is the calendar period at which a cohort first becomes
//!   treated. Never-treated units are encoded with the sentinel group `0`.
//! - `t` ranges over calendar periods `1 .. =T`.
//! - `Ȳ_S(t)` denotes the average outcome at period `t` over the unit set `S`.
//!
//! ## Group-time ATT
//!
//! For a treated group `g` and an *event* period `t ≥ g`, with a *base* period
//! `g − 1` (the last period before the cohort was treated), the group-time
//! average treatment effect under the **not-yet-treated** comparison group is
//!
//! ```text
//!   ATT(g, t) = [ Ȳ_g(t) − Ȳ_g(g−1) ]  −  [ Ȳ_C(t) − Ȳ_C(g−1) ]
//! ```
//!
//! where the comparison set `C = C(g, t)` consists of every unit whose own
//! first-treatment period is strictly greater than `max(g, t)` (this includes
//! never-treated units). This is the canonical "not-yet-treated" control group
//! of Callaway-Sant'Anna §3.1; it guarantees the comparison units are still
//! untreated at *both* the base period `g − 1` and the event period `t`, so the
//! parallel-trends assumption applies to a genuinely untreated counterfactual.
//!
//! ## Aggregation schemes
//!
//! The matrix of `ATT(g, t)` is summarised by three aggregation schemes
//! (Callaway-Sant'Anna §4), each weighting the post-treatment cells
//! (`t ≥ g`) by the cohort's share of the treated population `P(G = g)`:
//!
//! - [`Aggregation::Simple`] — population-share-weighted mean of every
//!   post-treatment `ATT(g, t)`. This is the overall average effect of having
//!   been treated.
//! - [`Aggregation::Dynamic`] — *event-study* profile: averages `ATT(g, t)`
//!   over cohorts at fixed *event time* `e = t − g` (periods since treatment),
//!   exposing how the effect evolves with exposure length.
//! - [`Aggregation::Group`] — one number per cohort `g`: the average of that
//!   cohort's own post-treatment effects, then population-share-weighted into a
//!   single summary.
//!
//! All computations are pure-Rust FP64 over a balanced panel.

use crate::error::{CausalError, CausalResult};

/// Sentinel group label for never-treated units.
pub const NEVER_TREATED: usize = 0;

/// Aggregation scheme for collapsing the `ATT(g, t)` matrix to a scalar
/// summary plus (for dynamic) an event-study profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Population-share-weighted mean over all post-treatment `(g, t)` cells.
    Simple,
    /// Event-study: average by event time `e = t − g`.
    Dynamic,
    /// Per-cohort average, then population-share-weighted summary.
    Group,
}

/// Configuration for [`callaway_santanna`].
#[derive(Debug, Clone)]
pub struct StaggeredDidConfig {
    /// Number of calendar periods `T` in the (balanced) panel.
    pub n_periods: usize,
    /// Aggregation scheme for the overall summary.
    pub aggregation: Aggregation,
}

impl Default for StaggeredDidConfig {
    fn default() -> Self {
        Self {
            n_periods: 0,
            aggregation: Aggregation::Simple,
        }
    }
}

/// A single group-time average treatment effect cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupTimeAtt {
    /// First-treatment period (cohort) `g ≥ 1`.
    pub group: usize,
    /// Calendar period `t ≥ 1`.
    pub period: usize,
    /// Estimated `ATT(g, t)` (NaN-free; `0.0` if the cell is not identified
    /// because no valid not-yet-treated comparison units exist).
    pub att: f64,
    /// Whether a valid not-yet-treated comparison group existed for this cell.
    /// When `false`, [`Self::att`] is reported as `0.0` and the cell is omitted
    /// from every aggregation.
    pub identified: bool,
}

/// One point of the dynamic (event-study) aggregation profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventStudyPoint {
    /// Event time `e = t − g` (periods relative to first treatment; `≥ 0` for
    /// post-treatment exposure).
    pub event_time: i64,
    /// Population-share-weighted average `ATT` at this event time.
    pub att: f64,
}

/// Output of [`callaway_santanna`].
#[derive(Debug, Clone)]
pub struct StaggeredDidResult {
    /// Every group-time cell `ATT(g, t)` with `t ≥ g` (post-treatment only),
    /// in ascending `(group, period)` order.
    pub group_time: Vec<GroupTimeAtt>,
    /// The scalar overall summary under the requested [`Aggregation`].
    pub overall_att: f64,
    /// Event-study profile (only populated when
    /// [`Aggregation::Dynamic`] is requested; empty otherwise), ascending in
    /// `event_time`.
    pub event_study: Vec<EventStudyPoint>,
}

/// Estimate staggered-adoption group-time ATTs and their aggregation.
///
/// # Parameters
/// - `outcomes`: row-major `n_units × n_periods` panel of outcomes
///   (length `n_units · cfg.n_periods`). Row `i` holds unit `i`'s outcomes
///   across periods `1 .. =T` in columns `0 .. T`.
/// - `groups`: length `n_units`; `groups[i]` is unit `i`'s first-treatment
///   period (`1 .. =T`) or [`NEVER_TREATED`] (`0`) if never treated.
/// - `n_units`: number of cross-sectional units. Must be `> 0`.
/// - `cfg`: see [`StaggeredDidConfig`].
///
/// # Errors
/// - [`CausalError::EmptyInput`] if `n_units == 0`, `cfg.n_periods == 0`, or
///   `outcomes` is empty.
/// - [`CausalError::DimensionMismatch`] if `outcomes.len() != n_units · T` or
///   `groups.len() != n_units`.
/// - [`CausalError::InvalidParameter`] if any `groups[i] > T`.
pub fn callaway_santanna(
    outcomes: &[f64],
    groups: &[usize],
    n_units: usize,
    cfg: &StaggeredDidConfig,
) -> CausalResult<StaggeredDidResult> {
    let t_max = cfg.n_periods;
    if n_units == 0 || t_max == 0 || outcomes.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    if outcomes.len() != n_units * t_max {
        return Err(CausalError::DimensionMismatch {
            expected: n_units * t_max,
            got: outcomes.len(),
        });
    }
    if groups.len() != n_units {
        return Err(CausalError::DimensionMismatch {
            expected: n_units,
            got: groups.len(),
        });
    }
    for &g in groups {
        if g > t_max {
            return Err(CausalError::InvalidParameter {
                reason: format!("group {g} exceeds n_periods {t_max}"),
            });
        }
    }

    // Distinct treated cohorts (g ≥ 1), ascending.
    let mut cohorts: Vec<usize> = Vec::new();
    for &g in groups {
        if g != NEVER_TREATED && !cohorts.contains(&g) {
            cohorts.push(g);
        }
    }
    cohorts.sort_unstable();

    // Population share weight P(G = g) over the *treated* population. Cells from
    // cohort g are weighted by this share in every aggregation.
    let n_treated_total: usize = groups.iter().filter(|&&g| g != NEVER_TREATED).count();

    // ---- compute group-time ATT(g, t) for every post-treatment cell --------
    let mut group_time: Vec<GroupTimeAtt> = Vec::new();
    for &g in &cohorts {
        // base period is g − 1 (1-indexed period g-1 → column g-2).
        // A cohort first treated at g = 1 has no pre-period; it cannot anchor a
        // clean 2×2 DiD, so all its cells are unidentified.
        let base_period = g - 1; // 1-indexed; 0 means "no valid base"
        for t in g..=t_max {
            let cell = group_time_att(outcomes, groups, n_units, t_max, g, t, base_period);
            group_time.push(cell);
        }
    }

    // ---- aggregation -------------------------------------------------------
    let (overall_att, event_study) = match cfg.aggregation {
        Aggregation::Simple => (
            aggregate_simple(&group_time, groups, n_treated_total),
            Vec::new(),
        ),
        Aggregation::Group => (
            aggregate_group(&group_time, &cohorts, groups, n_treated_total),
            Vec::new(),
        ),
        Aggregation::Dynamic => {
            let es = aggregate_dynamic(&group_time, groups, n_treated_total);
            // Overall = population-weighted mean of the event-study points,
            // matching the "simple" overall ATT under dynamic reporting.
            let overall = aggregate_simple(&group_time, groups, n_treated_total);
            (overall, es)
        }
    };

    Ok(StaggeredDidResult {
        group_time,
        overall_att,
        event_study,
    })
}

// =====================================================================
// group-time ATT — a single clean 2×2 DiD
// =====================================================================

/// Compute `ATT(g, t)` against the not-yet-treated comparison group.
///
/// `base_period` is the 1-indexed pre-treatment anchor `g − 1`; a value of `0`
/// signals that no valid base period exists (cohort treated in period 1).
fn group_time_att(
    outcomes: &[f64],
    groups: &[usize],
    n_units: usize,
    t_max: usize,
    g: usize,
    t: usize,
    base_period: usize,
) -> GroupTimeAtt {
    let unidentified = GroupTimeAtt {
        group: g,
        period: t,
        att: 0.0,
        identified: false,
    };
    if base_period == 0 {
        return unidentified;
    }

    // Comparison units: not-yet-treated at max(g, t), i.e. own group > max(g, t)
    // (never-treated, group 0, always qualifies).
    let horizon = g.max(t);

    // Treated-cohort averages.
    let (treat_t, n_treat) = period_mean(outcomes, groups, n_units, t_max, t, |gi| gi == g);
    let (treat_base, _) = period_mean(outcomes, groups, n_units, t_max, base_period, |gi| gi == g);

    // Comparison-group averages.
    let is_control = |gi: usize| gi == NEVER_TREATED || gi > horizon;
    let (ctrl_t, n_ctrl) = period_mean(outcomes, groups, n_units, t_max, t, is_control);
    let (ctrl_base, _) = period_mean(outcomes, groups, n_units, t_max, base_period, is_control);

    if n_treat == 0 || n_ctrl == 0 {
        return unidentified;
    }

    let att = (treat_t - treat_base) - (ctrl_t - ctrl_base);
    GroupTimeAtt {
        group: g,
        period: t,
        att,
        identified: true,
    }
}

/// Mean outcome at 1-indexed `period` over all units whose group satisfies
/// `pred`. Returns `(mean, count)`; `mean = 0.0` when `count == 0`.
fn period_mean(
    outcomes: &[f64],
    groups: &[usize],
    n_units: usize,
    t_max: usize,
    period: usize,
    pred: impl Fn(usize) -> bool,
) -> (f64, usize) {
    let col = period - 1; // 1-indexed period → 0-indexed column
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n_units {
        if pred(groups[i]) {
            sum += outcomes[i * t_max + col];
            count += 1;
        }
    }
    if count == 0 {
        (0.0, 0)
    } else {
        (sum / count as f64, count)
    }
}

// =====================================================================
// aggregation schemes
// =====================================================================

/// Count units in cohort `g`.
fn cohort_size(groups: &[usize], g: usize) -> usize {
    groups.iter().filter(|&&gi| gi == g).count()
}

/// Population share `P(G = g)` over the treated population.
fn cohort_share(groups: &[usize], g: usize, n_treated_total: usize) -> f64 {
    if n_treated_total == 0 {
        0.0
    } else {
        cohort_size(groups, g) as f64 / n_treated_total as f64
    }
}

/// `Aggregation::Simple`: population-share-weighted mean over all identified
/// post-treatment cells. Each cohort `g` contributes its cells weighted by
/// `P(G = g)`, then we renormalise by the total weight actually used so that
/// the result is a proper weighted average even when some cells drop out.
fn aggregate_simple(group_time: &[GroupTimeAtt], groups: &[usize], n_treated_total: usize) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for cell in group_time {
        if !cell.identified {
            continue;
        }
        let w = cohort_share(groups, cell.group, n_treated_total);
        num += w * cell.att;
        den += w;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// `Aggregation::Group`: average each cohort's own post-treatment effects, then
/// combine the per-cohort numbers with population-share weights.
fn aggregate_group(
    group_time: &[GroupTimeAtt],
    cohorts: &[usize],
    groups: &[usize],
    n_treated_total: usize,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for &g in cohorts {
        let mut g_sum = 0.0_f64;
        let mut g_cnt = 0_usize;
        for cell in group_time {
            if cell.group == g && cell.identified {
                g_sum += cell.att;
                g_cnt += 1;
            }
        }
        if g_cnt == 0 {
            continue;
        }
        let g_att = g_sum / g_cnt as f64;
        let w = cohort_share(groups, g, n_treated_total);
        num += w * g_att;
        den += w;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// `Aggregation::Dynamic`: event-study profile averaging `ATT(g, t)` at fixed
/// event time `e = t − g`, weighted across cohorts by population share. Returns
/// the profile ascending in event time.
fn aggregate_dynamic(
    group_time: &[GroupTimeAtt],
    groups: &[usize],
    n_treated_total: usize,
) -> Vec<EventStudyPoint> {
    // Distinct (post-treatment) event times.
    let mut event_times: Vec<i64> = Vec::new();
    for cell in group_time {
        if !cell.identified {
            continue;
        }
        let e = cell.period as i64 - cell.group as i64;
        if !event_times.contains(&e) {
            event_times.push(e);
        }
    }
    event_times.sort_unstable();

    let mut profile: Vec<EventStudyPoint> = Vec::with_capacity(event_times.len());
    for e in event_times {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for cell in group_time {
            if !cell.identified {
                continue;
            }
            if cell.period as i64 - cell.group as i64 == e {
                let w = cohort_share(groups, cell.group, n_treated_total);
                num += w * cell.att;
                den += w;
            }
        }
        let att = if den > 0.0 { num / den } else { 0.0 };
        profile.push(EventStudyPoint { event_time: e, att });
    }
    profile
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a balanced panel where every unit follows a common linear trend
    /// `Y_it = unit_intercept + trend·t`, plus a constant treatment effect
    /// `tau` applied additively in every period `t ≥ g` for treated unit `i`.
    ///
    /// Returns `(outcomes, groups)`.
    fn make_panel(
        unit_groups: &[usize],
        t_max: usize,
        trend: f64,
        tau: f64,
    ) -> (Vec<f64>, Vec<usize>) {
        let n = unit_groups.len();
        let mut outcomes = vec![0.0_f64; n * t_max];
        for (i, &g) in unit_groups.iter().enumerate() {
            let intercept = 1.0 + 0.5 * i as f64; // unit fixed effect
            for period in 1..=t_max {
                let mut y = intercept + trend * period as f64;
                if g != NEVER_TREATED && period >= g {
                    y += tau;
                }
                outcomes[i * t_max + (period - 1)] = y;
            }
        }
        (outcomes, unit_groups.to_vec())
    }

    fn cfg(t: usize, agg: Aggregation) -> StaggeredDidConfig {
        StaggeredDidConfig {
            n_periods: t,
            aggregation: agg,
        }
    }

    // -------------------- input validation ---------------------------------

    #[test]
    fn n_units_0_error() {
        let r = callaway_santanna(&[], &[], 0, &cfg(3, Aggregation::Simple));
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn n_periods_0_error() {
        let r = callaway_santanna(&[1.0], &[0], 1, &cfg(0, Aggregation::Simple));
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn outcomes_dim_mismatch_error() {
        // 2 units × 3 periods needs 6 entries, give 5.
        let r = callaway_santanna(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0, 2],
            2,
            &cfg(3, Aggregation::Simple),
        );
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn groups_dim_mismatch_error() {
        let (outcomes, _) = make_panel(&[0, 2, 3], 4, 1.0, 5.0);
        // groups has wrong length (2 instead of 3).
        let r = callaway_santanna(&outcomes, &[0, 2], 3, &cfg(4, Aggregation::Simple));
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn group_exceeds_periods_error() {
        let (outcomes, groups) = make_panel(&[0, 2, 5], 4, 1.0, 5.0);
        // group 5 > T=4.
        let r = callaway_santanna(&outcomes, &groups, 3, &cfg(4, Aggregation::Simple));
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    // -------------------- recovery / correctness ---------------------------

    /// Constant additive effect under exact parallel trends: every identified
    /// ATT(g, t) must equal tau and the overall summary must equal tau.
    #[test]
    fn recovers_constant_tau_simple() {
        let tau = 4.0;
        // Cohorts at g=2, g=3, plus never-treated controls; T=4.
        let (outcomes, groups) = make_panel(&[0, 0, 2, 2, 3, 3], 4, 1.5, tau);
        let res = callaway_santanna(&outcomes, &groups, 6, &cfg(4, Aggregation::Simple))
            .expect("valid constant-tau panel should compute successfully");
        // Every identified cell ≈ tau.
        for cell in &res.group_time {
            if cell.identified {
                assert!(
                    (cell.att - tau).abs() < 1e-9,
                    "ATT(g={}, t={}) = {} expected {tau}",
                    cell.group,
                    cell.period,
                    cell.att
                );
            }
        }
        assert!(
            (res.overall_att - tau).abs() < 1e-9,
            "overall = {} expected {tau}",
            res.overall_att
        );
    }

    /// Zero treatment effect → all ATTs and the summary are ≈ 0.
    #[test]
    fn null_effect_zero_att() {
        let (outcomes, groups) = make_panel(&[0, 0, 2, 3], 4, 2.0, 0.0);
        let res = callaway_santanna(&outcomes, &groups, 4, &cfg(4, Aggregation::Simple))
            .expect("value should be present");
        for cell in &res.group_time {
            if cell.identified {
                assert!(cell.att.abs() < 1e-9, "ATT = {}", cell.att);
            }
        }
        assert!(res.overall_att.abs() < 1e-9);
    }

    /// First-period cohort (g = 1) has no valid base period → its cells are
    /// flagged unidentified.
    #[test]
    fn group_one_unidentified() {
        let (outcomes, groups) = make_panel(&[0, 1, 3], 4, 1.0, 5.0);
        let res = callaway_santanna(&outcomes, &groups, 3, &cfg(4, Aggregation::Simple))
            .expect("value should be present");
        for cell in &res.group_time {
            if cell.group == 1 {
                assert!(!cell.identified, "g=1 cells must be unidentified");
                assert_eq!(cell.att, 0.0);
            }
        }
    }

    /// Only post-treatment cells (t ≥ g) are produced; the count matches.
    #[test]
    fn output_shape_post_treatment_only() {
        // g=2 → t∈{2,3,4} (3 cells); g=3 → t∈{3,4} (2 cells); T=4.
        let (outcomes, groups) = make_panel(&[0, 2, 3], 4, 1.0, 5.0);
        let res = callaway_santanna(&outcomes, &groups, 3, &cfg(4, Aggregation::Simple))
            .expect("value should be present");
        assert_eq!(res.group_time.len(), 3 + 2);
        for cell in &res.group_time {
            assert!(cell.period >= cell.group);
        }
    }

    /// All ATTs and the overall summary are finite for a generic panel.
    #[test]
    fn att_finite() {
        let (outcomes, groups) = make_panel(&[0, 0, 2, 3, 4, 4], 5, 0.7, 2.0);
        let res = callaway_santanna(&outcomes, &groups, 6, &cfg(5, Aggregation::Simple))
            .expect("value should be present");
        for cell in &res.group_time {
            assert!(cell.att.is_finite());
        }
        assert!(res.overall_att.is_finite());
    }

    /// Dynamic aggregation produces an ascending event-study profile and, for a
    /// constant effect, every event-time point equals tau.
    #[test]
    fn dynamic_event_study_constant() {
        let tau = 3.0;
        let (outcomes, groups) = make_panel(&[0, 0, 2, 2, 3, 4], 5, 1.0, tau);
        let res = callaway_santanna(&outcomes, &groups, 6, &cfg(5, Aggregation::Dynamic))
            .expect("value should be present");
        assert!(!res.event_study.is_empty());
        // ascending event_time
        for w in res.event_study.windows(2) {
            assert!(w[0].event_time < w[1].event_time);
        }
        // event times are e = t − g ≥ 0
        for p in &res.event_study {
            assert!(p.event_time >= 0);
            assert!(
                (p.att - tau).abs() < 1e-9,
                "event e={} att={} expected {tau}",
                p.event_time,
                p.att
            );
        }
    }

    /// Group aggregation equals tau under a constant effect, and equals the
    /// simple aggregation in that symmetric case.
    #[test]
    fn group_aggregation_constant() {
        let tau = 5.0;
        let (outcomes, groups) = make_panel(&[0, 0, 2, 2, 3, 3], 4, 1.2, tau);
        let res_g = callaway_santanna(&outcomes, &groups, 6, &cfg(4, Aggregation::Group))
            .expect("value should be present");
        assert!(
            (res_g.overall_att - tau).abs() < 1e-9,
            "group overall = {} expected {tau}",
            res_g.overall_att
        );
    }

    /// Not-yet-treated units serve as controls: a later cohort is a valid
    /// comparison for an earlier cohort's early event periods. We verify that
    /// removing all never-treated units still yields identified cells (the
    /// later cohort g=4 controls for g=2 at t=2,3).
    #[test]
    fn not_yet_treated_controls_used() {
        // No never-treated units; cohorts at g=2 and g=4 only.
        let (outcomes, groups) = make_panel(&[2, 2, 4, 4], 5, 1.0, 6.0);
        let res = callaway_santanna(&outcomes, &groups, 4, &cfg(5, Aggregation::Simple))
            .expect("value should be present");
        // ATT(2, 2) and ATT(2, 3): comparison = units with group > max(2,t).
        // For t=2: group > 2 → the g=4 units qualify. Identified.
        let cell_2_2 = res
            .group_time
            .iter()
            .find(|c| c.group == 2 && c.period == 2)
            .expect("cell (2,2) present");
        assert!(
            cell_2_2.identified,
            "g=2,t=2 should use not-yet-treated g=4 as control"
        );
        assert!((cell_2_2.att - 6.0).abs() < 1e-9);
    }

    /// Heterogeneous effects: a larger effect in the later cohort raises the
    /// overall summary above the smaller cohort's effect.
    #[test]
    fn heterogeneous_effects_weighted() {
        // Build manually: g=2 cohort has tau=2, g=3 cohort has tau=6.
        let t_max = 4;
        let unit_groups = vec![0usize, 0, 2, 2, 3, 3];
        let n = unit_groups.len();
        let mut outcomes = vec![0.0_f64; n * t_max];
        for (i, &g) in unit_groups.iter().enumerate() {
            let intercept = 1.0 + 0.5 * i as f64;
            let tau = if g == 2 {
                2.0
            } else if g == 3 {
                6.0
            } else {
                0.0
            };
            for period in 1..=t_max {
                let mut y = intercept + 1.0 * period as f64;
                if g != NEVER_TREATED && period >= g {
                    y += tau;
                }
                outcomes[i * t_max + (period - 1)] = y;
            }
        }
        let res = callaway_santanna(&outcomes, &unit_groups, n, &cfg(t_max, Aggregation::Simple))
            .expect("value should be present");
        // Overall must lie strictly between the two cohort effects.
        assert!(
            res.overall_att > 2.0 && res.overall_att < 6.0,
            "overall = {} should be between 2 and 6",
            res.overall_att
        );
    }

    /// Deterministic: identical inputs yield bit-identical results.
    #[test]
    fn deterministic() {
        let (outcomes, groups) = make_panel(&[0, 0, 2, 3, 4], 5, 0.9, 3.0);
        let r1 = callaway_santanna(&outcomes, &groups, 5, &cfg(5, Aggregation::Dynamic))
            .expect("value should be present");
        let r2 = callaway_santanna(&outcomes, &groups, 5, &cfg(5, Aggregation::Dynamic))
            .expect("value should be present");
        assert_eq!(r1.overall_att, r2.overall_att);
        assert_eq!(r1.group_time.len(), r2.group_time.len());
        for (a, b) in r1.event_study.iter().zip(r2.event_study.iter()) {
            assert_eq!(a.att, b.att);
            assert_eq!(a.event_time, b.event_time);
        }
    }

    /// All-never-treated panel: no cohorts, empty group-time, overall 0.
    #[test]
    fn all_never_treated_empty() {
        let (outcomes, groups) = make_panel(&[0, 0, 0], 3, 1.0, 0.0);
        let res = callaway_santanna(&outcomes, &groups, 3, &cfg(3, Aggregation::Simple))
            .expect("value should be present");
        assert!(res.group_time.is_empty());
        assert_eq!(res.overall_att, 0.0);
    }

    #[test]
    fn config_default_is_sane() {
        let c = StaggeredDidConfig::default();
        assert_eq!(c.aggregation, Aggregation::Simple);
    }
}
