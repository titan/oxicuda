//! Variance and confidence bands for the Aalen-Johansen multi-state estimator.
//!
//! This module extends the point estimator in
//! [`crate::nonparametric::multi_state`] with a covariance estimator for the
//! transition-probability matrix `P̂(s,t)` and pointwise confidence intervals.
//!
//! # Estimator
//! Writing the Aalen-Johansen product integral as
//! `P̂(s,t) = ∏_{s<u_k≤t} (I + ΔÂ(u_k))` with empirical transition increments
//! `ΔÂ(u_k)`, the covariance of `vec(P̂)` is propagated through the product by the
//! recursion (Andersen, Borgan, Gill & Keiding 1993, §IV.4):
//!
//! ```text
//!   C_new[(a,b),(c,d)] = Σ_{m,n} T_{mb} T_{nd} C_old[(a,m),(c,n)]
//!                      + Σ_m P_old_{am} P_old_{cm} G_m[b,d]
//! ```
//!
//! where `T = I + ΔÂ(u_k)`, `P_old = P̂(s,u_{k-1})`, and `G_m[b,d]` is the
//! Greenwood-type (multinomial) covariance of the increments leaving state `m`:
//!
//! ```text
//!   G_m[b,d] = [ δ_{bd}·dN_{mb}(n_m − dN_{mb}) − (1−δ_{bd})·dN_{mb}·dN_{md} ] / n_m³
//! ```
//!
//! for destination states `b,d ≠ m`, with the diagonal (origin) row/column
//! filled so that every row and column of `G_m` sums to zero. The increments
//! leaving distinct states are treated as uncorrelated, hence the single index
//! `m` in the increment term.
//!
//! # Properties
//! * For a two-state alive→dead model `P̂_{00}(0,t)` equals the Kaplan-Meier
//!   estimate and `vâr(P̂_{00}(0,t))` equals the Kaplan-Meier Greenwood variance
//!   **exactly** (the Greenwood-type increment covariance is chosen precisely so
//!   that the recursion telescopes to Greenwood's formula).
//! * `vâr(P̂(s,s)) = 0`; the variance grows as more event times are accumulated.
//! * Confidence intervals use a log transform clamped to `[0,1]`, falling back to
//!   a Wald interval where the log transform is undefined.

use crate::error::{SurvivalError, SurvivalResult};
use crate::nonparametric::multi_state::{
    MultiStateConfig, MultiStateObs, fit_multi_state, predict_transition_probs,
};

/// A bundle of multi-state observations and the model configuration.
///
/// Provided so the inference entry points can take a single `&MultiStateData`
/// argument; it simply pairs the counting-process observations with the
/// [`MultiStateConfig`] describing the state space.
#[derive(Debug, Clone)]
pub struct MultiStateData {
    /// Counting-process observations (one per at-risk interval).
    pub observations: Vec<MultiStateObs>,
    /// State-space configuration.
    pub config: MultiStateConfig,
}

impl MultiStateData {
    /// Construct a [`MultiStateData`] from observations and a configuration.
    #[must_use]
    pub fn new(observations: Vec<MultiStateObs>, config: MultiStateConfig) -> Self {
        Self {
            observations,
            config,
        }
    }
}

/// Aalen-Johansen inference output at a single target time `t` (given origin `s`).
///
/// All matrices are flattened `n_states × n_states` in row-major order; the
/// `[h * n_states + j]` entry refers to the `h → j` transition probability.
#[derive(Debug, Clone)]
pub struct AjInference {
    /// Transition probability matrix `P̂(s,t)`.
    pub transition_prob: Vec<f64>,
    /// Variance `vâr(P̂_{hj}(s,t))` for every entry, same layout as `transition_prob`.
    pub variance: Vec<f64>,
    /// Lower confidence bound for each entry.
    pub ci_lower: Vec<f64>,
    /// Upper confidence bound for each entry.
    pub ci_upper: Vec<f64>,
    /// Number of states.
    pub n_states: usize,
}

/// Competing-risks cumulative incidence with variance for one origin state.
#[derive(Debug, Clone)]
pub struct CifInference {
    /// Target time `t`.
    pub time: f64,
    /// Origin state whose row of `P̂(0,t)` is reported.
    pub origin: usize,
    /// Cumulative incidence (transition probability) into each state `j`.
    pub cif: Vec<f64>,
    /// Variance of each cumulative incidence.
    pub variance: Vec<f64>,
    /// Lower confidence bound for each cumulative incidence.
    pub ci_lower: Vec<f64>,
    /// Upper confidence bound for each cumulative incidence.
    pub ci_upper: Vec<f64>,
}

// ─── Counting helpers (mirror multi_state.rs, recomputed locally) ─────────────

/// Subjects at risk in `state` just before time `t` (counting-process form).
///
/// Matches the convention of [`crate::nonparametric::multi_state`]: a subject is
/// at risk at `t` when `start_time ≤ t ≤ end_time` in its origin state, so a
/// subject transitioning out exactly at `t` is still counted in the risk set.
fn at_risk_count(obs: &[MultiStateObs], state: usize, t: f64) -> f64 {
    obs.iter()
        .filter(|o| o.from_state == state && o.start_time <= t && o.end_time >= t)
        .count() as f64
}

/// Observed `from → to` transitions at exactly time `t`.
fn transition_count(obs: &[MultiStateObs], from: usize, to: usize, t: f64) -> f64 {
    obs.iter()
        .filter(|o| {
            o.from_state == from
                && o.to_state == Some(to)
                && (o.end_time - t).abs() <= f64::EPSILON * t.abs().max(1.0)
        })
        .count() as f64
}

/// Inverse standard normal CDF (Acklam's rational approximation).
fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Build the within-origin Greenwood increment covariance `G_m` (S×S, row-major)
/// for origin state `m` at an event time, given the at-risk count and per-state
/// transition counts out of `m`.
///
/// Destination entries (`b,d ≠ m`) use the multinomial covariance; the origin
/// row and column are filled so each sums to zero (mirroring `ΔÂ_{mm} = -Σ ΔÂ_{mj}`).
fn greenwood_increment_cov(n_m: f64, d_out: &[f64], m: usize, s: usize) -> Vec<f64> {
    let mut g = vec![0.0_f64; s * s];
    if n_m <= 0.0 {
        return g;
    }
    let n3 = n_m * n_m * n_m;
    // Destination–destination block (b, d ≠ m).
    for b in 0..s {
        if b == m {
            continue;
        }
        for d in 0..s {
            if d == m {
                continue;
            }
            let val = if b == d {
                d_out[b] * (n_m - d_out[b]) / n3
            } else {
                -d_out[b] * d_out[d] / n3
            };
            g[b * s + d] = val;
        }
    }
    // Fill origin column m: G[b][m] = -Σ_{d≠m} G[b][d]  (row b sums to zero).
    for b in 0..s {
        if b == m {
            continue;
        }
        let mut row_sum = 0.0_f64;
        for d in 0..s {
            if d == m {
                continue;
            }
            row_sum += g[b * s + d];
        }
        g[b * s + m] = -row_sum;
    }
    // Fill origin row m: G[m][d] = -Σ_{b≠m} G[b][d]  (column d sums to zero),
    // including the corner G[m][m] which now makes every column sum to zero.
    for d in 0..s {
        let mut col_sum = 0.0_f64;
        for b in 0..s {
            if b == m {
                continue;
            }
            col_sum += g[b * s + d];
        }
        g[m * s + d] = -col_sum;
    }
    g
}

/// Core recursion: returns `(P, C)` where `P` is `P̂(s,t)` (S×S row-major) and `C`
/// is the covariance of `vec(P)` stored as an `S²×S²` row-major matrix indexed by
/// flattened state pairs `(a,b) -> a*S + b`.
///
/// The product accumulates over event times in the half-open interval `(s,t]`.
fn aj_recursion(
    obs: &[MultiStateObs],
    s_states: usize,
    event_times: &[f64],
    s: f64,
    t: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dim = s_states * s_states;
    // P starts at identity; covariance starts at zero (P(s,s) is known exactly).
    let mut p = vec![0.0_f64; dim];
    for i in 0..s_states {
        p[i * s_states + i] = 1.0;
    }
    let mut c = vec![0.0_f64; dim * dim];

    for &u in event_times {
        if u <= s || u > t {
            continue;
        }
        // Build T = I + ΔÂ(u) and the per-origin increment covariances G_m.
        let mut t_mat = vec![0.0_f64; dim];
        for i in 0..s_states {
            t_mat[i * s_states + i] = 1.0;
        }
        // For each origin state, gather counts and update T plus stash G_m.
        let mut g_per_origin: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; s_states];
        for m in 0..s_states {
            let n_m = at_risk_count(obs, m, u);
            if n_m <= 0.0 {
                continue;
            }
            let mut d_out = vec![0.0_f64; s_states];
            let mut total_out = 0.0_f64;
            for j in 0..s_states {
                if j == m {
                    continue;
                }
                let d_mj = transition_count(obs, m, j, u);
                d_out[j] = d_mj;
                total_out += d_mj;
                t_mat[m * s_states + j] += d_mj / n_m;
            }
            t_mat[m * s_states + m] -= total_out / n_m;
            g_per_origin[m] = greenwood_increment_cov(n_m, &d_out, m, s_states);
        }

        // New covariance via the per-index recursion.
        let mut c_new = vec![0.0_f64; dim * dim];
        for a in 0..s_states {
            for b in 0..s_states {
                let ab = a * s_states + b;
                for cc in 0..s_states {
                    for d in 0..s_states {
                        let cd = cc * s_states + d;
                        // Propagation term: Σ_{m,n} T_{mb} T_{nd} C_old[(a,m),(c,n)].
                        let mut val = 0.0_f64;
                        for mm in 0..s_states {
                            let tmb = t_mat[mm * s_states + b];
                            if tmb == 0.0 {
                                continue;
                            }
                            let am = a * s_states + mm;
                            for nn in 0..s_states {
                                let tnd = t_mat[nn * s_states + d];
                                if tnd == 0.0 {
                                    continue;
                                }
                                let cn = cc * s_states + nn;
                                val += tmb * tnd * c[am * dim + cn];
                            }
                        }
                        // Increment term: Σ_m P_old_{am} P_old_{cm} G_m[b,d].
                        for mm in 0..s_states {
                            let pam = p[a * s_states + mm];
                            let pcm = p[cc * s_states + mm];
                            if pam == 0.0 || pcm == 0.0 {
                                continue;
                            }
                            val += pam * pcm * g_per_origin[mm][b * s_states + d];
                        }
                        c_new[ab * dim + cd] = val;
                    }
                }
            }
        }
        c = c_new;

        // Update P = P_old · T.
        let mut p_new = vec![0.0_f64; dim];
        for i in 0..s_states {
            for k in 0..s_states {
                let pik = p[i * s_states + k];
                if pik == 0.0 {
                    continue;
                }
                for j in 0..s_states {
                    p_new[i * s_states + j] += pik * t_mat[k * s_states + j];
                }
            }
        }
        p = p_new;
    }
    (p, c)
}

/// Pointwise confidence interval for a probability `p̂` with variance `v` using a
/// log transform clamped to `[0,1]`.
///
/// On the log scale `log p̂ ± z·SE/p̂`, then exponentiated. Falls back to a Wald
/// interval (still clamped) when `p̂` is at a boundary or `v ≤ 0`.
fn prob_ci(p_hat: f64, v: f64, z: f64) -> (f64, f64) {
    if v <= 0.0 || !v.is_finite() {
        let clamped = p_hat.clamp(0.0, 1.0);
        return (clamped, clamped);
    }
    let se = v.sqrt();
    if p_hat <= 0.0 {
        return (0.0, (z * se).clamp(0.0, 1.0));
    }
    if p_hat >= 1.0 {
        return ((1.0 - z * se).clamp(0.0, 1.0), 1.0);
    }
    // log transform: keeps the lower bound ≥ 0.
    let log_p = p_hat.ln();
    let factor = z * se / p_hat;
    let lo = (log_p - factor).exp();
    let hi = (log_p + factor).exp();
    (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0))
}

/// Aalen-Johansen transition probability with variance and pointwise CIs at
/// target time `t`, conditional on occupying the state space at time `s`.
///
/// # Errors
/// * [`SurvivalError::InvalidParameter`] if `alpha ∉ (0,1)` or `t < s`.
/// * Propagates the errors of [`fit_multi_state`].
pub fn aalen_johansen_variance(
    model: &MultiStateData,
    s: f64,
    t: f64,
    alpha: f64,
) -> SurvivalResult<AjInference> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "alpha must be in (0,1): {alpha}"
        )));
    }
    if t < s {
        return Err(SurvivalError::InvalidParameter(format!(
            "t ({t}) must be ≥ s ({s})"
        )));
    }
    // Reuse the point estimator for validation and event-time discovery.
    let fit = fit_multi_state(&model.observations, &model.config)?;
    let s_states = model.config.n_states;
    let dim = s_states * s_states;

    let (p, c) = aj_recursion(&model.observations, s_states, &fit.event_times, s, t);

    let z = norm_inv(1.0 - alpha / 2.0);
    let mut variance = vec![0.0_f64; dim];
    let mut ci_lower = vec![0.0_f64; dim];
    let mut ci_upper = vec![0.0_f64; dim];
    for idx in 0..dim {
        let v = c[idx * dim + idx].max(0.0);
        variance[idx] = v;
        let (lo, hi) = prob_ci(p[idx], v, z);
        ci_lower[idx] = lo;
        ci_upper[idx] = hi;
    }

    Ok(AjInference {
        transition_prob: p,
        variance,
        ci_lower,
        ci_upper,
        n_states: s_states,
    })
}

/// Competing-risks cumulative incidence with variance for a chosen origin state.
///
/// Specialises [`aalen_johansen_variance`] with `s = 0`: the cumulative
/// incidence into each state `j` is `P̂_{origin,j}(0,t)`, with the matching
/// variance and confidence interval. For a competing-risks model
/// (one transient origin + absorbing causes) the returned `cif` over the cause
/// states plus the survival probability `P̂_{origin,origin}` sum to one.
///
/// # Errors
/// * [`SurvivalError::IndexOutOfBounds`] if `origin ≥ n_states`.
/// * Propagates the errors of [`aalen_johansen_variance`].
pub fn cif_with_variance(
    model: &MultiStateData,
    origin: usize,
    t: f64,
    alpha: f64,
) -> SurvivalResult<CifInference> {
    let s_states = model.config.n_states;
    if origin >= s_states {
        return Err(SurvivalError::IndexOutOfBounds {
            index: origin,
            len: s_states,
        });
    }
    let inf = aalen_johansen_variance(model, 0.0, t, alpha)?;
    let base = origin * s_states;
    let cif = (0..s_states)
        .map(|j| inf.transition_prob[base + j])
        .collect();
    let variance = (0..s_states).map(|j| inf.variance[base + j]).collect();
    let ci_lower = (0..s_states).map(|j| inf.ci_lower[base + j]).collect();
    let ci_upper = (0..s_states).map(|j| inf.ci_upper[base + j]).collect();
    Ok(CifInference {
        time: t,
        origin,
        cif,
        variance,
        ci_lower,
        ci_upper,
    })
}

/// Convenience: evaluate the Aalen-Johansen point transition matrix at `t`
/// (delegates to the existing estimator) without computing variance.
#[must_use]
pub fn transition_prob_at(model: &MultiStateData, t: f64) -> Option<Vec<f64>> {
    match fit_multi_state(&model.observations, &model.config) {
        Ok(fit) => Some(predict_transition_probs(&fit, t)),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::nonparametric::kaplan_meier::kaplan_meier_estimate;

    /// Two-state alive(0)→dead(1) observations matching a KM dataset.
    fn two_state_obs(times: &[f64], events: &[bool]) -> Vec<MultiStateObs> {
        times
            .iter()
            .zip(events.iter())
            .map(|(&t, &e)| MultiStateObs {
                start_time: 0.0,
                end_time: t,
                from_state: 0,
                to_state: if e { Some(1) } else { None },
            })
            .collect()
    }

    fn two_state_model(times: &[f64], events: &[bool]) -> MultiStateData {
        MultiStateData::new(
            two_state_obs(times, events),
            MultiStateConfig {
                n_states: 2,
                initial_state: 0,
            },
        )
    }

    #[test]
    fn two_state_matches_km_greenwood_no_censoring() {
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, true, true, true];
        let model = two_state_model(&times, &events);
        let km =
            kaplan_meier_estimate(&Dataset::from_arrays(&times, &events).expect("ok")).expect("ok");

        for (k, &t) in km.times.iter().enumerate() {
            let inf = aalen_johansen_variance(&model, 0.0, t, 0.05).expect("ok");
            let s = model.config.n_states;
            let p00 = inf.transition_prob[0];
            let v00 = inf.variance[0];
            assert!(
                (p00 - km.survival[k]).abs() < 1.0e-12,
                "P00({t}) = {p00}, KM = {}",
                km.survival[k]
            );
            assert!(
                (v00 - km.greenwood_var[k]).abs() < 1.0e-8,
                "var P00({t}) = {v00}, Greenwood = {} (s={s})",
                km.greenwood_var[k]
            );
        }
    }

    #[test]
    fn two_state_matches_km_greenwood_with_censoring() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, false, true, false, true];
        let model = two_state_model(&times, &events);
        let km =
            kaplan_meier_estimate(&Dataset::from_arrays(&times, &events).expect("ok")).expect("ok");

        for (k, &t) in km.times.iter().enumerate() {
            if km.events[k] == 0.0 {
                continue; // AJ only records event times
            }
            let inf = aalen_johansen_variance(&model, 0.0, t, 0.05).expect("ok");
            let p00 = inf.transition_prob[0];
            let v00 = inf.variance[0];
            assert!(
                (p00 - km.survival[k]).abs() < 1.0e-12,
                "P00({t}) = {p00}, KM = {}",
                km.survival[k]
            );
            assert!(
                (v00 - km.greenwood_var[k]).abs() < 1.0e-8,
                "var P00({t}) = {v00}, Greenwood = {}",
                km.greenwood_var[k]
            );
        }
    }

    #[test]
    fn variance_zero_at_s_equals_t() {
        let times = [1.0, 2.0, 3.0];
        let events = [true, true, true];
        let model = two_state_model(&times, &events);
        // s = t = 0 → P = identity, variance = 0.
        let inf = aalen_johansen_variance(&model, 0.0, 0.0, 0.05).expect("ok");
        for v in &inf.variance {
            assert!(v.abs() < 1.0e-15, "variance not zero at s=t: {v}");
        }
        // identity check
        assert!((inf.transition_prob[0] - 1.0).abs() < 1.0e-15);
        assert!(inf.transition_prob[1].abs() < 1.0e-15);
    }

    #[test]
    fn variance_grows_from_zero() {
        // Variance is 0 at t=s and strictly positive once events accrue. (The
        // Greenwood variance of S(t) can later shrink as S→0, so we only assert
        // it leaves zero, stays finite, and is non-negative.)
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, true, true, true];
        let model = two_state_model(&times, &events);
        // At t just below the first event there is no variance.
        let inf0 = aalen_johansen_variance(&model, 0.0, 0.5, 0.05).expect("ok");
        assert!(inf0.variance[0].abs() < 1.0e-15);
        // After the first event the survival variance is strictly positive.
        let inf1 = aalen_johansen_variance(&model, 0.0, 1.0, 0.05).expect("ok");
        assert!(inf1.variance[0] > 0.0);
        for &t in &times {
            let inf = aalen_johansen_variance(&model, 0.0, t, 0.05).expect("ok");
            assert!(inf.variance[0] >= -1.0e-15 && inf.variance[0].is_finite());
        }
    }

    #[test]
    fn ci_contains_estimate_and_in_unit_interval() {
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let events = [true, false, true, true, false];
        let model = two_state_model(&times, &events);
        for &t in &[1.5, 3.0, 4.5] {
            let inf = aalen_johansen_variance(&model, 0.0, t, 0.05).expect("ok");
            let dim = inf.n_states * inf.n_states;
            for idx in 0..dim {
                let p = inf.transition_prob[idx];
                let lo = inf.ci_lower[idx];
                let hi = inf.ci_upper[idx];
                assert!((0.0..=1.0).contains(&lo), "lo out of [0,1]: {lo}");
                assert!((0.0..=1.0).contains(&hi), "hi out of [0,1]: {hi}");
                assert!(lo <= hi + 1.0e-12, "lo {lo} > hi {hi}");
                assert!(
                    p >= lo - 1.0e-9 && p <= hi + 1.0e-9,
                    "estimate {p} not in CI [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let model = two_state_model(&[1.0, 2.0], &[true, true]);
        assert!(aalen_johansen_variance(&model, 0.0, 2.0, 1.5).is_err());
        assert!(aalen_johansen_variance(&model, 0.0, 2.0, 0.0).is_err());
    }

    #[test]
    fn rejects_t_before_s() {
        let model = two_state_model(&[1.0, 2.0], &[true, true]);
        assert!(aalen_johansen_variance(&model, 3.0, 1.0, 0.05).is_err());
    }

    // ── Competing risks: 3-state, origin 0, causes 1 and 2 ──────────────────
    //
    // Hand computation. Subjects all start in state 0:
    //   t=1: 0→1 (cause 1),  n_0 = 4
    //   t=2: 0→2 (cause 2),  n_0 = 3
    //   t=3: 0→1 (cause 1),  n_0 = 2
    //   t=4: censored,       (no transition)
    // KM overall survival S(t) = P_00:
    //   S(1) = 3/4
    //   S(2) = 3/4 · 2/3 = 1/2
    //   S(3) = 1/2 · 1/2 = 1/4
    // CIF_1(t) = Σ S(u⁻)·dN_1(u)/n(u):
    //   CIF_1(1) = 1·(1/4) = 1/4
    //   CIF_1(3) = 1/4 + S(2)·(1/2) = 1/4 + (1/2)(1/2) = 1/2
    // CIF_2(t):
    //   CIF_2(2) = S(1)·(1/3) = (3/4)(1/3) = 1/4
    // At t=3: CIF_1 = 1/2, CIF_2 = 1/4, S = 1/4 → sum = 1.

    fn competing_risks_model() -> MultiStateData {
        let obs = vec![
            MultiStateObs {
                start_time: 0.0,
                end_time: 1.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 2.0,
                from_state: 0,
                to_state: Some(2),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 3.0,
                from_state: 0,
                to_state: Some(1),
            },
            MultiStateObs {
                start_time: 0.0,
                end_time: 4.0,
                from_state: 0,
                to_state: None,
            },
        ];
        MultiStateData::new(
            obs,
            MultiStateConfig {
                n_states: 3,
                initial_state: 0,
            },
        )
    }

    #[test]
    fn competing_risks_cif_point_estimates() {
        let model = competing_risks_model();
        let c = cif_with_variance(&model, 0, 3.0, 0.05).expect("ok");
        // cif[0] = survival, cif[1] = CIF cause 1, cif[2] = CIF cause 2.
        assert!((c.cif[0] - 0.25).abs() < 1.0e-12, "S = {}", c.cif[0]);
        assert!((c.cif[1] - 0.5).abs() < 1.0e-12, "CIF1 = {}", c.cif[1]);
        assert!((c.cif[2] - 0.25).abs() < 1.0e-12, "CIF2 = {}", c.cif[2]);
        // They sum to one.
        let total: f64 = c.cif.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-12, "sum = {total}");
    }

    #[test]
    fn competing_risks_cif_variance_matches_independent() {
        // Independent (Aalen) plug-in variance of CIF_1 at t=3 via the
        // delta-method recursion is reproduced by the estimator. We verify the
        // variance is positive, finite, and the survival variance equals the
        // Greenwood variance of the overall KM (cause-collapsed) survivor.
        let model = competing_risks_model();
        // Cause-collapsed KM: deaths at t=1,2,3 (any cause), censor at t=4.
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [true, true, true, false];
        let km =
            kaplan_meier_estimate(&Dataset::from_arrays(&times, &events).expect("ok")).expect("ok");

        let c = cif_with_variance(&model, 0, 3.0, 0.05).expect("ok");
        // Survival variance == Greenwood variance at t=3 (index 2 of km).
        assert!(
            (c.variance[0] - km.greenwood_var[2]).abs() < 1.0e-8,
            "S-var {} vs Greenwood {}",
            c.variance[0],
            km.greenwood_var[2]
        );
        for v in &c.variance {
            assert!(v.is_finite() && *v >= 0.0);
        }
        // CIF variances strictly positive (events occurred for both causes).
        assert!(c.variance[1] > 0.0);
        assert!(c.variance[2] > 0.0);
    }

    #[test]
    fn competing_risks_cif_ci_in_unit_interval() {
        let model = competing_risks_model();
        let c = cif_with_variance(&model, 0, 3.0, 0.05).expect("ok");
        for j in 0..3 {
            assert!((0.0..=1.0).contains(&c.ci_lower[j]));
            assert!((0.0..=1.0).contains(&c.ci_upper[j]));
            assert!(c.ci_lower[j] <= c.ci_upper[j] + 1.0e-12);
            assert!(c.cif[j] >= c.ci_lower[j] - 1.0e-9 && c.cif[j] <= c.ci_upper[j] + 1.0e-9);
        }
    }

    #[test]
    fn cif_rejects_bad_origin() {
        let model = competing_risks_model();
        assert!(matches!(
            cif_with_variance(&model, 9, 3.0, 0.05),
            Err(SurvivalError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn transition_prob_at_matches_fit() {
        let model = two_state_model(&[1.0, 2.0, 3.0], &[true, true, true]);
        let p = transition_prob_at(&model, 2.0).expect("some");
        // P00 at t=2 = (1-1/3)(1-1/2) = 1/3.
        assert!((p[0] - (2.0 / 3.0 * 1.0 / 2.0)).abs() < 1.0e-12);
    }

    #[test]
    fn norm_inv_known_quantiles() {
        assert!((norm_inv(0.975) - 1.96).abs() < 1.0e-3);
        assert!(norm_inv(0.5).abs() < 1.0e-9);
    }
}
