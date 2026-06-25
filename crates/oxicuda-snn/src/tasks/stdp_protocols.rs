//! Controlled spike-pair / pairing protocols that exercise the crate's real
//! STDP rules and let their textbook properties be *measured* rather than
//! assumed.
//!
//! * [`pair_stdp_window`] reproduces the classic STDP learning window
//!   `W(Δt)` (Bi & Poo 1998) from a single isolated pre/post spike pair: the
//!   sign flips at `Δt = 0` and the magnitude decays exponentially in `|Δt|`.
//! * [`pair_stdp_poisson_final_weight`] drives a synapse with Poisson spike
//!   trains; with the crate's default (slightly LTD-dominated) amplitudes,
//!   *uncorrelated* activity drives the weight **down**, while a causal pre→post
//!   correlation drives it **up** — the competitive convergence of
//!   Song-Miller-Abbott (2000).
//! * [`triplet_pairing_dw`] runs a frequency-controlled pre→post pairing
//!   protocol through the triplet rule. Because the triplet potentiation term
//!   scales with the slow post-synaptic trace, total potentiation grows with
//!   the pairing rate — the rate-dependent, BCM-like behaviour
//!   (Pfister & Gerstner 2006) that the pair rule on its own lacks.

use crate::error::SnnResult;
use crate::handle::LcgRng;
use crate::plasticity::stdp::{StdpConfig, StdpTraces, stdp_step};
use crate::plasticity::triplet_stdp::{TripletStdpConfig, TripletTraces, triplet_stdp_step};

/// Mid-range initial weight used by the window protocols (far from both clips,
/// so the tiny pair-rule updates are never masked by clamping).
const WINDOW_W_INIT: f32 = 0.5;

/// Measure the pair-STDP learning window `W(Δt)` from one isolated spike pair.
///
/// A single 1×1 synapse with freshly reset traces is given exactly one pre
/// spike and one post spike separated by `dt_steps` timesteps (positive ⇒ pre
/// precedes post). The returned value is the signed net weight change `Δw`
/// produced by the crate's real [`stdp_step`]:
///
/// * `dt_steps > 0` (pre→post) ⇒ potentiation, `Δw > 0`;
/// * `dt_steps < 0` (post→pre) ⇒ depression, `Δw < 0`;
/// * `|Δw|` decays like `exp(−|Δt|/τ)`.
///
/// # Errors
///
/// Propagates [`stdp_step`] validation errors (e.g. invalid time constants).
pub fn pair_stdp_window(dt_steps: i32, cfg: &StdpConfig) -> SnnResult<f32> {
    let mut w = vec![WINDOW_W_INIT];
    let mut traces = StdpTraces::new(1, 1);

    if dt_steps == 0 {
        stdp_step(&mut w, &mut traces, &[1.0], &[1.0], 1, 1, cfg)?;
        return Ok(w[0] - WINDOW_W_INIT);
    }

    let lag = dt_steps.unsigned_abs() as usize;
    // Which neuron fires first depends on the sign of Δt.
    let (first_pre, first_post, second_pre, second_post) = if dt_steps > 0 {
        (1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32) // pre leads
    } else {
        (0.0_f32, 1.0_f32, 1.0_f32, 0.0_f32) // post leads
    };

    stdp_step(&mut w, &mut traces, &[first_pre], &[first_post], 1, 1, cfg)?;
    for _ in 1..lag {
        stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, cfg)?;
    }
    stdp_step(
        &mut w,
        &mut traces,
        &[second_pre],
        &[second_post],
        1,
        1,
        cfg,
    )?;
    Ok(w[0] - WINDOW_W_INIT)
}

/// Drive a 1×1 synapse with Poisson pre/post spike trains and return the final
/// weight.
///
/// At every step the pre neuron spikes with probability `pre_rate` and the post
/// neuron with probability `post_rate`. When `correlation > 0`, a pre spike is
/// additionally copied (with that probability) onto the *next* step's post
/// neuron, injecting a causal pre→post (`Δt = 1`) correlation. The weight is
/// clamped to the config's `[w_min, w_max]` band each step, so the returned
/// value is the converged operating point under the chosen spike statistics.
///
/// # Errors
///
/// Propagates [`stdp_step`] validation errors.
pub fn pair_stdp_poisson_final_weight(
    pre_rate: f32,
    post_rate: f32,
    correlation: f32,
    w_init: f32,
    n_steps: usize,
    seed: u64,
    cfg: &StdpConfig,
) -> SnnResult<f32> {
    let mut rng = LcgRng::new(seed);
    let mut w = vec![w_init];
    let mut traces = StdpTraces::new(1, 1);
    let mut prev_pre = 0.0_f32;
    for _ in 0..n_steps {
        let pre = if rng.next_f32() < pre_rate { 1.0 } else { 0.0 };
        let mut post = if rng.next_f32() < post_rate { 1.0 } else { 0.0 };
        if correlation > 0.0 && prev_pre > 0.0 && rng.next_f32() < correlation {
            post = 1.0;
        }
        stdp_step(&mut w, &mut traces, &[pre], &[post], 1, 1, cfg)?;
        prev_pre = pre;
    }
    Ok(w[0])
}

/// Total weight change `Δw` from a frequency-controlled pre→post pairing
/// protocol run through the triplet rule.
///
/// `n_pairs` identical cycles of length `period_steps` are presented; within
/// each cycle the pre neuron spikes at local step `0` and the post neuron at
/// local step `dt_steps`. A *smaller* `period_steps` therefore means a *higher*
/// pairing frequency (and post firing rate). The weight starts at zero, so the
/// return value is the accumulated `Δw`; the caller should pass a `cfg` with a
/// wide `[w_min, w_max]` band so clamping never masks the measurement.
///
/// Setting `cfg.a2_plus = cfg.a2_minus = 0` reduces this to the pure pair rule,
/// giving an apples-to-apples baseline for the triplet rate dependence.
///
/// # Errors
///
/// Propagates [`triplet_stdp_step`] validation errors.
pub fn triplet_pairing_dw(
    n_pairs: usize,
    period_steps: usize,
    dt_steps: usize,
    cfg: &TripletStdpConfig,
) -> SnnResult<f32> {
    let mut w = vec![0.0_f32];
    let mut traces = TripletTraces::new(1, 1);
    let period = period_steps.max(dt_steps + 1);
    for _ in 0..n_pairs {
        for s in 0..period {
            let pre = if s == 0 { 1.0 } else { 0.0 };
            let post = if s == dt_steps { 1.0 } else { 0.0 };
            triplet_stdp_step(&mut w, &mut traces, &[pre], &[post], 1, 1, cfg)?;
        }
    }
    Ok(w[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default config but with a wide weight band so triplet pairing protocols
    /// accumulate without clamping.
    fn wide_triplet_cfg(a2: f32) -> TripletStdpConfig {
        let mut cfg = TripletStdpConfig {
            a2_plus: a2,
            a2_minus: a2,
            ..Default::default()
        };
        cfg.stdp.w_min = -100.0;
        cfg.stdp.w_max = 100.0;
        cfg
    }

    // ── Pair-STDP learning window: sign + exponential shape ─────────────────

    #[test]
    fn pair_window_sign_law() {
        let cfg = StdpConfig::default();
        // Pre before post (Δt > 0) ⇒ potentiation.
        for dt in [1, 2, 5, 10, 20] {
            let dw = pair_stdp_window(dt, &cfg).expect("window");
            assert!(dw > 0.0, "Δt=+{dt} should potentiate, got Δw={dw}");
        }
        // Post before pre (Δt < 0) ⇒ depression.
        for dt in [1, 2, 5, 10, 20] {
            let dw = pair_stdp_window(-dt, &cfg).expect("window");
            assert!(dw < 0.0, "Δt=-{dt} should depress, got Δw={dw}");
        }
    }

    #[test]
    fn pair_window_decays_with_lag() {
        let cfg = StdpConfig::default();
        // Magnitude must decrease monotonically with |Δt| on both branches.
        let lags = [1, 2, 4, 8, 16, 32];
        let mut prev_pot = f32::INFINITY;
        let mut prev_dep = f32::INFINITY;
        for &dt in &lags {
            let pot = pair_stdp_window(dt, &cfg).expect("pot").abs();
            let dep = pair_stdp_window(-dt, &cfg).expect("dep").abs();
            assert!(pot < prev_pot, "potentiation not decreasing at Δt=+{dt}");
            assert!(dep < prev_dep, "depression not decreasing at Δt=-{dt}");
            prev_pot = pot;
            prev_dep = dep;
        }
    }

    #[test]
    fn pair_window_matches_exponential_shape() {
        let cfg = StdpConfig::default();
        // W(Δt) = A₊·exp(−(Δt−1)/τ₊) for Δt ≥ 1 (the trace is decayed (Δt−1)
        // times between the first and second spike). Verify the ratio law.
        let w1 = pair_stdp_window(1, &cfg).expect("w1");
        for dt in [2, 4, 8, 16] {
            let wdt = pair_stdp_window(dt, &cfg).expect("wdt");
            let predicted = w1 * (-((dt - 1) as f32) / cfg.tau_plus).exp();
            let rel = (wdt - predicted).abs() / predicted.abs().max(1.0e-12);
            assert!(
                rel < 0.02,
                "Δt=+{dt}: measured {wdt} vs exp-law {predicted} (rel err {rel})"
            );
        }
    }

    #[test]
    fn pair_window_amplitude_asymmetry() {
        // The default rule is LTD-dominated (a_minus > a_plus): the depression
        // lobe at Δt=−1 must exceed the potentiation lobe at Δt=+1 in size.
        let cfg = StdpConfig::default();
        let pot = pair_stdp_window(1, &cfg).expect("pot");
        let dep = pair_stdp_window(-1, &cfg).expect("dep");
        assert!(
            dep.abs() > pot.abs(),
            "expected |LTD| > |LTP|: |{dep}| vs |{pot}|"
        );
    }

    // ── Pair-STDP convergence under Poisson statistics ──────────────────────

    #[test]
    fn pair_poisson_uncorrelated_depresses() {
        // Uncorrelated Poisson activity with the LTD-dominated default rule must
        // drive the weight below its starting point (competitive weakening).
        let cfg = StdpConfig::default();
        let w_init = 0.5_f32;
        let w_final = pair_stdp_poisson_final_weight(0.10, 0.10, 0.0, w_init, 6000, 42, &cfg)
            .expect("uncorr");
        assert!(w_final.is_finite());
        assert!(
            w_final < w_init,
            "uncorrelated Poisson should depress: {w_init} → {w_final}"
        );
    }

    #[test]
    fn pair_poisson_causal_correlation_potentiates() {
        // A causal pre→post correlation must converge the weight ABOVE the
        // uncorrelated operating point (timing structure drives potentiation).
        let cfg = StdpConfig::default();
        let w_init = 0.5_f32;
        let w_uncorr =
            pair_stdp_poisson_final_weight(0.10, 0.10, 0.0, w_init, 6000, 7, &cfg).expect("uncorr");
        let w_corr =
            pair_stdp_poisson_final_weight(0.10, 0.10, 0.9, w_init, 6000, 7, &cfg).expect("corr");
        assert!(w_corr.is_finite() && w_uncorr.is_finite());
        assert!(
            w_corr > w_uncorr,
            "causal correlation should raise the weight: corr={w_corr}, uncorr={w_uncorr}"
        );
    }

    #[test]
    fn pair_poisson_is_deterministic() {
        let cfg = StdpConfig::default();
        let a = pair_stdp_poisson_final_weight(0.1, 0.1, 0.5, 0.5, 2000, 99, &cfg).expect("a");
        let b = pair_stdp_poisson_final_weight(0.1, 0.1, 0.5, 0.5, 2000, 99, &cfg).expect("b");
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // ── Triplet-STDP rate-dependent (BCM-like) behaviour ────────────────────

    #[test]
    fn triplet_adds_potentiation_over_pair() {
        // At any pairing rate the triplet rule must produce a more potentiating
        // (larger) weight change than the pure pair rule (a2 = 0): the extra
        // third-factor term is potentiating. (The absolute Δw also carries the
        // inter-cycle post→pre depression shared by both rules, so the honest
        // comparison is triplet-vs-pair at matched frequency.)
        let n_pairs = 30;
        let dt = 2;
        let trip = wide_triplet_cfg(5.0e-3);
        let pair = wide_triplet_cfg(0.0);
        for period in [5usize, 50] {
            let dw_trip = triplet_pairing_dw(n_pairs, period, dt, &trip).expect("trip");
            let dw_pair = triplet_pairing_dw(n_pairs, period, dt, &pair).expect("pair");
            assert!(dw_trip.is_finite() && dw_pair.is_finite());
            assert!(
                dw_trip > dw_pair,
                "triplet should add potentiation at period={period}: trip={dw_trip}, pair={dw_pair}"
            );
        }
    }

    #[test]
    fn triplet_extra_potentiation_grows_with_rate() {
        // BCM-like rate dependence: the triplet-specific potentiation
        // (Δw_triplet − Δw_pair, isolating the third-factor term that scales with
        // the slow post-synaptic trace) must be positive AND grow with the
        // pairing rate — behaviour the pair rule structurally cannot produce.
        let n_pairs = 30;
        let dt = 2;
        let trip = wide_triplet_cfg(5.0e-3);
        let pair = wide_triplet_cfg(0.0);

        let extra_high = triplet_pairing_dw(n_pairs, 5, dt, &trip).expect("th")
            - triplet_pairing_dw(n_pairs, 5, dt, &pair).expect("ph");
        let extra_low = triplet_pairing_dw(n_pairs, 50, dt, &trip).expect("tl")
            - triplet_pairing_dw(n_pairs, 50, dt, &pair).expect("pl");

        assert!(
            extra_low > 0.0,
            "triplet term should potentiate even at low rate: {extra_low}"
        );
        assert!(
            extra_high > extra_low,
            "triplet potentiation must increase with rate: high={extra_high}, low={extra_low}"
        );
    }

    #[test]
    fn triplet_pairing_is_deterministic() {
        let cfg = wide_triplet_cfg(5.0e-3);
        let a = triplet_pairing_dw(20, 8, 2, &cfg).expect("a");
        let b = triplet_pairing_dw(20, 8, 2, &cfg).expect("b");
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
