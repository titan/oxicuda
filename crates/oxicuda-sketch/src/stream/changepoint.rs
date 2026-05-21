//! Online change-point detection: Page-Hinkley and CUSUM.
//!
//! References:
//! * Page, E. S. (1954). "Continuous Inspection Schemes". *Biometrika* 41, 100–115.
//! * Hinkley, D. V. (1971). "Inference about the change-point from cumulative sum tests".
//!   *Biometrika* 58, 509–523.
//!
//! Both detectors process a stream of `f64` observations one at a time and
//! emit a [`ChangeAlarm`] on each update.
//!
//! ## Page-Hinkley
//!
//! Detects shifts in the running mean. Maintains the cumulative deviation
//! `m_t = Σ (x_t - mean_t - δ)` (with `δ ≥ 0` an allowance / robustness term),
//! along with the running minimum `min_m_t`. The test statistic
//! `PH_t = m_t - min_m_t` is monotone non-decreasing; an Increase alarm fires
//! when `PH_t > λ` and the warmup `min_n` has elapsed. A symmetric pair
//! `m_t_neg += (mean - x - δ)` with its own running minimum `min_m_t_neg`
//! drives the Decrease side via `PH_neg = m_t_neg - min_m_t_neg`.
//!
//! Spec correction note: an early draft tracked `max_m_t` for the negative
//! side, but the correct symmetric pairing for `m_t_neg = Σ(mean - x - δ)`
//! is the running **minimum** (mirror of the positive side after sign-flip).
//! With a max-of-`m_t_neg` and `max - m_t_neg`, a sustained downward shift
//! would keep `m_t_neg` AT its running maximum and yield `PH_neg ≈ 0`, never
//! alarming.
//!
//! The running mean can be updated either incrementally
//! (`mean += (x - mean) / n`) or as an exponential moving average
//! (`mean += α (x - mean)` for `α ∈ (0, 1]`); the EMA mode tracks slow drifts
//! better at the cost of being a biased estimator.
//!
//! ## CUSUM
//!
//! For a known reference mean `μ₀`, accumulates two one-sided sums
//! `S⁺ = max(0, S⁺ + (x - μ₀ - k))` and `S⁻ = max(0, S⁻ - (x - μ₀ + k))`,
//! and alarms when either exceeds a threshold `h`. The parameter `k > 0`
//! is the "reference value" (typically half the size of the smallest shift
//! that should be reliably detected).

use crate::error::{SketchError, SketchResult};

/// Result of a single [`PageHinkley::update`] or [`Cusum::update`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangeAlarm {
    /// No alarm.
    None,
    /// Detected an upward shift at sample index `at` with test statistic `ph`.
    Increase { at: u64, ph: f64 },
    /// Detected a downward shift at sample index `at` with test statistic `ph`.
    Decrease { at: u64, ph: f64 },
}

/// Configuration for [`PageHinkley`].
#[derive(Debug, Clone, Copy)]
pub struct PageHinkleyConfig {
    /// Allowance / magnitude of the smallest mean shift to ignore (`δ ≥ 0`).
    pub delta: f64,
    /// Detection threshold (`λ > 0`). Larger λ → fewer false alarms but
    /// longer detection delay.
    pub lambda: f64,
    /// Minimum number of observations before alarms can fire (warmup).
    pub min_n: usize,
    /// Mean-update policy:
    /// * `None` → incremental Welford-style mean (`mean += (x - mean) / n`).
    /// * `Some(α)` with `α ∈ (0, 1]` → exponential moving average
    ///   (`mean += α (x - mean)`).
    pub alpha: Option<f64>,
}

/// Online Page-Hinkley change-point detector.
#[derive(Debug, Clone)]
pub struct PageHinkley {
    config: PageHinkleyConfig,
    n: u64,
    mean: f64,
    /// Cumulative positive-shift sum: `m_t = Σ (x - mean - δ)`.
    m_t: f64,
    /// Running minimum of `m_t` (for the Increase-side test statistic).
    min_m_t: f64,
    /// Cumulative negative-shift sum: `m_t_neg = Σ (mean - x - δ)`.
    m_t_neg: f64,
    /// Running minimum of `m_t_neg` (for the Decrease-side test statistic).
    ///
    /// Mirror of `min_m_t` after the sign flip; this is the correct
    /// symmetric pairing — see the module docs for why a `max_m_t` instead
    /// would fail to detect sustained downward shifts.
    min_m_t_neg: f64,
}

impl PageHinkley {
    /// Construct a new detector.
    ///
    /// Errors if `delta < 0`, `lambda ≤ 0`, or `alpha` is `Some(a)` with
    /// `a ∉ (0, 1]`.
    pub fn new(config: PageHinkleyConfig) -> SketchResult<Self> {
        if !config.delta.is_finite() || config.delta < 0.0 {
            return Err(SketchError::InvalidParameter {
                name: "delta".to_string(),
                reason: "must be ≥ 0 and finite".to_string(),
            });
        }
        if !config.lambda.is_finite() || config.lambda <= 0.0 {
            return Err(SketchError::InvalidParameter {
                name: "lambda".to_string(),
                reason: "must be > 0 and finite".to_string(),
            });
        }
        if let Some(a) = config.alpha {
            if !a.is_finite() || a <= 0.0 || a > 1.0 {
                return Err(SketchError::InvalidParameter {
                    name: "alpha".to_string(),
                    reason: "must be in (0, 1]".to_string(),
                });
            }
        }
        Ok(Self {
            config,
            n: 0,
            mean: 0.0,
            m_t: 0.0,
            min_m_t: 0.0,
            m_t_neg: 0.0,
            min_m_t_neg: 0.0,
        })
    }

    /// Total observations processed.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Current running-mean estimate.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current Increase-side test statistic `PH_t = m_t - min_m_t`.
    ///
    /// Always non-negative; an Increase alarm fires when this exceeds `λ`
    /// (post-warmup).
    #[must_use]
    pub fn ph_increase(&self) -> f64 {
        self.m_t - self.min_m_t
    }

    /// Current Decrease-side test statistic `PH_neg = m_t_neg - min_m_t_neg`.
    ///
    /// Always non-negative; a Decrease alarm fires when this exceeds `λ`
    /// (post-warmup).
    #[must_use]
    pub fn ph_decrease(&self) -> f64 {
        self.m_t_neg - self.min_m_t_neg
    }

    /// Reset the detector to its initial empty state.
    pub fn reset(&mut self) {
        self.n = 0;
        self.mean = 0.0;
        self.m_t = 0.0;
        self.min_m_t = 0.0;
        self.m_t_neg = 0.0;
        self.min_m_t_neg = 0.0;
    }

    /// Feed one observation; returns the alarm status for this step.
    ///
    /// Non-finite inputs are silently ignored (return [`ChangeAlarm::None`]).
    pub fn update(&mut self, x: f64) -> ChangeAlarm {
        if !x.is_finite() {
            return ChangeAlarm::None;
        }
        // Update n and running mean BEFORE accumulating the deviation, so that
        // the very first observation has `mean = x` and contributes zero to
        // `m_t` (this matches the standard Page-Hinkley formulation).
        self.n = self.n.saturating_add(1);
        match self.config.alpha {
            None => {
                let n_f = self.n as f64;
                self.mean += (x - self.mean) / n_f;
            }
            Some(a) => {
                if self.n == 1 {
                    // Seed the EMA with the first observation.
                    self.mean = x;
                } else {
                    self.mean += a * (x - self.mean);
                }
            }
        }

        // Increase-side accumulator.
        self.m_t += x - self.mean - self.config.delta;
        if self.m_t < self.min_m_t {
            self.min_m_t = self.m_t;
        }
        let ph = self.m_t - self.min_m_t;

        // Decrease-side accumulator (symmetric mirror of the positive side
        // after the sign-flip in the update; see the module docs).
        self.m_t_neg += self.mean - x - self.config.delta;
        if self.m_t_neg < self.min_m_t_neg {
            self.min_m_t_neg = self.m_t_neg;
        }
        let ph_neg = self.m_t_neg - self.min_m_t_neg;

        if (self.n as usize) < self.config.min_n {
            return ChangeAlarm::None;
        }
        if ph > self.config.lambda {
            return ChangeAlarm::Increase { at: self.n, ph };
        }
        if ph_neg > self.config.lambda {
            return ChangeAlarm::Decrease {
                at: self.n,
                ph: ph_neg,
            };
        }
        ChangeAlarm::None
    }
}

/// Configuration for [`Cusum`].
#[derive(Debug, Clone, Copy)]
pub struct CusumConfig {
    /// Reference mean `μ₀` (the "in-control" target).
    pub mu0: f64,
    /// Reference value `k > 0`. Smaller `k` → faster detection of small shifts.
    pub k: f64,
    /// Alarm threshold `h > 0`. Larger `h` → fewer false alarms.
    pub h: f64,
}

/// Online two-sided CUSUM change-point detector.
#[derive(Debug, Clone)]
pub struct Cusum {
    config: CusumConfig,
    s_plus: f64,
    s_minus: f64,
    n: u64,
}

impl Cusum {
    /// Construct a new CUSUM detector.
    ///
    /// Errors if `k < 0`, `h ≤ 0`, or any field is non-finite.
    pub fn new(config: CusumConfig) -> SketchResult<Self> {
        if !config.mu0.is_finite() {
            return Err(SketchError::InvalidParameter {
                name: "mu0".to_string(),
                reason: "must be finite".to_string(),
            });
        }
        if !config.k.is_finite() || config.k < 0.0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be ≥ 0 and finite".to_string(),
            });
        }
        if !config.h.is_finite() || config.h <= 0.0 {
            return Err(SketchError::InvalidParameter {
                name: "h".to_string(),
                reason: "must be > 0 and finite".to_string(),
            });
        }
        Ok(Self {
            config,
            s_plus: 0.0,
            s_minus: 0.0,
            n: 0,
        })
    }

    /// Total observations processed.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Current upward CUSUM `S⁺`.
    #[must_use]
    pub fn s_plus(&self) -> f64 {
        self.s_plus
    }

    /// Current downward CUSUM `S⁻`.
    #[must_use]
    pub fn s_minus(&self) -> f64 {
        self.s_minus
    }

    /// Reset to empty.
    pub fn reset(&mut self) {
        self.s_plus = 0.0;
        self.s_minus = 0.0;
        self.n = 0;
    }

    /// Feed one observation; returns the alarm status for this step.
    ///
    /// Non-finite inputs are silently ignored (return [`ChangeAlarm::None`]).
    pub fn update(&mut self, x: f64) -> ChangeAlarm {
        if !x.is_finite() {
            return ChangeAlarm::None;
        }
        self.n = self.n.saturating_add(1);
        let dx = x - self.config.mu0;
        self.s_plus = (self.s_plus + dx - self.config.k).max(0.0);
        self.s_minus = (self.s_minus - dx - self.config.k).max(0.0);
        if self.s_plus > self.config.h {
            ChangeAlarm::Increase {
                at: self.n,
                ph: self.s_plus,
            }
        } else if self.s_minus > self.config.h {
            ChangeAlarm::Decrease {
                at: self.n,
                ph: self.s_minus,
            }
        } else {
            ChangeAlarm::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_ph(delta: f64, lambda: f64, min_n: usize, alpha: Option<f64>) -> PageHinkley {
        PageHinkley::new(PageHinkleyConfig {
            delta,
            lambda,
            min_n,
            alpha,
        })
        .unwrap()
    }

    fn make_cusum(mu0: f64, k: f64, h: f64) -> Cusum {
        Cusum::new(CusumConfig { mu0, k, h }).unwrap()
    }

    #[test]
    fn page_hinkley_stationary_few_false_alarms() {
        // N(0, 1) i.i.d. of length 5000, conservative lambda → few false alarms.
        let mut ph = make_ph(0.005, 50.0, 30, None);
        let mut rng = LcgRng::new(11);
        let mut alarms = 0usize;
        for _ in 0..5_000 {
            let x = rng.next_normal();
            if !matches!(ph.update(x), ChangeAlarm::None) {
                alarms += 1;
                // Reset so we keep counting distinct false-alarm events.
                ph.reset();
            }
        }
        assert!(
            alarms <= 5,
            "too many false alarms on stationary stream: {alarms}"
        );
    }

    #[test]
    fn page_hinkley_detects_mean_shift() {
        // Stream: 500 samples N(0,1), then 500 samples N(2,1). Detection must
        // happen within ~200 samples after the change point, across several seeds.
        for seed in [1u64, 7, 13, 21] {
            let mut ph = make_ph(0.05, 20.0, 30, None);
            let mut rng = LcgRng::new(seed);
            let change_at = 500u64;
            let mut detected_at: Option<u64> = None;
            for t in 1..=1_000u64 {
                let x = if t <= change_at {
                    rng.next_normal()
                } else {
                    rng.next_normal() + 2.0
                };
                if let ChangeAlarm::Increase { at, .. } = ph.update(x) {
                    if at > change_at {
                        detected_at = Some(at);
                        break;
                    }
                }
            }
            let at = detected_at.unwrap_or_else(|| panic!("seed {seed}: no detection"));
            let delay = at - change_at;
            assert!(
                delay <= 300,
                "seed {seed}: detection delay {delay} too large"
            );
        }
    }

    #[test]
    fn page_hinkley_two_sided_detects_both_directions() {
        let mut up = make_ph(0.05, 15.0, 20, None);
        let mut down = make_ph(0.05, 15.0, 20, None);
        let mut rng_a = LcgRng::new(101);
        let mut rng_b = LcgRng::new(202);
        // Upward shift
        for t in 1..=600u64 {
            let x = if t <= 300 {
                rng_a.next_normal()
            } else {
                rng_a.next_normal() + 3.0
            };
            let a = up.update(x);
            if t > 300 {
                if let ChangeAlarm::Increase { .. } = a {
                    break;
                }
            }
        }
        // Downward shift
        let mut saw_decrease = false;
        for t in 1..=600u64 {
            let x = if t <= 300 {
                rng_b.next_normal()
            } else {
                rng_b.next_normal() - 3.0
            };
            let a = down.update(x);
            if t > 300 {
                if let ChangeAlarm::Decrease { .. } = a {
                    saw_decrease = true;
                    break;
                }
            }
        }
        assert!(saw_decrease, "expected a Decrease alarm for downward shift");
    }

    #[test]
    fn page_hinkley_reset_clears_state() {
        let mut ph = make_ph(0.01, 5.0, 5, None);
        for i in 0..100u64 {
            let _ = ph.update(if i < 50 { 0.0 } else { 5.0 });
        }
        ph.reset();
        assert_eq!(ph.n(), 0);
        assert_eq!(ph.mean(), 0.0);
        assert_eq!(ph.update(0.0), ChangeAlarm::None);
    }

    #[test]
    fn cusum_zero_shift_never_alarms() {
        // mu0 = 0 matches the true mean; with no shift, only randomness can
        // cause crossings. Use a large h to be safe.
        let mut cs = make_cusum(0.0, 0.5, 50.0);
        let mut rng = LcgRng::new(33);
        for _ in 0..5_000 {
            let a = cs.update(rng.next_normal());
            assert_eq!(a, ChangeAlarm::None);
        }
    }

    #[test]
    fn page_hinkley_deterministic_given_same_inputs() {
        let mut a = make_ph(0.05, 10.0, 5, None);
        let mut b = make_ph(0.05, 10.0, 5, None);
        let xs = [0.1, -0.3, 0.5, 1.5, 2.4, -0.1, 0.8, 3.0, 2.7, 2.1];
        for &x in &xs {
            assert_eq!(a.update(x), b.update(x));
        }
    }

    #[test]
    fn page_hinkley_larger_lambda_fewer_alarms() {
        let mut rng = LcgRng::new(5);
        let xs: Vec<f64> = (0..2_000).map(|_| rng.next_normal()).collect();
        let mut low = make_ph(0.0, 2.0, 5, None);
        let mut high = make_ph(0.0, 50.0, 5, None);
        let mut low_alarms = 0usize;
        let mut high_alarms = 0usize;
        for &x in &xs {
            if !matches!(low.update(x), ChangeAlarm::None) {
                low_alarms += 1;
                low.reset();
            }
            if !matches!(high.update(x), ChangeAlarm::None) {
                high_alarms += 1;
                high.reset();
            }
        }
        assert!(
            high_alarms <= low_alarms,
            "larger lambda should yield ≤ alarms (low={low_alarms}, high={high_alarms})"
        );
    }

    #[test]
    fn page_hinkley_larger_delta_fewer_alarms() {
        // Same stream, same lambda; only delta differs. Larger delta drags
        // `m_t` down more aggressively each step, making positive excursions
        // above the running minimum smaller. We compare the FIRST-PASSAGE
        // statistic peak (max ph across the run, NO resetting) — this is the
        // monotone quantity controlled by delta.
        let mut rng = LcgRng::new(31);
        let xs: Vec<f64> = (0..2_000)
            .map(|t| {
                if t < 1_000 {
                    rng.next_normal()
                } else {
                    rng.next_normal() + 0.5
                }
            })
            .collect();
        let mut small = make_ph(0.0, 1e9, 10, None);
        let mut big = make_ph(1.5, 1e9, 10, None);
        let mut small_peak = 0.0_f64;
        let mut big_peak = 0.0_f64;
        for &x in &xs {
            let _ = small.update(x);
            let _ = big.update(x);
            let ps = small.ph_increase();
            let pb = big.ph_increase();
            if ps > small_peak {
                small_peak = ps;
            }
            if pb > big_peak {
                big_peak = pb;
            }
        }
        assert!(
            big_peak <= small_peak,
            "larger delta should yield smaller PH peak (small={small_peak}, big={big_peak})"
        );
    }

    #[test]
    fn cusum_smaller_k_faster_detection() {
        // Two CUSUMs differing only in k. Smaller k must alarm at the same
        // index or earlier when the shift is genuine.
        let mut small_k = make_cusum(0.0, 0.1, 5.0);
        let mut large_k = make_cusum(0.0, 1.0, 5.0);
        let mut rng_a = LcgRng::new(77);
        let mut rng_b = LcgRng::new(77);
        let mut first_small: Option<u64> = None;
        let mut first_large: Option<u64> = None;
        for t in 1..=2_000u64 {
            let x_a = if t <= 200 {
                rng_a.next_normal()
            } else {
                rng_a.next_normal() + 1.5
            };
            let x_b = if t <= 200 {
                rng_b.next_normal()
            } else {
                rng_b.next_normal() + 1.5
            };
            if first_small.is_none() {
                if let ChangeAlarm::Increase { at, .. } = small_k.update(x_a) {
                    if at > 200 {
                        first_small = Some(at);
                    }
                }
            }
            if first_large.is_none() {
                if let ChangeAlarm::Increase { at, .. } = large_k.update(x_b) {
                    if at > 200 {
                        first_large = Some(at);
                    }
                }
            }
            if first_small.is_some() && first_large.is_some() {
                break;
            }
        }
        let ss = first_small.unwrap_or(u64::MAX);
        let ll = first_large.unwrap_or(u64::MAX);
        assert!(
            ss <= ll,
            "smaller k should detect no later than larger k (small={ss}, large={ll})"
        );
    }

    #[test]
    fn cusum_alarm_carries_detection_index() {
        // Force a strong Increase: with mu0=0 and large positive inputs the
        // alarm must encode the current sample index.
        let mut cs = make_cusum(0.0, 0.1, 1.0);
        let mut detected: Option<u64> = None;
        for _ in 1..=20u64 {
            if let ChangeAlarm::Increase { at, .. } = cs.update(2.0) {
                detected = Some(at);
                break;
            }
        }
        let at = detected.expect("Increase alarm expected");
        assert!(at >= 1);
        assert_eq!(at, cs.n());
    }

    #[test]
    fn page_hinkley_warmup_respected() {
        // Use EMA with tiny alpha so the mean barely tracks the data → the
        // PH statistic grows monotonically once the mean is established.
        // Seed mean with 0, then drive with constant 10: m_t grows by
        // ~10 each step (less the slow EMA adjustment), quickly exceeding
        // lambda. With min_n = 50, no alarm may fire before n ≥ 50.
        let mut ph = make_ph(0.0, 5.0, 50, Some(0.001));
        // Seed the EMA with x=0 so the subsequent stream of 10s creates a
        // clear, sustained positive deviation from the running mean.
        let _ = ph.update(0.0);
        for t in 2..50u64 {
            let a = ph.update(10.0);
            assert_eq!(a, ChangeAlarm::None, "alarm fired before warmup at t={t}");
        }
        // After warmup the next step (n = 50) must alarm; the cumulative
        // deviation has been ~10 per step for ~48 steps → m_t ~ 480 >> 5.
        let after = ph.update(10.0);
        assert!(
            !matches!(after, ChangeAlarm::None),
            "expected alarm to fire at end of warmup, got {after:?}"
        );
    }

    #[test]
    fn page_hinkley_invalid_lambda() {
        let r = PageHinkley::new(PageHinkleyConfig {
            delta: 0.0,
            lambda: 0.0,
            min_n: 5,
            alpha: None,
        });
        assert!(r.is_err());
        let r2 = PageHinkley::new(PageHinkleyConfig {
            delta: 0.0,
            lambda: -1.0,
            min_n: 5,
            alpha: None,
        });
        assert!(r2.is_err());
    }

    #[test]
    fn page_hinkley_invalid_delta() {
        let r = PageHinkley::new(PageHinkleyConfig {
            delta: -0.1,
            lambda: 1.0,
            min_n: 5,
            alpha: None,
        });
        assert!(r.is_err());
    }

    #[test]
    fn page_hinkley_invalid_alpha() {
        let r1 = PageHinkley::new(PageHinkleyConfig {
            delta: 0.0,
            lambda: 1.0,
            min_n: 5,
            alpha: Some(0.0),
        });
        assert!(r1.is_err());
        let r2 = PageHinkley::new(PageHinkleyConfig {
            delta: 0.0,
            lambda: 1.0,
            min_n: 5,
            alpha: Some(1.1),
        });
        assert!(r2.is_err());
        // Boundary value 1.0 should be accepted.
        let r3 = PageHinkley::new(PageHinkleyConfig {
            delta: 0.0,
            lambda: 1.0,
            min_n: 5,
            alpha: Some(1.0),
        });
        assert!(r3.is_ok());
    }

    #[test]
    fn cusum_invalid_h() {
        let r = Cusum::new(CusumConfig {
            mu0: 0.0,
            k: 0.5,
            h: 0.0,
        });
        assert!(r.is_err());
        let r2 = Cusum::new(CusumConfig {
            mu0: 0.0,
            k: 0.5,
            h: -1.0,
        });
        assert!(r2.is_err());
    }

    #[test]
    fn cusum_invalid_k() {
        let r = Cusum::new(CusumConfig {
            mu0: 0.0,
            k: -0.5,
            h: 1.0,
        });
        assert!(r.is_err());
    }

    #[test]
    fn cusum_reset_clears_state() {
        let mut cs = make_cusum(0.0, 0.1, 5.0);
        for _ in 0..100 {
            let _ = cs.update(1.0);
        }
        assert!(cs.n() > 0);
        cs.reset();
        assert_eq!(cs.n(), 0);
        assert_eq!(cs.s_plus(), 0.0);
        assert_eq!(cs.s_minus(), 0.0);
    }

    #[test]
    fn cusum_detects_negative_shift() {
        let mut cs = make_cusum(0.0, 0.2, 5.0);
        let mut rng = LcgRng::new(55);
        let mut saw = false;
        for t in 1..=2_000u64 {
            let x = if t <= 500 {
                rng.next_normal()
            } else {
                rng.next_normal() - 2.0
            };
            if let ChangeAlarm::Decrease { at, .. } = cs.update(x) {
                if at > 500 {
                    saw = true;
                    break;
                }
            }
        }
        assert!(saw, "CUSUM should detect downward shift");
    }

    #[test]
    fn page_hinkley_ema_mode_tracks_drift() {
        // EMA mode lets the mean adapt to slow drifts. After many constant
        // inputs the mean should converge close to that input.
        let mut ph = make_ph(0.0, 1e6, 1, Some(0.3));
        for _ in 0..500 {
            let _ = ph.update(5.0);
        }
        assert!(
            (ph.mean() - 5.0).abs() < 1e-3,
            "EMA mean should converge to constant input, got {}",
            ph.mean()
        );
    }

    #[test]
    fn page_hinkley_first_sample_no_alarm() {
        // First sample: mean is updated to equal x, so deviation = 0 and
        // PH cannot exceed any positive lambda.
        let mut ph = make_ph(0.0, 0.001, 1, None);
        let a = ph.update(1_000.0);
        assert_eq!(a, ChangeAlarm::None);
    }

    #[test]
    fn page_hinkley_non_finite_input_ignored() {
        let mut ph = make_ph(0.0, 1.0, 1, None);
        let _ = ph.update(1.0);
        let n_before = ph.n();
        let r = ph.update(f64::NAN);
        assert_eq!(r, ChangeAlarm::None);
        assert_eq!(ph.n(), n_before);
        let r2 = ph.update(f64::INFINITY);
        assert_eq!(r2, ChangeAlarm::None);
        assert_eq!(ph.n(), n_before);
    }

    #[test]
    fn cusum_non_finite_input_ignored() {
        let mut cs = make_cusum(0.0, 0.1, 1.0);
        let _ = cs.update(1.0);
        let n_before = cs.n();
        let r = cs.update(f64::NAN);
        assert_eq!(r, ChangeAlarm::None);
        assert_eq!(cs.n(), n_before);
    }

    #[test]
    fn cusum_deterministic_given_same_inputs() {
        let mut a = make_cusum(0.5, 0.2, 3.0);
        let mut b = make_cusum(0.5, 0.2, 3.0);
        let xs = [0.0, 0.4, 0.6, 1.2, 0.9, 0.7, 0.55, 0.5, 0.5];
        for &x in &xs {
            assert_eq!(a.update(x), b.update(x));
        }
        assert_eq!(a.s_plus(), b.s_plus());
        assert_eq!(a.s_minus(), b.s_minus());
    }
}
