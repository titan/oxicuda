//! Memory pressure monitoring with threshold callbacks and eviction hooks.
//!
//! A long-running allocator benefits from reacting *before* the device runs out
//! of memory: trimming idle pool pages, evicting cached buffers, or throttling
//! new work as free VRAM drops.  This module models that control loop on the
//! host side.
//!
//! [`MemoryPressureMonitor`] holds configurable warning / critical thresholds
//! (expressed as a *used fraction* of total device memory) and, on each sample,
//! classifies the current state into a [`PressureLevel`].  Transitions into a
//! more severe level fire user callbacks; entering the critical level fires an
//! eviction hook that returns the number of bytes reclaimed so the loop can
//! account for the relief.
//!
//! Samples are supplied as [`crate::memory_info::MemoryInfo`] values.  In
//! production the [`MemoryPressureMonitor::poll`] method fetches a fresh sample
//! via [`crate::memory_info::memory_info`] (requires a GPU); in tests the
//! [`MemoryPressureMonitor::observe`] method feeds synthetic samples so the
//! whole state machine is deterministic and hardware-free.
//!
//! # Example
//!
//! ```rust
//! # use oxicuda_memory::pool_pressure::*;
//! # use oxicuda_memory::memory_info::MemoryInfo;
//! let mut monitor = MemoryPressureMonitor::new(0.80, 0.95).expect("thresholds");
//!
//! // 70% used -> nominal.
//! let lvl = monitor.observe(MemoryInfo { free: 30, total: 100 });
//! assert_eq!(lvl, PressureLevel::Nominal);
//!
//! // 96% used -> critical.
//! let lvl = monitor.observe(MemoryInfo { free: 4, total: 100 });
//! assert_eq!(lvl, PressureLevel::Critical);
//! ```

use crate::memory_info::MemoryInfo;
use oxicuda_driver::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// PressureLevel
// ---------------------------------------------------------------------------

/// The classified memory-pressure state.
///
/// Ordered by severity so levels can be compared directly
/// (`Nominal < Warning < Critical`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    /// Used fraction below the warning threshold.
    Nominal,
    /// Used fraction at/above the warning threshold but below critical.
    Warning,
    /// Used fraction at/above the critical threshold.
    Critical,
}

impl PressureLevel {
    /// Returns `true` if this is the [`Critical`](PressureLevel::Critical) level.
    #[inline]
    #[must_use]
    pub fn is_critical(self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Returns `true` if this level is at least [`Warning`](PressureLevel::Warning).
    #[inline]
    #[must_use]
    pub fn is_elevated(self) -> bool {
        self >= Self::Warning
    }
}

impl std::fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nominal => write!(f, "nominal"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// PressureSample
// ---------------------------------------------------------------------------

/// The outcome of classifying one memory sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureSample {
    /// The classified level.
    pub level: PressureLevel,
    /// The used fraction (0.0–1.0) that produced the classification.
    pub used_fraction: f64,
    /// The previous level before this sample (for transition detection).
    pub previous: PressureLevel,
}

impl PressureSample {
    /// Returns `true` if this sample represents a transition into a strictly
    /// more severe level than the previous sample.
    #[inline]
    #[must_use]
    pub fn escalated(&self) -> bool {
        self.level > self.previous
    }

    /// Returns `true` if this sample represents a transition into a strictly
    /// less severe level than the previous sample.
    #[inline]
    #[must_use]
    pub fn de_escalated(&self) -> bool {
        self.level < self.previous
    }
}

// ---------------------------------------------------------------------------
// MemoryPressureMonitor
// ---------------------------------------------------------------------------

/// Type alias for the eviction hook: called on entering the critical level,
/// returns the number of bytes reclaimed.
type EvictionHook = Box<dyn FnMut() -> usize + Send>;

/// Type alias for a level-transition callback.
type TransitionHook = Box<dyn FnMut(PressureSample) + Send>;

/// Monitors device memory pressure and drives reactive trim / eviction.
///
/// Construct with [`new`](Self::new) supplying the warning and critical used
/// fractions.  Optionally attach an eviction hook with
/// [`on_eviction`](Self::on_eviction) and a transition callback with
/// [`on_transition`](Self::on_transition).  Feed samples via
/// [`observe`](Self::observe) (synthetic) or [`poll`](Self::poll) (live GPU).
pub struct MemoryPressureMonitor {
    warning_fraction: f64,
    critical_fraction: f64,
    current: PressureLevel,
    /// Total bytes the eviction hook has reclaimed over the monitor's lifetime.
    total_evicted: usize,
    /// Number of times the eviction hook has fired.
    eviction_count: u64,
    eviction_hook: Option<EvictionHook>,
    transition_hook: Option<TransitionHook>,
}

impl std::fmt::Debug for MemoryPressureMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPressureMonitor")
            .field("warning_fraction", &self.warning_fraction)
            .field("critical_fraction", &self.critical_fraction)
            .field("current", &self.current)
            .field("total_evicted", &self.total_evicted)
            .field("eviction_count", &self.eviction_count)
            .field("has_eviction_hook", &self.eviction_hook.is_some())
            .field("has_transition_hook", &self.transition_hook.is_some())
            .finish()
    }
}

impl MemoryPressureMonitor {
    /// Creates a monitor with the given warning and critical used-fraction
    /// thresholds.
    ///
    /// Both fractions must lie in `(0.0, 1.0]` and `warning_fraction` must be
    /// strictly less than `critical_fraction`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if either threshold is non-finite, outside
    ///   `(0.0, 1.0]`, or if `warning_fraction >= critical_fraction`.
    pub fn new(warning_fraction: f64, critical_fraction: f64) -> CudaResult<Self> {
        let valid = |x: f64| x.is_finite() && x > 0.0 && x <= 1.0;
        if !valid(warning_fraction)
            || !valid(critical_fraction)
            || warning_fraction >= critical_fraction
        {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self {
            warning_fraction,
            critical_fraction,
            current: PressureLevel::Nominal,
            total_evicted: 0,
            eviction_count: 0,
            eviction_hook: None,
            transition_hook: None,
        })
    }

    /// Attaches an eviction hook fired whenever the monitor *enters* the
    /// critical level.  The hook returns the number of bytes it reclaimed,
    /// which is accumulated into [`total_evicted`](Self::total_evicted).
    ///
    /// Builder-style; returns `self` for chaining.
    #[must_use]
    pub fn on_eviction<F>(mut self, hook: F) -> Self
    where
        F: FnMut() -> usize + Send + 'static,
    {
        self.eviction_hook = Some(Box::new(hook));
        self
    }

    /// Attaches a callback fired on every level transition (escalation or
    /// de-escalation).
    ///
    /// Builder-style; returns `self` for chaining.
    #[must_use]
    pub fn on_transition<F>(mut self, hook: F) -> Self
    where
        F: FnMut(PressureSample) + Send + 'static,
    {
        self.transition_hook = Some(Box::new(hook));
        self
    }

    /// The warning used-fraction threshold.
    #[inline]
    #[must_use]
    pub fn warning_fraction(&self) -> f64 {
        self.warning_fraction
    }

    /// The critical used-fraction threshold.
    #[inline]
    #[must_use]
    pub fn critical_fraction(&self) -> f64 {
        self.critical_fraction
    }

    /// The current pressure level (state from the most recent sample).
    #[inline]
    #[must_use]
    pub fn level(&self) -> PressureLevel {
        self.current
    }

    /// Total bytes reclaimed by the eviction hook over this monitor's lifetime.
    #[inline]
    #[must_use]
    pub fn total_evicted(&self) -> usize {
        self.total_evicted
    }

    /// Number of times the eviction hook has fired.
    #[inline]
    #[must_use]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Classifies a `used_fraction` (0.0–1.0) into a [`PressureLevel`] using
    /// this monitor's thresholds.
    ///
    /// The comparison is inclusive at the threshold (a fraction exactly equal
    /// to the critical threshold is [`Critical`](PressureLevel::Critical)).
    #[must_use]
    pub fn classify(&self, used_fraction: f64) -> PressureLevel {
        if used_fraction >= self.critical_fraction {
            PressureLevel::Critical
        } else if used_fraction >= self.warning_fraction {
            PressureLevel::Warning
        } else {
            PressureLevel::Nominal
        }
    }

    /// Feeds a synthetic memory sample through the monitor and returns the new
    /// level.
    ///
    /// On a transition the transition callback (if any) fires.  On *entering*
    /// the critical level the eviction hook (if any) fires and its return value
    /// is accumulated.  Re-observing the same level does not re-fire hooks.
    pub fn observe(&mut self, info: MemoryInfo) -> PressureLevel {
        let used_fraction = info.usage_fraction();
        let previous = self.current;
        let level = self.classify(used_fraction);
        self.current = level;

        if level != previous {
            let sample = PressureSample {
                level,
                used_fraction,
                previous,
            };
            if let Some(hook) = self.transition_hook.as_mut() {
                hook(sample);
            }
            // Fire eviction only on an *upward* transition into Critical.
            if level == PressureLevel::Critical && previous != PressureLevel::Critical {
                if let Some(hook) = self.eviction_hook.as_mut() {
                    let reclaimed = hook();
                    self.total_evicted = self.total_evicted.saturating_add(reclaimed);
                    self.eviction_count = self.eviction_count.saturating_add(1);
                }
            }
        }
        level
    }

    /// Polls live device memory via [`crate::memory_info::memory_info`] and
    /// feeds the sample through [`observe`](Self::observe).
    ///
    /// # Errors
    ///
    /// Forwards any error from the memory-info query (e.g. no current context).
    pub fn poll(&mut self) -> CudaResult<PressureLevel> {
        let info = crate::memory_info::memory_info()?;
        Ok(self.observe(info))
    }

    /// Resets the monitor's level to [`Nominal`](PressureLevel::Nominal) and
    /// clears eviction accounting, without changing thresholds or hooks.
    pub fn reset(&mut self) {
        self.current = PressureLevel::Nominal;
        self.total_evicted = 0;
        self.eviction_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/// Validates a pair of `(warning, critical)` used-fraction thresholds without
/// constructing a monitor.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] under the same conditions as
///   [`MemoryPressureMonitor::new`].
pub fn validate_thresholds(warning_fraction: f64, critical_fraction: f64) -> CudaResult<()> {
    MemoryPressureMonitor::new(warning_fraction, critical_fraction).map(|_| ())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    fn info(free: usize, total: usize) -> MemoryInfo {
        MemoryInfo { free, total }
    }

    #[test]
    fn pressure_level_ordering() {
        assert!(PressureLevel::Nominal < PressureLevel::Warning);
        assert!(PressureLevel::Warning < PressureLevel::Critical);
        assert!(PressureLevel::Critical.is_critical());
        assert!(!PressureLevel::Warning.is_critical());
        assert!(PressureLevel::Warning.is_elevated());
        assert!(!PressureLevel::Nominal.is_elevated());
        assert!(PressureLevel::Critical.is_elevated());
    }

    #[test]
    fn pressure_level_display() {
        assert_eq!(format!("{}", PressureLevel::Nominal), "nominal");
        assert_eq!(format!("{}", PressureLevel::Warning), "warning");
        assert_eq!(format!("{}", PressureLevel::Critical), "critical");
    }

    #[test]
    fn new_rejects_bad_thresholds() {
        // warning >= critical
        assert!(MemoryPressureMonitor::new(0.9, 0.9).is_err());
        assert!(MemoryPressureMonitor::new(0.95, 0.8).is_err());
        // out of range
        assert!(MemoryPressureMonitor::new(0.0, 0.9).is_err());
        assert!(MemoryPressureMonitor::new(0.8, 1.5).is_err());
        // non-finite
        assert!(MemoryPressureMonitor::new(f64::NAN, 0.9).is_err());
    }

    #[test]
    fn new_accepts_valid_thresholds() {
        let m = MemoryPressureMonitor::new(0.8, 0.95).expect("ok");
        assert!((m.warning_fraction() - 0.8).abs() < 1e-12);
        assert!((m.critical_fraction() - 0.95).abs() < 1e-12);
        assert_eq!(m.level(), PressureLevel::Nominal);
    }

    #[test]
    fn validate_thresholds_helper() {
        assert!(validate_thresholds(0.7, 0.9).is_ok());
        assert!(validate_thresholds(0.9, 0.7).is_err());
    }

    #[test]
    fn classify_boundaries() {
        let m = MemoryPressureMonitor::new(0.80, 0.95).expect("ok");
        // Below warning.
        assert_eq!(m.classify(0.50), PressureLevel::Nominal);
        assert_eq!(m.classify(0.799), PressureLevel::Nominal);
        // Exactly warning -> Warning (inclusive).
        assert_eq!(m.classify(0.80), PressureLevel::Warning);
        assert_eq!(m.classify(0.90), PressureLevel::Warning);
        // Just below critical.
        assert_eq!(m.classify(0.949), PressureLevel::Warning);
        // Exactly critical -> Critical (inclusive).
        assert_eq!(m.classify(0.95), PressureLevel::Critical);
        assert_eq!(m.classify(1.00), PressureLevel::Critical);
    }

    #[test]
    fn observe_transitions_through_levels() {
        let mut m = MemoryPressureMonitor::new(0.80, 0.95).expect("ok");
        // 50% used (free 50 / total 100) -> nominal.
        assert_eq!(m.observe(info(50, 100)), PressureLevel::Nominal);
        // 85% used -> warning.
        assert_eq!(m.observe(info(15, 100)), PressureLevel::Warning);
        // 97% used -> critical.
        assert_eq!(m.observe(info(3, 100)), PressureLevel::Critical);
        assert_eq!(m.level(), PressureLevel::Critical);
        // back down to 40% -> nominal.
        assert_eq!(m.observe(info(60, 100)), PressureLevel::Nominal);
    }

    #[test]
    fn eviction_hook_fires_on_entering_critical() {
        let evicted = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicU64::new(0));
        let e2 = Arc::clone(&evicted);
        let c2 = Arc::clone(&calls);
        let mut m = MemoryPressureMonitor::new(0.80, 0.95)
            .expect("ok")
            .on_eviction(move || {
                e2.fetch_add(1, Ordering::SeqCst);
                c2.fetch_add(1, Ordering::SeqCst);
                4096 // bytes reclaimed
            });

        // Nominal -> no eviction.
        m.observe(info(50, 100));
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        // Enter critical -> eviction fires once.
        m.observe(info(2, 100));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(m.total_evicted(), 4096);
        assert_eq!(m.eviction_count(), 1);

        // Stay critical -> no re-fire.
        m.observe(info(1, 100));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(m.total_evicted(), 4096);

        // Drop to nominal, then re-enter critical -> fires again.
        m.observe(info(80, 100));
        m.observe(info(1, 100));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(m.total_evicted(), 8192);
        assert_eq!(m.eviction_count(), 2);
    }

    #[test]
    fn eviction_does_not_fire_when_warning_to_critical_skips_nominal() {
        // Going Warning -> Critical (not via Nominal) should still fire once.
        let calls = Arc::new(AtomicU64::new(0));
        let c2 = Arc::clone(&calls);
        let mut m = MemoryPressureMonitor::new(0.80, 0.95)
            .expect("ok")
            .on_eviction(move || {
                c2.fetch_add(1, Ordering::SeqCst);
                100
            });
        m.observe(info(15, 100)); // warning
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        m.observe(info(2, 100)); // critical
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn transition_hook_records_escalation_and_de_escalation() {
        let escalations = Arc::new(AtomicU64::new(0));
        let de_escalations = Arc::new(AtomicU64::new(0));
        let up = Arc::clone(&escalations);
        let down = Arc::clone(&de_escalations);
        let mut m = MemoryPressureMonitor::new(0.80, 0.95)
            .expect("ok")
            .on_transition(move |sample: PressureSample| {
                if sample.escalated() {
                    up.fetch_add(1, Ordering::SeqCst);
                }
                if sample.de_escalated() {
                    down.fetch_add(1, Ordering::SeqCst);
                }
            });

        m.observe(info(50, 100)); // nominal, no transition (already nominal)
        assert_eq!(escalations.load(Ordering::SeqCst), 0);
        m.observe(info(15, 100)); // -> warning (escalate)
        m.observe(info(2, 100)); //  -> critical (escalate)
        assert_eq!(escalations.load(Ordering::SeqCst), 2);
        m.observe(info(50, 100)); // -> nominal (de-escalate)
        assert_eq!(de_escalations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn observe_same_level_does_not_transition() {
        let calls = Arc::new(AtomicU64::new(0));
        let c2 = Arc::clone(&calls);
        let mut m = MemoryPressureMonitor::new(0.80, 0.95)
            .expect("ok")
            .on_transition(move |_| {
                c2.fetch_add(1, Ordering::SeqCst);
            });
        m.observe(info(50, 100)); // nominal == initial, no transition
        m.observe(info(60, 100)); // still nominal, no transition
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn pressure_sample_transition_flags() {
        let escalate = PressureSample {
            level: PressureLevel::Critical,
            used_fraction: 0.97,
            previous: PressureLevel::Warning,
        };
        assert!(escalate.escalated());
        assert!(!escalate.de_escalated());

        let de = PressureSample {
            level: PressureLevel::Nominal,
            used_fraction: 0.20,
            previous: PressureLevel::Critical,
        };
        assert!(de.de_escalated());
        assert!(!de.escalated());
    }

    #[test]
    fn reset_clears_state_keeps_thresholds() {
        let mut m = MemoryPressureMonitor::new(0.80, 0.95)
            .expect("ok")
            .on_eviction(|| 1000);
        m.observe(info(1, 100)); // critical -> evict
        assert_eq!(m.total_evicted(), 1000);
        assert_eq!(m.level(), PressureLevel::Critical);
        m.reset();
        assert_eq!(m.level(), PressureLevel::Nominal);
        assert_eq!(m.total_evicted(), 0);
        assert_eq!(m.eviction_count(), 0);
        // thresholds intact
        assert!((m.warning_fraction() - 0.80).abs() < 1e-12);
    }

    #[test]
    fn observe_zero_total_is_nominal() {
        // usage_fraction returns 0.0 when total is 0.
        let mut m = MemoryPressureMonitor::new(0.80, 0.95).expect("ok");
        assert_eq!(m.observe(info(0, 0)), PressureLevel::Nominal);
    }

    #[test]
    fn poll_signature_compiles() {
        let _: fn(&mut MemoryPressureMonitor) -> CudaResult<PressureLevel> =
            MemoryPressureMonitor::poll;
    }

    #[test]
    fn debug_does_not_panic() {
        let m = MemoryPressureMonitor::new(0.8, 0.95)
            .expect("ok")
            .on_eviction(|| 0);
        let s = format!("{m:?}");
        assert!(s.contains("MemoryPressureMonitor"));
        assert!(s.contains("has_eviction_hook"));
    }
}
