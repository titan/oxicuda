//! Dynamic-programming beat tracking (Ellis 2007 / Böck 2011).
//!
//! Given an **onset-strength envelope** and a target tempo, the beat tracker
//! finds the set of beat frames `{b_0 < b_1 < … < b_{K-1}}` that maximises the
//! global objective
//!
//! ```text
//!   C({b_i}) = Σ_i  onset(b_i)  +  α · Σ_i  F(b_i − b_{i-1})
//! ```
//!
//! where `onset(·)` is the (z-normalised) onset envelope, `α` is the penalty
//! `tightness`, and the inter-beat transition cost
//!
//! ```text
//!   F(Δ) = −( ln(Δ / period) )²
//! ```
//!
//! is maximised (zero) when the spacing `Δ` equals the target beat `period`
//! (in frames) and grows quadratically in log-spacing otherwise. This is the
//! Ellis (2007) dynamic-programming formulation of the Böck (2011) dynamic-
//! Bayesian-network idea: a tempo prior (the period) combined with a Markov
//! inter-beat-interval penalty, solved exactly by a forward DP plus backtrace.
//!
//! The DP is `O(N · W)` for `N` envelope frames and a predecessor window of
//! width `W ≈ 1.5 · period`, deterministic, and allocation-light.
//!
//! ## Pipeline
//! 1. **Period** — from a supplied target BPM (`period = 60·fps / bpm`) or, if
//!    none is given, from autocorrelation via [`tempo_from_envelope`].
//! 2. **Normalise** — z-score the envelope so `tightness` is scale-independent.
//! 3. **Forward DP** — `cumscore(t) = onset(t) + max_τ [ cumscore(τ) + α·F(t−τ) ]`
//!    over predecessor frames `τ ∈ [t − 2·period, t − period/2]`. Predecessors
//!    before frame 0 act as a virtual chain-start with `cumscore = 0`, so an
//!    early strong onset can begin a beat chain without penalty.
//! 4. **Backtrace** — from the global `argmax` of `cumscore`.
//!
//! ## References
//! - Ellis, D. P. W. (2007). "Beat tracking by dynamic programming."
//!   *J. New Music Research* 36(1), 51–60.
//! - Böck, S. & Schedl, M. (2011). "Enhanced beat tracking with a dynamic
//!   Bayesian network." *Proc. ISMIR*.

use crate::error::{AudioError, AudioResult};
use crate::features::onset::{OnsetConfig, onset_strength, tempo_from_envelope};

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the [`BeatTracker`].
#[derive(Debug, Clone)]
pub struct BeatTrackerConfig {
    /// Tempo prior in beats per minute. When `Some(bpm)`, the beat period is
    /// `60·fps / bpm`. When `None`, the period is estimated from the envelope
    /// by autocorrelation over `[min_bpm, max_bpm]`.
    pub target_bpm: Option<f32>,
    /// Lower tempo bound (BPM) for autocorrelation when `target_bpm` is `None`.
    pub min_bpm: f32,
    /// Upper tempo bound (BPM) for autocorrelation when `target_bpm` is `None`.
    pub max_bpm: f32,
    /// Transition-penalty tightness `α` (`> 0`). Larger values pull beat
    /// spacings harder toward the target period. The Ellis/librosa default is
    /// `100`.
    pub tightness: f32,
}

impl Default for BeatTrackerConfig {
    fn default() -> Self {
        Self {
            target_bpm: None,
            min_bpm: 60.0,
            max_bpm: 240.0,
            tightness: 100.0,
        }
    }
}

impl BeatTrackerConfig {
    /// Build a config with an explicit tempo prior in BPM.
    #[must_use]
    pub fn with_target_bpm(bpm: f32) -> Self {
        Self {
            target_bpm: Some(bpm),
            ..Self::default()
        }
    }
}

// ─── Tracker ─────────────────────────────────────────────────────────────────────

/// Relative dynamic range below which the input is treated as having no onset
/// structure (a flat envelope), for which no beats are returned.
const FLAT_RANGE_EPS: f32 = 1e-6;

/// Dynamic-programming beat tracker (Ellis 2007).
#[derive(Debug, Clone)]
pub struct BeatTracker {
    config: BeatTrackerConfig,
}

impl BeatTracker {
    /// Construct a beat tracker from a configuration.
    ///
    /// # Errors
    /// - [`AudioError::Internal`] if `tightness` is not a finite positive value,
    ///   if `target_bpm` is `Some(b)` with `b ≤ 0`, or if the BPM search range
    ///   is not `0 < min_bpm < max_bpm`.
    pub fn new(config: BeatTrackerConfig) -> AudioResult<Self> {
        if !(config.tightness.is_finite() && config.tightness > 0.0) {
            return Err(AudioError::Internal(format!(
                "beat_tracker: tightness must be finite and > 0, got {}",
                config.tightness
            )));
        }
        if let Some(bpm) = config.target_bpm {
            if !(bpm.is_finite() && bpm > 0.0) {
                return Err(AudioError::Internal(format!(
                    "beat_tracker: target_bpm must be finite and > 0, got {bpm}"
                )));
            }
        }
        if !(config.min_bpm > 0.0 && config.max_bpm > config.min_bpm) {
            return Err(AudioError::Internal(format!(
                "beat_tracker: require 0 < min_bpm < max_bpm, got {}, {}",
                config.min_bpm, config.max_bpm
            )));
        }
        Ok(Self { config })
    }

    /// The configuration this tracker was built with.
    #[must_use]
    pub fn config(&self) -> &BeatTrackerConfig {
        &self.config
    }

    /// Resolve the target beat period (in frames) for an envelope.
    fn resolve_period(&self, onset_env: &[f32], frames_per_second: f32) -> AudioResult<f32> {
        let period = match self.config.target_bpm {
            Some(bpm) => 60.0 * frames_per_second / bpm,
            None => {
                let est = tempo_from_envelope(
                    onset_env,
                    frames_per_second,
                    self.config.min_bpm,
                    self.config.max_bpm,
                )?;
                est.period_frames as f32
            }
        };
        if !(period.is_finite() && period >= 1.0) {
            return Err(AudioError::Internal(format!(
                "beat_tracker: resolved beat period must be ≥ 1 frame, got {period}"
            )));
        }
        Ok(period)
    }

    /// Track beats on a precomputed onset-strength envelope.
    ///
    /// Returns the beat **frame indices** in strictly ascending order. A flat
    /// envelope (no onset structure) yields an empty vector.
    ///
    /// # Errors
    /// - [`AudioError::EmptyInput`] if `onset_env` is empty.
    /// - [`AudioError::Internal`] if `frames_per_second ≤ 0`, if a supplied
    ///   `target_bpm ≤ 0`, or if the resolved period is sub-frame.
    /// - Any error propagated from [`tempo_from_envelope`] when the tempo is
    ///   estimated (`target_bpm == None`).
    pub fn track(&self, onset_env: &[f32], frames_per_second: f32) -> AudioResult<Vec<usize>> {
        if onset_env.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "beat_tracker: empty onset envelope".into(),
            });
        }
        if !(frames_per_second.is_finite() && frames_per_second > 0.0) {
            return Err(AudioError::Internal(format!(
                "beat_tracker: frames_per_second must be finite and > 0, got {frames_per_second}"
            )));
        }

        let period = self.resolve_period(onset_env, frames_per_second)?;
        let n = onset_env.len();

        // ── Flat-envelope guard ───────────────────────────────────────────────
        // Detect a (near-)constant envelope by its dynamic range. This is robust
        // to the float rounding that a constant vector's computed standard
        // deviation suffers — rounding that z-normalisation would otherwise
        // amplify into spurious periodic structure.
        let max = onset_env.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = onset_env.iter().copied().fold(f32::INFINITY, f32::min);
        let range = max - min;
        let scale = max.abs().max(min.abs()).max(1.0);
        if !range.is_finite() || range <= FLAT_RANGE_EPS * scale {
            // No onset structure → no beats.
            return Ok(Vec::new());
        }

        // ── Z-normalise the envelope (scale-independent tightness) ────────────
        let mean = onset_env.iter().sum::<f32>() / n as f32;
        let var = onset_env
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        let std = var.sqrt();
        if !(std.is_finite() && std > 0.0) {
            return Ok(Vec::new());
        }
        let localscore: Vec<f32> = onset_env.iter().map(|&x| (x - mean) / std).collect();

        // ── Forward dynamic program ───────────────────────────────────────────
        // Predecessor inter-beat intervals are searched in [period/2, 2·period].
        let win_min = ((period * 0.5).round() as isize).max(1);
        let win_max = ((period * 2.0).round() as isize).max(win_min);

        let mut cumscore = vec![0.0_f32; n];
        let mut backlink = vec![-1_isize; n];

        for t in 0..n {
            let mut best = f32::NEG_INFINITY;
            let mut best_link = -1_isize;
            let mut interval = win_min;
            while interval <= win_max {
                let prev = t as isize - interval;
                // Transition cost: 0 at exactly `period`, negative otherwise.
                let txcost = -self.config.tightness * (interval as f32 / period).ln().powi(2);
                // Predecessors before frame 0 act as a zero-score virtual start.
                let cand = if prev >= 0 {
                    cumscore[prev as usize] + txcost
                } else {
                    txcost
                };
                if cand > best {
                    best = cand;
                    best_link = prev;
                }
                interval += 1;
            }
            cumscore[t] = localscore[t] + best;
            backlink[t] = if best_link >= 0 { best_link } else { -1 };
        }

        // ── Pick the optimal endpoint (global argmax) and backtrace ───────────
        let mut t_end = 0_usize;
        let mut best_cum = f32::NEG_INFINITY;
        for (t, &c) in cumscore.iter().enumerate() {
            if c > best_cum {
                best_cum = c;
                t_end = t;
            }
        }

        let mut beats: Vec<usize> = Vec::new();
        let mut cursor = t_end as isize;
        while cursor >= 0 {
            beats.push(cursor as usize);
            cursor = backlink[cursor as usize];
        }
        beats.reverse();
        Ok(beats)
    }

    /// Convenience: compute the onset envelope from a raw signal and then track.
    ///
    /// The onset envelope and frames-per-second are derived from `onset_cfg`
    /// via [`onset_strength`] and [`OnsetConfig::frames_per_second`].
    ///
    /// # Errors
    /// Propagates any error from [`onset_strength`] or [`BeatTracker::track`].
    pub fn track_signal(&self, signal: &[f32], onset_cfg: &OnsetConfig) -> AudioResult<Vec<usize>> {
        let env = onset_strength(signal, onset_cfg)?;
        self.track(&env, onset_cfg.frames_per_second())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────────

/// Convert beat **frame indices** to beat **times** in seconds.
///
/// `time = frame / frames_per_second`. A non-positive `frames_per_second`
/// yields an empty vector (no meaningful time base).
#[must_use]
pub fn beat_times(beats: &[usize], frames_per_second: f32) -> Vec<f32> {
    if !(frames_per_second.is_finite() && frames_per_second > 0.0) {
        return Vec::new();
    }
    beats
        .iter()
        .map(|&b| b as f32 / frames_per_second)
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic onset envelope with strong unit peaks every `period`
    /// frames (starting at `offset`) over a low background.
    fn peaked_envelope(period: usize, offset: usize, n: usize) -> Vec<f32> {
        let mut env = vec![0.05_f32; n];
        let mut t = offset;
        while t < n {
            env[t] = 1.0;
            t += period;
        }
        env
    }

    fn median_interval(beats: &[usize]) -> f32 {
        let mut diffs: Vec<f32> = beats.windows(2).map(|w| (w[1] - w[0]) as f32).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        if diffs.is_empty() {
            return 0.0;
        }
        let mid = diffs.len() / 2;
        if diffs.len() % 2 == 0 {
            0.5 * (diffs[mid - 1] + diffs[mid])
        } else {
            diffs[mid]
        }
    }

    #[test]
    fn beats_land_on_periodic_peaks() {
        // Peaks every 20 frames; fps=100, bpm=300 → period = 20 frames exactly.
        let period = 20_usize;
        let env = peaked_envelope(period, 0, 200);
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(300.0)).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        assert!(beats.len() >= 5, "expected several beats, got {beats:?}");
        // Each beat lands on (or adjacent to) a peak frame (multiple of period).
        for &b in &beats {
            let nearest = ((b as f32 / period as f32).round() as usize) * period;
            let dist = b.abs_diff(nearest);
            assert!(dist <= 1, "beat {b} not near a peak (nearest {nearest})");
        }
        // Median inter-beat interval matches the period.
        let med = median_interval(&beats);
        assert!(
            (med - period as f32).abs() <= 1.0,
            "median IBI {med} vs {period}"
        );
    }

    #[test]
    fn beats_with_offset_peaks() {
        // Peaks every 16 frames, starting at frame 7.
        let period = 16_usize;
        let env = peaked_envelope(period, 7, 240);
        // fps=100, bpm = 60*100/16 = 375.
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(375.0)).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        assert!(beats.len() >= 5, "got {beats:?}");
        for &b in &beats {
            // nearest peak is offset + k*period.
            let k = (((b as f32) - 7.0) / period as f32).round();
            let nearest = (7.0 + k * period as f32).max(0.0) as usize;
            assert!(b.abs_diff(nearest) <= 1, "beat {b} not near peak {nearest}");
        }
        let med = median_interval(&beats);
        assert!((med - period as f32).abs() <= 1.0, "median IBI {med}");
    }

    #[test]
    fn beats_strictly_increasing_and_bounded() {
        let env = peaked_envelope(25, 3, 300);
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(240.0)).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        for w in beats.windows(2) {
            assert!(w[1] > w[0], "beats must strictly increase: {beats:?}");
        }
        for &b in &beats {
            assert!(b < env.len(), "beat {b} out of bounds {}", env.len());
        }
    }

    #[test]
    fn flat_envelope_returns_no_beats() {
        let env = vec![0.7_f32; 200];
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(120.0)).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        assert!(
            beats.is_empty(),
            "flat envelope must yield no beats: {beats:?}"
        );
    }

    #[test]
    fn zero_envelope_returns_no_beats() {
        let env = vec![0.0_f32; 128];
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(120.0)).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        assert!(beats.is_empty());
    }

    #[test]
    fn empty_envelope_errors() {
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(120.0)).expect("config");
        assert!(matches!(
            bt.track(&[], 100.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn invalid_fps_errors() {
        let env = peaked_envelope(20, 0, 100);
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(120.0)).expect("config");
        assert!(matches!(
            bt.track(&env, 0.0).unwrap_err(),
            AudioError::Internal(_)
        ));
        assert!(matches!(
            bt.track(&env, -5.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn invalid_tempo_prior_errors() {
        assert!(matches!(
            BeatTracker::new(BeatTrackerConfig::with_target_bpm(0.0)).unwrap_err(),
            AudioError::Internal(_)
        ));
        assert!(matches!(
            BeatTracker::new(BeatTrackerConfig::with_target_bpm(-120.0)).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn invalid_tightness_errors() {
        let cfg = BeatTrackerConfig {
            tightness: 0.0,
            ..BeatTrackerConfig::default()
        };
        assert!(matches!(
            BeatTracker::new(cfg).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn invalid_bpm_range_errors() {
        let cfg = BeatTrackerConfig {
            target_bpm: None,
            min_bpm: 200.0,
            max_bpm: 100.0,
            tightness: 100.0,
        };
        assert!(matches!(
            BeatTracker::new(cfg).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn estimated_tempo_path_tracks_beats() {
        // No target_bpm → period estimated by autocorrelation. Use a long
        // envelope so the BPM range is representable.
        let period = 20_usize;
        let env = peaked_envelope(period, 0, 600);
        let cfg = BeatTrackerConfig {
            target_bpm: None,
            min_bpm: 100.0,
            max_bpm: 400.0,
            tightness: 100.0,
        };
        let bt = BeatTracker::new(cfg).expect("config");
        let beats = bt.track(&env, 100.0).expect("track");
        assert!(beats.len() >= 5, "got {beats:?}");
        let med = median_interval(&beats);
        // Estimated period should be the click spacing or a small multiple.
        let ratio = med / period as f32;
        assert!(
            (ratio - ratio.round()).abs() < 0.25,
            "median IBI {med} not harmonically related to {period}"
        );
    }

    #[test]
    fn beat_times_convert_seconds() {
        let beats = vec![0_usize, 20, 40, 60];
        let times = beat_times(&beats, 100.0);
        assert_eq!(times.len(), beats.len());
        assert!((times[1] - 0.2).abs() < 1e-6, "t={}", times[1]);
        assert!((times[3] - 0.6).abs() < 1e-6);
        // Non-positive fps yields an empty vector.
        assert!(beat_times(&beats, 0.0).is_empty());
    }

    #[test]
    fn track_signal_runs_onset_first() {
        // A click train should yield onsets and hence some beats.
        let cfg = OnsetConfig {
            sample_rate: 22_050.0,
            n_fft: 512,
            hop_length: 128,
        };
        let mut signal = vec![0.0_f32; 16_384];
        let mut t = 1024_usize;
        while t < signal.len() {
            for d in 0..16usize {
                if t + d < signal.len() {
                    let env = (-(d as f32) / 4.0).exp();
                    signal[t + d] += env * (((d * 31 + t) % 5) as f32 / 2.0 - 1.0);
                }
            }
            t += 1024;
        }
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(258.0)).expect("config");
        let beats = bt.track_signal(&signal, &cfg).expect("track_signal");
        // Beats are strictly increasing and in range of the envelope length.
        for w in beats.windows(2) {
            assert!(w[1] > w[0]);
        }
        assert!(beats.iter().all(|&b| b < 16_384 / 128));
    }

    #[test]
    fn deterministic() {
        let env = peaked_envelope(20, 0, 200);
        let bt = BeatTracker::new(BeatTrackerConfig::with_target_bpm(300.0)).expect("config");
        let a = bt.track(&env, 100.0).expect("track");
        let b = bt.track(&env, 100.0).expect("track");
        assert_eq!(a, b);
    }
}
