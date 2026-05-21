//! Voice-Activity Detection (VAD) — lightweight classical-feature detector.
//!
//! A Silero-style *frame-level inference path*, but driven entirely by classical
//! signal features (no learned weights). Each analysis frame is classified as
//! speech / non-speech using two cheap, complementary cues:
//!
//! - **Log-energy (dB)**: `10·log10(mean(x²) + eps)`. Speech is louder than the
//!   noise floor, so a frame must clear an energy threshold to be active.
//! - **Spectral flatness** (Wiener entropy): the ratio of the geometric mean to
//!   the arithmetic mean of the magnitude spectrum,
//!   `exp(mean(ln(mag + eps))) / (mean(mag) + eps) ∈ [0, 1]`. Tonal / voiced
//!   content has a peaky spectrum and *low* flatness; broadband noise has a flat
//!   spectrum and *high* flatness. A frame must fall *below* a flatness
//!   threshold to be active.
//!
//! A raw frame is **active** iff `energy_db ≥ energy_threshold_db` **and**
//! `flatness ≤ spectral_flatness_threshold`. The raw activity track is then
//! cleaned with two hysteresis stages mirroring streaming VAD front-ends:
//!
//! - **Onset**: a speech region only *starts* after `onset_frames` consecutive
//!   active frames, suppressing isolated single-frame blips.
//! - **Hangover**: once speech is declared, it keeps being marked for
//!   `hangover_frames` after activity drops, bridging short intra-word pauses.
//!
//! The magnitude spectrum is computed with an inline real discrete Fourier
//! transform (O(n²)); no FFT crate is required and no learned parameters are
//! used, keeping the detector dependency-free and fully deterministic.

use crate::error::{AudioError, AudioResult};

/// Numerical floor added inside logarithms / ratios to avoid `log(0)`.
const VAD_EPS: f32 = 1e-10;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`Vad`].
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// Analysis frame length in samples (must be ≥ 1).
    pub frame_len: usize,
    /// Hop (step) between consecutive frame starts in samples (must be ≥ 1).
    pub hop_len: usize,
    /// Energy threshold in dB; frames at or above this are energetic enough.
    pub energy_threshold_db: f32,
    /// Spectral-flatness threshold; frames at or below this are tonal enough.
    pub spectral_flatness_threshold: f32,
    /// Number of trailing frames to keep marking speech after activity drops
    /// (hangover hysteresis). May be `0`.
    pub hangover_frames: usize,
    /// Number of consecutive active frames required to *start* a speech region
    /// (onset hysteresis). Must be ≥ 1.
    pub onset_frames: usize,
}

impl VadConfig {
    /// A reasonable default configuration for 16 kHz speech: 25 ms frames with
    /// a 10 ms hop, a −35 dB energy gate, a 0.5 flatness gate, 5-frame hangover,
    /// and a 3-frame onset requirement.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            frame_len: 400,
            hop_len: 160,
            energy_threshold_db: -35.0,
            spectral_flatness_threshold: 0.5,
            hangover_frames: 5,
            onset_frames: 3,
        }
    }
}

// ─── Result ─────────────────────────────────────────────────────────────────

/// Result of running [`Vad::detect`] over a waveform.
#[derive(Debug, Clone, PartialEq)]
pub struct VadResult {
    /// Per-frame speech (`true`) / non-speech (`false`) decision after onset and
    /// hangover smoothing. Length equals the number of analysis frames.
    pub frame_flags: Vec<bool>,
    /// Contiguous speech regions as `(start_frame, end_frame)` half-open
    /// intervals (`end` is exclusive). Sorted ascending and non-overlapping.
    pub segments: Vec<(usize, usize)>,
}

// ─── Vad ──────────────────────────────────────────────────────────────────────

/// Classical energy + spectral-flatness voice-activity detector.
#[derive(Debug, Clone)]
pub struct Vad {
    cfg: VadConfig,
}

impl Vad {
    /// Construct a new detector from the given configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidSequenceLength`] if `frame_len == 0`.
    /// - [`AudioError::InvalidStride`] if `hop_len == 0`.
    /// - [`AudioError::Internal`] if `onset_frames == 0`.
    pub fn new(cfg: VadConfig) -> AudioResult<Self> {
        if cfg.frame_len == 0 {
            return Err(AudioError::InvalidSequenceLength(cfg.frame_len));
        }
        if cfg.hop_len == 0 {
            return Err(AudioError::InvalidStride(cfg.hop_len));
        }
        if cfg.onset_frames == 0 {
            return Err(AudioError::Internal(
                "onset_frames must be >= 1".to_string(),
            ));
        }
        // `hangover_frames >= 0` is guaranteed by the `usize` type.
        Ok(Self { cfg })
    }

    /// Borrow the configuration this detector was built with.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.cfg
    }

    /// Per-frame log-energy in dB: `10·log10(mean(x²) + eps)`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `frame.len() != frame_len`.
    pub fn frame_energy_db(&self, frame: &[f32]) -> AudioResult<f32> {
        self.check_frame_len(frame)?;
        let mut sum_sq = 0.0f32;
        for &x in frame {
            sum_sq += x * x;
        }
        let mean_sq = sum_sq / frame.len() as f32;
        Ok(10.0 * (mean_sq + VAD_EPS).log10())
    }

    /// Spectral flatness in `[0, 1]`: the geometric mean of the magnitude
    /// spectrum divided by its arithmetic mean,
    /// `exp(mean(ln(mag + eps))) / (mean(mag) + eps)`.
    ///
    /// Low for tonal / voiced frames, high for broadband (noise-like) frames.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `frame.len() != frame_len`.
    pub fn spectral_flatness(&self, frame: &[f32]) -> AudioResult<f32> {
        self.check_frame_len(frame)?;
        let mags = dft_magnitude(frame);
        // `mags` always has at least one bin because `frame_len >= 1`.
        let n = mags.len() as f32;
        let mut sum_ln = 0.0f32;
        let mut sum_lin = 0.0f32;
        for &m in &mags {
            let shifted = m + VAD_EPS;
            sum_ln += shifted.ln();
            sum_lin += m;
        }
        let geometric_mean = (sum_ln / n).exp();
        let arithmetic_mean = sum_lin / n;
        let flatness = geometric_mean / (arithmetic_mean + VAD_EPS);
        // Numerically clamp into the analytic [0, 1] range.
        Ok(flatness.clamp(0.0, 1.0))
    }

    /// Run VAD over a whole waveform: frame it, classify each frame, and apply
    /// onset + hangover smoothing before extracting contiguous speech segments.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `waveform` is empty.
    pub fn detect(&self, waveform: &[f32]) -> AudioResult<VadResult> {
        if waveform.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "waveform is empty".to_string(),
            });
        }

        let n_frames = num_frames(waveform.len(), self.cfg.frame_len, self.cfg.hop_len);
        if n_frames == 0 {
            // Waveform shorter than a single frame: no analysis frames at all.
            return Ok(VadResult {
                frame_flags: Vec::new(),
                segments: Vec::new(),
            });
        }

        // ── Stage 1: raw per-frame activity (energy AND flatness gates) ───────
        let mut raw_active = vec![false; n_frames];
        for (frame_idx, slot) in raw_active.iter_mut().enumerate() {
            let start = frame_idx * self.cfg.hop_len;
            let frame = &waveform[start..start + self.cfg.frame_len];
            let energy_db = self.frame_energy_db(frame)?;
            let flatness = self.spectral_flatness(frame)?;
            *slot = energy_db >= self.cfg.energy_threshold_db
                && flatness <= self.cfg.spectral_flatness_threshold;
        }

        // ── Stage 2: onset + hangover hysteresis ─────────────────────────────
        let frame_flags = self.smooth(&raw_active);

        // ── Stage 3: contiguous segment extraction ───────────────────────────
        let segments = extract_segments(&frame_flags);

        Ok(VadResult {
            frame_flags,
            segments,
        })
    }

    /// Apply onset and hangover smoothing to a raw activity track.
    ///
    /// Onset: speech only *starts* once `onset_frames` consecutive raw-active
    /// frames have been seen; the whole run (including the leading frames that
    /// satisfied the requirement) is marked as speech. Hangover: after speech
    /// has started, an inactive frame keeps being marked as speech while the
    /// hangover budget (`hangover_frames`) lasts, and any active frame refills
    /// the budget.
    fn smooth(&self, raw_active: &[bool]) -> Vec<bool> {
        let n = raw_active.len();
        let mut flags = vec![false; n];
        let mut in_speech = false;
        let mut run = 0usize;
        let mut hangover_left = 0usize;

        for i in 0..n {
            if raw_active[i] {
                if in_speech {
                    // Already in speech: stay, refill hangover budget.
                    flags[i] = true;
                    hangover_left = self.cfg.hangover_frames;
                } else {
                    run += 1;
                    if run >= self.cfg.onset_frames {
                        // Onset confirmed: declare speech and retroactively mark
                        // the (onset_frames - 1) preceding active frames in this
                        // run that we had left provisional.
                        in_speech = true;
                        hangover_left = self.cfg.hangover_frames;
                        let back = self.cfg.onset_frames.min(i + 1);
                        for f in flags.iter_mut().take(i + 1).skip(i + 1 - back) {
                            *f = true;
                        }
                    }
                }
            } else {
                // Inactive frame.
                run = 0;
                if in_speech {
                    if hangover_left > 0 {
                        flags[i] = true;
                        hangover_left -= 1;
                    } else {
                        in_speech = false;
                    }
                }
            }
        }
        flags
    }

    /// Validate that `frame.len() == frame_len`.
    fn check_frame_len(&self, frame: &[f32]) -> AudioResult<()> {
        if frame.len() != self.cfg.frame_len {
            return Err(AudioError::DimensionMismatch {
                expected: self.cfg.frame_len,
                got: frame.len(),
            });
        }
        Ok(())
    }
}

// ─── Free helpers ───────────────────────────────────────────────────────────

/// Number of full frames extractable from `len` samples with the given frame
/// length and hop. Returns `0` when `len < frame_len`.
#[inline]
fn num_frames(len: usize, frame_len: usize, hop_len: usize) -> usize {
    if len < frame_len {
        0
    } else {
        (len - frame_len) / hop_len + 1
    }
}

/// One-sided magnitude spectrum of a real frame via a direct (O(n²)) DFT.
///
/// Returns `floor(n / 2) + 1` non-negative magnitude bins (DC through Nyquist),
/// matching the standard real-FFT layout. For `n == 1` this is the single DC
/// bin (the absolute value of the lone sample).
fn dft_magnitude(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_bins = n / 2 + 1;
    let mut mags = vec![0.0f32; n_bins];
    let two_pi = 2.0 * std::f32::consts::PI;
    let inv_n = 1.0 / n as f32;
    for (k, mag) in mags.iter_mut().enumerate() {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let w = two_pi * k as f32 * inv_n;
        for (t, &x) in frame.iter().enumerate() {
            let angle = w * t as f32;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        *mag = (re * re + im * im).sqrt();
    }
    mags
}

/// Extract contiguous `true` runs from `flags` as `(start, end)` half-open
/// intervals (`end` exclusive). The result is sorted ascending and
/// non-overlapping by construction.
fn extract_segments(flags: &[bool]) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &f) in flags.iter().enumerate() {
        match (f, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                segments.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        segments.push((s, flags.len()));
    }
    segments
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn default_vad() -> Vad {
        Vad::new(VadConfig::tiny()).expect("tiny config is valid")
    }

    /// Build a `frame_len`-sample sine wave of the given amplitude.
    fn sine_frame(frame_len: usize, cycles: f32, amplitude: f32) -> Vec<f32> {
        (0..frame_len)
            .map(|t| amplitude * (2.0 * PI * cycles * t as f32 / frame_len as f32).sin())
            .collect()
    }

    /// Build a deterministic pseudo-random ("white-ish") frame in [-1, 1].
    fn noise_frame(frame_len: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::handle::LcgRng::new(seed);
        (0..frame_len)
            .map(|_| (rng.next_u32() as f32 / (u32::MAX as f32 + 1.0)) * 2.0 - 1.0)
            .collect()
    }

    /// Concatenate a loud-tone region then a silent region into one waveform.
    fn tone_then_silence(frame_len: usize, hop_len: usize) -> Vec<f32> {
        let tone_samples = frame_len + hop_len * 30;
        let silence_samples = frame_len + hop_len * 30;
        let mut wave = Vec::with_capacity(tone_samples + silence_samples);
        for t in 0..tone_samples {
            wave.push(0.8 * (2.0 * PI * 5.0 * t as f32 / frame_len as f32).sin());
        }
        wave.extend(std::iter::repeat_n(0.0f32, silence_samples));
        wave
    }

    // ── construction / validation ────────────────────────────────────────────

    #[test]
    fn new_valid_config_ok() {
        assert!(Vad::new(VadConfig::tiny()).is_ok());
    }

    #[test]
    fn new_frame_len_zero_err() {
        let mut cfg = VadConfig::tiny();
        cfg.frame_len = 0;
        assert_eq!(
            Vad::new(cfg).unwrap_err(),
            AudioError::InvalidSequenceLength(0)
        );
    }

    #[test]
    fn new_hop_len_zero_err() {
        let mut cfg = VadConfig::tiny();
        cfg.hop_len = 0;
        assert_eq!(Vad::new(cfg).unwrap_err(), AudioError::InvalidStride(0));
    }

    #[test]
    fn new_onset_frames_zero_err() {
        let mut cfg = VadConfig::tiny();
        cfg.onset_frames = 0;
        assert!(matches!(
            Vad::new(cfg).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn new_hangover_zero_ok() {
        let mut cfg = VadConfig::tiny();
        cfg.hangover_frames = 0;
        assert!(Vad::new(cfg).is_ok());
    }

    // ── frame_energy_db ────────────────────────────────────────────────────────

    #[test]
    fn energy_db_of_silence_is_very_low() {
        let vad = default_vad();
        let silence = vec![0.0f32; vad.config().frame_len];
        let db = vad.frame_energy_db(&silence).expect("ok");
        // 10*log10(eps) ≈ -100 dB for eps = 1e-10.
        assert!(db < -90.0, "silence energy too high: {db}");
    }

    #[test]
    fn energy_db_of_loud_sine_is_high() {
        let vad = default_vad();
        let frame = sine_frame(vad.config().frame_len, 5.0, 0.9);
        let db = vad.frame_energy_db(&frame).expect("ok");
        // RMS of a 0.9-amplitude sine ≈ 0.636 → ~ -3.9 dB; comfortably > -20.
        assert!(db > -20.0, "loud sine energy too low: {db}");
    }

    #[test]
    fn energy_db_louder_is_higher() {
        let vad = default_vad();
        let quiet = sine_frame(vad.config().frame_len, 5.0, 0.1);
        let loud = sine_frame(vad.config().frame_len, 5.0, 0.9);
        let q = vad.frame_energy_db(&quiet).expect("ok");
        let l = vad.frame_energy_db(&loud).expect("ok");
        assert!(l > q, "loud {l} should exceed quiet {q}");
    }

    #[test]
    fn energy_db_wrong_length_err() {
        let vad = default_vad();
        let short = vec![0.0f32; vad.config().frame_len - 1];
        assert!(matches!(
            vad.frame_energy_db(&short).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    // ── spectral_flatness ──────────────────────────────────────────────────────

    #[test]
    fn flatness_in_unit_range() {
        let vad = default_vad();
        for seed in 0..8u64 {
            let frame = noise_frame(vad.config().frame_len, seed + 1);
            let f = vad.spectral_flatness(&frame).expect("ok");
            assert!((0.0..=1.0).contains(&f), "flatness out of range: {f}");
        }
        let tone = sine_frame(vad.config().frame_len, 5.0, 0.8);
        let f = vad.spectral_flatness(&tone).expect("ok");
        assert!((0.0..=1.0).contains(&f), "tone flatness out of range: {f}");
    }

    #[test]
    fn flatness_noise_higher_than_tone() {
        let vad = default_vad();
        let tone = sine_frame(vad.config().frame_len, 5.0, 0.8);
        let noise = noise_frame(vad.config().frame_len, 12345);
        let tone_flat = vad.spectral_flatness(&tone).expect("ok");
        let noise_flat = vad.spectral_flatness(&noise).expect("ok");
        assert!(
            noise_flat > tone_flat,
            "noise flatness {noise_flat} should exceed tone flatness {tone_flat}"
        );
    }

    #[test]
    fn flatness_pure_tone_is_low() {
        let vad = default_vad();
        let tone = sine_frame(vad.config().frame_len, 5.0, 0.8);
        let f = vad.spectral_flatness(&tone).expect("ok");
        assert!(f < 0.3, "pure tone flatness should be low, got {f}");
    }

    #[test]
    fn flatness_wrong_length_err() {
        let vad = default_vad();
        let bad = vec![0.0f32; vad.config().frame_len + 3];
        assert!(matches!(
            vad.spectral_flatness(&bad).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    // ── detect: errors ─────────────────────────────────────────────────────────

    #[test]
    fn detect_empty_waveform_err() {
        let vad = default_vad();
        assert!(matches!(
            vad.detect(&[]).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    // ── detect: behaviour ──────────────────────────────────────────────────────

    #[test]
    fn detect_all_silence_no_speech() {
        let vad = default_vad();
        let cfg = vad.config().clone();
        let waveform = vec![0.0f32; cfg.frame_len + cfg.hop_len * 40];
        let result = vad.detect(&waveform).expect("ok");
        assert!(
            result.frame_flags.iter().all(|&f| !f),
            "silence must yield no speech frames"
        );
        assert!(result.segments.is_empty(), "silence must yield no segments");
    }

    #[test]
    fn detect_loud_tone_has_speech() {
        let vad = default_vad();
        let cfg = vad.config().clone();
        let mut waveform = Vec::new();
        let total = cfg.frame_len + cfg.hop_len * 40;
        for t in 0..total {
            waveform.push(0.8 * (2.0 * PI * 5.0 * t as f32 / cfg.frame_len as f32).sin());
        }
        let result = vad.detect(&waveform).expect("ok");
        assert!(
            result.frame_flags.iter().any(|&f| f),
            "loud tone must yield some speech frames"
        );
        assert!(
            !result.segments.is_empty(),
            "loud tone must yield at least one segment"
        );
    }

    #[test]
    fn detect_frame_flags_length_matches_n_frames() {
        let vad = default_vad();
        let cfg = vad.config().clone();
        let len = cfg.frame_len + cfg.hop_len * 25;
        let waveform = vec![0.1f32; len];
        let result = vad.detect(&waveform).expect("ok");
        let expected = num_frames(len, cfg.frame_len, cfg.hop_len);
        assert_eq!(result.frame_flags.len(), expected);
    }

    #[test]
    fn detect_segments_contiguous_sorted_non_overlapping() {
        let vad = default_vad();
        let waveform = tone_then_silence(vad.config().frame_len, vad.config().hop_len);
        let result = vad.detect(&waveform).expect("ok");
        // Each segment is a valid half-open interval, sorted and disjoint.
        for &(s, e) in &result.segments {
            assert!(s < e, "segment ({s},{e}) is not a valid interval");
            assert!(e <= result.frame_flags.len(), "segment end out of range");
        }
        for pair in result.segments.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "segments overlap or are unsorted: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
        // Every flagged frame must lie inside exactly one reported segment.
        for (i, &flag) in result.frame_flags.iter().enumerate() {
            let inside = result.segments.iter().any(|&(s, e)| i >= s && i < e);
            assert_eq!(flag, inside, "frame {i} flag/segment mismatch");
        }
    }

    #[test]
    fn detect_speech_then_silence_one_segment() {
        let vad = default_vad();
        let waveform = tone_then_silence(vad.config().frame_len, vad.config().hop_len);
        let result = vad.detect(&waveform).expect("ok");
        assert_eq!(
            result.segments.len(),
            1,
            "tone-then-silence should yield exactly one segment, got {:?}",
            result.segments
        );
        // The single segment must start early and end before the final frame
        // (silence at the tail is not speech, modulo hangover).
        let (start, end) = result.segments[0];
        assert!(start < end);
        assert!(
            end < result.frame_flags.len(),
            "segment should end before the silent tail (end={end}, n={})",
            result.frame_flags.len()
        );
    }

    #[test]
    fn detect_is_deterministic() {
        let vad = default_vad();
        let waveform = tone_then_silence(vad.config().frame_len, vad.config().hop_len);
        let a = vad.detect(&waveform).expect("ok");
        let b = vad.detect(&waveform).expect("ok");
        assert_eq!(a, b, "detect must be deterministic");
    }

    #[test]
    fn detect_short_waveform_no_frames() {
        let vad = default_vad();
        let short = vec![0.5f32; vad.config().frame_len - 1];
        let result = vad.detect(&short).expect("ok");
        assert!(result.frame_flags.is_empty());
        assert!(result.segments.is_empty());
    }

    // ── onset / hangover hysteresis (via the smooth path) ──────────────────────

    #[test]
    fn onset_suppresses_single_frame_blip() {
        let mut cfg = VadConfig::tiny();
        cfg.onset_frames = 2;
        cfg.hangover_frames = 0;
        let vad = Vad::new(cfg).expect("ok");
        // A single isolated active frame should not survive a 2-frame onset.
        let raw = [false, false, true, false, false];
        let flags = vad.smooth(&raw);
        assert!(
            flags.iter().all(|&f| !f),
            "isolated blip should be suppressed: {flags:?}"
        );
    }

    #[test]
    fn onset_admits_consecutive_active_run() {
        let mut cfg = VadConfig::tiny();
        cfg.onset_frames = 2;
        cfg.hangover_frames = 0;
        let vad = Vad::new(cfg).expect("ok");
        // Two consecutive active frames satisfy onset; both get marked.
        let raw = [false, true, true, false];
        let flags = vad.smooth(&raw);
        assert_eq!(flags, vec![false, true, true, false]);
    }

    #[test]
    fn hangover_extends_speech_region() {
        let mut cfg = VadConfig::tiny();
        cfg.onset_frames = 1;
        cfg.hangover_frames = 2;
        let vad = Vad::new(cfg).expect("ok");
        // One active frame then silence: hangover keeps speech for 2 more frames.
        let raw = [true, false, false, false, false];
        let flags = vad.smooth(&raw);
        assert_eq!(
            flags,
            vec![true, true, true, false, false],
            "hangover should extend speech by 2 frames"
        );
    }

    #[test]
    fn hangover_zero_no_extension() {
        let mut cfg = VadConfig::tiny();
        cfg.onset_frames = 1;
        cfg.hangover_frames = 0;
        let vad = Vad::new(cfg).expect("ok");
        let raw = [true, false, true, false];
        let flags = vad.smooth(&raw);
        assert_eq!(flags, vec![true, false, true, false]);
    }

    #[test]
    fn energy_threshold_boundary_active() {
        // Construct a config whose energy threshold sits just below a frame's
        // measured energy, with a permissive flatness gate, so the frame is
        // active exactly at the boundary.
        let frame_len = 64usize;
        let probe_vad = Vad::new(VadConfig {
            frame_len,
            hop_len: frame_len,
            energy_threshold_db: -200.0,
            spectral_flatness_threshold: 1.0,
            hangover_frames: 0,
            onset_frames: 1,
        })
        .expect("ok");
        let frame = sine_frame(frame_len, 4.0, 0.5);
        let measured = probe_vad.frame_energy_db(&frame).expect("ok");

        // Threshold exactly equal to measured energy → active (>= is inclusive).
        let cfg = VadConfig {
            frame_len,
            hop_len: frame_len,
            energy_threshold_db: measured,
            spectral_flatness_threshold: 1.0,
            hangover_frames: 0,
            onset_frames: 1,
        };
        let vad = Vad::new(cfg).expect("ok");
        let result = vad.detect(&frame).expect("ok");
        assert_eq!(result.frame_flags.len(), 1);
        assert!(
            result.frame_flags[0],
            "frame at the exact energy threshold should be active"
        );

        // Threshold a hair above measured energy → inactive.
        let cfg_above = VadConfig {
            frame_len,
            hop_len: frame_len,
            energy_threshold_db: measured + 1.0,
            spectral_flatness_threshold: 1.0,
            hangover_frames: 0,
            onset_frames: 1,
        };
        let vad_above = Vad::new(cfg_above).expect("ok");
        let result_above = vad_above.detect(&frame).expect("ok");
        assert!(
            !result_above.frame_flags[0],
            "frame below the energy threshold should be inactive"
        );
    }

    // ── helper-level coverage ──────────────────────────────────────────────────

    #[test]
    fn num_frames_matches_expected() {
        assert_eq!(num_frames(10, 4, 2), 4); // starts at 0,2,4,6
        assert_eq!(num_frames(3, 4, 2), 0); // shorter than a frame
        assert_eq!(num_frames(4, 4, 2), 1); // exactly one frame
    }

    #[test]
    fn extract_segments_basic() {
        let flags = [false, true, true, false, true, false];
        assert_eq!(extract_segments(&flags), vec![(1, 3), (4, 5)]);
        let trailing = [false, true, true];
        assert_eq!(extract_segments(&trailing), vec![(1, 3)]);
        let none = [false, false];
        assert!(extract_segments(&none).is_empty());
    }

    #[test]
    fn dft_magnitude_bin_count_and_nonneg() {
        let frame = sine_frame(16, 2.0, 1.0);
        let mags = dft_magnitude(&frame);
        assert_eq!(mags.len(), 16 / 2 + 1);
        assert!(mags.iter().all(|&m| m >= 0.0));
        // A length-1 frame yields a single DC bin equal to |sample|.
        let dc = dft_magnitude(&[-3.0]);
        assert_eq!(dc.len(), 1);
        assert!((dc[0] - 3.0).abs() < 1e-6);
    }
}
