//! Temporal encoding with continuous (real-valued) time embeddings.
//!
//! Classical sequence encoding in hyperdimensional computing represents *position* by repeated
//! application of a fixed permutation `ρ^{i}` (Kanerva 2009; see
//! [`crate::encoding::sequence_hd`]). That scheme is inherently *discrete*: it only knows about
//! integer slots `0, 1, 2, …`, so two events one millisecond apart and two events one hour apart
//! look identical if they occupy adjacent slots. Many real signals — EMG gestures, neural spike
//! trains, log streams, sensor telemetry — instead carry a *continuous* timestamp, and the
//! representation should make events that are *close in real time* end up *close in
//! hyperspace*.
//!
//! This module implements continuous-time encoding via **fractional power encoding** of the time
//! axis (Frady, Kleyko & Sommer 2021; Komer & Eliasmith spatial-semantic pointers; Rahimi 2016
//! for the EMG/time-series HDC pipeline). The construction is:
//!
//! 1. **Time embedding.** A fixed random Fourier-Holographic-Reduced-Representation (FHRR) base
//!    of `D` phases `φ_k` (a [`crate::vector::fpe::FpeBase`]) defines a continuous-time
//!    hypervector for any real timestamp `t`: phase `k` is `t' · φ_k` where `t' = t ·
//!    time_scale`. Because binding is phase addition, this `base^{t'}` is a *fractional binding*
//!    whose self-similarity kernel is a smooth, locality-preserving bump of `|t'₁ − t'₂|`: it
//!    peaks at `1.0` when the times coincide and decays as they separate. `time_scale` sets the
//!    kernel width (larger scale ⇒ finer temporal resolution / faster decay).
//! 2. **Symbol vocabulary.** Each of the `n_symbols` discrete event symbols is assigned a fixed
//!    random phasor-only FHRR hypervector (see [`crate::vector::fhrr`]), quasi-orthogonal to the
//!    others.
//! 3. **Event binding.** Each event `(t, sym)` is encoded as `bind(symbol_hv[sym],
//!    time_phase(t))` — the symbol's identity bound to *when* it occurred.
//! 4. **Bundling.** All per-event bindings are superposed with the FHRR circular-mean bundle,
//!    yielding a single fixed-width hypervector that softly stores the whole `(symbol, time)`
//!    set.
//!
//! Querying whether a symbol occurred near a time `t` re-creates `bind(symbol_hv[sym],
//! time_phase(t))` and measures its cosine similarity to the bundle: the value is high when that
//! `(sym, t)` pair (or a near-in-time occurrence of `sym`) is present, and falls off both for
//! absent symbols and for the right symbol at the wrong time.
//!
//! # Representation note
//!
//! [`crate::vector::fpe::FpeBase::encode`] returns the *interleaved* `[re, im]` FHRR layout of
//! length `2·D`, whereas the binding / bundling / similarity primitives in
//! [`crate::vector::fhrr`] operate on the *phase-only* layout of length `D`. To keep the
//! encode → bind → bundle pipeline dimensionally consistent, the time embedding used for binding
//! ([`TemporalHdEncoder::time_phase`]) is the phase-only twin `θ_k = (t' · φ_k) mod 2π` of the
//! same fractional encoding, derived from the identical base phases. The interleaved form is
//! still exposed via [`TemporalHdEncoder::time_hv`] for inspecting the continuous-time kernel
//! with [`crate::vector::fpe::fpe_similarity`].
//!
//! # References
//!
//! - A. Rahimi *et al.*, "Hyperdimensional Computing for Blind and One-Shot Classification of
//!   EEG / EMG Signals" (2016).
//! - E. P. Frady, D. Kleyko & F. T. Sommer, "Computing on Functions Using Randomized Vector
//!   Representations" (2021) — fractional power encoding.
//! - P. Kanerva, "Hyperdimensional Computing" (2009) — binding / bundling algebra.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::fhrr::{fhrr_bind, fhrr_bundle, fhrr_cosine, random_fhrr};
use crate::vector::fpe::FpeBase;

use std::f32::consts::TAU;

/// Wrap a phase angle into the canonical range `[0, 2π)` used by the phasor-only FHRR layout.
#[inline]
fn wrap_phase(theta: f32) -> f32 {
    let mut t = theta % TAU;
    if t < 0.0 {
        t += TAU;
    }
    if t >= TAU {
        t -= TAU;
    }
    t
}

/// Continuous-time hyperdimensional encoder over a fixed symbol vocabulary.
///
/// Holds the fractional-power time base, the random symbol hypervectors, and the temporal
/// `time_scale`. All hypervectors live in the phasor-only FHRR layout (`Vec<f32>` of length
/// `D`, each entry a phase in `[0, 2π)`) so that [`crate::vector::fhrr::fhrr_bind`],
/// [`crate::vector::fhrr::fhrr_bundle`] and [`crate::vector::fhrr::fhrr_cosine`] apply directly.
#[derive(Debug, Clone)]
pub struct TemporalHdEncoder {
    /// Number of complex components `D` (the phase-only hypervector length).
    dim: usize,
    /// Fractional-power encoder for the continuous time axis.
    time_base: FpeBase,
    /// One fixed random phasor-only FHRR hypervector per symbol; each has length `D`.
    symbol_hvs: Vec<Vec<f32>>,
    /// Multiplies timestamps before fractional encoding, controlling the kernel width.
    time_scale: f32,
}

impl TemporalHdEncoder {
    /// Build a temporal encoder with `n_symbols` random symbols of dimension `dim`.
    ///
    /// `time_scale` multiplies every timestamp before fractional encoding: a larger value
    /// shrinks the effective temporal kernel (events must be closer in real time to look
    /// similar), a smaller value widens it.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `n_symbols == 0`.
    /// - [`HdcError::InvalidProbability`] if `time_scale` is not finite and strictly positive.
    pub fn new(n_symbols: usize, dim: usize, time_scale: f32, rng: &mut LcgRng) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_symbols == 0 {
            return Err(HdcError::EmptyInput);
        }
        if !time_scale.is_finite() || time_scale <= 0.0 {
            return Err(HdcError::InvalidProbability(time_scale as f64));
        }
        let time_base = FpeBase::random(dim, rng)?;
        let mut symbol_hvs = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            symbol_hvs.push(random_fhrr(dim, rng)?);
        }
        Ok(Self {
            dim,
            time_base,
            symbol_hvs,
            time_scale,
        })
    }

    /// The number of complex components `D` (phase-only hypervector length).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The size of the symbol vocabulary.
    #[must_use]
    pub fn n_symbols(&self) -> usize {
        self.symbol_hvs.len()
    }

    /// The temporal scaling factor applied to timestamps before fractional encoding.
    #[must_use]
    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    /// Borrow the fixed random hypervector for symbol `sym` (phasor-only, length `D`).
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `sym >= n_symbols`.
    pub fn symbol_hv(&self, sym: usize) -> HdcResult<&[f32]> {
        self.symbol_hvs
            .get(sym)
            .map(Vec::as_slice)
            .ok_or(HdcError::FeatureIndexOutOfRange {
                feat: sym,
                max: self.symbol_hvs.len(),
            })
    }

    /// Continuous-time embedding of timestamp `t` in the **interleaved** `[re, im]` FHRR layout
    /// (length `2·D`), i.e. `time_base.encode(t · time_scale)`.
    ///
    /// This is the fractional power encoding `base^{t · time_scale}`. It is the natural input to
    /// [`crate::vector::fpe::fpe_similarity`] for inspecting the locality-preserving temporal
    /// kernel: two timestamps close in (scaled) time give a similarity near `1.0` that decays as
    /// they separate. For the binding pipeline use [`TemporalHdEncoder::time_phase`], the
    /// phase-only twin of this encoding.
    #[must_use]
    pub fn time_hv(&self, t: f32) -> Vec<f32> {
        self.time_base.encode(t * self.time_scale)
    }

    /// Continuous-time embedding of timestamp `t` in the **phasor-only** FHRR layout (length
    /// `D`): `θ_k = (t · time_scale · φ_k) mod 2π`.
    ///
    /// This is the phase-only twin of [`TemporalHdEncoder::time_hv`] (whose component `k` is
    /// `(cos θ_k, sin θ_k)`), derived from the same fractional-power base phases. It binds and
    /// bundles consistently with the symbol hypervectors, all of length `D`.
    #[must_use]
    pub fn time_phase(&self, t: f32) -> Vec<f32> {
        let scaled = t * self.time_scale;
        self.time_base
            .phases()
            .iter()
            .map(|&phi| wrap_phase(scaled * phi))
            .collect()
    }

    /// Encode a sequence of continuous-time events into one hypervector.
    ///
    /// Each event is a `(timestamp, symbol_index)` pair; timestamps are real-valued and need not
    /// be sorted or evenly spaced. The encoding bundles `bind(symbol_hv[sym], time_phase(t))`
    /// over all events, returning the phasor-only FHRR hypervector of length `D`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `events` is empty.
    /// - [`HdcError::FeatureIndexOutOfRange`] if any event's symbol index is `>= n_symbols`.
    /// - Propagates [`crate::vector::fhrr`] binding/bundling errors (dimension mismatches), which
    ///   cannot arise here because every hypervector is constructed with length `D`.
    pub fn encode(&self, events: &[(f32, usize)]) -> HdcResult<Vec<f32>> {
        if events.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let mut bound = Vec::with_capacity(events.len());
        for &(t, sym) in events {
            let symbol = self.symbol_hv(sym)?;
            let time = self.time_phase(t);
            bound.push(fhrr_bind(symbol, &time)?);
        }
        fhrr_bundle(&bound)
    }

    /// Probe whether symbol `sym` occurred near continuous time `t` in an encoded bundle.
    ///
    /// Re-creates the event binding `bind(symbol_hv[sym], time_phase(t))` and returns its cosine
    /// similarity ([`crate::vector::fhrr::fhrr_cosine`]) to `encoded`. The result is high (toward
    /// `1.0`) when that `(sym, t)` pair — or a near-in-time occurrence of `sym` — is present in
    /// the bundle, and lower when the symbol is absent or `sym` occurred only at distant times,
    /// because both the symbol mismatch (quasi-orthogonal symbol HVs) and the time mismatch (the
    /// decaying fractional-power kernel) reduce the overlap.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `sym >= n_symbols`.
    /// - [`HdcError::DimensionMismatch`] if `encoded.len() != D`.
    /// - [`HdcError::EmptyInput`] if `encoded` is empty.
    pub fn query_time(&self, encoded: &[f32], sym: usize, t: f32) -> HdcResult<f32> {
        let symbol = self.symbol_hv(sym)?;
        let time = self.time_phase(t);
        let probe = fhrr_bind(symbol, &time)?;
        fhrr_cosine(encoded, &probe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::fpe::fpe_similarity;

    fn rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    #[test]
    fn new_rejects_zero_dim() {
        let mut r = rng(0x1111_2222);
        let res = TemporalHdEncoder::new(8, 0, 1.0, &mut r);
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn new_rejects_zero_symbols() {
        let mut r = rng(0x3333_4444);
        let res = TemporalHdEncoder::new(0, 512, 1.0, &mut r);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn new_rejects_bad_time_scale() {
        let mut r = rng(0x5555_6666);
        // Zero, negative, NaN and infinite scales are all invalid.
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let res = TemporalHdEncoder::new(4, 256, bad, &mut r);
            assert!(
                matches!(res, Err(HdcError::InvalidProbability(_))),
                "time_scale {bad} should be rejected"
            );
        }
    }

    #[test]
    fn accessors_report_construction_parameters() {
        let mut r = rng(0x7777_8888);
        let enc = TemporalHdEncoder::new(12, 512, 2.5, &mut r).expect("encoder");
        assert_eq!(enc.dim(), 512);
        assert_eq!(enc.n_symbols(), 12);
        assert!((enc.time_scale() - 2.5).abs() < 1e-6);
        // Each symbol hypervector is a valid length-D phasor vector.
        for s in 0..enc.n_symbols() {
            let hv = enc.symbol_hv(s).expect("symbol");
            assert_eq!(hv.len(), 512);
            for &p in hv {
                assert!((0.0..TAU).contains(&p), "phase {p} out of range");
            }
        }
    }

    #[test]
    fn symbol_hv_out_of_range_errors() {
        let mut r = rng(0x9999_AAAA);
        let enc = TemporalHdEncoder::new(3, 256, 1.0, &mut r).expect("encoder");
        let res = enc.symbol_hv(3);
        assert!(matches!(
            res,
            Err(HdcError::FeatureIndexOutOfRange { feat: 3, max: 3 })
        ));
    }

    #[test]
    fn time_hv_self_similarity_is_one() {
        let mut r = rng(0xBBBB_CCCC);
        let enc = TemporalHdEncoder::new(4, 1024, 1.0, &mut r).expect("encoder");
        let hv = enc.time_hv(3.5);
        assert_eq!(hv.len(), 2 * 1024);
        let sim = fpe_similarity(&hv, &hv).expect("similarity");
        assert!((sim - 1.0).abs() < 1e-4, "self-similarity={sim}");
    }

    #[test]
    fn near_times_more_similar_than_far() {
        let mut r = rng(0xDDDD_EEEE);
        let enc = TemporalHdEncoder::new(4, 1024, 1.0, &mut r).expect("encoder");
        let anchor = enc.time_hv(10.0);
        let near = enc.time_hv(10.05);
        let far = enc.time_hv(13.0);
        let sim_near = fpe_similarity(&anchor, &near).expect("near");
        let sim_far = fpe_similarity(&anchor, &far).expect("far");
        assert!(
            sim_near > sim_far,
            "kernel should decay with time: near={sim_near} far={sim_far}"
        );
        assert!(sim_near > 0.9, "near times too dissimilar: {sim_near}");
    }

    #[test]
    fn time_scale_sharpens_kernel() {
        // A larger time_scale should make the same real-time gap look *less* similar
        // (narrower kernel), demonstrating the controllable kernel width.
        let mut r1 = rng(0x0102_0304);
        let mut r2 = rng(0x0102_0304);
        let wide = TemporalHdEncoder::new(2, 1024, 0.5, &mut r1).expect("wide");
        let narrow = TemporalHdEncoder::new(2, 1024, 4.0, &mut r2).expect("narrow");
        let gap = 0.5f32;
        let sim_wide = {
            let a = wide.time_hv(0.0);
            let b = wide.time_hv(gap);
            fpe_similarity(&a, &b).expect("wide sim")
        };
        let sim_narrow = {
            let a = narrow.time_hv(0.0);
            let b = narrow.time_hv(gap);
            fpe_similarity(&a, &b).expect("narrow sim")
        };
        assert!(
            sim_wide > sim_narrow,
            "larger time_scale should sharpen kernel: wide={sim_wide} narrow={sim_narrow}"
        );
    }

    #[test]
    fn encode_is_deterministic_for_same_input() {
        let mut r1 = rng(0x1357_2468);
        let mut r2 = rng(0x1357_2468);
        let enc1 = TemporalHdEncoder::new(6, 512, 1.5, &mut r1).expect("enc1");
        let enc2 = TemporalHdEncoder::new(6, 512, 1.5, &mut r2).expect("enc2");
        let events = [(0.0f32, 0usize), (1.2, 3), (2.7, 5), (4.1, 1)];
        let a = enc1.encode(&events).expect("encode a");
        let b = enc2.encode(&events).expect("encode b");
        assert_eq!(a, b, "same seed + same events must encode identically");
    }

    #[test]
    fn encode_output_has_dimension_d() {
        let mut r = rng(0x2468_1357);
        let enc = TemporalHdEncoder::new(5, 768, 1.0, &mut r).expect("encoder");
        let events = [(0.5f32, 0usize), (1.5, 2), (3.0, 4)];
        let hv = enc.encode(&events).expect("encode");
        assert_eq!(hv.len(), 768, "phasor-only bundle must have length D");
        for &p in &hv {
            assert!((0.0..TAU).contains(&p), "bundle phase {p} out of range");
        }
    }

    #[test]
    fn encode_rejects_empty_events() {
        let mut r = rng(0xABAB_CDCD);
        let enc = TemporalHdEncoder::new(4, 256, 1.0, &mut r).expect("encoder");
        let res = enc.encode(&[]);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn encode_rejects_out_of_range_symbol() {
        let mut r = rng(0xCDCD_ABAB);
        let enc = TemporalHdEncoder::new(3, 256, 1.0, &mut r).expect("encoder");
        // Symbol index 7 >= n_symbols (3).
        let res = enc.encode(&[(0.0f32, 0usize), (1.0, 7)]);
        assert!(matches!(
            res,
            Err(HdcError::FeatureIndexOutOfRange { feat: 7, max: 3 })
        ));
    }

    #[test]
    fn query_time_finds_present_event_and_rejects_absent() {
        let mut r = rng(0xFEED_BEEF);
        let enc = TemporalHdEncoder::new(8, 1024, 2.0, &mut r).expect("encoder");
        // Symbol 2 occurs at t = 5.0; symbol 6 never occurs.
        let events = [(1.0f32, 0usize), (5.0, 2), (9.0, 4)];
        let encoded = enc.encode(&events).expect("encode");

        // Present (sym, t) pair -> high similarity.
        let hit = enc.query_time(&encoded, 2, 5.0).expect("hit");
        // Absent symbol at the same time -> lower similarity.
        let miss_symbol = enc.query_time(&encoded, 6, 5.0).expect("miss symbol");
        // Right symbol at a distant time -> lower similarity.
        let miss_time = enc.query_time(&encoded, 2, 5.0 + 8.0).expect("miss time");

        assert!(
            hit > miss_symbol,
            "present event should beat absent symbol: hit={hit} miss={miss_symbol}"
        );
        assert!(
            hit > miss_time,
            "present event should beat wrong-time query: hit={hit} miss={miss_time}"
        );
        assert!(hit > 0.2, "present-event similarity too low: {hit}");
    }

    #[test]
    fn query_time_rejects_out_of_range_symbol() {
        let mut r = rng(0x0BAD_F00D);
        let enc = TemporalHdEncoder::new(4, 256, 1.0, &mut r).expect("encoder");
        let encoded = enc.encode(&[(0.0f32, 1usize)]).expect("encode");
        let res = enc.query_time(&encoded, 9, 0.0);
        assert!(matches!(
            res,
            Err(HdcError::FeatureIndexOutOfRange { feat: 9, max: 4 })
        ));
    }

    #[test]
    fn query_time_rejects_dimension_mismatch() {
        let mut r = rng(0xDEAD_C0DE);
        let enc = TemporalHdEncoder::new(4, 256, 1.0, &mut r).expect("encoder");
        // An encoded bundle of the wrong length must be rejected by the cosine step.
        let wrong = vec![0.0f32; 128];
        let res = enc.query_time(&wrong, 0, 0.0);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn time_phase_has_length_d_and_valid_range() {
        let mut r = rng(0x5A5A_5A5A);
        let enc = TemporalHdEncoder::new(2, 512, 1.5, &mut r).expect("encoder");
        let tp = enc.time_phase(7.3);
        assert_eq!(tp.len(), 512);
        for &p in &tp {
            assert!((0.0..TAU).contains(&p), "time phase {p} out of range");
        }
        // At t = 0 every phase collapses to 0 (the fractional-power identity).
        let zero = enc.time_phase(0.0);
        for &p in &zero {
            assert!(p.abs() < 1e-6, "t=0 phase should be 0, got {p}");
        }
    }
}
