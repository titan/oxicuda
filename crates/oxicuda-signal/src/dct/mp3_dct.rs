//! MP3 / MPEG-1 Audio Layer III aligned MDCT with block-type window switching.
//!
//! ISO/IEC 11172-3 specifies the hybrid filterbank's MDCT stage with two block
//! lengths and four window types selected per granule to trade frequency
//! resolution for time resolution around transients:
//!
//! | Block type | `block_type` | MDCT N | Window length | Window |
//! |------------|--------------|--------|---------------|--------|
//! | Normal (long)  | 0 | 18 | 36 | full sine          |
//! | Start          | 1 | 18 | 36 | long→short ramp     |
//! | Short          | 2 | 6  | 12 | three short windows |
//! | Stop           | 3 | 18 | 36 | short→long ramp     |
//!
//! The standard window definitions are (n = 0..window_length):
//!
//! ```text
//! type 0 (normal): w[n] = sin( π/36 · (n + 1/2) )
//! type 1 (start) : w[n] = sin( π/36 · (n + 1/2) )            0 ≤ n < 18
//!                  1.0                                       18 ≤ n < 24
//!                  sin( π/12 · (n − 18 + 1/2) )              24 ≤ n < 30
//!                  0.0                                       30 ≤ n < 36
//! type 3 (stop)  : 0.0                                       0 ≤ n < 6
//!                  sin( π/12 · (n − 6 + 1/2) )               6 ≤ n < 12
//!                  1.0                                       12 ≤ n < 18
//!                  sin( π/36 · (n + 1/2) )                   18 ≤ n < 36
//! type 2 (short) : w[n] = sin( π/12 · (n + 1/2) )            (per 12-sample sub-window)
//! ```
//!
//! For a short block the 36 input samples are processed as three overlapping
//! 12-sample short MDCTs producing 3 × 6 = 18 coefficients, interleaved in the
//! ISO order.
//!
//! References:
//!   ISO/IEC 11172-3:1993 §2.4.3.4 (hybrid filterbank, IMDCT, windowing).

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Canonical (TDAC-perfect) MDCT / IMDCT used by the MP3 hybrid filterbank.
//
//  These use the standard ISO/IEC 11172-3 phase term
//  `cos( π/N · (n + 1/2 + N/2) · (k + 1/2) )`, which — unlike a bare DCT-IV —
//  guarantees time-domain aliasing cancellation (TDAC) when adjacent windowed
//  blocks are overlap-added at hop `N`.  The module is therefore self-contained
//  and exactly invertible regardless of other DCT conventions in the crate.
// --------------------------------------------------------------------------- //

/// Forward MDCT of a length-`2N` block, producing `N` coefficients.
fn mdct_core(x: &[f64], n: usize) -> Vec<f64> {
    let n2 = 2 * n;
    let mut out = vec![0.0_f64; n];
    let phase0 = n as f64 / 2.0;
    for (k, ok) in out.iter_mut().enumerate() {
        let mut s = 0.0_f64;
        for (nn, &xn) in x.iter().enumerate().take(n2) {
            s += xn * (PI / n as f64 * (nn as f64 + 0.5 + phase0) * (k as f64 + 0.5)).cos();
        }
        *ok = s;
    }
    out
}

/// Inverse MDCT of `N` coefficients, producing a length-`2N` (aliased) block;
/// overlap-add of adjacent blocks cancels the aliasing (TDAC).
fn imdct_core(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    let n2 = 2 * n;
    let scale = 2.0 / n as f64;
    let phase0 = n as f64 / 2.0;
    let mut out = vec![0.0_f64; n2];
    for (nn, on) in out.iter_mut().enumerate() {
        let mut s = 0.0_f64;
        for (k, &ck) in coeffs.iter().enumerate() {
            s += ck * (PI / n as f64 * (nn as f64 + 0.5 + phase0) * (k as f64 + 0.5)).cos();
        }
        *on = s * scale;
    }
    out
}

/// MP3 granule block type, selecting MDCT length and window shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp3BlockType {
    /// Normal long block (`block_type = 0`): 36→18, full sine window.
    Normal,
    /// Start block (`block_type = 1`): long block transitioning to short.
    Start,
    /// Short block (`block_type = 2`): three 12→6 short MDCTs.
    Short,
    /// Stop block (`block_type = 3`): short block transitioning to long.
    Stop,
}

impl Mp3BlockType {
    /// The MPEG-1 numeric `block_type` field value.
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Mp3BlockType::Normal => 0,
            Mp3BlockType::Start => 1,
            Mp3BlockType::Short => 2,
            Mp3BlockType::Stop => 3,
        }
    }

    /// Number of MDCT coefficients produced for this block type (always 18 — a
    /// short block yields 3 × 6 interleaved coefficients).
    #[must_use]
    pub fn num_coeffs(self) -> usize {
        18
    }
}

/// Length of an MP3 long-block MDCT window (36 samples → 18 coefficients).
pub const MP3_LONG_WINDOW_LEN: usize = 36;
/// Length of an MP3 short-block MDCT window (12 samples → 6 coefficients).
pub const MP3_SHORT_WINDOW_LEN: usize = 12;
/// MP3 long-block MDCT size N (= 18).
pub const MP3_LONG_N: usize = 18;
/// MP3 short-block MDCT size N (= 6).
pub const MP3_SHORT_N: usize = 6;

// --------------------------------------------------------------------------- //
//  ISO 11172-3 window tables
// --------------------------------------------------------------------------- //

/// Build the 36-sample window for a given block type (per ISO 11172-3).
///
/// The short block returns the *single* 12-sample short window (the three
/// sub-windows are identical); see [`mp3_short_window`].
#[must_use]
pub fn mp3_window(block_type: Mp3BlockType) -> Vec<f64> {
    match block_type {
        Mp3BlockType::Normal => (0..MP3_LONG_WINDOW_LEN)
            .map(|n| (PI / 36.0 * (n as f64 + 0.5)).sin())
            .collect(),
        Mp3BlockType::Start => {
            let mut w = vec![0.0_f64; MP3_LONG_WINDOW_LEN];
            for (n, wn) in w.iter_mut().enumerate().take(18) {
                *wn = (PI / 36.0 * (n as f64 + 0.5)).sin();
            }
            for wn in w.iter_mut().take(24).skip(18) {
                *wn = 1.0;
            }
            for (n, wn) in w.iter_mut().enumerate().take(30).skip(24) {
                *wn = (PI / 12.0 * ((n - 18) as f64 + 0.5)).sin();
            }
            // 30..36 already 0.0
            w
        }
        Mp3BlockType::Stop => {
            let mut w = vec![0.0_f64; MP3_LONG_WINDOW_LEN];
            // 0..6 already 0.0
            for (n, wn) in w.iter_mut().enumerate().take(12).skip(6) {
                *wn = (PI / 12.0 * ((n - 6) as f64 + 0.5)).sin();
            }
            for wn in w.iter_mut().take(18).skip(12) {
                *wn = 1.0;
            }
            for (n, wn) in w.iter_mut().enumerate().take(36).skip(18) {
                *wn = (PI / 36.0 * (n as f64 + 0.5)).sin();
            }
            w
        }
        Mp3BlockType::Short => mp3_short_window(),
    }
}

/// The 12-sample MP3 short-block sine window `w[n] = sin(π/12 · (n + 1/2))`.
#[must_use]
pub fn mp3_short_window() -> Vec<f64> {
    (0..MP3_SHORT_WINDOW_LEN)
        .map(|n| (PI / 12.0 * (n as f64 + 0.5)).sin())
        .collect()
}

// --------------------------------------------------------------------------- //
//  Forward MDCT
// --------------------------------------------------------------------------- //

/// Forward MP3 MDCT of a 36-sample subband block, with windowing and block-type
/// switching, producing 18 frequency coefficients.
///
/// For long / start / stop blocks this windows the 36 input samples and applies
/// the length-18 MDCT.  For a short block it windows and transforms three
/// overlapping 12-sample sub-blocks (input samples `0..12`, `12..24`, `24..36`)
/// with the short window, producing 3 × 6 coefficients placed in the ISO
/// interleaved order `out[3·i + s] = short_coeff[s][i]`.
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `subband.len() != 36`.
pub fn mp3_mdct(subband: &[f64], block_type: Mp3BlockType) -> SignalResult<Vec<f64>> {
    if subband.len() != MP3_LONG_WINDOW_LEN {
        return Err(SignalError::InvalidSize(format!(
            "MP3 MDCT expects 36 input samples, got {}",
            subband.len()
        )));
    }
    match block_type {
        Mp3BlockType::Short => {
            let sw = mp3_short_window();
            let mut out = vec![0.0_f64; 18];
            for s in 0..3 {
                let base = s * 12;
                let windowed: Vec<f64> = (0..12).map(|n| subband[base + n] * sw[n]).collect();
                let coeffs = mdct_core(&windowed, MP3_SHORT_N); // 6 coefficients
                for (i, &c) in coeffs.iter().enumerate() {
                    out[3 * i + s] = c;
                }
            }
            Ok(out)
        }
        long_type => {
            let w = mp3_window(long_type);
            let windowed: Vec<f64> = subband
                .iter()
                .zip(w.iter())
                .map(|(&x, &wn)| x * wn)
                .collect();
            Ok(mdct_core(&windowed, MP3_LONG_N)) // 18 coefficients
        }
    }
}

// --------------------------------------------------------------------------- //
//  Inverse MDCT (IMDCT) + windowing
// --------------------------------------------------------------------------- //

/// Inverse MP3 MDCT of 18 frequency coefficients, producing the 36-sample
/// windowed time-domain block ready for overlap-add with the next granule.
///
/// For long / start / stop blocks the length-18 IMDCT yields 36 samples which
/// are multiplied by the block-type window.  For a short block the 18
/// coefficients are de-interleaved into three 6-coefficient sets, each
/// inverse-transformed to 12 samples, short-windowed and overlap-added at the
/// 6-sample stride (the standard short-block reconstruction), then zero-padded
/// at the head/tail to a 36-sample block.
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `coeffs.len() != 18`.
pub fn mp3_imdct(coeffs: &[f64], block_type: Mp3BlockType) -> SignalResult<Vec<f64>> {
    if coeffs.len() != MP3_LONG_N {
        return Err(SignalError::InvalidSize(format!(
            "MP3 IMDCT expects 18 coefficients, got {}",
            coeffs.len()
        )));
    }
    match block_type {
        Mp3BlockType::Short => {
            let sw = mp3_short_window();
            // De-interleave: short_coeff[s][i] = coeffs[3·i + s].
            let mut block = vec![0.0_f64; MP3_LONG_WINDOW_LEN];
            for s in 0..3 {
                let mut sub = [0.0_f64; 6];
                for (i, sv) in sub.iter_mut().enumerate() {
                    *sv = coeffs[3 * i + s];
                }
                let time = imdct_core(&sub); // 12 samples
                // Window and overlap-add at offset s·12 within the 36-block,
                // following the ISO short-block placement.
                let base = s * 12;
                for n in 0..12 {
                    block[base + n] += time[n] * sw[n];
                }
            }
            Ok(block)
        }
        long_type => {
            let time = imdct_core(coeffs); // 36 samples
            let w = mp3_window(long_type);
            Ok(time.iter().zip(w.iter()).map(|(&t, &wn)| t * wn).collect())
        }
    }
}

// --------------------------------------------------------------------------- //
//  Plan
// --------------------------------------------------------------------------- //

/// Execution plan for a batch of MP3 MDCT granule blocks of a fixed block type.
///
/// Mirrors [`crate::dct::MdctPlan`] but fixes the ISO 11172-3 window/length
/// parameters from a [`Mp3BlockType`], pre-computing the window table.
#[derive(Debug, Clone)]
pub struct Mp3MdctPlan {
    /// Block type controlling MDCT length and window.
    pub block_type: Mp3BlockType,
    /// Number of independent 36-sample granule blocks.
    pub batch: usize,
    /// Pre-computed 36-sample window (long types) or the short window repeated.
    pub window_coeffs: Vec<f64>,
}

impl Mp3MdctPlan {
    /// Create a new MP3 MDCT plan for the given block type and batch size.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidParameter`] if `batch == 0`.
    pub fn new(block_type: Mp3BlockType, batch: usize) -> SignalResult<Self> {
        if batch == 0 {
            return Err(SignalError::InvalidParameter(
                "batch must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            block_type,
            batch,
            window_coeffs: mp3_window(block_type),
        })
    }

    /// Number of MDCT coefficients per block (always 18).
    #[must_use]
    pub fn num_coeffs(&self) -> usize {
        self.block_type.num_coeffs()
    }

    /// Forward-transform a single 36-sample granule block.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidSize`] if `subband.len() != 36`.
    pub fn forward(&self, subband: &[f64]) -> SignalResult<Vec<f64>> {
        mp3_mdct(subband, self.block_type)
    }

    /// Inverse-transform 18 coefficients back to a windowed 36-sample block.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidSize`] if `coeffs.len() != 18`.
    pub fn inverse(&self, coeffs: &[f64]) -> SignalResult<Vec<f64>> {
        mp3_imdct(coeffs, self.block_type)
    }
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_block(seed: u64) -> Vec<f64> {
        let mut v = Vec::with_capacity(36);
        let mut s = seed;
        for _ in 0..36 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 32) as u32) as f64 / (u32::MAX as f64);
            v.push(2.0 * u - 1.0);
        }
        v
    }

    #[test]
    fn test_block_type_codes() {
        assert_eq!(Mp3BlockType::Normal.code(), 0);
        assert_eq!(Mp3BlockType::Start.code(), 1);
        assert_eq!(Mp3BlockType::Short.code(), 2);
        assert_eq!(Mp3BlockType::Stop.code(), 3);
        assert_eq!(Mp3BlockType::Short.num_coeffs(), 18);
    }

    #[test]
    fn test_normal_window_values() {
        let w = mp3_window(Mp3BlockType::Normal);
        assert_eq!(w.len(), 36);
        assert!((w[0] - (PI / 36.0 * 0.5).sin()).abs() < 1e-12);
        assert!((w[35] - (PI / 36.0 * 35.5).sin()).abs() < 1e-12);
        // Princen-Bradley perfect-reconstruction property: w[n]² + w[n+18]² = 1.
        for n in 0..18 {
            let pr = w[n] * w[n] + w[n + 18] * w[n + 18];
            assert!((pr - 1.0).abs() < 1e-12, "PR failed at n={n}: {pr}");
        }
    }

    #[test]
    fn test_start_window_shape() {
        let w = mp3_window(Mp3BlockType::Start);
        assert_eq!(w.len(), 36);
        // Flat unity region 18..24.
        for wn in w.iter().take(24).skip(18) {
            assert!((wn - 1.0).abs() < 1e-12);
        }
        // Zero tail 30..36.
        for wn in w.iter().skip(30) {
            assert!(wn.abs() < 1e-12);
        }
    }

    #[test]
    fn test_stop_window_shape() {
        let w = mp3_window(Mp3BlockType::Stop);
        assert_eq!(w.len(), 36);
        // Zero head 0..6.
        for wn in w.iter().take(6) {
            assert!(wn.abs() < 1e-12);
        }
        // Flat unity 12..18.
        for wn in w.iter().take(18).skip(12) {
            assert!((wn - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_start_stop_complementary_overlap() {
        // The start window's tapering-down half and the stop window's
        // tapering-up half must satisfy the PR identity at the short-window
        // junction: start[24+k]² + stop[6+k]² = 1 for k=0..6.
        let start = mp3_window(Mp3BlockType::Start);
        let stop = mp3_window(Mp3BlockType::Stop);
        for k in 0..6 {
            let s = start[24 + k] * start[24 + k] + stop[6 + k] * stop[6 + k];
            assert!((s - 1.0).abs() < 1e-12, "junction PR failed at k={k}: {s}");
        }
    }

    #[test]
    fn test_short_window() {
        let w = mp3_short_window();
        assert_eq!(w.len(), 12);
        for n in 0..6 {
            let pr = w[n] * w[n] + w[n + 6] * w[n + 6];
            assert!((pr - 1.0).abs() < 1e-12, "short PR at n={n}");
        }
    }

    #[test]
    fn test_mp3_mdct_long_dims() {
        let x = lcg_block(1);
        let c = mp3_mdct(&x, Mp3BlockType::Normal).expect("ok");
        assert_eq!(c.len(), 18);
        // Wrong input length must error.
        assert!(mp3_mdct(&x[..35], Mp3BlockType::Normal).is_err());
        assert!(mp3_imdct(&c[..17], Mp3BlockType::Normal).is_err());
    }

    #[test]
    fn test_mp3_mdct_short_dims_and_interleave() {
        let x = lcg_block(2);
        let c = mp3_mdct(&x, Mp3BlockType::Short).expect("ok");
        assert_eq!(c.len(), 18);
        // Recompute one sub-block manually and confirm the interleave placement.
        let sw = mp3_short_window();
        let windowed: Vec<f64> = (0..12).map(|n| x[n] * sw[n]).collect();
        let sub = mdct_core(&windowed, MP3_SHORT_N);
        for (i, &sv) in sub.iter().enumerate() {
            // Sub-block s=0 lands at indices 3·i + 0.
            assert!((c[3 * i] - sv).abs() < 1e-12, "interleave i={i}");
        }
    }

    #[test]
    fn test_mp3_long_block_tdac_roundtrip() {
        // Two consecutive long blocks overlapping by 18 samples reconstruct the
        // shared region exactly via time-domain aliasing cancellation (TDAC).
        // Build a 54-sample stream; block A = [0..36], block B = [18..54].
        let stream: Vec<f64> = (0..54)
            .map(|i| {
                let mut s = i as u64 + 17;
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = ((s >> 32) as u32) as f64 / (u32::MAX as f64);
                2.0 * u - 1.0
            })
            .collect();
        let block_a: Vec<f64> = stream[0..36].to_vec();
        let block_b: Vec<f64> = stream[18..54].to_vec();

        let ca = mp3_mdct(&block_a, Mp3BlockType::Normal).expect("ok");
        let cb = mp3_mdct(&block_b, Mp3BlockType::Normal).expect("ok");
        let ra = mp3_imdct(&ca, Mp3BlockType::Normal).expect("ok");
        let rb = mp3_imdct(&cb, Mp3BlockType::Normal).expect("ok");

        // Overlap-add: the second half of A overlaps the first half of B,
        // covering stream samples 18..36. TDAC -> exact reconstruction there.
        for k in 0..18 {
            let recon = ra[18 + k] + rb[k];
            assert!(
                (recon - stream[18 + k]).abs() < 1e-9,
                "TDAC failed at stream idx {}: recon={recon} orig={}",
                18 + k,
                stream[18 + k]
            );
        }
    }

    #[test]
    fn test_mp3_plan() {
        let plan = Mp3MdctPlan::new(Mp3BlockType::Normal, 4).expect("ok");
        assert_eq!(plan.batch, 4);
        assert_eq!(plan.num_coeffs(), 18);
        assert_eq!(plan.window_coeffs.len(), 36);
        let x = lcg_block(5);
        let c = plan.forward(&x).expect("fwd");
        assert_eq!(c.len(), 18);
        let r = plan.inverse(&c).expect("inv");
        assert_eq!(r.len(), 36);
        assert!(Mp3MdctPlan::new(Mp3BlockType::Short, 0).is_err());
    }
}
