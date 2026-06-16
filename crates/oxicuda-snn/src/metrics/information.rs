#![allow(clippy::needless_range_loop)]
//! Information-theoretic spike-train metrics: entropy and mutual information.
//!
//! Continuous spike trains are discretised into symbolic *words* before any
//! information measure is computed. The word-binning pipeline is:
//!
//! 1. Coarse-grain a single-neuron train of `t_steps` samples into
//!    `n_bins = floor(t_steps / bin_steps)` non-overlapping bins of `bin_steps`
//!    steps each (a trailing partial bin is discarded).
//! 2. Threshold every bin to one bit: `1` if any sample in the bin spiked,
//!    `0` otherwise.
//! 3. Pack `word_bits` consecutive bits, most-significant-bit first, into one
//!    integer symbol in `[0, 2^word_bits)`. Words are non-overlapping, giving
//!    `M = floor(n_bins / word_bits)` symbols over an alphabet of size
//!    `K = 2^word_bits`.
//!
//! Entropy is the Shannon entropy of the empirical symbol distribution (bits);
//! mutual information is computed from the joint symbol histogram of two aligned
//! trains, optionally bias-corrected by the Miller-Madow estimator.
//!
//! Single-neuron trains are plain `&[f32]` of length `t_steps`; a spike is any
//! value `!= 0.0`.

use crate::error::{SnnError, SnnResult};

/// Bias correction applied to mutual-information estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiCorrection {
    /// Plug-in (naive) estimator with tiny-negative round-off clamped to zero.
    None,
    /// Miller-Madow bias correction; the corrected value is returned unclamped.
    MillerMadow,
}

/// Validate the shared word-binning parameters.
fn validate_word_params(t_steps: usize, bin_steps: usize, word_bits: usize) -> SnnResult<()> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if bin_steps == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if !(1..=16).contains(&word_bits) {
        return Err(SnnError::OutOfRange {
            name: "word_bits".to_string(),
            val: word_bits as f32,
        });
    }
    Ok(())
}

/// Convert a single-neuron train into a vector of word symbols.
///
/// Returns the symbol list of length `M = floor(n_bins / word_bits)`. The caller
/// has already validated the parameters via [`validate_word_params`].
fn words(s: &[f32], bin_steps: usize, word_bits: usize) -> Vec<u32> {
    let t_steps = s.len();
    let n_bins = t_steps / bin_steps;
    // Threshold each bin to a single bit.
    let mut bits = vec![0_u8; n_bins];
    for b in 0..n_bins {
        let start = b * bin_steps;
        let end = start + bin_steps;
        let mut bit = 0_u8;
        for &x in &s[start..end] {
            if x != 0.0 {
                bit = 1;
                break;
            }
        }
        bits[b] = bit;
    }
    // Pack consecutive bits (MSB first) into non-overlapping words.
    let m = n_bins / word_bits;
    let mut out = Vec::with_capacity(m);
    for w in 0..m {
        let base = w * word_bits;
        let mut symbol = 0_u32;
        for k in 0..word_bits {
            symbol = (symbol << 1) | u32::from(bits[base + k]);
        }
        out.push(symbol);
    }
    out
}

/// Shannon entropy (bits) of a single-neuron spike train under word-binning.
///
/// Bins the train, packs bits into `word_bits`-wide symbols, and returns
/// `H = −Σ_k p_k log2 p_k` over the empirical symbol distribution.
///
/// # Errors
/// Returns [`SnnError::BadTimesteps`] for `t_steps == 0`, [`SnnError::BadDim`]
/// for `bin_steps == 0`, [`SnnError::OutOfRange`] if `word_bits` is outside
/// `1..=16`, and [`SnnError::BadShape`] if `s.len() != t_steps`.
pub fn spike_train_entropy(
    s: &[f32],
    t_steps: usize,
    bin_steps: usize,
    word_bits: usize,
) -> SnnResult<f32> {
    validate_word_params(t_steps, bin_steps, word_bits)?;
    if s.len() != t_steps {
        return Err(SnnError::BadShape {
            expected: t_steps,
            got: s.len(),
        });
    }
    let symbols = words(s, bin_steps, word_bits);
    let m = symbols.len();
    if m == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    let k = 1_usize << word_bits;
    let mut counts = vec![0_u32; k];
    for &sym in &symbols {
        counts[sym as usize] += 1;
    }
    let m_f = m as f32;
    let mut h = 0.0_f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / m_f;
            h -= p * p.log2();
        }
    }
    Ok(h)
}

/// Mutual information (bits) between two single-neuron spike trains.
///
/// Both trains are word-binned with identical parameters and truncated to the
/// shorter symbol sequence (`M = min(M_a, M_b)`). From the `K × K` joint
/// histogram, `MI = ΣΣ p(x,y) log2[p(x,y) / (p(x) p(y))]`.
///
/// With [`MiCorrection::MillerMadow`], the bias correction
/// `(K̂_X + K̂_Y − K̂_XY − 1) / (2 · M · ln2)` is added, where each `K̂` is the
/// number of symbols (or symbol pairs) observed with non-zero count; the
/// corrected value is returned unclamped (and may be slightly negative).
/// With [`MiCorrection::None`], tiny negative round-off is clamped to zero.
///
/// # Errors
/// Returns [`SnnError::BadTimesteps`] for `t_steps == 0`, [`SnnError::BadDim`]
/// for `bin_steps == 0`, [`SnnError::OutOfRange`] if `word_bits` is outside
/// `1..=16`, [`SnnError::IncompatibleLength`] if the two trains differ in
/// length, [`SnnError::BadShape`] if either differs from `t_steps`, and
/// [`SnnError::BadTimesteps`] if fewer than one word can be formed (`M < 1`).
pub fn mutual_information(
    s_a: &[f32],
    s_b: &[f32],
    t_steps: usize,
    bin_steps: usize,
    word_bits: usize,
    correction: MiCorrection,
) -> SnnResult<f32> {
    validate_word_params(t_steps, bin_steps, word_bits)?;
    if s_a.len() != s_b.len() {
        return Err(SnnError::IncompatibleLength {
            a: s_a.len(),
            b: s_b.len(),
        });
    }
    if s_a.len() != t_steps {
        return Err(SnnError::BadShape {
            expected: t_steps,
            got: s_a.len(),
        });
    }

    let words_a = words(s_a, bin_steps, word_bits);
    let words_b = words(s_b, bin_steps, word_bits);
    let m = words_a.len().min(words_b.len());
    if m == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }

    let k = 1_usize << word_bits;
    let mut joint = vec![0_u32; k * k];
    let mut marg_x = vec![0_u32; k];
    let mut marg_y = vec![0_u32; k];
    for idx in 0..m {
        let x = words_a[idx] as usize;
        let y = words_b[idx] as usize;
        joint[x * k + y] += 1;
        marg_x[x] += 1;
        marg_y[y] += 1;
    }

    let m_f = m as f32;
    let mut mi = 0.0_f32;
    for x in 0..k {
        if marg_x[x] == 0 {
            continue;
        }
        let px = marg_x[x] as f32 / m_f;
        for y in 0..k {
            let cxy = joint[x * k + y];
            if cxy == 0 || marg_y[y] == 0 {
                continue;
            }
            let pxy = cxy as f32 / m_f;
            let py = marg_y[y] as f32 / m_f;
            mi += pxy * (pxy / (px * py)).log2();
        }
    }

    match correction {
        MiCorrection::None => {
            // Plug-in MI is non-negative in exact arithmetic; clamp round-off.
            if mi < 0.0 {
                mi = 0.0;
            }
            Ok(mi)
        }
        MiCorrection::MillerMadow => {
            let k_x = marg_x.iter().filter(|&&c| c > 0).count();
            let k_y = marg_y.iter().filter(|&&c| c > 0).count();
            let k_xy = joint.iter().filter(|&&c| c > 0).count();
            let correction_term = (k_x as f32 + k_y as f32 - k_xy as f32 - 1.0)
                / (2.0 * m_f * std::f32::consts::LN_2);
            Ok(mi + correction_term)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_one_bit_for_fair_alternating() {
        let s = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let h = spike_train_entropy(&s, 8, 1, 1).expect("entropy");
        assert!((h - 1.0).abs() < 1e-6, "H={h}");
    }

    #[test]
    fn entropy_zero_for_all_zero() {
        let s = [0.0_f32; 8];
        let h = spike_train_entropy(&s, 8, 1, 1).expect("entropy");
        assert!(h.abs() < 1e-6, "H={h}");
    }

    #[test]
    fn mi_one_bit_for_identical() {
        let s = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mi = mutual_information(&s, &s, 8, 1, 1, MiCorrection::None).expect("mi");
        assert!((mi - 1.0).abs() < 1e-6, "MI={mi}");
    }

    #[test]
    fn mi_one_bit_for_anti_correlated() {
        let a = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let b = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mi = mutual_information(&a, &b, 8, 1, 1, MiCorrection::None).expect("mi");
        assert!((mi - 1.0).abs() < 1e-6, "MI={mi}");
    }

    #[test]
    fn mi_zero_for_one_constant() {
        let a = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let b = [0.0_f32; 8];
        let mi = mutual_information(&a, &b, 8, 1, 1, MiCorrection::None).expect("mi");
        assert!(mi.abs() < 1e-6, "MI={mi}");
    }

    #[test]
    fn mi_never_negative_plugin() {
        // A noisier, less symmetric pairing must still yield MI >= 0.
        let a = [0.0_f32, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let b = [1.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let mi = mutual_information(&a, &b, 8, 1, 1, MiCorrection::None).expect("mi");
        assert!(mi >= 0.0, "MI={mi}");
    }

    #[test]
    fn miller_madow_at_least_plugin() {
        let s = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let plugin = mutual_information(&s, &s, 8, 1, 1, MiCorrection::None).expect("plugin");
        let mm = mutual_information(&s, &s, 8, 1, 1, MiCorrection::MillerMadow).expect("mm");
        assert!(mm >= plugin - 1e-6, "mm={mm} plugin={plugin}");
    }

    #[test]
    fn length_and_param_errors() {
        let s = [0.0_f32; 8];
        assert!(matches!(
            spike_train_entropy(&s, 0, 1, 1),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            spike_train_entropy(&s, 8, 0, 1),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            spike_train_entropy(&s, 8, 1, 0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            spike_train_entropy(&s, 8, 1, 17),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            spike_train_entropy(&s, 7, 1, 1),
            Err(SnnError::BadShape { .. })
        ));
        let short = [0.0_f32; 6];
        assert!(matches!(
            mutual_information(&s, &short, 8, 1, 1, MiCorrection::None),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }
}
