//! TimesNet block: 1-D → 2-D → 2-D conv → 1-D, with FFT-based period detection.
//!
//! Reference: Wu et al. 2023 "TimesNet: Temporal 2D-Variation Modeling for
//! General Time Series Analysis" (ICLR 2023).
//!
//! CPU reference implementation using O(T²) DFT, depthwise 3×3 2-D conv,
//! and layer normalisation over the C dimension.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Private helper: DFT magnitudes ─────────────────────────────────────────

/// Compute DFT magnitudes for frequencies f = 1 ..= T/2 of a real sequence.
///
/// Uses the direct O(T²) formula:
/// `X[f] = Σ_{t=0}^{T-1} x[t] * exp(-2πi·f·t / T)`
/// `mag[f] = |X[f]|`
///
/// Returns a `Vec<f32>` of length `T/2` (DC at f=0 is excluded).
fn dft_magnitudes(x: &[f32]) -> Vec<f32> {
    let t = x.len();
    let n_freq = t / 2; // number of frequencies 1..=n_freq
    if n_freq == 0 {
        return Vec::new();
    }

    let two_pi_over_t = 2.0 * std::f32::consts::PI / t as f32;
    let mut mags = Vec::with_capacity(n_freq);

    for f in 1..=n_freq {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (ti, &xi) in x.iter().enumerate() {
            let angle = two_pi_over_t * (f * ti) as f32;
            re += xi * angle.cos();
            im -= xi * angle.sin(); // exp(-2πi·f·t/T)
        }
        mags.push((re * re + im * im).sqrt());
    }
    mags
}

// ─── Private helper: depthwise 2-D conv 3×3 ─────────────────────────────────

/// Depthwise 2-D convolution with a 3×3 kernel and same (symmetric zero) padding.
///
/// * `input`  — `[rows, cols, C]` row-major, C innermost.
/// * `weight` — `[C, 3, 3]` one filter per channel.
/// * `bias`   — `[C]`.
///
/// Returns `[rows, cols, C]` (same shape as input).
fn depthwise_conv2d_3x3(
    input: &[f32],
    rows: usize,
    cols: usize,
    c: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols * c];

    for (ch, &ch_bias) in bias.iter().enumerate() {
        // weight layout: [C, 3, 3] → filter for channel ch starts at ch * 9
        let w_base = ch * 9;

        for r in 0..rows {
            for col in 0..cols {
                let mut acc = ch_bias;
                // 3×3 kernel, centre at (kr=1, kc=1).
                // We compute the source coordinates as signed integers to handle
                // the same-padding (zero-fill) at borders naturally.
                let r_signed = r as isize;
                let col_signed = col as isize;
                for kr in 0..3isize {
                    let src_row = r_signed + kr - 1; // -1 offset for centre
                    if src_row < 0 || src_row >= rows as isize {
                        continue; // zero-pad
                    }
                    for kc in 0..3isize {
                        let src_col = col_signed + kc - 1;
                        if src_col < 0 || src_col >= cols as isize {
                            continue; // zero-pad
                        }
                        let w_idx = w_base + (kr as usize) * 3 + kc as usize;
                        let in_idx = (src_row as usize) * cols * c + (src_col as usize) * c + ch;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
                let out_idx = r * cols * c + col * c + ch;
                out[out_idx] = acc;
            }
        }
    }
    out
}

// ─── Private helper: layer norm over C ───────────────────────────────────────

/// Apply layer normalisation over the C (channel) dimension for each timestep.
///
/// For each t: normalise `x[t, :]` to zero-mean unit-variance, then
/// apply per-channel affine transform `γ * x̂ + β`.
///
/// Operates in-place on the flat `[T, C]` slice.
fn layer_norm_c(x: &mut [f32], t: usize, c: usize, gamma: &[f32], beta: &[f32]) {
    const EPS: f32 = 1e-5;
    for ti in 0..t {
        let base = ti * c;
        let slice = &mut x[base..base + c];

        // mean
        let mean: f32 = slice.iter().sum::<f32>() / c as f32;

        // variance
        let var: f32 = slice
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / c as f32;

        let inv_std = 1.0 / (var + EPS).sqrt();

        for (ci, s) in slice.iter_mut().enumerate() {
            *s = (*s - mean) * inv_std * gamma[ci] + beta[ci];
        }
    }
}

// ─── TimesBlock ──────────────────────────────────────────────────────────────

/// A single TimesNet block.
///
/// Applies FFT-based top-k period detection, 2-D reshape and 3×3 depthwise
/// convolution for each detected period, then sums amplitude-weighted outputs
/// back into 1-D and adds the input residual.
///
/// See the module-level documentation for the full algorithm description.
#[derive(Debug, Clone)]
pub struct TimesBlock {
    /// 2-D conv weight: `[C, 3, 3]` depthwise.
    pub conv_weight: Vec<f32>,
    /// 2-D conv bias: `[C]`.
    pub conv_bias: Vec<f32>,
    /// Layer norm weight `[C]`.
    pub norm_g: Vec<f32>,
    /// Layer norm bias `[C]`.
    pub norm_b: Vec<f32>,
    /// Number of channels.
    pub c: usize,
    /// Number of top periods.
    pub top_k: usize,
}

impl TimesBlock {
    /// Construct a `TimesBlock`.
    ///
    /// Conv weights are initialised with Kaiming He scaling; layer norm
    /// parameters are identity (`γ=1, β=0`).
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidNumVariates`]`(0)` when `c == 0`.
    /// - [`TsError::InvalidTopK`]`(top_k)` when `top_k == 0`.
    pub fn new(c: usize, top_k: usize, rng: &mut LcgRng) -> TsResult<Self> {
        if c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if top_k == 0 {
            return Err(TsError::InvalidTopK(0));
        }

        // Depthwise conv weight [C, 3, 3]: Kaiming He std = sqrt(2 / (3*3))
        let he_std = (2.0_f32 / 9.0).sqrt();
        let mut conv_weight = vec![0.0_f32; c * 9];
        rng.fill_normal(&mut conv_weight);
        for v in &mut conv_weight {
            *v *= he_std;
        }

        let conv_bias = vec![0.0_f32; c];
        let norm_g = vec![1.0_f32; c];
        let norm_b = vec![0.0_f32; c];

        Ok(Self {
            conv_weight,
            conv_bias,
            norm_g,
            norm_b,
            c,
            top_k,
        })
    }

    /// Forward pass: `[T, C] → [T, C]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`]`(t)` when `t < 2`.
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    pub fn forward(&self, x: &[f32], t: usize) -> TsResult<Vec<f32>> {
        if t < 2 {
            return Err(TsError::InvalidSequenceLength(t));
        }
        let expected = t * self.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let c = self.c;
        let n_freq = t / 2;

        // ── 1. Average DFT magnitudes over channels ───────────────────────────
        // For each channel, compute DFT on the time axis; then average.
        let mut avg_mag = vec![0.0_f32; n_freq];
        for ch in 0..c {
            // extract channel slice [T]
            let ch_slice: Vec<f32> = (0..t).map(|ti| x[ti * c + ch]).collect();
            let mags = dft_magnitudes(&ch_slice);
            for (m, &v) in avg_mag.iter_mut().zip(mags.iter()) {
                *m += v;
            }
        }
        let inv_c = 1.0 / c as f32;
        for m in &mut avg_mag {
            *m *= inv_c;
        }

        // ── 2. Select top-k frequencies by magnitude ─────────────────────────
        // top_k is clamped to n_freq so we never exceed FFT length
        let k = self.top_k.min(n_freq).max(1);

        // indices into avg_mag (0-based, corresponding to freq 1..=n_freq)
        let mut freq_indices: Vec<usize> = (0..n_freq).collect();
        // partial sort descending by magnitude
        freq_indices.sort_unstable_by(|&a, &b| {
            avg_mag[b]
                .partial_cmp(&avg_mag[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_freqs: Vec<usize> = freq_indices[..k].to_vec();

        // amplitudes for weighting (freq index → amplitude)
        let amp_sum: f32 = top_freqs
            .iter()
            .map(|&fi| avg_mag[fi])
            .sum::<f32>()
            .max(1e-12);
        let weights: Vec<f32> = top_freqs.iter().map(|&fi| avg_mag[fi] / amp_sum).collect();

        // ── 3. For each period: 2-D reshape → conv → reshape back → weight ───
        let mut output = vec![0.0_f32; t * c];

        for (idx, &fi) in top_freqs.iter().enumerate() {
            let freq = fi + 1; // fi=0 corresponds to freq 1
            // period p_i = T / freq, minimum 2
            let period = (t / freq).max(2);
            let num_periods = t.div_ceil(period);
            let t_pad = period * num_periods;

            // a. Pad x to t_pad by replicating the last timestep
            let mut padded = vec![0.0_f32; t_pad * c];
            // copy existing data
            padded[..t * c].copy_from_slice(x);
            // replicate last timestep for padding
            if t_pad > t {
                let last_start = (t - 1) * c;
                let last_row = x[last_start..last_start + c].to_vec();
                for ti in t..t_pad {
                    let base = ti * c;
                    padded[base..base + c].copy_from_slice(&last_row);
                }
            }

            // b. Reshape [t_pad, C] → [period, num_periods, C]
            //    The 2-D grid has rows=period, cols=num_periods, channels=C.
            //    Mapping: padded[ti, ch] → grid[ti % period, ti / period, ch]
            let mut grid = vec![0.0_f32; period * num_periods * c];
            for ti in 0..t_pad {
                let r = ti % period;
                let col = ti / period;
                let src_base = ti * c;
                let dst_base = r * num_periods * c + col * c;
                grid[dst_base..dst_base + c].copy_from_slice(&padded[src_base..src_base + c]);
            }

            // c. Depthwise 3×3 conv on [period, num_periods, C]
            let conv_out = depthwise_conv2d_3x3(
                &grid,
                period,
                num_periods,
                c,
                &self.conv_weight,
                &self.conv_bias,
            );

            // d. Reshape back [period, num_periods, C] → [t_pad, C], trim to [T, C]
            let w = weights[idx];
            for ti in 0..t {
                let r = ti % period;
                let col = ti / period;
                let src_base = r * num_periods * c + col * c;
                let dst_base = ti * c;
                for ch in 0..c {
                    output[dst_base + ch] += w * conv_out[src_base + ch];
                }
            }
        }

        // ── 4. Residual addition ──────────────────────────────────────────────
        for (o, &xi) in output.iter_mut().zip(x.iter()) {
            *o += xi;
        }

        // ── 5. Layer norm over C ─────────────────────────────────────────────
        layer_norm_c(&mut output, t, c, &self.norm_g, &self.norm_b);

        Ok(output)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn times_block_output_shape() {
        let mut rng = make_rng();
        let block = TimesBlock::new(8, 3, &mut rng).expect("ok");
        let t = 24;
        let x = vec![0.1_f32; t * 8];
        let out = block.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * 8, "output must have shape [T, C]");
    }

    #[test]
    fn times_block_output_finite() {
        let mut rng = make_rng();
        let block = TimesBlock::new(4, 3, &mut rng).expect("ok");
        let t = 32;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn times_block_zero_c_error() {
        let mut rng = make_rng();
        assert!(matches!(
            TimesBlock::new(0, 3, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn times_block_zero_topk_error() {
        let mut rng = make_rng();
        assert!(matches!(
            TimesBlock::new(4, 0, &mut rng).unwrap_err(),
            TsError::InvalidTopK(0)
        ));
    }

    #[test]
    fn times_block_short_seq_error() {
        let mut rng = make_rng();
        let block = TimesBlock::new(4, 3, &mut rng).expect("ok");
        let x = vec![0.0_f32; 4];
        assert!(matches!(
            block.forward(&x, 1).unwrap_err(),
            TsError::InvalidSequenceLength(1)
        ));
    }

    #[test]
    fn times_block_dim_mismatch() {
        let mut rng = make_rng();
        let block = TimesBlock::new(4, 3, &mut rng).expect("ok");
        // provide wrong number of elements (t=10 but only 8 elements for c=4 → expect 40)
        let x = vec![0.0_f32; 8];
        assert!(matches!(
            block.forward(&x, 10).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn times_block_residual_connection() {
        // With non-trivial weights the output must differ from the input.
        let mut rng = make_rng();
        let block = TimesBlock::new(4, 3, &mut rng).expect("ok");
        let t = 16;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("ok");
        // at least one element should differ (conv + layernorm transforms the signal)
        let any_diff = out
            .iter()
            .zip(x.iter())
            .any(|(&o, &xi)| (o - xi).abs() > 1e-6);
        assert!(
            any_diff,
            "output should differ from input after block transform"
        );
    }

    #[test]
    fn dft_magnitudes_dc_removed() {
        // Pure cosine at frequency 2 (over T=8) should produce a peak at f=2.
        // x[t] = cos(2π * 2 * t / 8)
        let t = 8_usize;
        let x: Vec<f32> = (0..t)
            .map(|ti| (2.0 * std::f32::consts::PI * 2.0 * ti as f32 / t as f32).cos())
            .collect();
        let mags = dft_magnitudes(&x);
        // mags has length T/2 = 4; f=1,2,3,4 → indices 0,1,2,3
        assert_eq!(mags.len(), 4);
        // magnitude at f=2 (index 1) should be largest
        let peak_idx = mags
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty");
        assert_eq!(
            peak_idx, 1,
            "peak should be at f=2 (index 1), mags={mags:?}"
        );
        // DC (f=0) must not appear — length is T/2, starting from f=1
        assert_eq!(mags.len(), t / 2);
    }
}
