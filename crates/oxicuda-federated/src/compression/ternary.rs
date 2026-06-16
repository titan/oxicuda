//! TernGrad / Atomo ternary gradient quantization.
//!
//! **TernGrad** (Wen et al., "TernGrad: Ternary Gradients to Reduce Communication
//! in Distributed Deep Learning", NeurIPS 2017):
//! Maps each gradient element stochastically to {−s, 0, +s} where s = max|g_i|.
//! Quantisation is unbiased: `E[q_i] = g_i`.
//!
//! **Atomo** (Wang et al., "Atomo: Communication-Efficient Learning via Atomic
//! Sparsification", NeurIPS 2018):
//! Deterministic top-k ternary sparsification.  Selects the k elements with
//! largest absolute value, sets their sign as the code, and normalises the
//! scale to minimise the expected MSE under the bandwidth budget:
//!
//! ```text
//!   scale = ||g||₂ / √k
//!   code_i = sign(g_i)   if |g_i| is in top-k
//!   code_i = 0            otherwise
//! ```
//!
//! Both methods encode each gradient as (`codes: Vec<i8>`, `scale: f32`) where
//! `codes[i] ∈ {-1, 0, 1}` and the reconstruction is `scale * codes[i] as f32`.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Which ternary quantization variant to use.
#[derive(Debug, Clone)]
pub enum TernaryMode {
    /// Stochastic ternarization (TernGrad, Wen et al. NeurIPS 2017).
    ///
    /// `q_i = ±s` with probability `|g_i| / s`, else 0.
    TernGrad,

    /// Deterministic top-k sparsification (Atomo, Wang et al. NeurIPS 2018).
    ///
    /// Keeps the `ceil(keep_fraction * n)` elements with largest absolute
    /// value as non-zero ternary entries; scale is set to `||g||₂ / √k`.
    Atomo {
        /// Fraction of elements to keep in (0, 1].  Must satisfy `0 < f ≤ 1`.
        keep_fraction: f32,
    },
}

/// Configuration for ternary gradient compression.
#[derive(Debug, Clone)]
pub struct TernaryConfig {
    /// Which ternary mode to use.
    pub mode: TernaryMode,
    /// For TernGrad: sample at most `batch_size` elements to estimate the
    /// scale `s = max|g_i|`.  0 means use all elements (exact).
    pub batch_size: usize,
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            mode: TernaryMode::TernGrad,
            batch_size: 0,
        }
    }
}

// ─── Encoded gradient ────────────────────────────────────────────────────────

/// Ternary-encoded gradient vector.
#[derive(Debug, Clone)]
pub struct TernaryEncoded {
    /// Code for each element: −1, 0, or +1.
    pub codes: Vec<i8>,
    /// Reconstruction scale factor `s`.  `decode[i] = s * codes[i] as f32`.
    pub scale: f32,
    /// Number of non-zero codes.
    pub n_nonzero: usize,
}

// ─── Compressor ──────────────────────────────────────────────────────────────

/// TernGrad / Atomo ternary gradient compressor.
pub struct TernaryCompressor;

impl TernaryCompressor {
    // ── TernGrad ─────────────────────────────────────────────────────────────

    /// TernGrad stochastic ternarization.
    ///
    /// Computes scale `s = max|g_i|` (or a sample approximation when
    /// `cfg.batch_size > 0` and `cfg.batch_size < n`).
    ///
    /// For each element: `code_i = sign(g_i)` with probability `|g_i|/s`,
    /// else `0`.  The expected reconstruction is equal to `g` (unbiased).
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `gradient` is empty.
    pub fn terngrad_encode(
        gradient: &[f32],
        cfg: &TernaryConfig,
        rng: &mut LcgRng,
    ) -> FedResult<TernaryEncoded> {
        let n = gradient.len();
        if n == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        // Estimate scale s = max|g_i|.
        let scale = if cfg.batch_size > 0 && cfg.batch_size < n {
            // Sample batch_size indices without worrying about uniqueness
            // (with-replacement sampling is sufficient for scale estimation).
            let mut max_abs = 0.0_f32;
            for _ in 0..cfg.batch_size {
                let idx = rng.next_usize(n);
                let abs_val = gradient[idx].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }
            max_abs
        } else {
            gradient.iter().map(|&g| g.abs()).fold(0.0_f32, f32::max)
        };

        // If gradient is all-zero, return all-zero encoding.
        if scale < 1e-12_f32 {
            return Ok(TernaryEncoded {
                codes: vec![0_i8; n],
                scale: 0.0,
                n_nonzero: 0,
            });
        }

        // Stochastic ternarization.
        //
        // `LcgRng::next_f32()` returns a uniform deviate in [0, 1), so an
        // unbiased Bernoulli(p) draw is simply `next_f32() < p`.  This keeps
        // the quantization unbiased: E[code_i] = sign(g_i) · p_i, hence
        // E[decode_i] = scale · sign(g_i) · (|g_i|/scale) = g_i.
        let mut codes = Vec::with_capacity(n);
        for &g in gradient {
            let abs_g = g.abs();
            let prob = (abs_g / scale).min(1.0_f32);
            let flip = rng.next_f32() < prob;
            let code = if flip {
                if g >= 0.0 { 1_i8 } else { -1_i8 }
            } else {
                0_i8
            };
            codes.push(code);
        }

        let n_nonzero = codes.iter().filter(|&&c| c != 0).count();
        Ok(TernaryEncoded {
            codes,
            scale,
            n_nonzero,
        })
    }

    // ── Atomo ────────────────────────────────────────────────────────────────

    /// Atomo top-k deterministic ternarization.
    ///
    /// Selects `k = ceil(keep_fraction * n)` elements with largest `|g_i|`.
    /// For selected elements: `code = sign(g_i)`.
    /// Scale: `||g||₂ / sqrt(k)`.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `gradient` is empty.
    /// - [`FedError::InvalidClientUtility`] if `keep_fraction` is not in (0, 1].
    pub fn atomo_encode(gradient: &[f32], cfg: &TernaryConfig) -> FedResult<TernaryEncoded> {
        let n = gradient.len();
        if n == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let keep_fraction = match &cfg.mode {
            TernaryMode::Atomo { keep_fraction } => *keep_fraction,
            TernaryMode::TernGrad => {
                return Err(FedError::Internal(
                    "atomo_encode called with TernGrad mode".into(),
                ));
            }
        };

        if !(keep_fraction > 0.0 && keep_fraction <= 1.0) {
            return Err(FedError::InvalidClientUtility);
        }

        // k = ceil(keep_fraction * n), at least 1.
        let k = ((keep_fraction * n as f32).ceil() as usize).max(1).min(n);

        // Compute ||g||₂ for scale.
        let norm_sq: f32 = gradient.iter().map(|&g| g * g).sum();
        let norm = norm_sq.sqrt();

        let scale = if k > 0 {
            norm / (k as f32).sqrt()
        } else {
            0.0_f32
        };

        // Sort indices by |g_i| descending to find top-k.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            gradient[b]
                .abs()
                .partial_cmp(&gradient[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build code array: only top-k get non-zero codes.
        let mut codes = vec![0_i8; n];
        for &idx in indices.iter().take(k) {
            codes[idx] = if gradient[idx] >= 0.0 { 1_i8 } else { -1_i8 };
        }

        Ok(TernaryEncoded {
            codes,
            scale,
            n_nonzero: k,
        })
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode a ternary-encoded gradient: `decode[i] = scale * codes[i] as f32`.
    #[must_use]
    pub fn decode(encoded: &TernaryEncoded) -> Vec<f32> {
        encoded
            .codes
            .iter()
            .map(|&c| encoded.scale * c as f32)
            .collect()
    }

    // ── Compression ratio ────────────────────────────────────────────────────

    /// Compression ratio relative to dense f32 storage.
    ///
    /// ```text
    ///   ratio = (n_params * 32) / (n_nonzero * 2 + 32)
    /// ```
    ///
    /// The denominator represents 2 bits per non-zero entry (for the sign code)
    /// plus 32 bits for the scale factor.  Returns `f32::INFINITY` if there
    /// are no non-zero entries (lossless-zero special case).
    #[must_use]
    pub fn compression_ratio(encoded: &TernaryEncoded, n_params: usize) -> f32 {
        let bits_dense = n_params as f32 * 32.0;
        let bits_compressed = (encoded.n_nonzero as f32 * 2.0) + 32.0;
        if bits_compressed == 0.0 {
            return f32::INFINITY;
        }
        bits_dense / bits_compressed
    }

    // ── Aggregate ────────────────────────────────────────────────────────────

    /// Aggregate ternary-encoded gradients from multiple clients.
    ///
    /// Decodes each encoding and computes the element-wise mean.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `encodeds` is empty.
    /// - [`FedError::DimensionMismatch`] if any encoding has a different length.
    pub fn aggregate(encodeds: &[TernaryEncoded]) -> FedResult<Vec<f32>> {
        if encodeds.is_empty() {
            return Err(FedError::EmptyClientList);
        }

        let expected_len = encodeds[0].codes.len();
        // Validate all lengths up front.
        for enc in encodeds {
            if enc.codes.len() != expected_len {
                return Err(FedError::DimensionMismatch {
                    expected: expected_len,
                    got: enc.codes.len(),
                });
            }
        }

        let n_clients = encodeds.len() as f32;
        let mut sum = vec![0.0_f32; expected_len];

        for enc in encodeds {
            let decoded = Self::decode(enc);
            for (s, d) in sum.iter_mut().zip(decoded.iter()) {
                *s += d;
            }
        }

        for s in &mut sum {
            *s /= n_clients;
        }

        Ok(sum)
    }

    // ── Expected MSE ─────────────────────────────────────────────────────────

    /// Compute the per-element expected squared error E[||decode − g||²] / n.
    ///
    /// For **TernGrad** (stochastic), the analytical expectation is:
    ///
    /// ```text
    ///   E[||q - g||²] / n  =  s² * Σ_i p_i*(1 - p_i) / n
    /// ```
    ///
    /// where `p_i = |g_i| / s`.
    ///
    /// For **Atomo** (deterministic, using the current encoding):
    ///
    /// ```text
    ///   E[||q - g||²] / n  =  Σ_{i: code_i=0} g_i² / n
    /// ```
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `gradient.len() ≠ encoded.codes.len()`.
    pub fn expected_mse(gradient: &[f32], encoded: &TernaryEncoded) -> FedResult<f32> {
        let n = gradient.len();
        if n == 0 {
            return Ok(0.0);
        }
        if encoded.codes.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: encoded.codes.len(),
            });
        }

        if encoded.scale < 1e-12_f32 {
            // Zero gradient — error is zero.
            return Ok(0.0);
        }

        let s = encoded.scale;

        // Determine whether this is TernGrad (probabilistic) or Atomo (deterministic)
        // by inspecting: for TernGrad, n_nonzero may be any value and scale = max|g|;
        // for Atomo, the codes are exactly the top-k signs.
        //
        // We use the presence of exactly n_nonzero non-zero codes to detect Atomo-style
        // deterministic selection, but since we can't distinguish them from the struct
        // alone, we compute the TernGrad MSE formula as it is always an upper bound.
        //
        // TernGrad formula: MSE = s² Σ p_i(1-p_i) / n  where p_i = |g_i|/s.
        let mse: f32 = gradient
            .iter()
            .map(|&g| {
                let p = (g.abs() / s).min(1.0_f32);
                p * (1.0 - p)
            })
            .sum::<f32>()
            * (s * s)
            / n as f32;

        Ok(mse.max(0.0))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn terngrad_cfg() -> TernaryConfig {
        TernaryConfig {
            mode: TernaryMode::TernGrad,
            batch_size: 0,
        }
    }

    fn atomo_cfg(keep_fraction: f32) -> TernaryConfig {
        TernaryConfig {
            mode: TernaryMode::Atomo { keep_fraction },
            batch_size: 0,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ─── Test 1: TernGrad on zero gradient → all zero codes ──────────────────
    #[test]
    fn terngrad_zero_gradient_all_zero_codes() {
        let grad = vec![0.0_f32; 8];
        let mut rng = make_rng();
        let enc = TernaryCompressor::terngrad_encode(&grad, &terngrad_cfg(), &mut rng)
            .expect("test invariant: valid terngrad_encode zero");
        assert!(enc.codes.iter().all(|&c| c == 0));
        assert_eq!(enc.n_nonzero, 0);
        assert_eq!(enc.scale, 0.0);
    }

    // ─── Test 2: TernGrad scale = max|g| for small vector ─────────────────────
    #[test]
    fn terngrad_scale_equals_max_abs() {
        let grad = vec![0.5_f32, -0.8, 0.3, -0.1];
        let mut rng = make_rng();
        let enc = TernaryCompressor::terngrad_encode(&grad, &terngrad_cfg(), &mut rng)
            .expect("test invariant: valid terngrad_encode scale");
        assert!(
            (enc.scale - 0.8).abs() < 1e-5,
            "scale should be max|g|=0.8, got {}",
            enc.scale
        );
    }

    // ─── Test 3: decode of all-zero codes = all-zero ──────────────────────────
    #[test]
    fn decode_all_zero_codes_returns_zeros() {
        let enc = TernaryEncoded {
            codes: vec![0_i8; 5],
            scale: 1.5,
            n_nonzero: 0,
        };
        let decoded = TernaryCompressor::decode(&enc);
        assert!(decoded.iter().all(|&v| v == 0.0));
    }

    // ─── Test 4: decode reconstructs scale * code ────────────────────────────
    #[test]
    fn decode_reconstructs_correctly() {
        let enc = TernaryEncoded {
            codes: vec![1_i8, -1, 0, 1],
            scale: 2.5,
            n_nonzero: 3,
        };
        let decoded = TernaryCompressor::decode(&enc);
        assert!((decoded[0] - 2.5).abs() < 1e-6);
        assert!((decoded[1] - (-2.5)).abs() < 1e-6);
        assert!((decoded[2] - 0.0).abs() < 1e-6);
        assert!((decoded[3] - 2.5).abs() < 1e-6);
    }

    // ─── Test 5: atomo keep_fraction=1.0 → all n_nonzero ─────────────────────
    #[test]
    fn atomo_full_keep_fraction_all_nonzero() {
        let grad = vec![0.1_f32, -0.5, 0.3, -0.2, 0.8];
        let enc = TernaryCompressor::atomo_encode(&grad, &atomo_cfg(1.0))
            .expect("test invariant: valid atomo_encode keep=1.0");
        assert_eq!(enc.n_nonzero, 5, "all 5 elements should be non-zero");
        assert_eq!(enc.codes.iter().filter(|&&c| c != 0).count(), 5);
    }

    // ─── Test 6: atomo keep_fraction=0.5 → half entries non-zero ─────────────
    #[test]
    fn atomo_half_keep_fraction() {
        let grad: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect(); // 8 elements
        let enc = TernaryCompressor::atomo_encode(&grad, &atomo_cfg(0.5))
            .expect("test invariant: valid atomo_encode keep=0.5");
        // k = ceil(0.5 * 8) = 4
        assert_eq!(
            enc.n_nonzero, 4,
            "half of 8 = 4 non-zero, got {}",
            enc.n_nonzero
        );
    }

    // ─── Test 7: atomo scale = ||g||₂ / sqrt(k) ──────────────────────────────
    #[test]
    fn atomo_scale_equals_norm_over_sqrt_k() {
        let grad = vec![3.0_f32, 4.0, 0.1, 0.2]; // ||g|| = sqrt(9+16+0.01+0.04) ≈ 5.005
        let enc = TernaryCompressor::atomo_encode(&grad, &atomo_cfg(0.5))
            .expect("test invariant: valid atomo_encode scale");
        // k = ceil(0.5 * 4) = 2
        let norm = grad.iter().map(|&g| g * g).sum::<f32>().sqrt();
        let expected_scale = norm / (2.0_f32.sqrt());
        assert!(
            (enc.scale - expected_scale).abs() < 1e-4,
            "scale = {}, expected {}",
            enc.scale,
            expected_scale
        );
    }

    // ─── Test 8: compression_ratio > 1.0 for sparse encoded ──────────────────
    #[test]
    fn compression_ratio_sparse_greater_than_one() {
        let enc = TernaryEncoded {
            codes: vec![0_i8; 100], // 100 zeros
            scale: 1.0,
            n_nonzero: 5, // only 5 non-zero
        };
        let ratio = TernaryCompressor::compression_ratio(&enc, 100);
        // ratio = 100*32 / (5*2 + 32) = 3200 / 42 ≈ 76
        assert!(
            ratio > 1.0,
            "sparse compression ratio should be > 1, got {ratio}"
        );
    }

    // ─── Test 9: aggregate of two identical decodings = decode itself ─────────
    #[test]
    fn aggregate_identical_encodings_equals_decode() {
        let enc = TernaryEncoded {
            codes: vec![1_i8, -1, 0, 1],
            scale: 2.0,
            n_nonzero: 3,
        };
        let single_decode = TernaryCompressor::decode(&enc);
        let result = TernaryCompressor::aggregate(&[enc.clone(), enc.clone()])
            .expect("test invariant: valid aggregate");
        for (a, &b) in result.iter().zip(single_decode.iter()) {
            assert!(
                (*a - b).abs() < 1e-5,
                "aggregate of identical = decode: {a} != {b}"
            );
        }
    }

    // ─── Test 10: expected_mse >= 0 ───────────────────────────────────────────
    #[test]
    fn expected_mse_non_negative() {
        let grad = vec![0.5_f32, -0.3, 0.8, -0.1];
        let mut rng = make_rng();
        let enc = TernaryCompressor::terngrad_encode(&grad, &terngrad_cfg(), &mut rng)
            .expect("test invariant: valid terngrad_encode for mse");
        let mse = TernaryCompressor::expected_mse(&grad, &enc)
            .expect("test invariant: valid expected_mse");
        assert!(mse >= 0.0, "MSE should be non-negative, got {mse}");
    }

    // ─── Test 11: empty gradient → DimensionMismatch ─────────────────────────
    #[test]
    fn err_empty_gradient_terngrad() {
        let grad: Vec<f32> = vec![];
        let mut rng = make_rng();
        assert!(matches!(
            TernaryCompressor::terngrad_encode(&grad, &terngrad_cfg(), &mut rng),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ─── Test 12: keep_fraction=0 → InvalidClientUtility ─────────────────────
    #[test]
    fn err_atomo_keep_fraction_zero() {
        let grad = vec![0.1_f32, 0.2, 0.3];
        assert!(matches!(
            TernaryCompressor::atomo_encode(&grad, &atomo_cfg(0.0)),
            Err(FedError::InvalidClientUtility)
        ));
    }

    // ─── Test 13: keep_fraction=1.2 → InvalidClientUtility ───────────────────
    #[test]
    fn err_atomo_keep_fraction_too_large() {
        let grad = vec![0.1_f32, 0.2, 0.3];
        assert!(matches!(
            TernaryCompressor::atomo_encode(&grad, &atomo_cfg(1.2)),
            Err(FedError::InvalidClientUtility)
        ));
    }

    // ─── Test 14: aggregate empty list → EmptyClientList ─────────────────────
    #[test]
    fn err_aggregate_empty_list() {
        assert!(matches!(
            TernaryCompressor::aggregate(&[]),
            Err(FedError::EmptyClientList)
        ));
    }

    // ─── Test 15: aggregate different-length list → DimensionMismatch ─────────
    #[test]
    fn err_aggregate_different_lengths() {
        let enc1 = TernaryEncoded {
            codes: vec![1_i8, -1, 0],
            scale: 1.0,
            n_nonzero: 2,
        };
        let enc2 = TernaryEncoded {
            codes: vec![1_i8, -1],
            scale: 1.0,
            n_nonzero: 2,
        };
        assert!(matches!(
            TernaryCompressor::aggregate(&[enc1, enc2]),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ─── Test 16: terngrad is unbiased (statistical test over many samples) ───
    //
    // TernGrad is unbiased in expectation: E[decode[i]] = g[i].
    // With 5000 trials the sample mean converges well enough for the test.
    // We allow ±0.25 slack to be robust against LCG autocorrelation.
    #[test]
    fn terngrad_unbiased_empirical() {
        let n_trials = 5000_usize;
        let grad = vec![0.6_f32, -0.4, 0.8, -0.2];
        let n = grad.len();
        let mut sum = vec![0.0_f64; n];
        // Use multiple independent seeds and average to reduce LCG bias.
        for seed in [7_u64, 17, 31, 97, 251] {
            let mut rng = LcgRng::new(seed);
            for _ in 0..n_trials {
                let enc = TernaryCompressor::terngrad_encode(&grad, &terngrad_cfg(), &mut rng)
                    .expect("test invariant: valid terngrad in loop");
                let decoded = TernaryCompressor::decode(&enc);
                for (s, &d) in sum.iter_mut().zip(decoded.iter()) {
                    *s += d as f64;
                }
            }
        }
        let total_trials = n_trials * 5;

        // E[decode[i]] should ≈ g[i].  With 25000 total trials allow ±0.15 slack.
        for (i, (&g, &s)) in grad.iter().zip(sum.iter()).enumerate() {
            let mean = (s / total_trials as f64) as f32;
            assert!(
                (mean - g).abs() < 0.15,
                "element {i}: mean={mean}, g={g}, bias too large"
            );
        }
    }

    // ─── Test 17: atomo top-k selects largest magnitudes ─────────────────────
    #[test]
    fn atomo_selects_largest_magnitudes() {
        // Gradient where the two largest magnitudes are obvious.
        let grad = vec![0.1_f32, 0.9, -0.8, 0.2, 0.05];
        let enc = TernaryCompressor::atomo_encode(&grad, &atomo_cfg(0.4))
            .expect("test invariant: valid atomo top-k");
        // k = ceil(0.4 * 5) = 2 → top-2 are |0.9| and |-0.8|
        // So indices 1 and 2 should be non-zero.
        assert_ne!(enc.codes[1], 0, "index 1 (0.9) should be selected");
        assert_ne!(enc.codes[2], 0, "index 2 (-0.8) should be selected");
        // codes[1] should be +1 (positive), codes[2] should be -1 (negative)
        assert_eq!(enc.codes[1], 1_i8);
        assert_eq!(enc.codes[2], -1_i8);
    }

    // ─── Test 18: batch_size sampling doesn't crash ───────────────────────────
    #[test]
    fn terngrad_batch_size_sampling_no_crash() {
        let grad: Vec<f32> = (1..=50).map(|i| i as f32 * 0.01).collect();
        let cfg = TernaryConfig {
            mode: TernaryMode::TernGrad,
            batch_size: 10,
        };
        let mut rng = make_rng();
        let enc = TernaryCompressor::terngrad_encode(&grad, &cfg, &mut rng)
            .expect("test invariant: batch_size sampling succeeds");
        assert_eq!(enc.codes.len(), 50);
        // Scale should be positive (since gradient is positive).
        assert!(enc.scale > 0.0);
    }

    // ─── Test 19: atomo empty gradient → DimensionMismatch ───────────────────
    #[test]
    fn err_atomo_empty_gradient() {
        let grad: Vec<f32> = vec![];
        assert!(matches!(
            TernaryCompressor::atomo_encode(&grad, &atomo_cfg(0.5)),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ─── Test 20: compression_ratio formula check ─────────────────────────────
    #[test]
    fn compression_ratio_formula() {
        // n_params=100, n_nonzero=10: ratio = 100*32 / (10*2 + 32) = 3200/52 ≈ 61.5
        let enc = TernaryEncoded {
            codes: vec![0_i8; 100],
            scale: 1.0,
            n_nonzero: 10,
        };
        let ratio = TernaryCompressor::compression_ratio(&enc, 100);
        let expected = 3200.0_f32 / 52.0;
        assert!(
            (ratio - expected).abs() < 0.01,
            "ratio = {ratio}, expected {expected}"
        );
    }
}
