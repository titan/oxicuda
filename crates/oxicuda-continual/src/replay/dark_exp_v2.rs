//! Enhanced DER++ with fused loss kernel (DER V2).
//!
//! Extends the original Dark Experience Replay (DER++) with:
//! - A **consistency regulariser** based on KL divergence between stored and
//!   current logit distributions (temperature-scaled).
//! - A **fused single-pass** loss that accumulates MSE, CE and KL without
//!   materialising intermediate buffers per term.
//!
//! Loss formula for a replayed mini-batch of size B:
//!
//! ```text
//!   L = α · (1/B) Σ MSE(z_cur_i, z_stored_i)
//!     + β · (1/B) Σ CE(z_cur_i, y_i)
//!     + γ · (1/B) Σ KL(softmax(z_stored_i / T) ‖ softmax(z_cur_i))
//! ```
//!
//! The buffer uses Vitter's reservoir sampling (identical to `er.rs` and
//! `dark_exp.rs`) so no external RNG crate is required.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the enhanced DER++ (V2) buffer.
#[derive(Debug, Clone)]
pub struct DerV2Config {
    /// Weight for logit MSE distillation term (α).
    pub alpha: f64,
    /// Weight for cross-entropy on stored labels (β).
    pub beta: f64,
    /// Weight for KL-divergence consistency regulariser (γ).
    pub gamma: f64,
    /// Temperature for the stored logit distribution in KL (T).
    pub temperature: f64,
    /// Maximum number of samples the replay buffer holds.
    pub capacity: usize,
}

impl Default for DerV2Config {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 1.0,
            gamma: 0.1,
            temperature: 2.0,
            capacity: 256,
        }
    }
}

impl DerV2Config {
    /// Validate configuration parameters.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.alpha < 0.0 || !self.alpha.is_finite() {
            return Err(ContinualError::InvalidLambda {
                lambda: self.alpha as f32,
            });
        }
        if self.beta < 0.0 || !self.beta.is_finite() {
            return Err(ContinualError::InvalidLambda {
                lambda: self.beta as f32,
            });
        }
        if self.gamma < 0.0 || !self.gamma.is_finite() {
            return Err(ContinualError::InvalidLambda {
                lambda: self.gamma as f32,
            });
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(ContinualError::InvalidLambda {
                lambda: self.temperature as f32,
            });
        }
        if self.capacity == 0 {
            return Err(ContinualError::BufferCapacityTooSmall);
        }
        Ok(())
    }
}

// ─── Replay buffer ────────────────────────────────────────────────────────────

/// Enhanced DER++ replay buffer storing features, teacher logits, and labels.
#[derive(Debug, Clone)]
pub struct DerV2Buffer {
    /// Stored input features.
    pub samples: Vec<Vec<f64>>,
    /// Stored teacher logits at the time the sample was first inserted.
    pub logits: Vec<Vec<f64>>,
    /// Stored class labels.
    pub labels: Vec<usize>,
    /// Maximum buffer capacity.
    pub capacity: usize,
    /// Total number of samples seen (for reservoir sampling book-keeping).
    pub n_seen: usize,
}

impl DerV2Buffer {
    /// Create a new empty buffer.
    pub fn new(config: &DerV2Config) -> ContinualResult<Self> {
        config.validate()?;
        Ok(Self {
            samples: Vec::with_capacity(config.capacity),
            logits: Vec::with_capacity(config.capacity),
            labels: Vec::with_capacity(config.capacity),
            capacity: config.capacity,
            n_seen: 0,
        })
    }

    /// Add a sample using Vitter's reservoir sampling (Algorithm R).
    ///
    /// - While the buffer is not full the sample is always inserted.
    /// - Once full a random slot is replaced with probability `capacity / (n_seen + 1)`.
    pub fn add(&mut self, features: Vec<f64>, logits: Vec<f64>, label: usize, rng: &mut LcgRng) {
        let n = self.n_seen;
        if n < self.capacity {
            self.samples.push(features);
            self.logits.push(logits);
            self.labels.push(label);
        } else {
            let slot = rng.next_usize(n + 1);
            if slot < self.capacity {
                self.samples[slot] = features;
                self.logits[slot] = logits;
                self.labels[slot] = label;
            }
        }
        self.n_seen += 1;
    }

    /// Compute the fused DER V2 loss over the entire buffer.
    ///
    /// `current_logits[i]` must correspond to `self.samples[i]` — the caller is
    /// responsible for running the current model on each stored sample and
    /// passing the resulting logits here.
    ///
    /// Returns `0.0` if the buffer is empty.
    ///
    /// # Errors
    /// Returns [`ContinualError::DimensionMismatch`] if `current_logits` length
    /// differs from the buffer size, or if any logit vector has a different
    /// number of classes than the stored logits.
    pub fn fused_loss(
        &self,
        current_logits: &[Vec<f64>],
        config: &DerV2Config,
    ) -> ContinualResult<f64> {
        config.validate()?;

        if self.is_empty() {
            return Ok(0.0);
        }

        let buf_len = self.len();
        if current_logits.len() != buf_len {
            return Err(ContinualError::DimensionMismatch {
                expected: buf_len,
                got: current_logits.len(),
            });
        }

        let mut total_mse = 0.0_f64;
        let mut total_ce = 0.0_f64;
        let mut total_kl = 0.0_f64;

        let stored_logits_iter = self.logits.iter();
        let stored_labels_iter = self.labels.iter();

        for ((z_cur, z_sto), &label) in current_logits
            .iter()
            .zip(stored_logits_iter)
            .zip(stored_labels_iter)
        {
            let n_cls = z_sto.len();

            if z_cur.len() != n_cls {
                return Err(ContinualError::DimensionMismatch {
                    expected: n_cls,
                    got: z_cur.len(),
                });
            }

            // ── MSE term ──────────────────────────────────────────────────────
            let mse: f64 = z_cur
                .iter()
                .zip(z_sto.iter())
                .map(|(&c, &s)| (c - s).powi(2))
                .sum::<f64>()
                / n_cls as f64;

            // ── Cross-entropy term ────────────────────────────────────────────
            if label >= n_cls {
                return Err(ContinualError::TaskIndexOutOfRange {
                    index: label,
                    n_tasks: n_cls,
                });
            }
            let probs_cur = softmax_temperature(z_cur, 1.0); // standard softmax
            let p_label = probs_cur[label].max(1e-30);
            let ce = -p_label.ln();

            // ── KL divergence term ────────────────────────────────────────────
            // p = softmax(z_stored / T)   (teacher / target distribution)
            // q = softmax(z_current)      (student distribution)
            // KL(p ‖ q) = Σ p_i · log(p_i / q_i)
            let p_teacher = softmax_temperature(z_sto, config.temperature);
            let kl = kl_divergence(&p_teacher, &probs_cur);

            total_mse += mse;
            total_ce += ce;
            total_kl += kl;
        }

        let b = buf_len as f64;
        let loss = config.alpha * (total_mse / b)
            + config.beta * (total_ce / b)
            + config.gamma * (total_kl / b);

        if !loss.is_finite() {
            return Err(ContinualError::NanEncountered {
                location: "DerV2Buffer::fused_loss",
            });
        }

        Ok(loss)
    }

    /// Number of samples currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True if the buffer holds no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Maximum number of samples this buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Softmax with temperature scaling.
///
/// `p_i = exp((z_i − max_z) / T) / Σ_j exp((z_j − max_z) / T)`
///
/// Numerical stability: subtract the maximum before exponentiation.
pub(crate) fn softmax_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let temp = temperature.max(1e-12);
    let max_z = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| ((z - max_z) / temp).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence KL(p ‖ q) = Σ_i p_i · ln(p_i / q_i).
///
/// Convention: `0 · ln 0 = 0`.  `p_i / q_i` is clamped away from zero to
/// avoid −∞ contributions when q_i = 0 and p_i ≈ 0.
pub(crate) fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-30 {
                0.0
            } else {
                pi * (pi / qi.max(1e-30)).ln()
            }
        })
        .sum()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(1234)
    }

    fn uniform_logits(n: usize) -> Vec<f64> {
        vec![0.0_f64; n]
    }

    // ── Test 1: Buffer capacity is strictly respected ─────────────────────────

    #[test]
    fn buffer_capacity_respected() {
        let mut rng = make_rng();
        let cfg = DerV2Config {
            capacity: 8,
            ..Default::default()
        };
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        for i in 0..50_usize {
            buf.add(vec![i as f64], vec![0.0, 1.0], i % 2, &mut rng);
        }
        assert_eq!(buf.len(), 8);
        assert_eq!(buf.n_seen, 50);
    }

    // ── Test 2: fused_loss is finite and non-negative ─────────────────────────

    #[test]
    fn fused_loss_finite_and_non_negative() {
        let mut rng = make_rng();
        let cfg = DerV2Config::default();
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        buf.add(vec![0.5, -0.5], vec![1.2, -0.3], 0, &mut rng);
        buf.add(vec![-0.3, 0.7], vec![-0.1, 0.8], 1, &mut rng);

        let cur = vec![vec![0.9_f64, -0.1], vec![0.2_f64, 0.6]];
        let loss = buf
            .fused_loss(&cur, &cfg)
            .expect("DER v2 fused loss should compute with valid inputs");
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    // ── Test 3: alpha=beta=gamma=0 → loss = 0 ────────────────────────────────

    #[test]
    fn all_weights_zero_gives_zero_loss() {
        let mut rng = make_rng();
        let cfg = DerV2Config {
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            ..Default::default()
        };
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        buf.add(vec![1.0], vec![0.5, -0.5], 0, &mut rng);

        let cur = vec![vec![0.3_f64, 0.7]];
        let loss = buf
            .fused_loss(&cur, &cfg)
            .expect("DER v2 fused loss should compute with valid inputs");
        assert!(
            loss.abs() < 1e-12,
            "zero-weight loss should be 0, got {loss}"
        );
    }

    // ── Test 4: alpha=1, beta=0, gamma=0 → MSE only ──────────────────────────

    #[test]
    fn alpha_only_gives_mse_loss() {
        let mut rng = make_rng();
        let cfg = DerV2Config {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            ..Default::default()
        };
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        let stored = vec![1.0_f64, 0.0];
        buf.add(vec![0.0], stored.clone(), 0, &mut rng);

        let current = vec![0.0_f64, 1.0]; // differs from stored
        let expected_mse = ((1.0_f64 - 0.0).powi(2) + (0.0_f64 - 1.0).powi(2)) / 2.0;
        let loss = buf
            .fused_loss(&[current], &cfg)
            .expect("DER v2 fused loss should compute with valid inputs");
        assert!(
            (loss - expected_mse).abs() < 1e-10,
            "MSE-only loss should equal expected_mse={expected_mse}, got {loss}"
        );
    }

    // ── Test 5: beta=1, alpha=0, gamma=0 → CE only ───────────────────────────

    #[test]
    fn beta_only_gives_ce_loss() {
        let mut rng = make_rng();
        let cfg = DerV2Config {
            alpha: 0.0,
            beta: 1.0,
            gamma: 0.0,
            ..Default::default()
        };
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        buf.add(vec![0.0], uniform_logits(3), 1, &mut rng);

        let logits = vec![0.0_f64, 10.0, 0.0]; // confident in class 1
        // CE = -log(softmax(logits)[1]) ≈ 0 (very confident)
        let loss = buf
            .fused_loss(&[logits], &cfg)
            .expect("DER v2 fused loss should compute with valid inputs");
        assert!(
            loss >= 0.0 && loss.is_finite(),
            "CE-only loss should be valid"
        );
        assert!(
            loss < 0.1,
            "Confident prediction should have small CE, got {loss}"
        );
    }

    // ── Test 6: KL divergence of identical distributions = 0 ─────────────────

    #[test]
    fn kl_divergence_identical_is_zero() {
        let p = vec![0.2_f64, 0.5, 0.3];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-12, "KL(p‖p) should be 0, got {kl}");
    }

    // ── Test 7: KL divergence with very different distributions is large ──────

    #[test]
    fn kl_divergence_different_distributions_is_large() {
        // p = [1, 0, 0], q = [0, 0, 1] (numerically regularised)
        let p = vec![1.0 - 1e-10, 0.5e-10, 0.5e-10];
        let q = vec![0.5e-10, 0.5e-10, 1.0 - 1e-10];
        let kl = kl_divergence(&p, &q);
        assert!(
            kl > 10.0,
            "KL of near-orthogonal distributions should be large, got {kl}"
        );
    }

    // ── Test 8: Higher temperature softens the distribution ───────────────────

    #[test]
    fn higher_temperature_gives_more_uniform_distribution() {
        let logits = vec![3.0_f64, 0.0, -3.0];

        let p_low = softmax_temperature(&logits, 0.5);
        let p_high = softmax_temperature(&logits, 5.0);

        // Max probability decreases with higher temperature.
        let max_low: f64 = p_low.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_high: f64 = p_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_high < max_low,
            "High temperature should give smaller max prob: {max_low} vs {max_high}"
        );
    }

    // ── Test 9: Temperature=1 equals standard softmax ─────────────────────────

    #[test]
    fn temperature_one_equals_standard_softmax() {
        let logits = vec![1.0_f64, 2.0, 3.0];
        let p1 = softmax_temperature(&logits, 1.0);

        // Manual standard softmax.
        let max_z = 3.0_f64;
        let exps: Vec<f64> = logits.iter().map(|&z| (z - max_z).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let p_std: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

        for (a, b) in p1.iter().zip(p_std.iter()) {
            assert!((a - b).abs() < 1e-12, "T=1 softmax mismatch: {a} vs {b}");
        }
    }

    // ── Test 10: Empty buffer fused_loss returns 0 ────────────────────────────

    #[test]
    fn empty_buffer_fused_loss_returns_zero() {
        let cfg = DerV2Config::default();
        let buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        let loss = buf
            .fused_loss(&[], &cfg)
            .expect("DER v2 fused loss should compute with valid inputs");
        assert_eq!(loss, 0.0, "empty buffer should return loss=0");
    }

    // ── Test 11: Reservoir sampling keeps capacity ────────────────────────────

    #[test]
    fn reservoir_sampling_keeps_capacity() {
        let mut rng = make_rng();
        let cfg = DerV2Config {
            capacity: 10,
            ..Default::default()
        };
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        for i in 0..200_usize {
            buf.add(vec![i as f64], vec![0.0, 1.0, 2.0], i % 3, &mut rng);
        }
        assert_eq!(buf.len(), 10, "buffer must not exceed capacity");
        assert_eq!(buf.n_seen, 200);
    }

    // ── Test 12: Invalid alpha returns error ──────────────────────────────────

    #[test]
    fn invalid_alpha_returns_error() {
        let cfg = DerV2Config {
            alpha: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err(), "negative alpha should be an error");
    }

    // ── Test 13: Logit count mismatch returns error ───────────────────────────

    #[test]
    fn logit_count_mismatch_returns_error() {
        let mut rng = make_rng();
        let cfg = DerV2Config::default();
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        buf.add(vec![0.0], vec![0.5, 0.5], 0, &mut rng);

        // current_logits has 3 classes but stored has 2
        let cur = vec![vec![0.3_f64, 0.4, 0.3]];
        assert!(
            buf.fused_loss(&cur, &cfg).is_err(),
            "mismatched logit dims should return error"
        );
    }

    // ── Test 14: Current logits batch length mismatch returns error ───────────

    #[test]
    fn batch_length_mismatch_returns_error() {
        let mut rng = make_rng();
        let cfg = DerV2Config::default();
        let mut buf =
            DerV2Buffer::new(&cfg).expect("DER v2 buffer should initialize with valid config");
        buf.add(vec![0.0], vec![0.5, 0.5], 0, &mut rng);
        buf.add(vec![1.0], vec![0.6, 0.4], 1, &mut rng);

        // Only 1 current logit row for 2 buffer entries.
        let cur = vec![vec![0.5_f64, 0.5]];
        assert!(
            buf.fused_loss(&cur, &cfg).is_err(),
            "wrong batch length should return error"
        );
    }
}
