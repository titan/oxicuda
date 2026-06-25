//! Label-smoothing cross-entropy regularisation — Szegedy et al., 2016.
//!
//! "Rethinking the Inception Architecture for Computer Vision"
//! (arXiv:1512.00567), §7.
//!
//! Hard one-hot targets encourage a classifier to drive the logit of the
//! correct class to `+∞` and all others to `−∞`, which over-confidently
//! over-fits.  **Label smoothing** replaces the one-hot target with a soft
//! distribution that mixes in a uniform prior `1/K` (over `K` classes):
//!
//! ```text
//! q_k = (1 − α)·[k == y] + α / K
//! ```
//!
//! The loss is the cross-entropy between this smoothed target and the
//! softmax of the logits:
//!
//! ```text
//! p = softmax(z)
//! L = − Σ_k q_k · log p_k
//! ```
//!
//! Because the cross-entropy is taken against a soft target, its gradient with
//! respect to the logits has the familiar closed form `∂L/∂z_k = p_k − q_k`,
//! which this module computes directly (numerically stable via the log-sum-exp
//! trick).  Setting `α = 0` recovers ordinary cross-entropy exactly.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`LabelSmoothingCrossEntropy`].
#[derive(Debug, Clone)]
pub struct LabelSmoothingConfig {
    /// Smoothing strength `α ∈ [0, 1)`.  `0` is plain cross-entropy.
    pub smoothing: f64,
    /// Number of classes `K` (must be ≥ 2).
    pub num_classes: usize,
}

impl LabelSmoothingConfig {
    /// Create and validate a configuration.
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] if `smoothing ∉ [0, 1)` or `num_classes < 2`.
    pub fn new(smoothing: f64, num_classes: usize) -> TrainResult<Self> {
        if !(0.0..1.0).contains(&smoothing) {
            return Err(TrainError::Internal {
                msg: format!("smoothing must be in [0, 1), got {smoothing}"),
            });
        }
        if num_classes < 2 {
            return Err(TrainError::Internal {
                msg: format!("num_classes must be >= 2, got {num_classes}"),
            });
        }
        Ok(Self {
            smoothing,
            num_classes,
        })
    }
}

// ─── Loss ─────────────────────────────────────────────────────────────────────

/// Label-smoothing cross-entropy loss operating on raw logit rows.
#[derive(Debug, Clone)]
pub struct LabelSmoothingCrossEntropy {
    config: LabelSmoothingConfig,
}

impl LabelSmoothingCrossEntropy {
    /// Build the loss from a validated configuration.
    #[must_use]
    pub fn new(config: LabelSmoothingConfig) -> Self {
        Self { config }
    }

    /// Number of classes `K`.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.config.num_classes
    }

    /// Smoothing coefficient `α`.
    #[must_use]
    pub fn smoothing(&self) -> f64 {
        self.config.smoothing
    }

    /// Numerically stable softmax of `logits` into `out` (both length `K`),
    /// returning the log-sum-exp normaliser for reuse.
    fn softmax_into(logits: &[f64], out: &mut [f64]) -> f64 {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (o, &z) in out.iter_mut().zip(logits.iter()) {
            let e = (z - max).exp();
            *o = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
        max + sum.ln()
    }

    /// Compute the label-smoothing cross-entropy loss for one example.
    ///
    /// * `logits` – raw scores `z ∈ ℝ^K`.
    /// * `target` – ground-truth class index `y ∈ [0, K)`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ShapeMismatch`] if `logits.len() != K`.
    /// * [`TrainError::Internal`] if `target >= K`.
    pub fn loss(&self, logits: &[f32], target: usize) -> TrainResult<f64> {
        self.check(logits, target)?;
        let k = self.config.num_classes;
        let alpha = self.config.smoothing;
        let logits64: Vec<f64> = logits.iter().map(|&z| f64::from(z)).collect();
        // log-softmax via the lse trick: log p_k = z_k − lse.
        let max = logits64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits64.iter().map(|&z| (z - max).exp()).sum::<f64>().ln();
        // L = −Σ q_k log p_k, q_k = (1−α)[k=y] + α/K.
        let uniform = alpha / k as f64;
        let mut loss = 0.0;
        for (idx, &z) in logits64.iter().enumerate() {
            let log_p = z - lse;
            let q = if idx == target {
                (1.0 - alpha) + uniform
            } else {
                uniform
            };
            loss -= q * log_p;
        }
        Ok(loss)
    }

    /// Compute the loss **and** write `∂L/∂z = p − q` into `grad` for one
    /// example.
    ///
    /// # Errors
    ///
    /// As [`LabelSmoothingCrossEntropy::loss`], plus [`TrainError::ShapeMismatch`]
    /// if `grad.len() != K`.
    pub fn loss_and_grad(
        &self,
        logits: &[f32],
        target: usize,
        grad: &mut [f32],
    ) -> TrainResult<f64> {
        self.check(logits, target)?;
        if grad.len() != self.config.num_classes {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.config.num_classes],
                got: vec![grad.len()],
            });
        }
        let k = self.config.num_classes;
        let alpha = self.config.smoothing;
        let uniform = alpha / k as f64;
        let logits64: Vec<f64> = logits.iter().map(|&z| f64::from(z)).collect();
        let mut p = vec![0.0_f64; k];
        let lse = Self::softmax_into(&logits64, &mut p);
        let mut loss = 0.0;
        for idx in 0..k {
            let log_p = logits64[idx] - lse;
            let q = if idx == target {
                (1.0 - alpha) + uniform
            } else {
                uniform
            };
            loss -= q * log_p;
            grad[idx] = (p[idx] - q) as f32;
        }
        Ok(loss)
    }

    /// Mean loss over a batch of `n` examples laid out row-major in
    /// `logits` (`n × K`) with class indices `targets` (length `n`).
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `targets` is empty.
    /// * [`TrainError::ShapeMismatch`] if `logits.len() != n·K`.
    /// * [`TrainError::Internal`] if any target is out of range.
    pub fn batch_loss(&self, logits: &[f32], targets: &[usize]) -> TrainResult<f64> {
        let n = targets.len();
        if n == 0 {
            return Err(TrainError::EmptyParams);
        }
        let k = self.config.num_classes;
        if logits.len() != n * k {
            return Err(TrainError::ShapeMismatch {
                expected: vec![n * k],
                got: vec![logits.len()],
            });
        }
        let mut total = 0.0;
        for (i, &t) in targets.iter().enumerate() {
            total += self.loss(&logits[i * k..(i + 1) * k], t)?;
        }
        Ok(total / n as f64)
    }

    /// Validate logits length and target range.
    fn check(&self, logits: &[f32], target: usize) -> TrainResult<()> {
        if logits.len() != self.config.num_classes {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.config.num_classes],
                got: vec![logits.len()],
            });
        }
        if target >= self.config.num_classes {
            return Err(TrainError::Internal {
                msg: format!(
                    "target {target} out of range for {} classes",
                    self.config.num_classes
                ),
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn ce(smoothing: f64, k: usize) -> LabelSmoothingCrossEntropy {
        LabelSmoothingCrossEntropy::new(
            LabelSmoothingConfig::new(smoothing, k).expect("valid config"),
        )
    }

    #[test]
    fn rejects_bad_config() {
        assert!(LabelSmoothingConfig::new(1.0, 5).is_err());
        assert!(LabelSmoothingConfig::new(-0.1, 5).is_err());
        assert!(LabelSmoothingConfig::new(0.1, 1).is_err());
    }

    /// With α=0 the loss equals plain cross-entropy `−log p_y`.
    #[test]
    fn alpha_zero_is_plain_ce() {
        let loss = ce(0.0, 3);
        let logits = [1.0_f32, 2.0, 0.5];
        // softmax then −log p_1.
        let m = 2.0_f64;
        let denom: f64 = logits.iter().map(|&z| (f64::from(z) - m).exp()).sum();
        let log_p1 = (f64::from(logits[1]) - m) - denom.ln();
        let expect = -log_p1;
        let got = loss.loss(&logits, 1).expect("ok");
        assert!((got - expect).abs() < 1e-12, "got {got} vs {expect}");
    }

    /// For uniform logits the loss equals −log(1/K) regardless of α (the
    /// smoothed target's mass on every coordinate sees the same log p = −log K).
    #[test]
    fn uniform_logits_loss() {
        let k = 4;
        let loss = ce(0.1, k);
        let logits = vec![0.0_f32; k];
        let got = loss.loss(&logits, 2).expect("ok");
        let expect = (k as f64).ln(); // −log(1/K)
        assert!((got - expect).abs() < 1e-12, "got {got} vs {expect}");
    }

    /// Smoothing raises the loss of a confident correct prediction (it now
    /// penalises over-confidence).
    #[test]
    fn smoothing_increases_confident_loss() {
        let logits = [10.0_f32, 0.0, 0.0, 0.0];
        let hard = ce(0.0, 4).loss(&logits, 0).expect("ok");
        let soft = ce(0.2, 4).loss(&logits, 0).expect("ok");
        assert!(soft > hard, "smoothed {soft} should exceed hard {hard}");
    }

    /// The analytic gradient `p − q` matches a central finite-difference of the
    /// loss for several random logit rows.
    #[test]
    fn gradient_matches_finite_difference() {
        let k = 5;
        let loss = ce(0.15, k);
        let mut rng = LcgRng::new(31);
        for _ in 0..10 {
            let logits: Vec<f32> = (0..k).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
            let target = (rng.next_u32() as usize) % k;
            let mut grad = vec![0.0_f32; k];
            loss.loss_and_grad(&logits, target, &mut grad).expect("ok");
            let h = 1e-4_f32;
            for j in 0..k {
                let mut plus = logits.clone();
                let mut minus = logits.clone();
                plus[j] += h;
                minus[j] -= h;
                let lp = loss.loss(&plus, target).expect("ok");
                let lm = loss.loss(&minus, target).expect("ok");
                let fd = ((lp - lm) / (2.0 * f64::from(h))) as f32;
                assert!(
                    (grad[j] - fd).abs() < 2e-2,
                    "grad[{j}] = {} vs fd {fd}",
                    grad[j]
                );
            }
        }
    }

    /// The gradient sums to (approximately) zero — both p and q are
    /// probability distributions, so Σ(p−q) = 0.
    #[test]
    fn gradient_sums_to_zero() {
        let k = 6;
        let loss = ce(0.1, k);
        let logits: Vec<f32> = (0..k).map(|i| i as f32 * 0.3).collect();
        let mut grad = vec![0.0_f32; k];
        loss.loss_and_grad(&logits, 3, &mut grad).expect("ok");
        let s: f32 = grad.iter().sum();
        assert!(s.abs() < 1e-5, "gradient should sum to ~0, got {s}");
    }

    #[test]
    fn batch_loss_is_mean() {
        let k = 3;
        let loss = ce(0.1, k);
        let row0 = [1.0_f32, 2.0, 0.0];
        let row1 = [0.0_f32, 1.0, 3.0];
        let l0 = loss.loss(&row0, 1).expect("ok");
        let l1 = loss.loss(&row1, 2).expect("ok");
        let mut logits = Vec::new();
        logits.extend_from_slice(&row0);
        logits.extend_from_slice(&row1);
        let batch = loss.batch_loss(&logits, &[1, 2]).expect("ok");
        assert!((batch - 0.5 * (l0 + l1)).abs() < 1e-12);
    }

    #[test]
    fn out_of_range_target_errors() {
        let loss = ce(0.1, 3);
        assert!(matches!(
            loss.loss(&[0.0, 0.0, 0.0], 5),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn wrong_logits_len_errors() {
        let loss = ce(0.1, 3);
        assert!(matches!(
            loss.loss(&[0.0, 0.0], 1),
            Err(TrainError::ShapeMismatch { .. })
        ));
    }
}
